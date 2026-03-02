# Bootstrap Verification Meta-Rules and Verification Index

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md`. `DESIGN.md` does not exist in the current tree, so this plan uses repository documentation style in `PLANS.md` and `ARCHITECTURE.md` as fallback design baseline for tone and detail. `VERIFICATION.md` is being created by this plan.

## Purpose / Big Picture

After this change, the repository will have a dedicated verification meta-policy document (`VERIFICATION.md`) and a concrete verification documentation entrypoint (`docs/verification/index.md`). Agents will have explicit rules for where verification-event documents live, what to read first, and how to write actionable, executable verification steps.

## Progress

- [x] (2026-03-02 01:12Z) Read `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md` to align policy tone and structure.
- [x] (2026-03-02 01:13Z) Created `VERIFICATION.md` with required meta-rules, directory/index requirements, and actionability constraints.
- [x] (2026-03-02 01:13Z) Created `docs/verification/index.md` as mandatory index/entrypoint and added initial event mapping.
- [x] (2026-03-02 01:13Z) Added concrete event documents under `docs/verification/` with executable actions:
  - `event_docs_only_changes.md`
  - `event_cpu_behavior_changes.md`
  - `event_gpu_cuda_changes.md`
- [x] (2026-03-02 01:14Z) Validated references and file layout consistency.
- [x] (2026-03-02 01:14Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: `AGENTS.md` already requires reading `VERIFICATION.md`, but the file does not yet exist.
  Evidence: `AGENTS.md` has a `### Verification (VERIFICATION.md)` section while repository root has no `VERIFICATION.md`.

## Decision Log

- Decision: Create both `VERIFICATION.md` and `docs/verification/index.md` in one change.
  Rationale: The requested policy text requires `docs/verification/index.md` to be mandatory and first-read; creating only `VERIFICATION.md` would leave that requirement immediately unsatisfied.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Verification policy baseline is now present and operational. The repository has a root verification meta-rule document (`VERIFICATION.md`), a mandatory verification entry index (`docs/verification/index.md`), and initial event-specific verification runbooks with concrete commands and success criteria. This closes the prior policy gap where `AGENTS.md` required `VERIFICATION.md` but the file did not exist.

No source-code behavior changed; this update is documentation policy and runbook scaffolding.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none (no `DESIGN.md` exists).
- Created/modified: none.
- Why unchanged: this task defines verification policy, not design policy.

Architecture documents:

- Referenced: `ARCHITECTURE.md` (tone/detail baseline only).
- Created/modified: none.
- Why unchanged: no architecture-boundary change is introduced.

Verification documents:

- Referenced: none (target file missing at start).
- Created/modified:
  - `VERIFICATION.md`
  - `docs/verification/index.md`
  - `docs/verification/event_docs_only_changes.md`
  - `docs/verification/event_cpu_behavior_changes.md`
  - `docs/verification/event_gpu_cuda_changes.md`
- Why: establish first-class verification policy, mandatory docs index, and actionable event runbooks.

## Context and Orientation

The repository currently has explicit policy documents for planning (`PLANS.md`) and architecture (`ARCHITECTURE.md`), and `AGENTS.md` already instructs agents to read `VERIFICATION.md` for verification requirements. However, no verification policy document or verification documentation directory currently exists. This plan creates that missing policy layer and the required index file under `docs/verification/`.

## Plan of Work

Author a new root-level `VERIFICATION.md` that matches the strict, long-lived policy tone of existing meta-rule documents, and includes explicit rules requested by the user: verification event docs live in `docs/verification/`, filenames describe events clearly, `docs/verification/index.md` is mandatory and must be read first, and each verification document must contain concrete agent-executable actions. Then create `docs/verification/index.md` that functions as a decision index for which file to read per event class. Validate resulting structure and move the plan to completed.

## Concrete Steps

Run from `.`:

    mkdir -p docs/verification
    cat > VERIFICATION.md << '...'
    cat > docs/verification/index.md << '...'
    find docs/verification -maxdepth 2 -type f | sort
    sed -n '1,260p' VERIFICATION.md
    sed -n '1,260p' docs/verification/index.md

## Validation and Acceptance

Acceptance criteria:

1. Root-level `VERIFICATION.md` exists and defines verification meta-rules in repository-policy tone.
2. `VERIFICATION.md` states that verification-event docs live under `docs/verification/` and filenames must describe events clearly.
3. `VERIFICATION.md` states that `docs/verification/index.md` is mandatory and must be read first by agents.
4. `VERIFICATION.md` states that each verification document must contain concrete executable actions for agents.
5. `docs/verification/index.md` exists and works as an index that explains which file to read for which case.

## Idempotence and Recovery

This change is additive documentation only. Re-running creation commands is safe with file overwrite. If rollback is needed, remove or revert `VERIFICATION.md` and `docs/verification/` docs.

## Artifacts and Notes

Expected artifacts:

    VERIFICATION.md
    docs/verification/index.md

## Interfaces and Dependencies

No production code interfaces are changed. This plan introduces documentation interfaces only: policy entrypoint (`VERIFICATION.md`) and verification index entrypoint (`docs/verification/index.md`).

Revision note (2026-03-02, Codex): Initial plan created to bootstrap verification policy and index.
Revision note (2026-03-02, Codex): Updated progress/outcomes after creating verification policy, index, and initial event documents.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.
