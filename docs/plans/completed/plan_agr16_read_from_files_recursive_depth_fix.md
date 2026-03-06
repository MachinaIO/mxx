# Fix AGR16 read_from_files to Support Recursive Auxiliary Depth

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `9843a7c801b76f31fb05b73f55dc8c31231fd74b`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_read_from_files_depth_fix.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, `Agr16PublicKey::read_from_files` will load recursive auxiliary levels consistently with the current vector-based AGR16 model, instead of hardcoding two levels. Persisted keys for depth > 2 can be reconstructed through the public API.

## Progress

- [x] (2026-03-03 00:05Z) Read latest reviewer comment and identified target finding in `src/agr16/public_key.rs` (`read_from_files` hardcodes 2 levels).
- [x] (2026-03-03 00:08Z) Ran pre-creation context checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed branch/PR scope alignment.
- [x] (2026-03-03 00:08Z) Created active PR tracking file `docs/prs/active/pr_feat_agr16_read_from_files_depth_fix.md`.
- [x] (2026-03-03 00:09Z) Created this ExecPlan.
- [x] (2026-03-03 00:11Z) Updated `Agr16PublicKey::read_from_files` to accept `recursive_depth` and load recursive vector levels with legacy-name fallback for level 0/1.
- [x] (2026-03-03 00:11Z) Added file-loading tests in `src/agr16/mod.rs`:
  - `test_agr16_pubkey_read_from_files_supports_recursive_depth`
  - `test_agr16_pubkey_read_from_files_supports_legacy_two_level_names`
- [x] (2026-03-03 00:12Z) Ran verification commands:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
- [x] (2026-03-03 00:14Z) Posted reviewer follow-up response comment: `https://github.com/MachinaIO/mxx/pull/60#issuecomment-3987779322`.
- [x] (2026-03-03 00:14Z) Ran post-completion readiness action `gh pr ready 60` (already ready) and moved plan/PR tracking docs to completed.
- [x] (2026-03-03 00:16Z) Persisted final post-completion state with commit `253dd55` and push `feat/agr16_encoding -> origin/feat/agr16_encoding`.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `implement read_from_files recursive depth support` -> run `cargo test -r --lib agr16`.
- Action `add/read_from_files tests` -> rerun `cargo test -r --lib agr16`.
- Action `complete follow-up scope` -> run `cargo test -r --lib`.
- Action `finalize lifecycle` -> run `gh pr ready`, move docs to completed, commit, push.

## Surprises & Discoveries

- Observation: Current `read_from_files` uses legacy fixed IDs (`cts_pk`/`ctss_pk`, `s2_pk`/`s2s_pk`) and does not encode recursive depth in its interface.
  Evidence: `src/agr16/public_key.rs`.

- Observation: Matrix-file block-size naming differs between matrix implementations (`dcrt` uses configured block size; GPU path can use compacted size), so existence checks must support both naming conventions.
  Evidence: `src/matrix/dcrt_poly.rs` vs `src/matrix/gpu_dcrt_poly.rs`.

## Decision Log

- Decision: Keep this follow-up on PR #60 and current branch.
  Rationale: The requested fix is a direct reviewer finding in the same feature scope.
  Date/Author: 2026-03-03 / Codex

- Decision: Keep backward compatibility by falling back to legacy level IDs (`cts_pk`/`ctss_pk`, `s2_pk`/`s2s_pk`) when recursive level files are absent for level 0/1.
  Rationale: Existing persisted two-level artifacts should remain readable while enabling recursive-depth persisted keys.
  Date/Author: 2026-03-03 / Codex

## Outcomes & Retrospective

Completed. `read_from_files` now supports recursive-depth persisted keys (with legacy two-level compatibility), reviewer response is posted, and lifecycle evidence is persisted in commit `253dd55`.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`, `docs/design/agr16_recursive_auxiliary_chain.md`.
- Modified/Created: none (no design contract change; this aligns existing persistence API with the already-defined recursive design).

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`.
- Modified/Created: none (no boundary/dependency change).

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Policy updates: none.

## Context and Orientation

AGR16 moved from fixed auxiliary fields to recursive vectors in key/encoding state. Sampler and tests already use configurable depth, but persisted-file loading in `Agr16PublicKey::read_from_files` still loads exactly two levels. This creates API inconsistency for depth > 2 persisted keys.

## Plan of Work

Change `Agr16PublicKey::read_from_files` to accept a recursive auxiliary depth and read vector levels in a loop. For compatibility with historical 2-level files, keep fallback IDs for the first two levels when explicit recursive level filenames are absent.

Add unit tests that generate temporary matrix files, then assert the method can load:
1. recursive naming (`*_cts_pk_{level}`, `*_s_power_pk_{level}`) for depth > 2,
2. legacy two-level naming for depth 2 compatibility.

## Concrete Steps

Run from repository root (`.`):

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Lifecycle closure commands:

    gh pr comment 60 --body "<review response summary>"
    gh pr ready
    mv docs/prs/active/pr_feat_agr16_read_from_files_depth_fix.md docs/prs/completed/pr_feat_agr16_read_from_files_depth_fix.md
    mv docs/plans/active/plan_agr16_read_from_files_recursive_depth_fix.md docs/plans/completed/plan_agr16_read_from_files_recursive_depth_fix.md
    git add -A
    git commit -m "fix: align agr16 read_from_files with recursive depth model"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance criteria:
1. `read_from_files` does not hardcode two levels and supports recursive depth loading.
2. Legacy two-level persisted naming remains loadable.
3. `cargo test -r --lib agr16` and `cargo test -r --lib` pass.

## Idempotence and Recovery

Changes are scoped to AGR16 key loading and tests. If loading logic fails on a naming branch, retry after isolating ID resolution helper tests before changing arithmetic logic.

## Artifacts and Notes

Expected touched files:
- `src/agr16/public_key.rs`
- `src/agr16/mod.rs`
- `docs/prs/completed/pr_feat_agr16_read_from_files_depth_fix.md`
- `docs/plans/completed/plan_agr16_read_from_files_recursive_depth_fix.md`

## Interfaces and Dependencies

Public interface impact:
- `Agr16PublicKey::read_from_files` gains an explicit recursive depth parameter.

No external dependencies are added.

Revision note (2026-03-03 00:12Z): Updated plan with implemented code/test changes, verification outcomes, matrix block-size naming discovery, and legacy compatibility decision.
Revision note (2026-03-03 00:14Z): Updated completed-path linkage, recorded PR response/readiness actions, and split final persistence as the remaining lifecycle step.
Revision note (2026-03-03 00:16Z): Recorded final commit/push evidence and marked lifecycle fully completed.
