# Align AGR16 Public Evaluation with Paper Semantics

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `d48a469`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_paper_alignment_followup.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `ARCHITECTURE.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, AGR16 ciphertext/public-key homomorphic multiplication will remain executable as a public operation even when wire plaintexts are not revealed. This aligns implementation behavior with paper semantics where evaluation uses public encodings/advice rather than requiring plaintext access.

## Progress

- [x] (2026-03-03 04:43Z) Reviewed current AGR16 implementation and identified plaintext-gated panic in `Agr16Encoding::mul` as a paper-semantic mismatch risk.
- [x] (2026-03-03 04:43Z) Ran pre-creation checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed this follow-up is aligned with PR #60.
- [x] (2026-03-03 04:43Z) Created active PR tracking document for this follow-up.
- [x] (2026-03-03 04:45Z) Removed plaintext-gated panic from `Agr16Encoding::mul` so ciphertext multiplication can run with hidden plaintext inputs.
- [x] (2026-03-03 04:45Z) Added regression test `test_agr16_mul_eval_works_without_revealed_plaintexts` to validate Eq. 5.1 ciphertext consistency and hidden output plaintext behavior.
- [x] (2026-03-03 04:46Z) Ran verification:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
- [x] (2026-03-03 04:47Z) Ran post-completion readiness check (`gh pr ready 60`) and moved plan/PR tracking docs to completed paths.
- [x] (2026-03-03 04:48Z) Persisted final lifecycle state with commit/push (`2b62c82`).

## Surprises & Discoveries

- Observation: `Agr16Encoding::mul` does not use plaintext values for ciphertext arithmetic, but still panics when left plaintext is hidden.
  Evidence: `src/agr16/encoding.rs` panic guard before arithmetic.

## Decision Log

- Decision: Treat plaintext-gated multiplication panic as the primary paper-alignment bug in this follow-up.
  Rationale: EvalCT in Section 5 is public and should not require plaintext reveal bits to compute ciphertext outputs.
  Date/Author: 2026-03-03 / Codex

## Outcomes & Retrospective

AGR16 multiplication now executes without plaintext reveal dependency, matching public-evaluation behavior expected by the paper-style EvalCT flow. A dedicated regression test now covers hidden-plaintext multiplication and keeps Eq. 5.1 ciphertext validation in place.

Completed-path lifecycle updates and persistence are done.

## Design/Architecture/Verification Document Summary

Design docs:
- Referenced: `DESIGN.md`.
- Modified/Created: none expected (localized behavior fix).

Architecture docs:
- Referenced: `ARCHITECTURE.md`.
- Modified/Created: none expected (no boundary/module layout changes).

Verification docs:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Policy updates: none expected.

## Context and Orientation

AGR16 arithmetic lives in `src/agr16/public_key.rs` and `src/agr16/encoding.rs`. The sampler in `src/agr16/sampler.rs` controls reveal flags (`reveal_plaintext`) for wire encodings. Tests in `src/agr16/mod.rs` currently use revealed plaintext inputs and therefore do not fail on plaintext-gated multiplication behavior.

## Plan of Work

Update `src/agr16/encoding.rs` so multiplication no longer panics on hidden plaintext and uses plaintext only for optional bookkeeping (`Option<P>` output).

Add a regression test in `src/agr16/mod.rs` that samples non-revealed inputs, evaluates a multiplication-containing circuit, checks Eq. 5.1 ciphertext consistency against known sampled plaintexts, and asserts output plaintext remains hidden.

## Concrete Steps

Run from repository root (`.`):

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

## Validation and Acceptance

Acceptance criteria:
1. AGR16 multiplication executes without requiring revealed plaintext on the left operand.
2. New non-revealed-input test passes and confirms ciphertext Eq. 5.1 relation.
3. Output plaintext stays `None` when multiplication combines hidden inputs.

## Idempotence and Recovery

Changes are local and additive. If a new test fails, adjust only AGR16 arithmetic/test files and re-run `cargo test -r --lib agr16` before broader verification.

## Artifacts and Notes

Expected touched files:
- `src/agr16/encoding.rs`
- `src/agr16/mod.rs`
- `docs/plans/completed/plan_agr16_paper_alignment_followup.md`
- `docs/prs/completed/pr_feat_agr16_paper_alignment_followup.md`

## Interfaces and Dependencies

No public API type changes expected.
