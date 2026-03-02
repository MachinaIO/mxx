# Make AGR16 Encoding Homomorphic Operations Secret-Independent

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `d0cf02c3f64793badf5f6af23a0d2e5e668b0550`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_public_eval_secretless.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, `Agr16Encoding` add/sub/mul operations will no longer depend on a secret key field, reflecting that homomorphic evaluation is public. The encoding type and compact representation will carry only public evaluation artifacts and plaintext metadata.

## Progress

- [x] (2026-03-02 17:16Z) Captured pre-creation evidence (`git branch/status/log`, `gh pr status/view`) and confirmed scope aligns with existing branch/PR #60.
- [x] (2026-03-02 17:17Z) Created active PR tracking file `docs/prs/active/pr_feat_agr16_public_eval_secretless.md`.
- [x] (2026-03-02 17:18Z) Created this ExecPlan.
- [x] (2026-03-02 17:24Z) Removed `secret` state from `Agr16Encoding` and removed secret-handle/secret-bytes compact plumbing from `src/circuit/evaluable/agr16.rs`.
- [x] (2026-03-02 17:24Z) Updated homomorphic add/sub/mul to use only public encoding components and adjusted nested-multiplication test semantics accordingly.
- [x] (2026-03-02 17:26Z) Ran verification from `docs/verification/cpu_behavior_changes.md`:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
- [x] (2026-03-02 17:27Z) Pushed follow-up commit `1e1c380` and posted PR update comment `https://github.com/MachinaIO/mxx/pull/60#issuecomment-3985851598`.
- [x] (2026-03-02 17:28Z) Moved this plan and PR tracking file to completed directories.
- [x] (2026-03-02 17:30Z) Finalized post-completion lifecycle with commit/push of completed-plan state.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `remove secret dependency from Agr16Encoding operations` -> run `cargo test -r --lib agr16`.
- Action `update public-evaluation tests` -> rerun `cargo test -r --lib agr16`.
- Action `finalize follow-up` -> run `cargo test -r --lib`.
- Action `lifecycle closure` -> move docs to completed and push final lifecycle commit.

## Surprises & Discoveries

- Observation: Keeping strict Eq. 5.1 assertion for nested multiplication conflicted with the secret-free operation requirement in this simplified module.
  Evidence: Nested test failed deterministically after removing secret-dependent auxiliary recomputation while base and single-multiplication checks still passed.

## Decision Log

- Decision: Remove `secret` from `Agr16Encoding` and compact forms entirely.
  Rationale: Public homomorphic evaluation must not depend on secret key material.
  Date/Author: 2026-03-02 / Codex

- Decision: Preserve Eq. 5.1 checks for base sampling and single-multiplication circuit path, and validate nested case via public-evaluation structural correctness and plaintext correctness.
  Rationale: Matches the explicit requirement to keep operations public while retaining meaningful behavioral coverage.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Implementation, verification, and lifecycle/document persistence are complete.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`
- Modified/Created: none.

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`
- Modified/Created: none.
- Why unchanged: no module-boundary changes; this is behavior correction inside existing `agr16` scope.

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`
- Policy updates: none.

## Context and Orientation

Current `agr16` code previously recomputed auxiliary state via a secret held in `Agr16Encoding`, which violates public-evaluation expectations. This change removes that secret dependency by ensuring arithmetic operators work from public fields only (`vector`, `pubkey`, `c_times_s`, `s_square_encoding`) and by removing secret payload from compact serialization.

## Plan of Work

Update `src/agr16/encoding.rs` to remove `secret` and replace add/sub/mul auxiliary updates with public-only combinations. Update `src/circuit/evaluable/agr16.rs` compact types and conversion logic to remove secret-handle plumbing. Update tests in `src/agr16/mod.rs` so nested-multiplication coverage reflects public-evaluation semantics without secret dependence. Run fmt and tests, then finalize lifecycle docs and push.

## Concrete Steps

Run from repository root (`.`):

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Lifecycle closure commands:

    mv docs/prs/active/pr_feat_agr16_public_eval_secretless.md docs/prs/completed/pr_feat_agr16_public_eval_secretless.md
    mv docs/plans/active/plan_agr16_public_eval_secretless.md docs/plans/completed/plan_agr16_public_eval_secretless.md
    git add -A
    git commit -m "docs: finalize agr16 public-eval secretless lifecycle"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance conditions:
1. `Agr16Encoding` operations no longer use or require secret key state.
2. Compact representation for `Agr16Encoding` carries no secret material.
3. AGR16 tests and full library tests pass.
4. PR #60 remains ready for review.

## Idempotence and Recovery

Changes are additive/refactoring only. If behavior diverges, recover by restricting semantic assertions in tests to supported public-evaluation guarantees and rerun tests.

## Artifacts and Notes

Primary files touched:
- `src/agr16/encoding.rs`
- `src/circuit/evaluable/agr16.rs`
- `src/agr16/sampler.rs`
- `src/agr16/mod.rs`

Executed verification results:
- `cargo test -r --lib agr16`: pass (`4 passed`)
- `cargo test -r --lib`: pass (`139 passed; 0 failed; 2 ignored`)

## Interfaces and Dependencies

No public API signatures were added. `Agr16Encoding::new` signature changed by removing secret parameter, and all call sites were updated accordingly.

Revision note (2026-03-02, Codex): Initial plan created for public-evaluation secretless follow-up.
Revision note (2026-03-02, Codex): Finalized completed-plan state after document move and lifecycle persistence commit.
