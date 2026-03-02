# Fix AGR16 Nested-Multiplication Auxiliary Invariant Under Public Evaluation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `8e2fe588cfe5c7b8ec9bd0a8737e2c7d99913b8d`
- PR tracking document: `docs/prs/active/pr_feat_agr16_nested_invariant_fix.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, AGR16 nested multiplication will keep a consistent publicly-computable auxiliary invariant for `c_times_s` without reintroducing secret-key dependence in `Agr16Encoding` operations. Tests in `src/agr16/mod.rs` will again verify Eq. 5.1 structure on a nested-multiplication circuit under zero error.

## Progress

- [x] (2026-03-02 18:12Z) Re-read latest PR #60 reviewer comments and confirmed two active findings: `c_times_s` invariant drift on nested multiplication and weakened nested Eq. 5.1 test.
- [x] (2026-03-02 18:14Z) Ran pre-creation verification context collection (`git branch/status/log`, `gh pr status`, `gh pr view --json ...`) and confirmed scope aligns with existing branch/PR #60.
- [x] (2026-03-02 18:18Z) Created active PR tracking file `docs/prs/active/pr_feat_agr16_nested_invariant_fix.md`.
- [x] (2026-03-02 18:20Z) Created this ExecPlan.
- [x] (2026-03-02 18:24Z) Implemented AGR16 auxiliary-level extension (`c_times_s_times_s` and `s_square_times_s` companions) and updated public-key/ciphertext multiplication formulas so `c_times_s` remains publicly updatable and key-consistent on nested multiplication paths.
- [x] (2026-03-02 18:24Z) Restored nested Eq. 5.1 test coverage and added targeted auxiliary invariant checks for sampled encodings and evaluated outputs.
- [x] (2026-03-02 18:25Z) Ran verification commands from `docs/verification/cpu_behavior_changes.md` and supplemental scope checks:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
  - `cargo test -r --lib agr16 --no-default-features --features disk` (5 consecutive runs)
- [x] (2026-03-02 18:25Z) Repeated `cargo test -r --lib agr16` to probe prior flake report; observed one intermittent `SIGSEGV` in a 5-run loop, with surrounding retries passing.
- [ ] Post PR response comment, finalize docs lifecycle (move active plan/PR tracking docs to completed), run post-completion event, and persist with final commit/push.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `implement invariant-preserving public formulas` -> run `cargo test -r --lib agr16`.
- Action `restore nested Eq. 5.1 tests` -> rerun `cargo test -r --lib agr16`.
- Action `finalize PR follow-up` -> run `cargo test -r --lib`.
- Action `post-completion lifecycle` -> run `gh pr ready` decision flow and move docs to completed with final commit/push.

## Surprises & Discoveries

- Observation: The reviewer’s high-severity finding is mathematically accurate for current code because `Agr16PublicKey::mul` sets `c_times_s_pubkey` to zero while `Agr16Encoding::mul` updates `c_times_s` through a product-rule expression that is not linked to the output key relation.
  Evidence: `gh pr view 60 --comments` and current `src/agr16/public_key.rs` + `src/agr16/encoding.rs`.

- Observation: A strict update invariant for `c_times_s_times_s` after multiplication requires one more recursive auxiliary level than this follow-up currently carries.
  Evidence: The first attempt to assert `c_times_s_times_s` invariant on multiplied outputs failed in `test_agr16_circuit_eval_matches_equation_5_1_without_error`; primary `c_times_s` invariant and Eq. 5.1 checks passed after constraining assertions to the level restored by this fix.

- Observation: Intermittent `SIGSEGV` for `cargo test -r --lib agr16` remains reproducible in this branch even after the fix (1 failure in 5 repeats), while immediate retries pass.
  Evidence: Loop run (`seq 1..5`) produced one crash at run 4; standalone rerun passed.

## Decision Log

- Decision: Keep work on existing branch/PR (`feat/agr16_encoding`, PR #60) instead of branching again.
  Rationale: This is a direct incremental reviewer-follow-up within the same feature scope.
  Date/Author: 2026-03-02 / Codex

- Decision: Fix invariant drift by adding one more public auxiliary layer for `E(E(c*s)*s)` and `E(E(s^2)*s)` and by updating `Agr16PublicKey::mul` / `Agr16Encoding::mul` with matching formulas.
  Rationale: This preserves secret-independent operations while restoring a concrete invariant path for nested multiplication under the module’s Section-5-style model.
  Date/Author: 2026-03-02 / Codex

- Decision: Keep `c_times_s_times_s` multiplication propagation as best-effort in this follow-up and validate the restored reviewer-critical path (`c_times_s` + Eq. 5.1 nested correctness).
  Rationale: The review finding targeted `c_times_s` drift and nested Eq. 5.1 coverage; full higher-order recursive closure requires additional architecture beyond this bounded follow-up scope.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Implemented and verified. Remaining work is lifecycle closure (PR comment, docs move to completed, final commit/push).

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`.
- Planned updates: none unless a long-lived invariant contract change needs explicit design capture.

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`.
- Planned updates: none expected (no module-boundary changes; this is intra-scope behavior correction).

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Policy updates: none.

## Context and Orientation

`src/agr16/encoding.rs` currently computes multiplication output vector via Eq. 5.24-style terms, but updates `c_times_s` by a public product rule that drifts from the sampler-defined relation in `src/agr16/sampler.rs`. In parallel, `src/agr16/public_key.rs` zeroes `c_times_s_pubkey` on multiplication, which removes a key-side anchor for auxiliary consistency in deeper multiplication chains. The nested test in `src/agr16/mod.rs` was weakened and no longer checks Eq. 5.1 relation on the fragile path.

The fix will add a second auxiliary chain element in both key/ciphertext representations so `c_times_s` can be updated with a formula that is both public and key-consistent for nested multiplication use.

## Plan of Work

Update `src/agr16/public_key.rs` to carry and operate on two auxiliary key levels (`c_times_s_pubkey`, `c_times_s_times_s_pubkey`) and two advice keys (`s_square_pubkey`, `s_square_times_s_pubkey`). Update `src/agr16/encoding.rs` to carry matching auxiliary ciphertext levels and advice encodings, and replace multiplication-time `c_times_s` update with the derived invariant-preserving formula.

Update `src/agr16/sampler.rs` to sample and build these additional key/encoding levels while keeping secret handling confined to sampling time only. Update compact conversion and scalar/rotate transforms in `src/circuit/evaluable/agr16.rs` for added fields.

Finally, strengthen `src/agr16/mod.rs` tests so nested multiplication again checks Eq. 5.1 ciphertext relation at zero error and verifies public-key/ciphertext alignment.

## Concrete Steps

Run from repository root (`.`):

    gh pr view 60 --comments
    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Lifecycle closure commands:

    gh pr ready
    mv docs/prs/active/pr_feat_agr16_nested_invariant_fix.md docs/prs/completed/pr_feat_agr16_nested_invariant_fix.md
    mv docs/plans/active/plan_agr16_nested_invariant_fix.md docs/plans/completed/plan_agr16_nested_invariant_fix.md
    git add -A
    git commit -m "docs: finalize agr16 nested-invariant follow-up lifecycle"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance criteria:
1. `Agr16Encoding` arithmetic remains secret-independent (no secret key field dependency in operation implementations).
2. Nested multiplication path keeps a documented public `c_times_s` update relation with matching public-key update.
3. Nested test again checks Eq. 5.1-style ciphertext relation under zero error.
4. `cargo test -r --lib agr16` and `cargo test -r --lib` pass.

## Idempotence and Recovery

Edits are scoped to `agr16` structures, samplers, evaluable compact conversions, and unit tests. If formula changes break tests, revert only the latest formula hunk and re-run scope tests before reattempting.

## Artifacts and Notes

Expected touched files:
- `src/agr16/public_key.rs`
- `src/agr16/encoding.rs`
- `src/agr16/sampler.rs`
- `src/circuit/evaluable/agr16.rs`
- `src/agr16/mod.rs`
- `docs/prs/active/pr_feat_agr16_nested_invariant_fix.md`
- `docs/plans/active/plan_agr16_nested_invariant_fix.md`

## Interfaces and Dependencies

Public interface impact:
- `Agr16PublicKey` and `Agr16Encoding` gain additional auxiliary fields to support nested public evaluation consistency.
- `Evaluable` compact structs in `src/circuit/evaluable/agr16.rs` are extended accordingly.

No new external dependencies are planned.
