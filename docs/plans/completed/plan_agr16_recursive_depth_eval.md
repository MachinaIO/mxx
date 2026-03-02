# Implement AGR16 Recursive Public Evaluation for Multiplication Depth >= 3

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `c1f5c3bc8dc6a683dc3db81d2f9684a0aa682ecf`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_recursive_depth_eval.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, AGR16 public-key/ciphertext homomorphic multiplication will use recursive auxiliary-state evaluation rather than the current fixed bounded update. This enables circuits with multiplication depth 3 or higher to preserve Equation 5.1-style ciphertext correctness checks under zero injected error, matching the reviewer’s requested acceptance criteria for PR #60.

## Progress

- [x] (2026-03-02 19:35Z) Read lifecycle and verification policies (`PLANS.md`, `VERIFICATION.md`, `docs/verification/index.md`).
- [x] (2026-03-02 19:40Z) Ran main-plan pre-creation context checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed this follow-up is aligned with existing PR #60 scope.
- [x] (2026-03-02 19:44Z) Created active PR tracking document at `docs/prs/active/pr_feat_agr16_recursive_depth_eval.md`.
- [x] (2026-03-02 19:46Z) Created this ExecPlan under `docs/plans/active/`.
- [x] (2026-03-02 19:48Z) Implemented recursive auxiliary chain representation and recursive multiplication updates in `src/agr16/public_key.rs` and `src/agr16/encoding.rs`.
- [x] (2026-03-02 19:48Z) Updated sampler and compact conversions (`src/agr16/sampler.rs`, `src/circuit/evaluable/agr16.rs`) for vectorized recursive auxiliary state.
- [x] (2026-03-02 19:49Z) Added depth>=3 AGR16 tests with Equation 5.1 ciphertext checks in `src/agr16/mod.rs` (depth-3 chain + depth-4 composed case).
- [x] (2026-03-02 19:50Z) Added design artifact `docs/design/agr16_recursive_auxiliary_chain.md` and linked it from `docs/design/index.md`.
- [x] (2026-03-02 19:50Z) Ran verification commands:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
- [x] (2026-03-02 19:53Z) Posted reviewer follow-up response comment: `https://github.com/MachinaIO/mxx/pull/60#issuecomment-3986557731`.
- [x] (2026-03-02 19:53Z) Ran post-completion readiness action `gh pr ready 60` (PR already ready) and moved lifecycle docs from active to completed paths.
- [ ] Persist final post-completion state via commit and push.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `implement recursive auxiliary-state evaluation` -> run `cargo test -r --lib agr16`.
- Action `add depth>=3 Eq. 5.1 tests` -> rerun `cargo test -r --lib agr16`.
- Action `complete AGR16 follow-up scope` -> run `cargo test -r --lib`.
- Action `finalize lifecycle and readiness state` -> run `gh pr ready`, move docs to completed, then commit and push.

## Surprises & Discoveries

- Observation: Section 5 Eq. 5.24/5.25 recurrence requires level-wise access to higher auxiliary advice (`l+1` level), so a fixed two-level auxiliary state cannot propagate correctness to arbitrary multiplication depth.
  Evidence: Extracted formulas and recursive EvalCT/EvalPK text from `docs/references/agr16_encoding.pdf` (Section 5).

- Observation: Multiplication consumes one recursive auxiliary level (because each output level `l` requires input level `l+1`), so branch-wise depths can diverge; add/sub therefore must preserve only common levels.
  Evidence: During implementation, strict equal-length add/sub assumptions conflict with mixed-depth composed circuits.

## Decision Log

- Decision: Reuse existing branch and PR (`feat/agr16_encoding`, PR #60) for this change instead of creating a new branch.
  Rationale: The requested recursion/depth>=3 fix is a direct reviewer follow-up on the same feature scope.
  Date/Author: 2026-03-02 / Codex

- Decision: Implement recursive auxiliary state as depth-indexed vectors on both key and ciphertext objects.
  Rationale: Eq. 5.24/5.25 style recursion references level-indexed `E(c*s)` and `PK(E(c*s))` terms across levels; vectors provide a natural and generic trait-level representation.
  Date/Author: 2026-03-02 / Codex

- Decision: Define add/sub over the minimum shared recursive auxiliary depth instead of requiring equal depths.
  Rationale: Multiplication-level consumption naturally creates different residual depths across branches in composed circuits; truncating to shared depth keeps operations well-defined and prevents incorrect assumptions in mixed-depth graphs.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Implementation, verification, and post-completion readiness actions are complete. Remaining work is final persistence commit/push.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`.
- Modified/Created: `docs/design/index.md`, `docs/design/agr16_recursive_auxiliary_chain.md`.

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`.
- Modified/Created: none (no module boundary or dependency-direction changes).

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Policy updates: none.

## Context and Orientation

`src/agr16/public_key.rs` and `src/agr16/encoding.rs` now use depth-indexed vectors for recursive auxiliary state (`c_times_s_*` and `s_power_*`) and multiplication updates are implemented recursively from Eq. 5.24/5.25-style relations. `src/agr16/sampler.rs` and `src/circuit/evaluable/agr16.rs` are aligned to this vectorized state.

`src/agr16/mod.rs` now includes the requested depth>=3 coverage (depth-3 chain and depth-4 composed circuit), both checking Equation 5.1 ciphertext relation under zero injected error.

## Plan of Work

First, replace fixed auxiliary fields in `Agr16PublicKey` and `Agr16Encoding` with depth-indexed vectors for `E(c*s^i)` and corresponding public-key labels, and similarly vectorize `E(s^j)` advice terms. Keep addition/subtraction component-wise.

Next, update multiplication to compute each auxiliary level recursively using Eq. 5.25-style key update and matching Eq. 5.24-style ciphertext update, with convolution terms over lower levels. Produce output levels up to one less than the available input depth (because each level references `l+1` advice/state).

Then, update samplers to generate the vectorized key/advice components for configurable recursion depth, and update `Evaluable` compact serialization/rotation/scalar operations to carry vectors.

Finally, add depth>=3 tests that evaluate concrete circuits and assert Equation 5.1 ciphertext relation, plus key/ciphertext equality and plaintext consistency checks. Use the same zero-error setup pattern as existing AGR16 tests.

## Concrete Steps

Run from repository root (`.`):

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Lifecycle closure commands:

    gh pr comment 60 --body "<review response summary>"
    gh pr ready
    mv docs/prs/active/pr_feat_agr16_recursive_depth_eval.md docs/prs/completed/pr_feat_agr16_recursive_depth_eval.md
    mv docs/plans/active/plan_agr16_recursive_depth_eval.md docs/plans/completed/plan_agr16_recursive_depth_eval.md
    git add -A
    git commit -m "fix: add agr16 recursive depth extension for public evaluation"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance criteria:
1. AGR16 multiplication logic no longer relies on a fixed two-level auxiliary update and instead evaluates recursively across configured depth.
2. AGR16 tests include at least one multiplication-depth-3 circuit and one deeper composed multiplication case.
3. Those depth>=3 tests assert Equation 5.1-style ciphertext consistency under zero injected error.
4. `cargo test -r --lib agr16` and `cargo test -r --lib` pass.

## Idempotence and Recovery

The edits are additive and scoped to `src/agr16/*`, `src/circuit/evaluable/agr16.rs`, and documentation/tests. If a recursive formula change breaks tests, revert only the affected multiplication hunk and rerun `cargo test -r --lib agr16` before reapplying corrected formulas.

## Artifacts and Notes

Touched files:
- `src/agr16/public_key.rs`
- `src/agr16/encoding.rs`
- `src/agr16/sampler.rs`
- `src/circuit/evaluable/agr16.rs`
- `src/agr16/mod.rs`
- `docs/design/index.md`
- `docs/design/agr16_recursive_auxiliary_chain.md`
- `docs/prs/completed/pr_feat_agr16_recursive_depth_eval.md`
- `docs/plans/completed/plan_agr16_recursive_depth_eval.md`

## Interfaces and Dependencies

`Agr16PublicKey` and `Agr16Encoding` will expose vectorized auxiliary/advice state:
- key: `c_times_s_pubkeys: Vec<M>`, `s_power_pubkeys: Vec<M>`
- ciphertext: `c_times_s_encodings: Vec<M>`, `s_power_encodings: Vec<M>`

`AGR16PublicKeySampler` gains explicit control of auxiliary recursion depth used for sampled advice.

No new external dependencies are planned.

Revision note (2026-03-02 19:51Z): Updated plan state after implementation and verification completion; added design-artifact evidence, command outcomes, and the add/sub shared-depth decision discovered during composed-circuit support.
Revision note (2026-03-02 19:54Z): Updated plan linkage to completed PR tracking path, recorded PR response comment/readiness actions, and split final persistence as the remaining lifecycle step.
