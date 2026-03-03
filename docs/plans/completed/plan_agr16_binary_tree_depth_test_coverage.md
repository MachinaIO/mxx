# Add AGR16 Complete Binary-Tree Multiplication Test Coverage

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `ce2f5d707aafa732e6c207b420b2bbc77f575664`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_binary_tree_test_coverage.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, AGR16 tests will include a complete binary-tree multiplication circuit at depth >= 3 and verify Equation 5.1 ciphertext consistency on its output. This closes the reviewer’s topology coverage gap beyond chain/composed-path tests.

## Progress

- [x] (2026-03-03 00:53Z) Read latest review comment and confirmed missing test coverage target: complete binary-tree multiplication depth >= 3.
- [x] (2026-03-03 00:58Z) Ran pre-creation context checks (`git branch --show-current`, `git status --short`, `git log --oneline --decorate --max-count=20`, `gh pr status`, `gh pr view --json ...`) and confirmed scope alignment with PR #60.
- [x] (2026-03-03 00:59Z) Created active PR tracking file `docs/prs/active/pr_feat_agr16_binary_tree_test_coverage.md`.
- [x] (2026-03-03 01:00Z) Created this ExecPlan.
- [x] (2026-03-03 01:03Z) Added complete binary-tree multiplication depth-3 test with Eq. 5.1 output consistency check in `src/agr16/mod.rs` (`test_agr16_complete_binary_tree_depth3_preserves_equation_5_1_without_error`).
- [x] (2026-03-03 01:05Z) Ran verification commands:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
- [x] (2026-03-03 01:08Z) Posted reviewer follow-up response comment: `https://github.com/MachinaIO/mxx/pull/60#issuecomment-3987936514`.
- [x] (2026-03-03 01:08Z) Ran post-completion readiness action `gh pr ready 60` (already ready) and moved plan/PR tracking docs to completed.
- [ ] Persist final post-completion state via commit and push.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `add binary-tree multiplication topology test` -> run `cargo test -r --lib agr16`.
- Action `complete follow-up scope` -> run `cargo test -r --lib`.
- Action `finalize lifecycle` -> run `gh pr ready`, move docs to completed, commit, push.

## Surprises & Discoveries

- Observation: Existing depth>=3 tests cover chain and mixed composed topology, but not balanced full binary multiplication fan-in.
  Evidence: Current tests in `src/agr16/mod.rs`.

## Decision Log

- Decision: Keep this fix to tests only, without changing AGR16 arithmetic formulas.
  Rationale: Review finding requests topology coverage gap closure, not behavior/formula change.
  Date/Author: 2026-03-03 / Codex

## Outcomes & Retrospective

Implementation, validation, and post-completion readiness actions are complete. Remaining work is final persistence commit/push.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`, `docs/design/agr16_recursive_auxiliary_chain.md`.
- Modified/Created: none (coverage-only follow-up).

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`.
- Modified/Created: none (no structural change).

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`.
- Policy updates: none.

## Context and Orientation

AGR16 already has Equation 5.1 tests for sampling, a depth-3 multiplication chain, and a depth-4 composed path. Reviewer requested one more topology: complete binary-tree multiplication at depth >= 3, which stresses fan-in balancing and auxiliary-level accounting differently from single-path-dominant circuits.

## Plan of Work

Add a new unit test in `src/agr16/mod.rs` that constructs a complete binary-tree multiplication circuit of depth 3 (8 leaves), evaluates both public keys and encodings, and checks Equation 5.1 output consistency using the existing helper assertion path.

Reuse existing fixture and helper assertions to keep consistency with the current validation style.

## Concrete Steps

Run from repository root (`.`):

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Lifecycle closure commands:

    gh pr comment 60 --body "<review response summary>"
    gh pr ready
    mv docs/prs/active/pr_feat_agr16_binary_tree_test_coverage.md docs/prs/completed/pr_feat_agr16_binary_tree_test_coverage.md
    mv docs/plans/active/plan_agr16_binary_tree_depth_test_coverage.md docs/plans/completed/plan_agr16_binary_tree_depth_test_coverage.md
    git add -A
    git commit -m "test: add agr16 complete binary-tree depth coverage"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance criteria:
1. New AGR16 unit test covers complete binary-tree multiplication depth >= 3.
2. The new test asserts Eq. 5.1 ciphertext consistency on output.
3. `cargo test -r --lib agr16` and `cargo test -r --lib` pass.

## Idempotence and Recovery

The change is test-focused. If the new circuit topology assertion fails, isolate whether the issue is test wiring vs implementation behavior by printing intermediate expected/plaintext values before adjusting assertions.

## Artifacts and Notes

Expected touched files:
- `src/agr16/mod.rs`
- `docs/prs/completed/pr_feat_agr16_binary_tree_test_coverage.md`
- `docs/plans/completed/plan_agr16_binary_tree_depth_test_coverage.md`

## Interfaces and Dependencies

No public interface changes expected.

Revision note (2026-03-03 01:05Z): Updated plan with completed binary-tree test implementation and verification outcomes; left only lifecycle closure steps pending.
Revision note (2026-03-03 01:08Z): Updated completed-path linkage and recorded PR response/readiness actions; left final commit/push as remaining lifecycle step.
