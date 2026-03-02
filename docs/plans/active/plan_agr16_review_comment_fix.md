# Address PR60 Reviewer Findings for AGR16 Compact Security and Sampler Input Validation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `c8654aa728ef63cc2862f93432ce4bf3c1986749`
- PR tracking document: `docs/prs/active/pr_feat_agr16_encoding_review_fix.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, PR #60 will no longer expose AGR16 secret material through compact serialization and will reject invalid empty secret input at sampler construction time. This restores expected secrecy boundaries for `Agr16Encoding` and prevents silent insecure misconfiguration.

## Progress

- [x] (2026-03-02 16:58Z) Captured pre-creation evidence from `docs/verification/main_execplan_pre_creation.md`: branch/status/log and PR context (`gh pr status`, `gh pr view`).
- [x] (2026-03-02 16:59Z) Added active PR tracking file `docs/prs/active/pr_feat_agr16_encoding_review_fix.md` for review-fix lifecycle work on existing PR #60.
- [x] (2026-03-02 17:00Z) Created this main ExecPlan and linked active PR tracking path.
- [x] (2026-03-02 17:05Z) Removed direct secret-byte serialization from `Agr16Encoding` compact representation by replacing `secret_bytes` with process-local opaque `secret_handle` cache rehydration.
- [x] (2026-03-02 17:06Z) Enforced non-empty `secrets` input in `AGR16EncodingSampler::new` and added panic test coverage for empty input.
- [x] (2026-03-02 17:08Z) Ran verification from `docs/verification/cpu_behavior_changes.md`:
  - `cargo +nightly fmt --all`
  - `cargo test -r --lib agr16`
  - `cargo test -r --lib`
- [ ] Push follow-up commits and post review-response comment on PR #60.
- [ ] Complete post-completion lifecycle (`docs/verification/main_execplan_post_completion.md`): readiness decision, PR tracking move to completed, final commit/push for plan lifecycle state.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `compact serialization fix` -> run `cargo test -r --lib agr16`.
- Action `sampler validation fix` -> rerun `cargo test -r --lib agr16`.
- Action `finalize review-fix` -> run `cargo test -r --lib`.
- Action `lifecycle closure` -> run `gh pr ready` decision flow + move PR tracking file + final commit/push.

## Surprises & Discoveries

- Observation: GitHub issue-comment API call was intermittently unavailable, while `gh pr view --comments` succeeded and provided the requested reviewer findings.
  Evidence: `gh api repos/.../issues/comments/...` returned connectivity error; `gh pr view 60 --comments` returned full reviewer text.

- Observation: Removing secret bytes from compact form requires an internal rehydration path because `PolyCircuit::eval` always round-trips inputs through `to_compact`/`from_compact`.
  Evidence: `src/circuit/mod.rs` converts `one` and each input to compact then immediately reconstructs values before gate evaluation.

## Decision Log

- Decision: Keep this work on existing branch/PR (`feat/agr16_encoding`, PR #60) rather than creating a new PR.
  Rationale: Requested scope is direct follow-up to reviewer comments on the same PR and is not independently reviewable work.
  Date/Author: 2026-03-02 / Codex

- Decision: Replace `secret_bytes` with opaque `secret_handle` and a process-local secret cache in `src/circuit/evaluable/agr16.rs`.
  Rationale: This removes direct secret exfiltration via public compact output while preserving required `from_compact` reconstruction semantics during circuit evaluation.
  Date/Author: 2026-03-02 / Codex

- Decision: Fail-fast for empty `secrets` in `AGR16EncodingSampler::new`.
  Rationale: Silent fallback to `s = 0` is insecure misconfiguration and must be rejected explicitly.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

The two reviewer findings are addressed in code and covered by tests:
- compact output no longer contains raw secret bytes;
- sampler now rejects empty secret input with an explicit panic and test.

Remaining lifecycle work is operational: commit/push, post PR response, and close ExecPlan lifecycle documents.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`
- Planned updates: none (no long-lived new design policy; this is a correctness/security fix in existing scope).

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/agr16.md`
- Planned updates: likely none unless interface boundary changes materially.

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`
- Policy updates: none.

## Context and Orientation

`src/circuit/evaluable/agr16.rs` defines compact serialization for `Agr16Encoding`. Current code stores `secret_bytes` directly in compact form, which makes recovering the secret trivial from `to_compact` output. `src/agr16/sampler.rs` currently allows `AGR16EncodingSampler::new` with an empty secret slice and silently substitutes `s = 0`, which is unsafe configuration behavior.

This plan fixes both while preserving existing AGR16 arithmetic behavior and tests.

## Plan of Work

First, refactor AGR16 compact representation so it no longer serializes raw secret bytes. Keep evaluation functional by replacing raw secret transport with an internal opaque handle backed by process-local secret cache that is inaccessible from external API callers. Then modify `from_compact` to rehydrate the secret through this opaque handle.

Second, update `AGR16EncodingSampler::new` to reject empty `secrets` input explicitly (assert/panic with clear message), and add a unit test that verifies this failure mode.

Finally, run formatting and tests, update plan progress, push follow-up commit(s), and close post-completion lifecycle steps.

## Concrete Steps

Run from repository root (`.`):

    gh pr view 60 --comments
    # edit src/circuit/evaluable/agr16.rs and src/agr16/sampler.rs (+ tests)
    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Lifecycle closure commands:

    gh pr ready
    mv docs/prs/active/pr_feat_agr16_encoding_review_fix.md docs/prs/completed/pr_feat_agr16_encoding_review_fix.md
    git status --short
    git add -A
    git commit -m "docs: finalize agr16 review-fix execplan lifecycle"
    git push origin $(git branch --show-current)

## Validation and Acceptance

Acceptance is met when:

1. `Agr16EncodingCompact` no longer contains raw secret bytes.
2. Reviewer-identified secret extraction path via `to_compact` is removed.
3. `AGR16EncodingSampler::new` rejects empty `secrets` input with explicit failure.
4. AGR16 tests and full library tests pass.
5. PR remains/returns ready-for-review after follow-up fixes.

## Idempotence and Recovery

Edits are safe and additive. If compact-cache rehydration fails in tests, keep failure explicit (`panic!`) so bugs cannot silently degrade to insecure fallback values.

## Artifacts and Notes

Primary files expected:
- `src/circuit/evaluable/agr16.rs`
- `src/agr16/sampler.rs`
- `src/agr16/mod.rs` (test update, if needed)

Commands executed:

    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib

Verification outcomes:
- `cargo test -r --lib agr16`: pass (`4 passed`)
- `cargo test -r --lib`: pass (`139 passed; 0 failed; 2 ignored`)

## Interfaces and Dependencies

Target interfaces remain:
- `Evaluable for Agr16Encoding<M>`
- `AGR16EncodingSampler<S>::new(...)`

Internal dependency additions may include standard synchronization primitives for opaque secret-handle cache (e.g. `OnceLock`, `DashMap`, `AtomicU64`) without changing public API signatures.

Revision note (2026-03-02, Codex): Initial plan created for PR #60 reviewer follow-up on compact-secret leakage and empty-secret sampler validation.
Revision note (2026-03-02, Codex): Updated with implemented code fixes, verification evidence, and final-lifecycle remaining steps.
