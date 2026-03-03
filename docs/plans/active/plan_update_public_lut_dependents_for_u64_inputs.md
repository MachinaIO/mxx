# Update PublicLut Dependents For u64 Inputs

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

`PLANS.md` is checked into this repository at `PLANS.md`; this document is maintained in accordance with that file.

Repository-document context: design/architecture/verification guidance was reviewed from `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/lookup.md`, `docs/architecture/scope/bgg.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, and `docs/verification/cpu_behavior_changes.md`. This change is an implementation follow-up for an already-decided interface shift in `src/lookup/mod.rs` and does not introduce a new long-lived design or architecture decision, so no design/architecture document update is planned unless implementation evidence contradicts that assumption.

PR tracking document: `docs/prs/active/pr_feat_semi_packing_public_lut_update.md`

ExecPlan start context:
- Branch: `feat/semi-packing`
- Start commit: `cdeb008389b69ebda0d867856dbee20601bf7779`

## Purpose / Big Picture

After this change, all lookup evaluators consistently consume `PublicLut` entries by `u64` input keys, matching the updated `PublicLut::get` contract. A user can run lookup-related tests and observe that polynomial lookup evaluation returns per-coefficient mapped outputs, while BGG lookup paths continue to work under the temporary assumption that revealed plaintext inputs are constant polynomials with a `u64`-fit constant term.

## Progress

- [x] (2026-03-03 09:18Z) Completed main ExecPlan pre-creation checks from `docs/verification/main_execplan_pre_creation.md`: captured branch/status/history/PR state, decided scope is aligned with `feat/semi-packing`, and confirmed no branch switch is required.
- [x] (2026-03-03 09:20Z) Added PR tracking document at `docs/prs/active/pr_feat_semi_packing_public_lut_update.md` and created this main ExecPlan in `docs/plans/active/`.
- [x] (2026-03-03 10:47Z) Implemented evaluator updates for `PublicLut::get(params, u64)` across lookup modules and removed all `new_from_usize_range` call sites without introducing compatibility wrappers (sequential, no sub ExecPlans).
- [x] (2026-03-03 10:50Z) Ran CPU behavior verification commands from `docs/verification/cpu_behavior_changes.md`: formatting and lookup-focused unit tests passed (sequential, after code edits).
- [x] (2026-03-03 11:08Z) Completed final validation depth by running full library tests (`cargo test -r --lib`) after scoped lookup tests (sequential, completion-level verification).
- [x] (2026-03-03 10:53Z) Updated this ExecPlan with final outcomes, command results, and retrospective (sequential, after verification).
- [x] (2026-03-03 15:16Z) Updated unit-test helper LUT constructors to emit `k % 2` directly (without building `y_lsb`) and renamed helper function names to match behavior; reran formatting and lookup-scoped unit tests.

## Surprises & Discoveries

- Observation: `PublicLut::get` in `src/lookup/mod.rs` was already migrated to `u64` input, but multiple evaluator call sites still pass polynomials.
  Evidence: `rg -n "\.get\(params," src/lookup -S` showed mixed old/new call patterns during plan setup.

- Observation: `PublicLut::new_from_usize_range` had already been removed, which caused wide compile fallout in lookup/unit-test helper LUT builders.
  Evidence: `cargo test -r --lib -- lookup --no-run` initially failed with multiple `new_from_usize_range` not found errors.

## Decision Log

- Decision: Treat this work as a single main ExecPlan without sub-plans.
  Rationale: The affected files are tightly coupled lookup evaluator implementations with overlapping call paths; parallel sub-plans would create same-file conflict risk.
  Date/Author: 2026-03-03 / Codex

- Decision: Keep design and architecture documents unchanged unless implementation forces a new reusable contract.
  Rationale: The user request asks for follow-up implementation behavior under explicit temporary assumptions, not a new long-lived repository policy.
  Date/Author: 2026-03-03 / Codex

- Decision: Do not restore `new_from_usize_range`; update all call sites to `PublicLut::new` directly.
  Rationale: User explicitly requested no compatibility layer and a minimal code path.
  Date/Author: 2026-03-03 / Codex

- Decision: Rename unit-test LUT helper names from `setup_lsb_constant_binary_plt` to `setup_lsb_bit_lut` (and GPU variant) and make LUT output explicitly `k % 2`.
  Rationale: This matches the implemented behavior directly and removes unnecessary construction of intermediate LSB polynomials in test setup paths.
  Date/Author: 2026-03-03 / Codex

## Outcomes & Retrospective

Implemented end-to-end migration for lookup evaluator dependencies of `PublicLut`:

- `PolyPltEvaluator::public_lookup` now performs coefficient-wise LUT queries and rebuilds the output polynomial from returned `P::Elem` values.
- BGG encoding lookup evaluators now query LUT rows with `u64` keys derived from constant plaintext assumptions and convert LUT output elements into constant polynomials only where matrix/plaintext APIs require `P`.
- All remaining `new_from_usize_range` call sites in `src/` and `tests/` were rewritten to `PublicLut::new` closures that return `Option<(u64, P::Elem)>`.

Verification passed for formatting and lookup-scoped unit tests. Design/architecture docs were intentionally left unchanged because no new long-lived contract was introduced beyond adapting existing code to the already-edited `PublicLut` signature.
Additional completion-level verification (`cargo test -r --lib`) also passed with no failures.

## Context and Orientation

`src/lookup/mod.rs` defines `PublicLut<P>`, which now stores a function keyed by `u64` and exposes `get(&P::Params, u64) -> Option<(u64, P::Elem)>`. Evaluator implementations in `src/lookup/poly.rs`, `src/lookup/lwe_eval.rs`, `src/lookup/ggh15_eval.rs`, and `src/lookup/commit_eval.rs` still include legacy call sites that pass `P` polynomials directly. The required behavior for this task is:

1. In `PolyPltEvaluator::public_lookup`, read every input polynomial coefficient.
2. Convert each coefficient to `u64` under the assumption that it is sufficiently small.
3. Use `PublicLut::get` with that `u64` and place the returned `y_i` as the corresponding output coefficient.

For BGG evaluator paths (`BggPublicKey` and `BggEncoding` `PltEvaluator` implementations), assume lookup plaintext inputs are constant polynomials whose constant term coefficient fits in `u64`, then query the LUT by that scalar.

## Plan of Work

Edit `src/lookup/poly.rs` so `PolyPltEvaluator::public_lookup` constructs an output polynomial from per-coefficient LUT queries. The implementation will iterate over `input.coeffs()`, convert each coefficient value to `u64`, call `plt.get(params, x_i)`, and collect returned `P::Elem` values into `P::from_coeffs`.

Then update evaluator call sites in `src/lookup/lwe_eval.rs`, `src/lookup/ggh15_eval.rs`, and `src/lookup/commit_eval.rs` to pass `u64` LUT keys. Where current code has a plaintext polynomial `x`, extract its constant coefficient with the temporary assumption (`x.coeffs()[0]` fits `u64`). Where code currently builds constant polynomials from indices only to call `plt.get`, remove that conversion and pass the numeric index directly.

Finally run formatting and scope-focused unit tests and record results in this plan.

## Concrete Steps

Working directory for all commands: repository root (`.`).

Planned implementation/verification commands:

    rg -n "\.get\(params," src/lookup -S
    cargo +nightly fmt --all
    cargo test -r --lib -- lookup

Commands run so far (pre-creation + planning evidence):

    git branch --show-current
    git status --short
    git log --oneline --decorate --max-count=20
    gh pr status
    rg --files | rg 'PLANS.md|DESIGN.md|ARCHITECTURE.md|VERIFICATION.md|lookup|plt|lut|BGG|bgg'

Commands run during implementation/verification:

    rg -n "new_from_usize_range" src tests -S
    cargo test -r --lib -- lookup --no-run    # failed: '--no-run' interpreted as test binary option due argument order
    cargo test -r --lib --no-run
    cargo +nightly fmt --all
    cargo test -r --lib -- lookup
    cargo test -r --lib
    cargo +nightly fmt --all
    cargo test -r --lib -- lookup

## Validation and Acceptance

Acceptance criteria:

- `PolyPltEvaluator::public_lookup` returns a polynomial whose i-th coefficient equals LUT output `y_i` queried by the i-th input coefficient interpreted as `u64`.
- BGG evaluator lookup paths compile and run with the temporary constant-plaintext assumption for LUT key extraction.
- Formatting passes and lookup-related library tests pass with:

    cargo +nightly fmt --all
    cargo test -r --lib -- lookup

If tests fail, record failing test names and likely causes in this document before completion.

## Idempotence and Recovery

The edits are source-level and idempotent: reapplying with the same file contents causes no further changes. If a test command fails, fix code and rerun the same command until success or until a reproducible pre-existing failure is isolated and documented. No destructive git operations are required for this task.

## Artifacts and Notes

Pre-creation alignment evidence:

    branch: feat/semi-packing
    status: M src/lookup/mod.rs, ?? docs/references/
    gh pr status: no PR associated with current branch

## Interfaces and Dependencies

The target interface contract is already defined in `src/lookup/mod.rs`:

    pub fn get(&self, params: &P::Params, x: u64) -> Option<(u64, P::Elem)>

Files that must match this contract after implementation:

- `src/lookup/poly.rs` (`impl PltEvaluator<P> for PolyPltEvaluator`)
- `src/lookup/lwe_eval.rs` (`impl PltEvaluator<BggEncoding<M>> for LWEBGGEncodingPltEvaluator`)
- `src/lookup/ggh15_eval.rs` (BGG lookup preprocessing and `GGH15BGGEncodingPltEvaluator::public_lookup`)
- `src/lookup/commit_eval.rs` (commit lookup evaluator and LUT layout/message-stream helpers)

Revision note (2026-03-03): Initial plan creation with pre-creation verification evidence and implementation scope mapping for `PublicLut` u64-input migration.

Revision note (2026-03-03): Updated implementation status to complete, recorded the user-directed no-compatibility decision (`new_from_usize_range` not restored), and added final verification command outcomes.

Revision note (2026-03-03): Updated lookup unit-test helper LUT constructors to direct `k % 2` output and renamed helper functions accordingly; added rerun validation commands.
