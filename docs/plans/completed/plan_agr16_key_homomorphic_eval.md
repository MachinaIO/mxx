# Implement AGR16 Section 5 Key-Homomorphic Evaluation Module

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan follows `PLANS.md`.

ExecPlan start context:
- Branch at start: `feat/agr16_encoding`
- Commit at start: `cdeb008389b69ebda0d867856dbee20601bf7779`
- PR tracking document: `docs/prs/completed/pr_feat_agr16_encoding.md`

Repository-document context used for this plan: `PLANS.md`, `DESIGN.md`, `docs/design/index.md`, `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/bgg.md`, `docs/architecture/scope/root_modules.md`, `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, and `docs/verification/main_execplan_post_completion.md`.

## Purpose / Big Picture

After this change, the repository will include a new `src/agr16` module that implements Section 5 key-homomorphic public-key and ciphertext evaluation interfaces from `docs/references/agr16_encoding.pdf` using repository-generic `Poly`/`PolyMatrix` abstractions. Users will be able to sample AGR16 public keys/encodings, evaluate arithmetic circuits on them via `Evaluable`, and verify that when injected error is zero, evaluated outputs satisfy Equation (5.1)-style relation for both public-key labels and ciphertext encodings.

## Progress

- [x] (2026-03-02 15:54Z) Ran main ExecPlan pre-creation checks from `docs/verification/main_execplan_pre_creation.md`: captured branch/status/log and PR context (`gh pr status`, `gh pr view`), and confirmed scope is aligned with current feature branch.
- [x] (2026-03-02 15:56Z) Attempted draft PR bootstrap; `gh pr create --draft` failed because branch had no committed diff yet. Pushed branch to origin and recorded this as an expected pre-implementation condition.
- [x] (2026-03-02 15:59Z) Created PR tracking document `docs/prs/active/pr_feat_agr16_encoding.md` with current metadata and deferred PR creation note.
- [x] (2026-03-02 16:02Z) Created this main ExecPlan under `docs/plans/active/` and linked PR tracking path.
- [x] (2026-03-02 16:07Z) Read `src/bgg/*`, `src/circuit/evaluable/*`, and extracted AGR16 Section 5 equations (5.1, 5.7, 5.11, 5.17, 5.24, 5.25) to map implementation semantics.
- [x] (2026-03-02 16:09Z) Implemented `src/agr16` module (`public_key`, `encoding`, `sampler`, tests) using generic `Poly`/`PolyMatrix` traits and `s * PK` convention.
- [x] (2026-03-02 16:09Z) Added `Evaluable` implementations for `Agr16PublicKey` and `Agr16Encoding` in `src/circuit/evaluable/agr16.rs` and registered module exports (`src/circuit/evaluable/mod.rs`, `src/lib.rs`).
- [x] (2026-03-02 16:10Z) Updated architecture scope documentation for new `src/agr16` scope and changed root/circuit/scope index maps.
- [x] (2026-03-02 16:11Z) Ran verification mapped from `docs/verification/cpu_behavior_changes.md`:
  - `cargo +nightly fmt --all`
  - scope-targeted tests for `agr16`/`circuit::evaluable::agr16`
  - `cargo test -r --lib` (feature completion and foundational module addition)
- [x] (2026-03-02 16:12Z) Created draft PR `https://github.com/MachinaIO/mxx/pull/60` and updated PR tracking metadata (`docs/prs/active/pr_feat_agr16_encoding.md`).
- [x] (2026-03-02 16:12Z) Moved this plan to `docs/plans/completed/` after implementation and verification were finalized.
- [x] (2026-03-02 16:12Z) Executed post-ExecPlan verification from `docs/verification/main_execplan_post_completion.md`: PR scope reviewed as complete, PR `#60` set to ready for review, and PR tracking file moved to `docs/prs/completed/pr_feat_agr16_encoding.md`.
- [x] (2026-03-02 16:13Z) Persisted post-completion state in git with final commit/push.

Main-ExecPlan validation mapping (PLANS.md lifecycle step 3):
- Action `Implement new src/agr16 module` -> event `cpu_behavior_changes.md`: run fmt + scoped unit tests after implementation.
- Action `Add Evaluable implementations and wire modules` -> event `cpu_behavior_changes.md`: rerun scoped tests including circuit evaluation path.
- Action `Finalize feature` -> event `cpu_behavior_changes.md`: run full `cargo test -r --lib`.
- Action `Lifecycle closure` -> event `main_execplan_post_completion.md`: PR readiness decision + PR tracking state move + final commit/push.

## Surprises & Discoveries

- Observation: `gh pr create --draft` cannot create a PR when branch has no committed diff from base branch.
  Evidence: CLI returned `GraphQL: No commits between main and feat/agr16_encoding (createPullRequest)`.

- Observation: `parallel_iter!` usage requires importing Rayon prelude traits in module scope; otherwise `map` is resolved as `Iterator` and fails to compile.
  Evidence: Initial build error `E0599: rayon::range::Iter<usize> is not an iterator` in `src/agr16/sampler.rs`.

- Observation: Generic compact structs in `Evaluable` implementations require an explicit marker field when generic parameter `M` is only represented through serialized bytes.
  Evidence: Build error `E0392: type parameter M is never used` in `src/circuit/evaluable/agr16.rs`, fixed by adding `PhantomData<M>`.

## Decision Log

- Decision: Reuse branch `feat/agr16_encoding` instead of switching branches.
  Rationale: Branch objective matches requested AGR16 feature scope and satisfies pre-creation alignment rule.
  Date/Author: 2026-03-02 / Codex

- Decision: Use `src/agr16` (not `src/arg16`) as module path.
  Rationale: Request body references `Agr16*` type names and `src/agr16` tests; `src/arg16` is treated as a typo.
  Date/Author: 2026-03-02 / Codex

- Decision: Model AGR16 wire labels/encodings as generic `PolyMatrix` values but operate on scalar-style `1x1` sampled matrices.
  Rationale: Section 5 equations assume commutative ring multiplication; keeping sampled labels/encodings scalar-like preserves those identities while still using repository-generic matrix traits.
  Date/Author: 2026-03-02 / Codex

- Decision: Keep auxiliary advice labels (`PK(E(c*s))`, `PK(E(s^2))`) explicit in `Agr16PublicKey` and carry corresponding advice encodings in `Agr16Encoding`.
  Rationale: This lets multiplication follow Eq. (5.24)/(5.25)-style key/ciphertext evaluation directly and keeps 5.1 checks explicit in tests.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Implemented the AGR16 module end-to-end (`src/agr16/*` + `Evaluable` wiring + circuit tests). The new tests demonstrate Section 5.1 behavior in zero-error mode for sampled encodings and for circuit-evaluated outputs, including nested multiplication.

Post-completion lifecycle actions are complete. PR `#60` is open and ready for review with tracking moved to completed state.

## Design/Architecture/Verification Document Summary

Design documents:
- Referenced: `DESIGN.md`, `docs/design/index.md`
- Modified/Created: none.
- Why unchanged: implementation follows existing crate-local pattern (`bgg`-style type/sampler/evaluable decomposition) without adding a new reusable design policy beyond this feature scope.

Architecture documents:
- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/bgg.md`, `docs/architecture/scope/root_modules.md`
- Created: `docs/architecture/scope/agr16.md`
- Modified: `docs/architecture/scope/index.md`, `docs/architecture/scope/circuit.md`, `docs/architecture/scope/root_modules.md`
- Why: this change adds new top-level scope `src/agr16` and adds `circuit` dependency on `agr16` via `src/circuit/evaluable/agr16.rs`.

Verification documents:
- Referenced: `VERIFICATION.md`, `docs/verification/index.md`, `docs/verification/main_execplan_pre_creation.md`, `docs/verification/cpu_behavior_changes.md`, `docs/verification/main_execplan_post_completion.md`
- Policy updates: none.

## Context and Orientation

`src/bgg` currently provides key-homomorphic public keys and encodings with samplers and arithmetic operations used by `PolyCircuit` through `Evaluable`. The new work adds parallel functionality under `src/agr16`, but with formulas aligned to AGR16 Section 5 Regev-encoding style key/ciphertext evaluation. The implementation must remain generic over `PolyMatrix` and `Poly` traits (not DCRT-only), while tests can instantiate `DCRTPoly*` concrete types.

Section 5 target behavior used in this plan:
- Equation (5.1): evaluated ciphertext has form `CT(f(x)) = PK_f * s + p_{d-1} * eta + mu_f(x) + f(x)`.
- Addition evaluation: key and ciphertext add linearly.
- Multiplication evaluation: key/ciphertext use quadratic-method form with terms equivalent to `c_i c_j + u_i u_j E(s^2) - u_j E(c_i s) - u_i E(c_j s)` and corresponding public-key equation.

Repository implementation convention requested by user:
- Write multiplication terms in `s * PK` order instead of `PK * s` order where expression structure allows it.

## Plan of Work

Create `src/agr16/mod.rs`, `src/agr16/public_key.rs`, `src/agr16/encoding.rs`, and `src/agr16/sampler.rs` modeled after `src/bgg` but with AGR16-specific fields and multiplication equations. `Agr16PublicKey` will carry the public label matrix and reveal flag. `Agr16Encoding` will carry ciphertext component, associated `Agr16PublicKey`, optional plaintext, and auxiliary encodings needed by AGR16 quadratic multiplication (`E(s^2)` and `E(c*s)` terms). Add arithmetic trait implementations for both types matching Section 5 add/mul equations.

Add circuit integration in `src/circuit/evaluable/agr16.rs` by implementing `Evaluable` for both `Agr16PublicKey` and `Agr16Encoding`, including compact serialization layout analogous to BGG compact types. Update `src/circuit/evaluable/mod.rs` and `src/lib.rs` to export new modules.

Implement samplers in `src/agr16/sampler.rs` following `src/bgg/sampler.rs`: a hash-based public-key sampler and a uniform-based encoding sampler that can inject optional Gaussian error. Ensure sampler outputs include auxiliary AGR16 terms required for multiplication.

Add unit tests in `src/agr16/mod.rs` and/or per-file test modules. Construct small circuits (addition/multiplication and mixed depth) using `PolyCircuit` and verify that evaluated output public key and encoding satisfy Equation (5.1) in the zero-error setting (`gauss_sigma=None`) by directly checking `ct == s*pk + plaintext`-style relation in repository matrix form. Use `src/bgg` tests as structural reference.

Update architecture docs by adding `docs/architecture/scope/agr16.md` and updating `docs/architecture/scope/index.md` and `docs/architecture/scope/root_modules.md` mappings.

## Concrete Steps

Run from repository root (`.`):

    rg -n "BGGPublicKey|BggEncoding|Evaluable" src/bgg src/circuit/evaluable
    pdftotext docs/references/agr16_encoding.pdf - | nl -ba | sed -n '860,1845p'
    # edit src/agr16/*, src/circuit/evaluable/*, src/lib.rs, docs/architecture/scope/*
    cargo +nightly fmt --all
    cargo test -r --lib agr16
    cargo test -r --lib circuit::evaluable::agr16
    cargo test -r --lib

Post-completion lifecycle commands:

    gh pr create --draft --title "feat: add AGR16 key-homomorphic evaluation module" --body "Implements AGR16 Section 5 public-key/ciphertext key-homomorphic evaluation under generic Poly/PolyMatrix abstractions with tests."
    gh pr ready
    mv docs/prs/active/pr_feat_agr16_encoding.md docs/prs/completed/pr_feat_agr16_encoding.md
    git status --short
    git add -A
    git commit -m "feat: implement agr16 key-homomorphic evaluation module"
    git push origin $(git branch --show-current)

Commands already run (pre-implementation phase):

    git branch --show-current
    git status --short
    git log --oneline --decorate --max-count=20
    gh pr status
    gh pr view --json number,title,body,state,headRefName,baseRefName,url
    gh pr create --draft --fill
    git push -u origin feat/agr16_encoding
    gh pr create --draft --title "feat: add AGR16 key-homomorphic evaluation module" --body "Implement AGR16 Section 5 key-homomorphic evaluation algorithms and tests."

Commands executed during implementation/verification:

    cargo test -r --lib agr16
    cargo test -r --lib circuit::evaluable::agr16
    cargo test -r --lib
    cargo +nightly fmt --all
    gh pr create --draft --title "feat: add AGR16 key-homomorphic evaluation module" --body-file /tmp/pr_body_agr16.md
    gh pr ready
    mv docs/prs/active/pr_feat_agr16_encoding.md docs/prs/completed/pr_feat_agr16_encoding.md

## Validation and Acceptance

Acceptance requires all of the following:

1. `src/agr16` exists with `Agr16PublicKey`, `Agr16Encoding`, and samplers implemented against `Poly`/`PolyMatrix` traits.
2. `Agr16PublicKey` and `Agr16Encoding` implement `Evaluable` and can be evaluated by `PolyCircuit`.
3. Tests under `src/agr16` verify Section 5.1 relation under zero injected error for evaluated circuit outputs.
4. Formatting and unit tests in `docs/verification/cpu_behavior_changes.md` pass.
5. PR is set to ready for review and lifecycle closure steps are completed per `docs/verification/main_execplan_post_completion.md`.

## Idempotence and Recovery

File edits are additive and can be retried safely. If PR creation fails before the first feature commit, retry after committing implementation changes. If any test fails, capture failing test names in this plan, fix incrementally, and rerun only affected scope tests before full `cargo test -r --lib`.

## Artifacts and Notes

Planned artifact files:
- `src/agr16/mod.rs`
- `src/agr16/public_key.rs`
- `src/agr16/encoding.rs`
- `src/agr16/sampler.rs`
- `src/circuit/evaluable/agr16.rs`
- `docs/architecture/scope/agr16.md`

Current evidence snapshot:
- Branch: `feat/agr16_encoding`
- PR: `https://github.com/MachinaIO/mxx/pull/60` (`OPEN`, `ready for review`)
- PR tracking file: `docs/prs/completed/pr_feat_agr16_encoding.md`
- Verification snapshot:
  - `cargo test -r --lib agr16`: pass (`3 passed`)
  - `cargo test -r --lib circuit::evaluable::agr16`: pass (`0 tests`, compile/selection check)
  - `cargo test -r --lib`: pass (`138 passed; 0 failed; 2 ignored`)
  - `cargo +nightly fmt --all`: pass

## Interfaces and Dependencies

Interfaces required at completion:
- `crate::agr16::public_key::Agr16PublicKey<M: PolyMatrix>`
- `crate::agr16::encoding::Agr16Encoding<M: PolyMatrix>`
- `crate::agr16::sampler::AGR16PublicKeySampler<K, S>` where `S: PolyHashSampler<K>`
- `crate::agr16::sampler::AGR16EncodingSampler<S>` where `S: PolyUniformSampler`
- `impl Evaluable for Agr16PublicKey<M>`
- `impl Evaluable for Agr16Encoding<M>`

Dependencies reused:
- `matrix::PolyMatrix`
- `poly::Poly` and params traits
- `sampler::{PolyHashSampler, PolyUniformSampler, DistType}`
- `circuit::PolyCircuit` test harness

Revision note (2026-03-02, Codex): Initial plan created with pre-creation evidence, validation mapping, and implementation milestones.
Revision note (2026-03-02, Codex): Updated progress with implemented AGR16 code/docs, recorded verification outcomes, and captured compile-time discoveries/decisions.
Revision note (2026-03-02, Codex): Recorded post-ExecPlan readiness transition (`gh pr ready`), moved PR tracking file to completed, and updated plan state after move to `docs/plans/completed/`.
