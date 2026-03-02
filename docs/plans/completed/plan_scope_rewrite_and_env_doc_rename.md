# Rewrite Scope Docs by Directory and Rename Runtime Feature Doc

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md`. `DESIGN.md` and `VERIFICATION.md` do not exist in the current tree, so this plan uses source-of-truth repository files (`docs/architecture/*`, `src/*`, `tests/*`, `benches/*`, `Cargo.toml`, `build.rs`) as fallback context.

## Purpose / Big Picture

After this change, architecture scope documentation will be directory-aligned and dependency-correct: every `src` top-level directory has its own scope document, `tests` and `benches` are independent scopes, and dependency direction is documented as “consumer depends on provider” (for example, `lookup` depends on `poly` and `matrix`). Also, feature runtime switch documentation will be renamed from `runtime_switches.md` to `env.md`.

## Progress

- [x] (2026-03-02 00:47Z) Read `PLANS.md` and current architecture docs, confirmed current scope docs are coarse-grained and dependency direction is wrong for practical usage.
- [x] (2026-03-02 00:47Z) Collected current `src` top-level directories/files and verified `tests`/`benches` layout.
- [x] (2026-03-02 00:48Z) Renamed `docs/architecture/features/runtime_switches.md` to `docs/architecture/features/env.md` and updated links.
- [x] (2026-03-02 00:50Z) Rewrote `docs/architecture/scope/index.md` from scratch with directory-level scopes and corrected dependency direction.
- [x] (2026-03-02 00:50Z) Replaced old scope domain docs with new per-directory scope docs for each `src/*` top-level directory.
- [x] (2026-03-02 00:50Z) Added independent scope docs for `tests/` and `benches/`.
- [x] (2026-03-02 00:50Z) Documented CUDA files inside Rust scopes that invoke them (`src_matrix` and `src_poly`).
- [x] (2026-03-02 00:50Z) Added anti-misinterpretation lesson to `docs/architecture/index.md`.
- [x] (2026-03-02 00:51Z) Validated links/files.
- [x] (2026-03-02 00:52Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: Earlier scope documentation encoded dependency direction opposite to implementation ownership expectations, causing statements like “math runtime depends on protocol workflows.”
  Evidence: `docs/architecture/scope/index.md` currently states `domain_math_runtime` depends on `domain_protocol_workflows`.
- Observation: Current feature index links `runtime_switches.md`, so renaming requires link update to avoid broken navigation.
  Evidence: `docs/architecture/features/index.md` currently links `[runtime_switches.md](./runtime_switches.md)`.
- Observation: `src_circuit` and `src_lookup` have a real mutual dependency in current code, so a strictly acyclic scope map would be inaccurate.
  Evidence: `src/circuit/mod.rs` imports `lookup::{PltEvaluator, PublicLut}` and lookup modules import `circuit` types such as `GateId`/`Evaluable`.

## Decision Log

- Decision: Use one scope document per `src` top-level directory and separate docs for `tests` and `benches`.
  Rationale: This matches requested granularity and maps architecture docs directly to the repository tree.
  Date/Author: 2026-03-02 / Codex
- Decision: Explicitly document dependency direction as “consumer depends on provider” in the rewritten scope index.
  Rationale: This aligns with implementation reasoning and resolves the documented lookup/poly/matrix contradiction.
  Date/Author: 2026-03-02 / Codex
- Decision: Document `src` top-level files (`env.rs`, `utils.rs`, `openfhe_guard.rs`, `rlwe_enc.rs`, `lib.rs`) as a dedicated scope file in addition to directory scopes.
  Rationale: This keeps scope coverage complete without forcing those files into unrelated directory scope docs.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Scope documentation is now directory-aligned and dependency direction is corrected to usage direction (consumer -> provider). The prior broad-domain scope set was removed, and each `src` top-level directory is represented explicitly. `tests` and `benches` are separate independent scopes.

`runtime_switches.md` was renamed to `env.md` as requested and linked from `features/index.md`.

Remaining gap: global `DESIGN.md` and `VERIFICATION.md` policy files still do not exist; plans continue to use explicit fallback references.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none (no `DESIGN.md` exists).
- Created/modified: none.
- Why unchanged: this task is architecture documentation correction, not design-policy authoring.

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`, existing `docs/architecture/scope/*`, `docs/architecture/features/*`.
- Created:
  - `docs/architecture/scope/src_root_modules.md`
  - `docs/architecture/scope/src_bgg.md`
  - `docs/architecture/scope/src_circuit.md`
  - `docs/architecture/scope/src_commit.md`
  - `docs/architecture/scope/src_element.md`
  - `docs/architecture/scope/src_gadgets.md`
  - `docs/architecture/scope/src_lookup.md`
  - `docs/architecture/scope/src_matrix.md`
  - `docs/architecture/scope/src_poly.md`
  - `docs/architecture/scope/src_sampler.md`
  - `docs/architecture/scope/src_simulator.md`
  - `docs/architecture/scope/src_storage.md`
  - `docs/architecture/scope/tests.md`
  - `docs/architecture/scope/benches.md`
- Renamed:
  - `docs/architecture/features/runtime_switches.md` -> `docs/architecture/features/env.md`
- Modified:
  - `docs/architecture/features/index.md`
  - `docs/architecture/scope/index.md`
  - `docs/architecture/index.md`
- Deleted (replaced):
  - `docs/architecture/scope/domain_math_runtime.md`
  - `docs/architecture/scope/domain_protocol_workflows.md`
  - `docs/architecture/scope/domain_storage_artifacts.md`

Verification documents:

- Referenced: none (no `VERIFICATION.md` exists).
- Created/modified: none.
- Why unchanged: no verification-policy update is needed for documentation-only architecture corrections.

## Context and Orientation

Current scope docs are grouped into three broad domains and do not preserve dependency direction as requested by the user. The repository has clear top-level implementation domains under `src/`: `bgg`, `circuit`, `commit`, `element`, `gadgets`, `lookup`, `matrix`, `poly`, `sampler`, `simulator`, `storage`, plus top-level support files (`env.rs`, `lib.rs`, `openfhe_guard.rs`, `rlwe_enc.rs`, `utils.rs`). CUDA implementation lives under `cuda/` but is invoked through Rust GPU paths (`src/poly/dcrt/gpu.rs`, `src/matrix/gpu_dcrt_poly.rs`, and build integration in `build.rs`).

## Plan of Work

First rename `runtime_switches.md` to `env.md` and update feature index links. Next replace the scope index and old broad domain docs with a directory-based set: one document per `src` top-level directory, plus additional independent scope documents for `tests` and `benches`. In each scope doc, include implementation mapping, dependency relations, and interface/implementation distinction where applicable. For Rust GPU-related scopes, include explicit mapping to `cuda/` files called through Rust FFI/bindings. Then add a lessons section to `docs/architecture/index.md` that abstracts the initial dependency-direction misunderstanding into reusable guidance for future agents. Finally validate file/link consistency and move this plan to `completed`.

## Concrete Steps

Run from `.`:

    find src -maxdepth 1 -type d | sort
    find src -maxdepth 1 -type f | sort
    find tests -maxdepth 1 -type f | sort
    find benches -maxdepth 1 -type f | sort
    find cuda -maxdepth 3 -type f | sort
    rg -n "^use crate::|^use crate::\\{" src/* | sed -n '1,240p'

Edit/create:

    docs/architecture/features/env.md
    docs/architecture/features/index.md
    docs/architecture/scope/index.md
    docs/architecture/scope/src_root_modules.md
    docs/architecture/scope/src_bgg.md
    docs/architecture/scope/src_circuit.md
    docs/architecture/scope/src_commit.md
    docs/architecture/scope/src_element.md
    docs/architecture/scope/src_gadgets.md
    docs/architecture/scope/src_lookup.md
    docs/architecture/scope/src_matrix.md
    docs/architecture/scope/src_poly.md
    docs/architecture/scope/src_sampler.md
    docs/architecture/scope/src_simulator.md
    docs/architecture/scope/src_storage.md
    docs/architecture/scope/tests.md
    docs/architecture/scope/benches.md
    docs/architecture/index.md

Retire old broad docs:

    docs/architecture/scope/domain_math_runtime.md
    docs/architecture/scope/domain_protocol_workflows.md
    docs/architecture/scope/domain_storage_artifacts.md

Verify:

    find docs/architecture -maxdepth 3 -type f | sort
    sed -n '1,260p' docs/architecture/scope/index.md
    sed -n '1,260p' docs/architecture/features/index.md
    sed -n '1,260p' docs/architecture/index.md

Move completed plan:

    mv docs/plans/active/plan_scope_rewrite_and_env_doc_rename.md docs/plans/completed/plan_scope_rewrite_and_env_doc_rename.md

## Validation and Acceptance

Acceptance criteria:

1. `runtime_switches.md` is renamed to `env.md` and all in-repo links are updated.
2. Scope docs are rewritten so each `src` top-level directory has a corresponding scope file.
3. `tests` and `benches` each have independent scope docs.
4. Scope dependency statements reflect actual consumer->provider direction (for example `lookup` depends on `poly`/`matrix`, not inverse).
5. CUDA files are documented inside the Rust scopes that call them.
6. `docs/architecture/index.md` includes a general lesson that prevents recurrence of the initial dependency-direction mistake.

## Idempotence and Recovery

All changes are markdown edits and file renames and are safe to re-run. If dependency mapping needs correction, update `scope/index.md` and affected scope docs in place. If rollback is needed, restore deleted/replaced scope docs and revert the feature-document rename.

## Artifacts and Notes

Expected artifact set after completion:

    docs/architecture/features/env.md
    docs/architecture/scope/index.md
    docs/architecture/scope/src_*.md
    docs/architecture/scope/tests.md
    docs/architecture/scope/benches.md
    docs/architecture/index.md

## Interfaces and Dependencies

The rewritten scope docs must preserve clear interface-vs-implementation separation for:

- math/runtime interfaces: `crate::element::PolyElem`, `crate::poly::{PolyParams, Poly}`, `crate::matrix::PolyMatrix`
- protocol interfaces: sampler traits, `Evaluable`, `PltEvaluator`
- GPU boundary: Rust FFI/binding call sites (`src/poly/dcrt/gpu.rs`, `src/matrix/gpu_dcrt_poly.rs`) mapped to concrete `cuda/include/*` and `cuda/src/*` files.

Revision note (2026-03-02, Codex): Initial plan created to rename runtime feature doc and rewrite scope docs at directory granularity with corrected dependency direction.
Revision note (2026-03-02, Codex): Updated progress/outcomes and architecture-summary sections after completing rename and full scope rewrite.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.
