# Create Architecture Index and Scope Domain Documents

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md`. `DESIGN.md` and `VERIFICATION.md` do not exist in the current tree, so this plan uses `AGENTS.md` plus repository facts from `Cargo.toml`, `build.rs`, and source/test layout as fallback policy context.

## Purpose / Big Picture

After this change, a contributor who has not read implementation code can start at `docs/architecture/index.md`, discover architecture document locations, and navigate scope-domain documentation that maps each domain to real code paths and interface boundaries. The change makes architecture guidance actionable instead of only meta-rules.

## Progress

- [x] (2026-03-02 00:29Z) Read `PLANS.md` and `ARCHITECTURE.md`, confirmed `docs/architecture/` directories exist but no architecture files are populated.
- [x] (2026-03-02 00:35Z) Explored the codebase (`src/`, `cuda/`, `tests/`, `benches/`, `Cargo.toml`, `build.rs`) to identify concrete scope domains and implementation boundaries.
- [x] (2026-03-02 00:32Z) Drafted and added `docs/architecture/index.md` as the architecture table of contents and entrypoint.
- [x] (2026-03-02 00:32Z) Drafted and added `docs/architecture/scope/index.md` with relative links and explicit inter-domain dependency statements.
- [x] (2026-03-02 00:32Z) Added concrete scope domain documents under `docs/architecture/scope/` mapped to current implementation paths and interface/implementation splits.
- [x] (2026-03-02 00:32Z) Validated created documents for path correctness.
- [x] (2026-03-02 00:33Z) Moved this ExecPlan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: `docs/architecture/` existed as directories only and had no markdown content.
  Evidence: `find docs/architecture -maxdepth 3 -type f | sort` returned no files before this change.
- Observation: `AGENTS.md` requires design/verification policy references, but `DESIGN.md` and `VERIFICATION.md` are currently absent.
  Evidence: `rg --files | rg '(^|/)DESIGN\.md$|(^|/)VERIFICATION\.md$'` returned no matches.
- Observation: `structure/`, `features/`, and `dependencies/` directories were empty and would not persist in git without files.
  Evidence: `.gitkeep` files were needed so `find docs/architecture -maxdepth 3 -type f | sort` consistently shows those directories as tracked.

## Decision Log

- Decision: Define scope domains using code ownership and interface boundaries rather than one-file-per-module granularity.
  Rationale: Domain-level docs stay stable longer and match architecture-document longevity requirements.
  Date/Author: 2026-03-02 / Codex
- Decision: Keep `structure/`, `features/`, and `dependencies/` directories as reserved locations and focus this task on requested `index.md` plus concrete `scope` docs.
  Rationale: User asked specifically for `docs/architecture/index.md` and concrete scope docs; adding extra placeholders would add noise.
  Date/Author: 2026-03-02 / Codex
- Decision: Use three scope domains (`math_runtime`, `protocol_workflows`, `storage_artifacts`) instead of one file per crate module.
  Rationale: This granularity preserves long-lived architecture value while still mapping to concrete implementation paths.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Created `docs/architecture/index.md` and concrete scope documentation under `docs/architecture/scope/` that maps the current codebase to stable architecture domains and interface boundaries. The result satisfies the immediate outcome: a contributor can navigate architecture documentation without reading source first.

Remaining gap: `DESIGN.md` and `VERIFICATION.md` do not exist yet, so architecture plans currently use explicit fallback assumptions from `AGENTS.md` and repository structure.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none (no `DESIGN.md` exists).
- Created/modified: none.
- Why unchanged: this task was architecture indexing/scope mapping only.

Architecture documents:

- Referenced: `ARCHITECTURE.md`.
- Created: `docs/architecture/index.md`, `docs/architecture/scope/index.md`, `docs/architecture/scope/domain_math_runtime.md`, `docs/architecture/scope/domain_protocol_workflows.md`, `docs/architecture/scope/domain_storage_artifacts.md`.
- Modified/added for required layout persistence: `docs/architecture/structure/.gitkeep`, `docs/architecture/features/.gitkeep`, `docs/architecture/dependencies/.gitkeep`.

Verification documents:

- Referenced: none (no `VERIFICATION.md` exists).
- Created/modified: none.
- Why unchanged: no meta verification policy exists yet in repository.

## Context and Orientation

The repository is a Rust library centered around lattice-cryptography primitives and evaluators. Core modules are exposed from `src/lib.rs`, with math abstractions in `src/element`, `src/poly`, and `src/matrix`; protocol/evaluation layers in `src/bgg`, `src/circuit`, `src/lookup`, `src/commit`, `src/gadgets`, and `src/rlwe_enc.rs`; optional GPU paths in `src/matrix/gpu_dcrt_poly.rs`, `src/poly/dcrt/gpu.rs`, `src/sampler/gpu.rs`, `src/sampler/trapdoor/gpu.rs`, and `cuda/*`; persistence in `src/storage`; and operational entrypoints in `tests` and `benches`.

“Scope” in this repository means structure-domain documentation under `docs/architecture/scope/` that maps implementation domains to concrete paths and explains how domains relate.

## Plan of Work

First, create `docs/architecture/index.md` as the mandatory reading entrypoint, with concise orientation and links to `scope`, `structure`, `features`, and `dependencies`. Next, create `docs/architecture/scope/index.md` that lists each scope document with relative links and states domain dependencies using the dependency-direction rule defined in `ARCHITECTURE.md`. Then add concrete domain documents that map current implementation paths, explain what each domain owns, and separate interface contracts from implementation variants when multiple implementations exist (for example CPU/GPU matrix backends and evaluator abstractions). Finally, verify path consistency and move this plan from `active` to `completed`.

## Concrete Steps

Run these commands from repository root `.`:

    sed -n '1,260p' AGENTS.md
    sed -n '1,260p' ARCHITECTURE.md
    find src -maxdepth 2 -type d | sort
    find cuda -maxdepth 3 -type d | sort
    find tests -maxdepth 2 -type f | sort
    find benches -maxdepth 2 -type f | sort
    sed -n '1,260p' Cargo.toml
    sed -n '1,260p' build.rs

Create and fill:

    docs/architecture/index.md
    docs/architecture/scope/index.md
    docs/architecture/scope/domain_math_runtime.md
    docs/architecture/scope/domain_protocol_workflows.md
    docs/architecture/scope/domain_storage_artifacts.md

Verify:

    find docs/architecture -maxdepth 3 -type f | sort
    sed -n '1,260p' docs/architecture/index.md
    sed -n '1,320p' docs/architecture/scope/index.md

Move completed plan:

    mv docs/plans/active/plan_architecture_scope_bootstrap.md docs/plans/completed/plan_architecture_scope_bootstrap.md

## Validation and Acceptance

Acceptance criteria:

1. `docs/architecture/index.md` exists and acts as a real table of contents with links to architecture sub-areas.
2. `docs/architecture/scope/index.md` exists, links to each scope domain doc by relative path, and includes explicit domain-dependency statements.
3. Scope docs map to real implementation paths and document interface-vs-implementation distinction where relevant.
4. The plan file is moved to `docs/plans/completed/` after work is done.

Human-verifiable outcome: a new contributor can open `docs/architecture/index.md`, follow links into scope docs, and locate corresponding code directories/files without reading source first.

## Idempotence and Recovery

All file creation and edits are additive and idempotent. Re-running the plan updates the same markdown files. If a scope split is unsatisfactory, recovery is safe: edit scope docs in place and keep links stable in `scope/index.md`. If needed, delete only newly created architecture docs and recreate them from this plan.

## Artifacts and Notes

Expected verification snapshot:

    docs/architecture/index.md
    docs/architecture/scope/index.md
    docs/architecture/scope/domain_math_runtime.md
    docs/architecture/scope/domain_protocol_workflows.md
    docs/architecture/scope/domain_storage_artifacts.md

## Interfaces and Dependencies

The scope docs must describe these stable interface anchors and their implementations:

- `crate::poly::PolyParams`, `crate::poly::Poly`, `crate::matrix::PolyMatrix`, `crate::sampler::{PolyHashSampler, PolyUniformSampler, PolyTrapdoorSampler}`, and `crate::circuit::evaluable::Evaluable`.
- CPU matrix path in `src/matrix/dcrt_poly.rs` and GPU matrix path in `src/matrix/gpu_dcrt_poly.rs` as separate implementations of the shared matrix abstraction.
- GPU FFI boundary between `src/poly/dcrt/gpu.rs` plus `src/matrix/gpu_dcrt_poly.rs` and CUDA sources under `cuda/include` and `cuda/src`.

Revision note (2026-03-02, Codex): Initial plan created to implement architecture index and concrete scope docs based on current codebase exploration.
Revision note (2026-03-02, Codex): Updated progress, decisions, and outcomes after creating architecture index and scope domain documents; added explicit design/architecture/verification document summary before completion move.
Revision note (2026-03-02, Codex): Corrected plan artifact filenames to match actually created scope documents (`domain_math_runtime`, `domain_protocol_workflows`, `domain_storage_artifacts`).
