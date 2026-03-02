# Add Architecture Feature and Dependency Documents

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan is maintained under `PLANS.md` and follows that file’s rules.

Repository-document context used for this plan: `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md`. `DESIGN.md` and `VERIFICATION.md` do not exist in the current tree, so this plan uses repository source-of-truth files (`Cargo.toml`, `build.rs`, `src/env.rs`, and `.github/workflows/ci.yml`) as fallback policy context.

## Purpose / Big Picture

After this change, a reader can open `docs/architecture/index.md` and find concrete, current architecture documentation for both dependency model and feature model, instead of empty placeholder directories. This makes architectural impact reviews easier before editing implementation code.

## Progress

- [x] (2026-03-02 00:36Z) Read `PLANS.md`, `AGENTS.md`, and `ARCHITECTURE.md` and confirmed required architecture-document structure.
- [x] (2026-03-02 00:36Z) Collected feature and dependency facts from `Cargo.toml`, `build.rs`, `src/env.rs`, and `.github/workflows/ci.yml`.
- [x] (2026-03-02 00:38Z) Created dependency architecture documents under `docs/architecture/dependencies/`.
- [x] (2026-03-02 00:38Z) Created feature architecture documents under `docs/architecture/features/`.
- [x] (2026-03-02 00:38Z) Updated `docs/architecture/index.md` to link to concrete `features/` and `dependencies/` docs.
- [x] (2026-03-02 00:38Z) Validated file/link set for new architecture docs.
- [x] (2026-03-02 00:39Z) Moved this plan to `docs/plans/completed/`.

## Surprises & Discoveries

- Observation: `features/` and `dependencies/` currently contain only `.gitkeep`, so architecture navigation for those areas is effectively empty.
  Evidence: `find docs/architecture/features docs/architecture/dependencies -maxdepth 2 -type f | sort` returned only `.gitkeep` files.
- Observation: CI validates `disk` feature explicitly, but does not run GPU builds/tests.
  Evidence: `.github/workflows/ci.yml` includes `cargo +nightly clippy --workspace --all-targets --no-default-features --features disk` and no `--features gpu` command.

## Decision Log

- Decision: Split architecture docs into index + focused topic docs (features and dependencies) rather than one large mixed file.
  Rationale: This improves long-term maintainability and keeps cross-references stable when only one area changes.
  Date/Author: 2026-03-02 / Codex
- Decision: Document both compile-time Cargo features and runtime environment switches under the features domain.
  Rationale: In this repository, observable feature behavior is controlled by both Cargo features and environment variables in `src/env.rs`.
  Date/Author: 2026-03-02 / Codex
- Decision: Keep `features/.gitkeep` and `dependencies/.gitkeep` alongside real docs.
  Rationale: Keeping tracked placeholders is harmless and preserves directory tracking even if docs are reorganized later.
  Date/Author: 2026-03-02 / Codex

## Outcomes & Retrospective

Added concrete architecture documents for both dependencies and features, and connected them from `docs/architecture/index.md`. This closes the gap where those areas previously had placeholders only.

Remaining gap: repository-level `DESIGN.md` and `VERIFICATION.md` are still missing, so ExecPlans continue to use fallback references from `AGENTS.md` and code/CI files.

## Design/Architecture/Verification Document Summary

Design documents:

- Referenced: none (no `DESIGN.md` exists).
- Created/modified: none.
- Why unchanged: task scope is architecture documentation expansion, not design-policy change.

Architecture documents:

- Referenced: `ARCHITECTURE.md`, `docs/architecture/index.md`.
- Created:
  - `docs/architecture/dependencies/index.md`
  - `docs/architecture/dependencies/rust_crates.md`
  - `docs/architecture/dependencies/native_and_toolchain.md`
  - `docs/architecture/features/index.md`
  - `docs/architecture/features/cargo_features.md`
  - `docs/architecture/features/env.md`
- Modified:
  - `docs/architecture/index.md`
- Left unchanged:
  - `docs/architecture/scope/*` (no structure-domain change in this task).

Verification documents:

- Referenced: none (no `VERIFICATION.md` exists).
- Created/modified: none.
- Why unchanged: task scope is architecture docs; verification policy remains unchanged.

## Context and Orientation

The repository is a Rust library with optional `disk` and `gpu` compile-time features in `Cargo.toml`, plus runtime tuning knobs in `src/env.rs`. External/native dependencies are defined in `Cargo.toml` and `build.rs`, with CI behavior in `.github/workflows/ci.yml`. Architecture docs for scope already exist, but features and dependencies docs are placeholders; this plan fills those areas with code-backed documentation.

## Plan of Work

First create dependency docs that explain Rust crate dependencies, optional dependencies, native/library/toolchain dependencies, and where each is consumed in code/build. Next create feature docs that explain compile-time features (`default`, `disk`, `gpu`) and runtime environment-controlled behavior. Then update the top-level architecture index so readers can navigate directly to these new documents. Finally verify file/link integrity and move this plan to `completed`.

## Concrete Steps

Run from `.`:

    sed -n '1,260p' Cargo.toml
    sed -n '1,260p' build.rs
    sed -n '1,260p' src/env.rs
    sed -n '1,260p' .github/workflows/ci.yml
    rg -n "cfg\\(feature = \\\"gpu\\\"\\)|cfg\\(feature = \\\"disk\\\"\\)" src tests benches

Create:

    docs/architecture/dependencies/index.md
    docs/architecture/dependencies/rust_crates.md
    docs/architecture/dependencies/native_and_toolchain.md
    docs/architecture/features/index.md
    docs/architecture/features/cargo_features.md
    docs/architecture/features/env.md

Update:

    docs/architecture/index.md

Verify:

    find docs/architecture -maxdepth 3 -type f | sort
    sed -n '1,260p' docs/architecture/index.md
    sed -n '1,260p' docs/architecture/features/index.md
    sed -n '1,260p' docs/architecture/dependencies/index.md

Move completed plan:

    mv docs/plans/active/plan_architecture_features_dependencies_docs.md docs/plans/completed/plan_architecture_features_dependencies_docs.md

## Validation and Acceptance

Acceptance criteria:

1. `docs/architecture/features/index.md` and `docs/architecture/dependencies/index.md` exist and provide clear navigation.
2. Feature docs capture both Cargo features and runtime environment switches with concrete code-path references.
3. Dependency docs capture Rust, native, and toolchain dependencies with concrete ownership points.
4. `docs/architecture/index.md` links to the new docs.
5. The plan is moved from `docs/plans/active/` to `docs/plans/completed/`.

Observable result: someone new to the repository can identify supported features and dependency boundaries from architecture docs before opening source files.

## Idempotence and Recovery

Edits are additive markdown updates and can be applied repeatedly. If structure is unsatisfactory, update the same files in place while keeping index links stable. Recovery is safe: remove newly created docs and restore `docs/architecture/index.md` links.

## Artifacts and Notes

Expected artifact set:

    docs/architecture/dependencies/index.md
    docs/architecture/dependencies/rust_crates.md
    docs/architecture/dependencies/native_and_toolchain.md
    docs/architecture/features/index.md
    docs/architecture/features/cargo_features.md
    docs/architecture/features/env.md
    docs/architecture/index.md

## Interfaces and Dependencies

Key interfaces and boundaries to document:

- Cargo feature interface (`default`, `disk`, `gpu`) from `Cargo.toml`.
- Runtime control interface via `src/env.rs` functions (`circuit_parallel_gates`, `lut_preimage_chunk_size`, `ggh15_gate_parallelism`, etc.).
- Build/runtime native boundary in `build.rs` (OpenFHE link args and CUDA build path).
- CI dependency-validation behavior from `.github/workflows/ci.yml`.

Revision note (2026-03-02, Codex): Initial plan created for adding concrete architecture documents in `features/` and `dependencies/`.
Revision note (2026-03-02, Codex): Updated progress/outcomes and document summary after creating features/dependencies architecture docs and wiring index links.
Revision note (2026-03-02, Codex): Marked completion and moved this plan from `active` to `completed`.
Revision note (2026-03-02, Codex): Updated artifact references after feature runtime document rename to `env.md`.
