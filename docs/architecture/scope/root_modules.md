# Scope: `src` top-level modules

## Purpose

Covers top-level Rust modules under `src/` that are not inside a subdirectory and provide cross-cutting wiring, environment behavior, and shared helpers.

## Implementation mapping

- `src/lib.rs`
- `src/env.rs`
- `src/openfhe_guard.rs`
- `src/rlwe_enc.rs`
- `src/utils.rs`

## Interface vs implementation

- `src/lib.rs` exposes crate module boundaries.
- `src/env.rs` defines runtime environment-variable controls.
- `src/openfhe_guard.rs` implements OpenFHE warmup behavior.
- `src/rlwe_enc.rs` provides RLWE encryption helper logic.
- `src/utils.rs` provides shared helper functions/macros used across scopes.

## Depends on scopes

This scope is cross-cutting and references multiple scopes through helper utilities, including:

- `poly`
- `matrix`
- `sampler`
- `bgg`

## Used by scopes

Cross-cutting consumers include:

- `circuit` (via `env`)
- `lookup` (via `env`)
- `commit` (via `env`)
- `storage` (via `env`)
- `poly` and `matrix` (via `utils`)
- `sampler` (via `openfhe_guard`)
