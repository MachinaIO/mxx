# Features Architecture Index

This section documents feature behavior and feature boundaries for this repository.

Read this file first before individual feature documents.

## Reading order

1. Read this file.
2. Read [cargo_features.md](./cargo_features.md) for compile-time feature flags.
3. Read [env.md](./env.md) for environment-variable runtime controls.

## Feature model in this repository

Feature behavior has two layers:

- compile-time feature selection in `Cargo.toml` (`default`, `disk`, `gpu`),
- runtime controls in `src/env.rs` that tune concurrency and artifact sizing.

Both layers must be kept architecture-consistent with implementation and verification behavior.

## Maintenance rule

Update feature documentation when:

- `Cargo.toml` feature definitions change,
- `#[cfg(feature = ...)]` boundaries change in major paths,
- runtime environment variable behavior changes in `src/env.rs`,
- CI feature coverage changes in `.github/workflows/ci.yml`.
