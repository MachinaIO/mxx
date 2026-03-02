# Scope: `src/commit`

## Purpose

Provides commitment workflows and cache/public-parameter orchestration (currently centered on Wee25 flows).

## Implementation mapping

- `src/commit/mod.rs`
- `src/commit/wee25.rs`

## Interface vs implementation

- Main implementation: `Wee25Commit` and related parameter/cache types in `src/commit/wee25.rs`
- This scope consumes sampler/matrix/poly/storage abstractions rather than redefining them.

## Depends on scopes

- `matrix`
- `poly`
- `sampler`
- `storage`

## Used by scopes

- `lookup`
- `tests`
