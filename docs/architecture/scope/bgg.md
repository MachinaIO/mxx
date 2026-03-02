# Scope: `src/bgg`

## Purpose

Implements BGG+ public-key and encoding structures and their samplers, built on matrix/poly/sampler abstractions.

## Implementation mapping

- `src/bgg/mod.rs`
- `src/bgg/public_key.rs`
- `src/bgg/encoding.rs`
- `src/bgg/sampler.rs`
- `src/bgg/digits_to_int.rs`

## Interface vs implementation

- Public interfaces/types:
  - `BggPublicKey`
  - `BggEncoding`
  - `BGGPublicKeySampler`
  - `BGGEncodingSampler`
- Implementations rely on shared interfaces from matrix/poly/sampler scopes.

## Depends on scopes

- `matrix`
- `poly`
- `sampler`

## Used by scopes

- `lookup`
- `tests`
