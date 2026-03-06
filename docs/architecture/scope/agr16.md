# Scope: `src/agr16`

## Purpose

Implements AGR16 Section 5-style key-homomorphic public-key and ciphertext evaluation structures and samplers.

## Implementation mapping

- `src/agr16/mod.rs`
- `src/agr16/public_key.rs`
- `src/agr16/encoding.rs`
- `src/agr16/sampler.rs`

## Interface vs implementation

- Public interfaces/types:
  - `Agr16PublicKey`
  - `Agr16Encoding`
  - `AGR16PublicKeySampler`
  - `AGR16EncodingSampler`
- Implementations are generic over `PolyMatrix` / `Poly` traits and are not restricted to DCRT concrete types.

## Depends on scopes

- `matrix`
- `poly`
- `sampler`

## Used by scopes

- `circuit` (via `src/circuit/evaluable/agr16.rs`)
- `tests`
