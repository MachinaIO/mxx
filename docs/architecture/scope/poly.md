# Scope: `src/poly`

## Purpose

Provides polynomial interfaces, CRT parameterization, and concrete polynomial implementations used across the repository.

## Implementation mapping

- `src/poly/mod.rs`
- `src/poly/dcrt/mod.rs`
- `src/poly/dcrt/element.rs`
- `src/poly/dcrt/params.rs`
- `src/poly/dcrt/poly.rs`
- `src/poly/dcrt/gpu.rs`

CUDA files included in this scope (called through Rust GPU bindings):

- `cuda/src/Runtime.cu`
- `cuda/src/ChaCha.cu`
- `cuda/include/Runtime.cuh`
- `cuda/include/ChaCha.cuh`

## Interface vs implementation

- Interfaces: `crate::poly::PolyParams`, `crate::poly::Poly`
- Concrete implementations:
  - CPU/host polynomial path in `src/poly/dcrt/poly.rs`
  - GPU binding path in `src/poly/dcrt/gpu.rs`

## Depends on scopes

- `element`

## Used by scopes

- `matrix`
- `sampler`
- `bgg`
- `circuit`
- `lookup`
- `commit`
- `gadgets`
- `simulator`
- `storage`
