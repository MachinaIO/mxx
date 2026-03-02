# Scope: `src/sampler`

## Purpose

Provides sampling interfaces and implementations for hash-based, uniform, and trapdoor-based sampling, including GPU-aware variants.

## Implementation mapping

- `src/sampler/mod.rs`
- `src/sampler/hash.rs`
- `src/sampler/uniform.rs`
- `src/sampler/gpu.rs`
- `src/sampler/trapdoor/mod.rs`
- `src/sampler/trapdoor/sampler.rs`
- `src/sampler/trapdoor/gpu.rs`
- `src/sampler/trapdoor/utils.rs`

## Interface vs implementation

- Interfaces:
  - `crate::sampler::PolyHashSampler`
  - `crate::sampler::PolyUniformSampler`
  - `crate::sampler::PolyTrapdoorSampler`
- Concrete implementations:
  - DCRT CPU samplers
  - GPU-enabled samplers and trapdoor paths (feature-gated)

## CUDA Boundary Contract

- GPU trapdoor/sampling kernels operate on packed-byte limb storage via matrix metadata (`stride_bytes`, `coeff_bytes`).
- Sampler-side CUDA launch paths must not assume fixed 8-byte coefficients for persisted limb buffers.

## Depends on scopes

- `matrix`
- `poly`

## Used by scopes

- `bgg`
- `lookup`
- `commit`
- `benches`
