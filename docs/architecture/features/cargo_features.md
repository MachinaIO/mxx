# Compile-Time Cargo Features

This document defines compile-time feature behavior from `Cargo.toml` and where each feature changes implementation paths.

## Declared features

`Cargo.toml` currently declares:

- `default = []`
- `disk = ["libc", "memmap2"]`
- `gpu = []`

## Feature: `default`

`default` enables no additional feature flags.

Architecture implication: baseline CPU path is the default behavior, with no disk-backed matrix path and no CUDA path unless explicitly enabled.

## Feature: `disk`

`disk` enables optional dependencies `libc` and `memmap2` and unlocks disk-backed matrix storage paths.

Primary implementation boundaries:

- `src/matrix/base/mod.rs` (`#[cfg(feature = "disk")]` module wiring),
- `src/matrix/base/disk.rs` (memory-mapped matrix storage implementation),
- `src/matrix/dcrt_poly.rs` and `src/sampler/uniform.rs` conditional paths.

CI coverage: clippy explicitly checks `--no-default-features --features disk`.

## Feature: `gpu`

`gpu` enables CUDA-aware modules and GPU implementations behind `#[cfg(feature = "gpu")]`.

Primary implementation boundaries:

- `src/matrix/gpu_dcrt_poly.rs`
- `src/poly/dcrt/gpu.rs`
- `src/sampler/gpu.rs`
- `src/sampler/trapdoor/gpu.rs`
- GPU branches in `src/circuit/mod.rs`, `src/lookup/ggh15_eval.rs`, and related evaluable paths
- GPU tests/benches under `tests/test_gpu_*` and `benches/*_gpu.rs`
- native build boundary in `build.rs` that compiles `cuda/src/*` and links CUDA runtime libraries

Architecture implication: `gpu` is an interface/implementation split on top of shared traits (`Poly`, `PolyMatrix`, sampler traits, evaluable traits), not a separate protocol API.

## Interface vs implementation distinction

Shared interfaces:

- `crate::poly::Poly` and `crate::poly::PolyParams`
- `crate::matrix::PolyMatrix`
- `crate::sampler::{PolyHashSampler, PolyUniformSampler, PolyTrapdoorSampler}`
- `crate::circuit::evaluable::Evaluable`

Implementation variants:

- CPU implementations are always available.
- GPU implementations are compiled only when `gpu` is enabled and native CUDA requirements are satisfied.

Any change that moves feature-gated code across domain boundaries must update architecture scope docs and this feature document.
