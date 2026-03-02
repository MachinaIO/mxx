# Runtime Feature Switches (`src/env.rs`)

This document defines environment-variable runtime controls that shape behavior after compilation.

## Why runtime switches are architecture-relevant

Even with fixed Cargo features, runtime behavior can change significantly based on environment variables. These switches control parallelism, batching, and artifact chunking across circuit/lookup/commit/storage paths.

## Runtime controls and ownership

- `MXX_CIRCUIT_PARALLEL_GATES`
  - Source: `env::circuit_parallel_gates`
  - Main consumer: `src/circuit/mod.rs`
  - Behavior: max gates processed in parallel per circuit level; GPU path enforces value <= detected GPU device count.

- `LUT_PREIMAGE_CHUNK_SIZE`
  - Source: `env::lut_preimage_chunk_size`
  - Main consumers: `src/lookup/ggh15_eval.rs`, `src/lookup/lwe_eval.rs`
  - Behavior: batch size for LUT/gate preimage workloads.

- `GGH15_GATE_PARALLELISM`
  - Source: `env::ggh15_gate_parallelism`
  - Main consumer: `src/lookup/ggh15_eval.rs`
  - Behavior: parallel gate-preimage processing limit.

- `BLOCK_SIZE`
  - Source: `env::block_size`
  - Main consumers: matrix/block-oriented paths (`src/matrix/base/*`, `src/matrix/dcrt_poly.rs`, `src/matrix/gpu_dcrt_poly.rs`, utility wrapper in `src/utils.rs`)
  - Behavior: generic block/chunk size for processing loops.

- `LUT_BYTES_LIMIT`
  - Source: `env::lut_bytes_limit`
  - Main consumer: `src/storage/write.rs`
  - Behavior: optional cap for lookup-table payload chunking.

- `WEE25_TOPJ_PARALLEL_BATCH`
  - Source: `env::wee25_topj_parallel_batch`
  - Main consumer: `src/commit/wee25.rs`
  - Behavior: batch size for Wee25 `top_j` generation.

- `WEE25_COMMIT_CACHE_PERSIST_BATCH`
  - Source: `env::wee25_commit_cache_persist_batch`
  - Main consumer: `src/commit/wee25.rs`
  - Behavior: commit-cache buffering threshold before persistence.

## CPU vs GPU default behavior

`src/env.rs` defines different defaults depending on whether `gpu` is enabled:

- GPU-enabled defaults generally derive from detected GPU device count for parallel/batch knobs.
- Non-GPU defaults use fixed numeric fallbacks (for example 30 for certain lookup batching knobs).

Architecture implication: runtime tuning defaults are part of feature behavior and must be documented whenever default values or constraints change.

## Update triggers

Update this document when:

- any variable is added, removed, renamed, or semantics/defaults change in `src/env.rs`,
- a variable gains new consumers in another domain,
- GPU/CPU divergence rules for defaults or validation change.
