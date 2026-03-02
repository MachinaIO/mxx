# Scope: `benches`

## Purpose

Contains benchmark entrypoints for CPU/GPU matrix multiplication and preimage workflows.

## Implementation mapping

- `benches/bench_matrix_mul_cpu.rs`
- `benches/bench_matrix_mul_gpu.rs`
- `benches/bench_preimage_cpu.rs`
- `benches/bench_preimage_gpu.rs`

## Interface vs implementation

- This scope does not define production interfaces.
- It measures implementation behavior of runtime/protocol paths under benchmark harnesses.

## Depends on scopes

- `matrix`
- `poly`
- `sampler`
