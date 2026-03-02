# Scope: `tests`

## Purpose

Holds integration tests that exercise cross-scope behavior and end-to-end flows for commit, lookup, LWE, GGH15, and GPU-specific paths.

## Implementation mapping

- `tests/test_commit_modp_chain.rs`
- `tests/test_commit_modq_arith.rs`
- `tests/test_ggh15_modp_chain.rs`
- `tests/test_ggh15_modq_arith.rs`
- `tests/test_lwe_modp_chain.rs`
- `tests/test_lwe_modq_arith.rs`
- `tests/test_gpu_ggh15_modp_chain.rs`
- `tests/test_gpu_ggh15_modp_multi.rs`
- `tests/test_gpu_ggh15_modq_arith.rs`

## Interface vs implementation

- This scope does not define production interfaces.
- It validates behavior by consuming public APIs from runtime/protocol scopes.

## Depends on scopes

- `bgg`
- `circuit`
- `commit`
- `gadgets`
- `lookup`
- `matrix`
- `poly`
- `sampler`
- `simulator`
- `storage`
