# Scope: `src/matrix`

## Purpose

Defines matrix interfaces and concrete matrix implementations for polynomial and utility matrices, including GPU-backed implementations.

## Implementation mapping

- `src/matrix/mod.rs`
- `src/matrix/base/`
- `src/matrix/dcrt_poly.rs`
- `src/matrix/gpu_dcrt_poly.rs`
- `src/matrix/i64.rs`
- `src/matrix/cpp_matrix.rs`

CUDA files included in this scope (used via Rust GPU matrix implementation and FFI bindings):

- `cuda/src/matrix/Matrix.cu`
- `cuda/src/matrix/MatrixArith.cu`
- `cuda/src/matrix/MatrixData.cu`
- `cuda/src/matrix/MatrixDecompose.cu`
- `cuda/src/matrix/MatrixNTT.cu`
- `cuda/src/matrix/MatrixSampling.cu`
- `cuda/src/matrix/MatrixSerde.cu`
- `cuda/src/matrix/MatrixTrapdoor.cu`
- `cuda/src/matrix/MatrixUtils.cu`
- `cuda/include/matrix/*.cuh`

## Interface vs implementation

- Interfaces:
  - `crate::matrix::MatrixParams`
  - `crate::matrix::MatrixElem`
  - `crate::matrix::PolyMatrix`
- Concrete implementations:
  - CPU DCRT matrix: `DCRTPolyMatrix` in `src/matrix/dcrt_poly.rs`
  - GPU DCRT matrix: `GpuDCRTPolyMatrix` in `src/matrix/gpu_dcrt_poly.rs`
  - Auxiliary implementations: `I64Matrix` and C++ interop matrix

## Depends on scopes

- `poly`
- `element`

## Used by scopes

- `sampler`
- `bgg`
- `lookup`
- `commit`
- `storage`
- `benches`
