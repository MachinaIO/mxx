# Native and Toolchain Dependencies

This document captures non-Rust dependencies and build/runtime toolchain requirements.

## Native library dependency boundary

`build.rs` unconditionally links against OpenFHE shared libraries:

- `-lOPENFHEpke`
- `-lOPENFHEbinfhe`
- `-lOPENFHEcore`
- search path `usr/local/lib`
- runtime rpath `usr/local/lib`

`build.rs` also links OpenMP (`-fopenmp`).

Architecture implication: CPU build and test paths assume OpenFHE and OpenMP libraries are available in the expected linker/runtime locations.

## CUDA dependency boundary (`gpu` feature)

When `CARGO_FEATURE_GPU` is set, `build.rs`:

- compiles CUDA/C++ sources under `cuda/src/` with headers in `cuda/include/`,
- uses `cc::Build::cuda(true)` and NVCC flags (`-std=c++17`, `-lineinfo`, `-Xcompiler -fPIC`, `-arch=sm_<CUDA_ARCH>`),
- links CUDA runtime libraries (`cudart`, `cudadevrt`),
- consumes environment variables `CUDA_ARCH`, `CUDA_HOME`, `CUDA_LIB_DIR`, and optionally `NVCC`.

Architecture implication: `gpu` is not only a Rust-level feature; it is a native toolchain boundary requiring CUDA SDK, NVCC, and compatible GPU/runtime libraries.

## CI dependency model

`.github/workflows/ci.yml` installs and builds native prerequisites:

- system packages: `build-essential`, `cmake`, `libgmp-dev`, `libasound2-dev`,
- OpenFHE source checkout/build/install from `MachinaIO/openfhe-development`,
- `ldconfig` update for `usr/local/lib`.

CI currently runs formatting, clippy (including `--features disk` path), and release lib tests without `gpu`.

## Autonomous orchestration tooling boundary

Repository-local autonomous PR orchestration under `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh` and `.agents/skills/eternal-cycler/scripts/run_builder_reviewer_doctor.sh` depends on CLI toolchain components in addition to Rust/CUDA toolchains:

- `git` (branch synchronization and worktree isolation),
- `gh` (PR metadata queries and PR comment posting),
- `codex` (non-interactive builder/reviewer agent execution),
- `jq` (JSON field extraction from `gh` output).

Architecture implication: autonomous-loop operation is an execution-time tooling boundary. Missing or broken CLI dependencies should be treated as runtime blockers for that scope even when Rust build/test paths are healthy.

## Update triggers

Update this document whenever:

- OpenFHE/CUDA/OpenMP link behavior changes in `build.rs`,
- required system packages or CI install/build steps change,
- environment-variable contract for native builds changes,
- GPU or native runtime assumptions change.
