# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/` as a Rust library (`lib.rs`) with domain modules such as `matrix/`, `poly/`, `sampler/`, `lookup/`, `circuit/`, and `commit/`. Integration tests are in `tests/` and follow scenario-oriented files like `test_lwe_modq_arith.rs` and `test_gpu_ggh15_modq_arith.rs`. Benchmarks are in `benches/` (custom harness binaries). CUDA sources and headers are in `cuda/src` and `cuda/include`, compiled via `build.rs` when the `gpu` feature is enabled. Use `test_data/` for generated artifacts and `docs/` for design/performance notes.

## Build, Test, and Development Commands
- `cargo build`: Build CPU path.
- `cargo build --features gpu`: Build with CUDA support (`nvcc` required).
- `cargo +nightly fmt --all --check`: CI formatting gate.
- `cargo test -r --lib`: Run all unit tests.
- `cargo test -r --lib --features gpu`: Run all unit tests with the `gpu` feature.
- `cargo test -r gpu --lib --features gpu`: Run all gpu-specific unit tests with the `gpu` feature.
- `cargo test --test test_gpu_ggh15_modq_arith --features gpu -- --nocapture`: Run a GPU integration test.
- `cargo bench --bench bench_preimage_gpu --features gpu`: Run GPU benchmark binary.

## Coding Style & Naming Conventions
Use Rust 2024 idioms and format with `rustfmt.toml` (nightly rustfmt options are used). Follow 4-space indentation and keep code rustfmt-clean before opening a PR. Naming: `snake_case` for modules/files/functions, `CamelCase` for types/traits, and `SCREAMING_SNAKE_CASE` for constants. Keep feature-gated code explicit (`#[cfg(feature = "gpu")]`) and colocated with CPU equivalents where practical.

## Testing Guidelines (Definition of Done)
Prefer targeted test runs first (`cargo test --test <name>`), then run all unit tests. GPU tests require GPUs and CUDA runtime. After you finish modifying cuda or gpu-specific rust codes, run all gpu-specific unit tests 300 times (outside a sandbox) by compiling the modfied codes once and then running the binary sequentially.

## Commit & Pull Request Guidelines
Recent history favors short, imperative, lowercase subjects (for example: `fix device id bugs`, `cache ntt constants`). Keep commits scoped to one logical change. PRs should include: purpose, key implementation notes, commands run (`fmt`, `clippy`, tests), and environment details (CPU/GPU, CUDA arch). Attach benchmark deltas for performance-sensitive changes.
