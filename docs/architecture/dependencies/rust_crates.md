# Rust Crate Dependencies

This document describes direct Rust dependencies declared in `Cargo.toml` and how they map to implementation areas.

## Dependency groups

### Core runtime dependencies

- `openfhe` (Git dependency): core polynomial/matrix arithmetic backend and FFI surface.
- `num-bigint`, `num-traits`, `bigdecimal`: integer/decimal math for ring arithmetic and analysis code.
- `rayon`, `itertools`: parallel and iterator-heavy compute paths across matrix/poly and protocol code.
- `serde`, `serde_json`, `bincode`: serialization and artifact encoding.
- `dashmap`: concurrent map for evaluator/registry/cache state.
- `tokio`: async file/testing paths used in storage and several integration tests.
- `tracing`, `tracing-subscriber`: observability/logging.
- `rand`, `digest`, `bitvec`, `keccak-asm`: sampling/hash and bit-level processing.
- `memory-stats`: runtime memory introspection.
- `tempfile`: temporary artifact/test storage support.
- `thiserror`: error typing support.

### Optional dependencies tied to Cargo features

- `libc` and `memmap2` are optional and activated via the `disk` feature.
  - Primary implementation path: `src/matrix/base/disk.rs`.

### Development and build dependencies

- `sequential-test` (`dev-dependencies`): test sequencing anchor in `src/lib.rs` and test behavior control.
- `cc` (`build-dependencies`): native build bridge in `build.rs`, including CUDA compilation path when `gpu` is enabled.

## Direct feature-dependency relationship

From `Cargo.toml`:

- `default = []` (no implicit feature dependency)
- `disk = ["libc", "memmap2"]`
- `gpu = []` (code-path gated; native/toolchain requirements are enforced by build/runtime environment, not by extra Rust crates in `Cargo.toml`)

## Ownership boundaries

Rust crate dependencies should remain implementation details of domain modules and should not leak as architecture ambiguity. When adding or changing a crate:

- document the architectural reason in this dependencies section,
- document which domain owns the dependency,
- update feature docs when dependency behavior changes with feature flags.
