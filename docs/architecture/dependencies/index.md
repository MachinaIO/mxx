# Dependencies Architecture Index

This section documents external dependencies and the repository boundaries where they are used.

Read this file first before individual dependency documents.

## Reading order

1. Read this file.
2. Read [rust_crates.md](./rust_crates.md) for Rust-level dependency structure.
3. Read [native_and_toolchain.md](./native_and_toolchain.md) for native library and build/runtime toolchain dependencies.

## Scope

Dependency architecture in this repository includes:

- Rust crate dependencies from `Cargo.toml` (direct, optional, dev, and build dependencies),
- native libraries linked by `build.rs`,
- external tools and system packages required for build/test paths,
- CI environment assumptions that enforce dependency behavior.

## Maintenance rule

Update these documents when any of the following changes:

- `Cargo.toml` dependency set or feature dependency graph,
- native link arguments or CUDA/OpenFHE build behavior in `build.rs`,
- CI dependency installation/build steps in `.github/workflows/ci.yml`,
- system-level dependency assumptions for local development or testing.
