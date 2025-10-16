# Repository Guidelines

During development, consult and follow any scope-specific `*.md` guides under `agents/`.
Write all comments and documentation in English regardless of the language used in user requests.

## Project Structure & Module Organization
- `Cargo.toml` defines the optional `disk` feature and the default `debug` profile.
- `src/` holds the lattice primitives: `matrix/`, `sampler/`, `circuit/`, `arithmetic/`, plus support files like `rlwe_enc.rs` and `utils.rs`.
- `tests/` stores integration suites such as `test_arithmetic_circuit.rs`; unit tests live beside modules under `#[cfg(test)]`.
- `test_data/` keeps deterministic fixtures consumed by integration tests.
- `build.rs` emits linker flags for OpenFHE (`OPENFHE*`) and OpenMP; adjust it when adding native libs.

## Build, Test, and Development Commands
- `cargo check` validates compilation quickly with the `debug` feature; run before committing.
- `cargo build [--release]` produces artifacts; add `--features disk` when working on disk-backed storage.
- `cargo test [--features disk]` runs unit and integration suites; append `-- --nocapture` to surface println output.
- `cargo fmt --all` enforces the shared formatting rules.

## Coding Style & Naming Conventions
- Follow Rust defaults: 4-space indentation, `snake_case` modules/functions, `PascalCase` types, and `SCREAMING_SNAKE_CASE` constants.
- Run `rustfmt` with the bundled `rustfmt.toml` (crate-scope import reordering, wrapped comments at 100 columns).
- Use `pub(crate)` to limit visibility unless APIs must be public.

## Testing Guidelines
- Favor deterministic seeds (`StdRng`, explicit `seed_from_u64`) for reproducibility.
- Integration tests belong in `tests/` with descriptive `test_*` filenames; co-locate fast unit tests inside their modules.
- Place shared fixtures in `test_data/` and document formats in test headers.
- Run `cargo test --all-features` before review when feature flags or native deps change.

## Commit & Pull Request Guidelines
- Match recent history: start with a scope or type (`fix:`, `Feat/...`) followed by a concise summary and optional PR number.
- Keep subject lines under ~72 characters; describe rationale, benchmarks, and feature flags in the body.
- Pull requests should link issues, list validation commands (`cargo test`, etc.), and attach evidence when behavior changes.

## Security & Configuration Tips
- Install the Machina-iO OpenFHE fork (branch `feat/improve_determinant`) to `/usr/local/lib`; `build.rs` already sets `rpath`, so no `LD_LIBRARY_PATH` hacks are needed.
- Avoid committing secrets or local overrides; prefer updating `.env.example` or the README when new settings emerge.
- Document additional native requirements (compiler flags, pkg-config hints) in this guide and the README so downstream users stay aligned.
