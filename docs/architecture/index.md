# Architecture Documentation Index

Read this file first before reading any other architecture document.

This index is the entrypoint for architecture context in this repository. It explains where architecture information lives and where to start when a change may affect structure, feature boundaries, or external dependencies.

## Reading order

1. Read this file.
2. Read [scope/index.md](./scope/index.md) for implementation-domain mapping.
3. Read [features/index.md](./features/index.md) for compile-time and runtime feature model.
4. Read [dependencies/index.md](./dependencies/index.md) for Rust/native/toolchain dependency boundaries.
5. Read documents under `structure/` when that area is populated.

## Document map

- Scope (implementation structure): [scope/index.md](./scope/index.md)
- Features: [features/index.md](./features/index.md)
- Dependencies: [dependencies/index.md](./dependencies/index.md)
- Structure (reserved): `docs/architecture/structure/`

## Current implementation anchors

The current architecture scope documents are mapped to these implementation roots:

- Rust library and modules: `src/`
- CUDA/C++ boundary code: `cuda/`
- Build integration: `build.rs`
- Test and benchmark entrypoints: `tests/`, `benches/`
- ExecPlan verification automation: `.agents/skills/`, `scripts/`

## Maintenance rule

Architecture documentation is long-lived and must stay synchronized with implementation. When structural rules, module boundaries, feature-flag shape, dependency/tooling assumptions, or FFI/CUDA/build boundaries change, update architecture documents in the same pull request.

To prevent dependency-model mistakes in scope documentation, apply these maintenance rules during every architecture update:

1. Determine dependencies from implementation usage first (imports, trait bounds, function calls, and file ownership), not from naming or conceptual layer labels.
2. Record dependency direction as "consumer depends on provider."
   Example: if `lookup` imports or uses `poly` and `matrix`, then `lookup` depends on `poly` and `matrix`.
3. For mixed-language paths, include native files in the Rust scope that invokes them.
   Example: CUDA kernels belong to Rust GPU scopes that call them through bindings or FFI.

If a dependency statement is not backed by concrete code-path evidence, do not document it as architecture fact.
