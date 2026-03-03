# Scope Index

This directory maps implementation scopes to concrete paths.

Read this file before reading individual scope documents.

## Dependency direction used in this index

Dependency statements in this scope index use implementation direction:

- Scope A depends on scope B if code in A imports, calls, or otherwise requires code in B.
- Example: `lookup` depends on `poly` and `matrix`.

## Scope documents

`src` directory scopes (one per top-level directory):

- [root_modules.md](./root_modules.md)
- [bgg.md](./bgg.md)
- [circuit.md](./circuit.md)
- [commit.md](./commit.md)
- [element.md](./element.md)
- [gadgets.md](./gadgets.md)
- [lookup.md](./lookup.md)
- [matrix.md](./matrix.md)
- [poly.md](./poly.md)
- [sampler.md](./sampler.md)
- [simulator.md](./simulator.md)
- [storage.md](./storage.md)

Additional independent scopes:

- [tests.md](./tests.md)
- [benches.md](./benches.md)
- [automation_orchestration.md](./automation_orchestration.md)

## Current scope dependency statements

Foundation scopes:

- `poly` depends on `element`.
- `matrix` depends on `poly` and `element`.
- `sampler` depends on `matrix` and `poly`.

Protocol and workflow scopes:

- `root_modules` is a cross-cutting support scope used by multiple directories.
- `bgg` depends on `matrix`, `poly`, and `sampler`.
- `storage` depends on `matrix` and `poly`.
- `commit` depends on `matrix`, `poly`, `sampler`, and `storage`.
- `lookup` depends on `bgg`, `circuit`, `matrix`, `poly`, `sampler`, and `storage`.
- `circuit` depends on `poly` and `lookup`.
- `gadgets` depends on `circuit`, `lookup`, and `poly`.
- `simulator` depends on `circuit`, `lookup`, and `poly`.

Operational scopes:

- `tests` depends on protocol/foundation scopes and validates end-to-end behavior.
- `benches` depends on performance-critical runtime scopes (`matrix`, `poly`, `sampler`).
- `automation_orchestration` depends on repository toolchain boundaries (`git`, `gh`, `codex`, `jq`) and event-skill infrastructure.

Known mutual dependency:

- `circuit` and `lookup` currently depend on each other through evaluator traits and lookup integration points. This is documented behavior and should be treated carefully in refactors.

## CUDA boundary note

CUDA files are documented inside Rust scopes that invoke them:

- `matrix` scope includes CUDA matrix kernels under `cuda/src/matrix/*` and `cuda/include/matrix/*`.
- `poly` scope includes runtime/utility CUDA bindings under `cuda/src/{Runtime,ChaCha}.cu` and `cuda/include/{Runtime,ChaCha}.cuh`.
