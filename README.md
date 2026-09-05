# mxx

`mxx` is a Rust and CUDA workspace for lattice-cryptography research. It contains
polynomial and matrix primitives, bounded samplers, an executable graph IR and DSL, reusable BGG+
and circuit gadgets, runtime backends, and application-specific Lean correctness proofs.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `mxx-primitives` | Polynomial/matrix operations, CPU/GPU kernels, and concrete samplers. |
| `mxx-ir-core` | Executable DAG, protocol declarations, structural validation, artifact manifests, and Lean claim generation. |
| `mxx-dsl` | Typed graph construction and sampler-free ideal/predicate builders. |
| `mxx-runtime` | CPU/GPU graph execution, transcripts, sessions, and in-memory artifacts. |
| `mxx-bench-estimator` | Validated-graph cost and memory composition. |
| `mxx-gadgets` | BGG-independent circuits and reusable circuit gadgets. |
| `mxx-bgg` | BGG+ keys, encodings, sampling, evaluation, decoding, lookup, slot transfer, and refresh. |
| `mxx-we` | Witness-encryption interfaces and parameterized dynamic-circuit Diamond WE. |
| `mxx-func-enc`, `mxx-io` | Application interfaces whose protocol modules are currently disabled. |

The retired symbolic IR and probabilistic noise simulator are not part of the workspace.
Correctness uses enforced integer coefficient cutoffs and deterministic worst-case bounds. CPU
samplers implement the current runtime-correspondence contract; GPU cutoff enforcement is tracked
as a follow-up. Lattice-security estimation intentionally continues to model the corresponding
ordinary untruncated distributions separately.

See `docs/architecture.md`, `docs/dsl.md`, `docs/ir-core.md`, `docs/runtime.md`, and
`docs/correctness/operational-protocol-inventory.md`.

## Requirements

- Rust with edition 2024 support.
- OpenFHE and OpenMP.
- CUDA toolkit for the optional `gpu` feature.

Rust formatting uses `cargo +nightly fmt --all`.
