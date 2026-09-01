# mxx

`mxx` is a Rust and CUDA workspace for lattice-cryptography research. It contains
polynomial and matrix primitives, bounded samplers, an executable graph IR and DSL, reusable BGG+
and circuit gadgets, runtime backends, and a concrete executable-IR noise simulator.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `mxx-primitives` | Polynomial/matrix operations, CPU/GPU kernels, and concrete samplers. |
| `mxx-ir-core` | Serializable executable DAG, parameter/type validation, and artifact manifests. |
| `mxx-dsl` | Typed graph construction, rank-N families, and relation-bearing preimages. |
| `mxx-runtime` | CPU/GPU graph execution, transcripts, sessions, and in-memory artifacts. |
| `mxx-noise-simulator` | Concrete coefficient-noise interpretation of frozen executable graphs. |
| `mxx-bench-estimator` | Validated-graph cost and memory composition. |
| `mxx-gadgets` | BGG-independent circuits and reusable circuit gadgets. |
| `mxx-bgg` | BGG+ keys, encodings, sampling, evaluation, decoding, lookup, slot transfer, and refresh. |
| `mxx-we` | Witness-encryption interfaces and parameterized dynamic-circuit Diamond WE. |
| `mxx-func-enc`, `mxx-io` | Application interfaces whose protocol modules are currently disabled. |

The simulator uses enforced integer coefficient cutoffs and deterministic worst-case bounds. It
returns numeric bounds rather than proving functional correctness or decoder acceptance;
applications apply their own decoder policy. Lattice-security estimation separately models the
corresponding ordinary untruncated distributions.

See `docs/architecture.md`, `docs/dsl.md`, `docs/ir-core.md`, `docs/runtime.md`, and
`docs/noise-simulator-spec.md`.

## Requirements

- Rust with edition 2024 support.
- OpenFHE and OpenMP.
- CUDA toolkit for the optional `gpu` feature.

Rust formatting uses `cargo +nightly fmt --all`.
