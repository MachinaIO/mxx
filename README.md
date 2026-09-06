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
| `mxx-func-enc`, `mxx-io` | Functional-encryption and iO interfaces; protocol implementations have been removed. |

The retired symbolic IR and probabilistic noise simulator are not part of the workspace.
Correctness uses enforced integer coefficient cutoffs and deterministic worst-case bounds. CPU
samplers implement the current runtime-correspondence contract; GPU cutoff enforcement is tracked
as a follow-up. Lattice-security estimation intentionally continues to model the corresponding
ordinary untruncated distributions separately.

See `docs/architecture.md`, `docs/dsl.md`, `docs/ir-core.md`, `docs/runtime.md`, and
`docs/correctness/operational-protocol-inventory.md`.

## Diamond iO and AKY24 iO implementations

Diamond iO and AKY24 iO were removed from this branch as part of the migration to a
DSL-based design. The disabled AKY24 functional-encryption implementation was also removed.
The latest implementations of Diamond iO and AKY24 iO remain on the
[`main` branch](https://github.com/MachinaIO/mxx/tree/main):
[Diamond iO](https://github.com/MachinaIO/mxx/blob/main/src/io/diamond_io.rs) and
[AKY24 iO](https://github.com/MachinaIO/mxx/blob/main/src/io/aky24_io.rs).
For a fixed reference, `main` pointed to
[`d5d6fba26f1d20f11d4648a3fd1c9b35241ff4a9`](https://github.com/MachinaIO/mxx/tree/d5d6fba26f1d20f11d4648a3fd1c9b35241ff4a9)
when this removal was made. Diamond WE remains available in this branch.

## Requirements

- Rust with edition 2024 support.
- OpenFHE and OpenMP.
- CUDA toolkit for the optional `gpu` feature.

Rust formatting uses `cargo +nightly fmt --all`.
