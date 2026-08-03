# mxx

`mxx` is a Rust, CUDA, and Lean workspace for lattice-cryptography research. It contains
polynomial and matrix primitives, bounded samplers, an executable graph IR and DSL, reusable BGG+
and circuit gadgets, runtime backends, and a kernel-checked perfect-correctness pipeline.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `mxx-primitives` | Polynomial/matrix operations, CPU/GPU kernels, and concrete samplers. |
| `mxx-ir-core` | Serializable executable DAG, parameter/type validation, and artifact manifests. |
| `mxx-dsl` | Typed graph construction and sampler-free ideal/predicate builders. |
| `mxx-runtime` | CPU/GPU graph execution, transcripts, sessions, and in-memory artifacts. |
| `mxx-correctness` | Common correctness declarations, Lean emission, and theorem verification helpers. |
| `mxx-bench-estimator` | Validated-graph cost and memory composition. |
| `mxx-gadgets` | BGG-independent circuits and reusable circuit gadgets. |
| `mxx-bgg` | BGG+ keys, encodings, sampling, evaluation, decoding, lookup, slot transfer, and refresh. |
| `mxx-func-enc`, `mxx-we`, `mxx-io` | Application interfaces; protocol modules are temporarily disabled during correctness migration. |

The retired symbolic IR and probabilistic noise simulator are not part of the workspace.
Correctness uses enforced integer coefficient cutoffs and deterministic worst-case bounds. CPU
samplers implement the current runtime-correspondence contract; GPU cutoff enforcement is tracked
as a follow-up. Lattice-security estimation intentionally continues to model the corresponding
ordinary untruncated distributions separately.

See `docs/architecture.md`, `docs/dsl.md`, `docs/ir-core.md`, `docs/runtime.md`, and
`docs/lean.md`.

## Requirements

- Rust with edition 2024 support.
- Lean 4.32.0 for the checked correctness package.
- OpenFHE and OpenMP.
- CUDA toolkit for the optional `gpu` feature.

Rust formatting uses `cargo +nightly fmt --all`. Run `scripts/verified_build.sh` when a build must
also regenerate and verify the checked-in Lean definitions and proofs.
