# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate. Consumers depend directly
on the crate that owns an abstraction.

## Dependency layers

```text
mxx-runtime              -> mxx-ir-core, mxx-primitives
mxx-bench-estimator      -> mxx-ir-core, mxx-runtime; optional mxx-primitives
mxx-dsl                  -> mxx-ir-core
mxx-gadgets              -> mxx-dsl, mxx-ir-core, mxx-primitives, mxx-runtime
mxx-bgg                  -> mxx-dsl, mxx-gadgets, mxx-ir-core, mxx-primitives
mxx-fhe                  -> mxx-dsl, mxx-ir-core, mxx-primitives
mxx-we                   -> mxx-bgg, mxx-ir-core, mxx-gadgets, mxx-runtime
mxx-func-enc/io          -> interface-only crates with no dependencies
```

Application crates never depend on one another. Diamond WE is active in `mxx-we`; functional
encryption and iO protocol implementations have been removed during the DSL migration.

## Responsibilities

### `mxx-primitives`

Owns polynomial and matrix representations, OpenFHE integration, concrete sampling, and native
CUDA. CPU Gaussian sampling resamples individual coefficients outside the authoritative integer
cutoff. CPU preimage sampling rejects a whole candidate outside its cutoff so `B * K = P` is
preserved. GPU Gaussian sampling enforces the same cutoff per coefficient in CUDA. Batched GPU
preimage sampling rejects a whole GPU-generated candidate after full-CRT centered-norm checking,
preserving both the preimage equation and the authoritative cutoff.

### `mxx-ir-core`

Owns the canonical executable graph, compile expressions, artifact metadata, parameter/type/shape
validation, execution ordering, and liveness. `derive_param_constraints` is the shared source of
decidable compile-parameter conditions consumed by concrete validation. Sampler
nodes serialize required integer coefficient cutoffs. Subgraph and parallel-loop bodies are
structural and stored once.

`protocol` owns protocol declarations, input contracts, frozen graph annotations, sampler-free
ideal/predicate specifications, and structural validation of linked workflows. These are core
graph data and checks, independent of the DSL used to construct a graph. There is no separate
correctness crate and no generic symbolic noise simulator.

The Lean exporter owns primitive execution-relation generation and application-independent linked
claim assembly. It receives explicit graph connections and endpoint semantics, not a WE protocol
implementation, and does not infer noise bounds or expand structural families into individual lanes.
`lean::protocol` converts a protocol declaration into exported roots and a linked claim;
`lean::claim` renders the final proposition. Applications supply backend bindings and decoder
semantics, while their mathematical bounds and proofs remain application-owned.

### `mxx-dsl`

Creates immutable core nodes immediately. It has no symbolic reinterpretation layer.
The constructed graphs feed core-owned `IdealSpec` and `PurePredicateSpec` validation.
Indexed `Family<T>` values preserve composite element schemas. `parallel` and `iterate` create
structural loops; lexical reads become explicit core dependencies with inferred member indexing.

### `mxx-runtime`

Executes validated schedules on CPU or GPU primitive backends and owns runtime values, sampling
transcripts, sessions, artifacts, and bounded parallel waves.
Temporary families never spill to disk. The GPU fleet completes one family member
before starting the next and stages intermediate family results in host RAM. Device
parallelism is internal to primitives: calibrated column widths share the configured
VRAM percentage and are recalculated against current residency for each invocation.
CPU backends retain bounded sibling waves, including across subgraph calls. Values
are released at their final live reference, including captures and member aliases.
Root Family exports stream directly to their final artifact paths; the manifest is published
only after execution succeeds. Other exports write their RAM payloads to final storage.
Durable exported artifacts remain available for online execution.

The common estimator tracks runtime storage boundaries across Family, capture,
subgraph, and sequential-loop values. It measures raw-RNS RAM staging/reloading
and compact artifact encoding/writing/reading/decoding with the production backend
and artifact store, then adds their invocation-weighted wall times to
`total_time_seconds`. GPU-resident edges have no added transfer charge;
`total_work_seconds` remains primitive GPU work. The runtime and estimator share
the matrix-Family staging rule and artifact codecs.

`dataflow` and `transfers` report the extra costs and counts separately. CPU IR
dispatch is a measured scalar-node proxy, not a full protocol execution trace.
Artifact import calibration includes public content-hash verification (also
charged conservatively to private imports). Measurements sum boundary costs;
they do not predict overlap with unrelated operations, cold-cache storage
behavior, or contention in a complete execution. Large transfer shapes use a
VRAM-bounded representative wave and round the final partial wave upward.

Compact serialization remains the durable-export format. GPU RNS transfers expose D2H
start/completion separately, and H2D sources are owned by the existing event-based pinned
memory reclaimer. This permits a previous output store and next input load to overlap.
Fleet snapshots pipeline two shards and reuse two pinned buffers, bounding transfer
scratch independently of the number of column chunks.
Completed scalar exports are written immediately and retained as artifact handles, including
members projected from batch families.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns BGG+-specific keys, encodings, sampling, evaluation, lookup, decoding, artifacts,
slot transfer, and refresh. Both build executable graphs through `mxx-dsl`.

### Application crates

`mxx-fhe` builds Ring Regev/Ring-GSW and leveled BGV graphs, including CRT modulus
switching, hybrid RNS key switching over QP, relinearization, and rotations. BGV encrypt/decrypt exchange SIMD slots
by default, with internal encoding and zero-padding of short inputs. Cryptographic arithmetic
and sampling execute through the DSL runtime; runtime is a test-only dependency.
It tracks coefficient noise bounds per ciphertext and reuses primitive ring parameters and DSL
matrix handles. Bootstrapping is out of scope. CPU and GPU backends share the same
FHE graphs. GPU centered basis conversion uses native unsigned CRT residues and
stream-ordered INTT/lift/NTT operations without a host coefficient round trip.
Hybrid RNS ModUp/ModDown use dedicated graph nodes with an explicit ordered
source basis, checked by the runtime against registered parameters. CPU and CUDA
primitives fuse CRT accumulation between one input INTT and one output NTT per
digit, preserving the approximate centered-sum semantics and noise bounds.
FHE artifacts stay in memory or enter the protocol as direct runtime inputs.

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search uses deterministic
worst-case bounds and accepts a candidate only after Lean checks the generated theorem for the
same frozen workflow, backend layout, and concrete parameter environment. The selected candidate
retains its checked artifact; numerical rejection and checker failures remain distinct.

`mxx-func-enc` and `mxx-io` expose only their common interface traits. The disabled AKY24 FE,
AKY24 iO, and Diamond iO modules and their exclusive BGG helpers have been removed. See the
README for the `main` branch containing the latest iO implementations. Reusable implementations
in `mxx-gadgets` remain available.

Tall's old-simulator-dependent parameter search and noisy verification modes are explicitly
unavailable pending a Tall-specific correctness implementation. The independent noiseless runtime
round-trip remains available; it is not a substitute for a proved noisy bound.

## Generated Lean artifacts

Each crate keeps its handwritten Lean modules directly under `lean/`, without a nested package-name
directory. Shared modules have crate-qualified filenames such as `PrimitivesBounds.lean` and
`RuntimeMatrixOps.lean`, avoiding collisions when several packages share one Lean search path.
Lake libraries list their module roots explicitly. The `MxxPrimitives.lean`, `MxxRuntime.lean`,
`MxxIR.lean`, `MxxGadgets.lean`, and `MxxBgg.lean` entry modules collect reusable imports; mathematical
namespaces and theorem names are independent of this file layout.

Diamond parameter search generates and checks Lean artifacts through the production library API;
the GPU integration test uses that same search. No separate example executable is required.
Crates do not contain example targets: reusable extraction fixtures live in ordinary unit-test
modules, and generated files belong under ignored `test_data` or temporary artifact directories.
