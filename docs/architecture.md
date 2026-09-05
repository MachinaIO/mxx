# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate. Consumers depend directly
on the crate that owns an abstraction.

## Dependency layers

```text
mxx-runtime              -> mxx-ir-core, mxx-primitives
mxx-bench-estimator      -> mxx-ir-core, mxx-runtime
mxx-dsl                  -> mxx-ir-core
mxx-gadgets              -> mxx-dsl, mxx-ir-core, mxx-primitives, mxx-runtime
mxx-bgg                  -> mxx-dsl, mxx-gadgets, mxx-ir-core
mxx-we                   -> mxx-bgg, mxx-ir-core, mxx-gadgets, mxx-runtime
mxx-func-enc/io          -> lower layers when their application modules are enabled
```

Application crates never depend on one another. Diamond WE is active in `mxx-we`; functional
encryption and iO protocol modules remain disabled.

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
Indexed `Family` operations create structural parallel loops.

### `mxx-runtime`

Executes validated schedules on CPU or GPU primitive backends and owns runtime values, sampling
transcripts, sessions, artifacts, and bounded parallel waves.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns BGG+-specific keys, encodings, sampling, evaluation, lookup, decoding, artifacts,
slot transfer, and refresh. Both build executable graphs through `mxx-dsl`.

### Application crates

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search uses deterministic
worst-case bounds and accepts a candidate only after Lean checks the generated theorem for the
same frozen workflow, backend layout, and concrete parameter environment. The selected candidate
retains its checked artifact; numerical rejection and checker failures remain distinct.

`mxx-func-enc` and `mxx-io` currently expose compiling interface shells. Their protocol modules
remain disabled. They need application-owned correctness implementations before being enabled.

Tall's old-simulator-dependent parameter search and noisy verification modes are explicitly
unavailable pending a Tall-specific correctness implementation. The independent noiseless runtime
round-trip remains available; it is not a substitute for a proved noisy bound.

## Generated Lean artifacts

Diamond parameter search generates and checks Lean artifacts through the production library API;
the GPU integration test uses that same search. No separate example executable is required.
Crates do not contain example targets: reusable extraction fixtures live in ordinary unit-test
modules, and generated files belong under ignored `test_data` or temporary artifact directories.
