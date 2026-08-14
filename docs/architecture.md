# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate. Consumers depend directly
on the crate that owns an abstraction.

## Dependency layers

```text
mxx-runtime              -> mxx-ir-core, mxx-primitives
mxx-bench-estimator      -> mxx-ir-core, mxx-runtime
mxx-dsl                  -> mxx-ir-core
mxx-correctness          -> mxx-ir-core, mxx-dsl
mxx-gadgets              -> mxx-dsl, mxx-ir-core, mxx-primitives, mxx-runtime
mxx-bgg                  -> mxx-dsl, mxx-gadgets, mxx-ir-core
mxx-we                   -> mxx-bgg, mxx-correctness, mxx-gadgets, mxx-runtime
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
decidable compile-parameter conditions consumed by concrete validation and Lean emission. Sampler
nodes serialize required integer coefficient cutoffs. Subgraph and parallel-loop bodies are
structural and stored once.

### `mxx-dsl`

Creates immutable core nodes immediately. It has no symbolic reinterpretation layer.
`IdealSpec` and `PurePredicateSpec` accept only sampler-free graphs for correctness declarations.
Indexed `Family` operations create structural parallel loops.

### `mxx-runtime`

Executes validated schedules on CPU or GPU primitive backends and owns runtime values, sampling
transcripts, sessions, artifacts, and bounded parallel waves.

### `mxx-correctness`

The library validates linked workflow declarations, emits checked-in Lean closed-protocol terms,
and verifies source, workflow, and toolkit hashes together with theorem axiom dependencies. Emission is mechanical from `ProtocolDecl`;
generated IR contains a Lean constructor tree, not an embedded JSON string. Each crate owns its
protocol declarations, generated Lean modules, proofs, and small generation and verification
examples. There is no central protocol registry or parameter-check executable. The shared Lean
semantics remain in `lean/Mxx`; the ownership and build flow are documented in `docs/lean.md`.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns BGG+-specific keys, encodings, sampling, evaluation, lookup, decoding, artifacts,
slot transfer, and refresh. Both build executable graphs through `mxx-dsl`.

### Application crates

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search uses deterministic
worst-case bounds and accepts a candidate only after the generic Rust operational checker accepts
the frozen workflow and concrete parameter environment.

`mxx-func-enc` and `mxx-io` currently expose compiling interface shells. Their protocol modules
remain disabled. Diamond iO must migrate to the Rust operational checker before it is enabled.
