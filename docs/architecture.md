# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate. Consumers depend directly
on the crate that owns an abstraction.

## Dependency layers

```text
mxx-runtime              -> mxx-ir-core, mxx-primitives
mxx-bench-estimator      -> mxx-ir-core, mxx-runtime
mxx-dsl                  -> mxx-ir-core
mxx-correctness          -> mxx-ir-core, mxx-dsl
mxx-gadgets              -> mxx-correctness, mxx-dsl, mxx-ir-core, mxx-primitives, mxx-runtime
mxx-bgg                  -> mxx-dsl, mxx-gadgets, mxx-ir-core
mxx-func-enc/we/io       -> lower layers when their application modules are enabled
```

Application crates never depend on one another. Their protocol modules are temporarily disabled
while application-specific hard-bound recurrences or certified checkers are developed.

## Responsibilities

### `mxx-primitives`

Owns polynomial and matrix representations, OpenFHE integration, concrete sampling, and native
CUDA. CPU Gaussian sampling resamples individual coefficients outside the authoritative integer
cutoff. CPU preimage sampling rejects a whole candidate outside its cutoff so `B * K = P` is
preserved. GPU enforcement is not yet part of runtime correspondence.

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

The library validates linked workflow declarations, emits checked-in Lean terms and statements,
and verifies theorem hashes and axiom dependencies. Emission is mechanical from `ProtocolDecl`;
generated IR contains a Lean constructor tree, not an embedded JSON string. Each crate owns its
protocol declarations, generated Lean modules, proofs, and small generation and verification
examples. There is no central protocol registry or parameter-check executable. The shared Lean
semantics remain in `lean/Mxx`; the ownership and build flow are documented in `docs/lean.md`.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns BGG+-specific keys, encodings, sampling, evaluation, lookup, decoding, artifacts,
slot transfer, and refresh. Both build executable graphs through `mxx-dsl`.

### Application crates

`mxx-func-enc`, `mxx-we`, and `mxx-io` currently expose compiling interface shells. Their
protocol graphs and parameter searches remain source-disabled until migrated to application-specific
hard bounds or verified correctness checkers.
