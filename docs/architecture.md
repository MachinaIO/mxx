# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate.
Consumers depend directly on the crate that owns an abstraction.

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

Application crates do not depend on one another. The authoritative crate list
is the workspace member list in `Cargo.toml`; the dependency rules are also
summarized in `docs/architecture.md`.

## Responsibilities

### `mxx-primitives`

Owns polynomial and matrix representations, OpenFHE integration, concrete
sampling, and native CUDA. CPU and GPU samplers enforce the authoritative
integer and centered-norm cutoffs. GPU matrix operations expose low-level
saved-layout bind primitives and native completion dependencies; they do not
own graph scheduling or runtime binding checks.

### `mxx-ir-core`

Owns the canonical executable graph, compile expressions, artifact metadata,
parameter/type/shape validation, execution ordering, and liveness.
`derive_param_constraints` is the shared source of decidable compile-parameter
conditions. Sampler nodes serialize their integer coefficient cutoffs.
Subgraph and parallel-loop bodies are structural and stored once.

The `protocol` modules own declarations, input contracts, frozen graph
annotations, sampler-free ideal/predicate specifications, and structural
validation of linked workflows. The Lean exporter generates primitive
execution relations and linked claims from explicit graph endpoints; it does
not infer noise bounds or expand structural families into lanes.

### `mxx-dsl`

Creates immutable core nodes immediately and has no symbolic reinterpretation
layer. `Family<T>` preserves composite element schemas. `parallel` and
`iterate` create structural loops, and lexical reads become explicit core
dependencies with inferred member indexing.

### `mxx-runtime`

Executes validated schedules on CPU or on the prepared GPU primitive backend.
CPU execution is independent of GPU preparation. GPU execution has exactly
three boundaries:

1. Compile validates the graph and lowers it to a fixed typed recipe.
2. Warmup resolves saved native descriptors, exact owners and layouts, fixed
   preimage lanes, output lifetimes, and reusable slot capacity.
3. Execute validates runtime bindings and replays the published commands.

Warmup uses one resolver transaction for retained owners, workspace, and all
execution slots. A failed transaction releases its complete partial state.
Execution can reach only `PreparedGpuProgram::run_with_runtime_bindings`; an
unprepared or structurally mismatched request returns `NotPrepared` before
slot acquisition and GPU submission. There is no dynamic graph walk, runtime
resource decision, implicit warmup, dynamic scheduler, or fallback runner in
the production GPU path.

Native operation binds consume resolver-provided saved descriptors. Matrix
contracts include rows, columns, ordered CRT basis and parameters, level,
representation, device/context, and owner layout. Families and trapdoors also
carry their fixed shape and construction metadata. These checks happen before
slot acquisition and never replace a fixed owner with a mismatched value.

Preimage resources use fixed lanes. Fresh and Record sampling use submit-time
freshness; Record stores the actual accepted native payload and Replay validates
and uploads the recorded payload without resampling. Runtime scalar values use
an arbitrary-precision `BigInt` capacity ledger. A value beyond the prepared
capacity is an input error, not a request to replan geometry.

Prepared slots advance through acquire, submit, publish, and terminal-event
retirement. Reader and writer events—not host handle drop—control reuse.
Input/capacity/exhaustion failures are recoverable. A native failure poisons
the slot and removes it from reuse. `max_parallel_instances` and
`max_live_gpu_executions` bound different resources and are accounted for
separately. A pool exhaustion wait observes existing terminal events and does
not make a second reservation.

### `mxx-bench-estimator`

The estimator builds prepared mini-programs for representative operation and
wave classes. It measures each missing class during explicit setup, freezes
the resulting table, and performs CPU-only lookup during report generation.
It reports aggregate device work, cumulative prepared-wave time, dependency
latency, workspace observations, and separately owned dataflow/materialization
costs. It never executes the application graph to discover a class and never
uses a missing measurement as a guessed singleton.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns reusable circuit gadgets. `mxx-bgg` owns BGG+-specific keys,
encodings, sampling, evaluation, lookup, decoding, artifacts, and refresh.
Both construct executable graphs through `mxx-dsl`.

### Application crates

`mxx-fhe` builds Ring Regev/Ring-GSW and leveled BGV graphs, including CRT
modulus conversion and hybrid RNS key switching. CPU and prepared GPU backends
execute the same FHE graphs; GPU centered basis conversion uses native CRT
residues and stream-ordered transforms without a host coefficient round trip.

`mxx-we` owns the implementation-independent witness-encryption declarations
and Diamond protocol. Parameter search retains the artifact checked by Lean.
`mxx-func-enc` and `mxx-io` expose common interface traits only.

## Generated Lean artifacts

Each crate keeps handwritten Lean modules directly under `lean/`, without a
nested package-name directory. Shared modules use crate-qualified filenames.
Lake libraries list module roots explicitly. Entry modules such as
`MxxPrimitives.lean`, `MxxRuntime.lean`, and `MxxIR.lean` collect reusable
imports. Generated files belong under ignored test-data or temporary artifact
directories; no example executable is required.

## Validation requirements

For runtime or GPU changes, run `cargo +nightly fmt --all`,
`git diff --check`, the relevant crate library checks, and targeted `--no-run`
tests for primitives and runtime. GPU behavior and multi-device claims require
the applicable hardware tests. Integration tests are not part of the default
validation and require explicit approval.
