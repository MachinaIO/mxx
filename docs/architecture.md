# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate. Consumers depend directly
on the crate that owns an abstraction.

## Dependency layers

```text
mxx-runtime              -> mxx-ir-core, mxx-primitives
mxx-bench-estimator      -> mxx-ir-core, mxx-runtime; optional mxx-primitives
mxx-dsl                  -> mxx-ir-core
mxx-noise-simulator      -> mxx-ir-core
mxx-gadgets              -> mxx-dsl, mxx-ir-core, mxx-primitives; optional mxx-runtime
mxx-bgg                  -> mxx-dsl, mxx-gadgets, mxx-ir-core, mxx-primitives
mxx-we                   -> mxx-bench-estimator, mxx-bgg, mxx-dsl, mxx-gadgets,
                            mxx-ir-core, mxx-noise-simulator, mxx-primitives, mxx-runtime
mxx-func-enc/io          -> lower layers when their application modules are enabled
```

Application crates never depend on one another. `mxx-noise-simulator` is a lower-layer primitive
that may be consumed by applications, but it never depends on an application crate. Diamond WE is
active in `mxx-we`; functional encryption and iO protocol modules remain disabled.

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
decidable compile-parameter conditions consumed by concrete validation and noise simulation.
Sampler nodes serialize required integer coefficient cutoffs. Subgraph, `ParallelGrid`, and
`SequentialLoop` bodies are structural and stored once. Rank-N `Family` values retain logical axes
while runtime storage remains flat and row-major. `Preimage` is a distinct wire type consumed by
`ApplyPreimage`; ordinary matrix multiplication does not consume relations.

### `mxx-dsl`

Creates immutable core nodes immediately. It has no symbolic reinterpretation layer. Typed
`Preimage` and `Decomposition` wrappers retain relation identity, and rank-N `Family` operations
create structural `ParallelGrid` nodes.

### `mxx-runtime`

Executes validated schedules on CPU or GPU primitive backends and owns runtime values, sampling
transcripts, sessions, artifacts, and bounded parallel waves.

### `mxx-noise-simulator`

Interprets frozen `mxx-ir-core` graphs under one concrete parameter environment and returns maximum
absolute coefficient-error bounds for requested matrix roots. It has no dependency on the DSL,
runtime, gadgets, BGG+, or applications. Decoder policy and functional correctness remain
application responsibilities.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns only the generic BGG+ key, encoding, sampler, and safe known-value operations (plus
the pre-existing BGG modules). Application-specific roles, setup identities, orchestration, and
refresh semantics remain outside the reusable BGG layer. The generic `mxx-noise-simulator` remains
protocol-agnostic and may perform structural or request validation without becoming an application
correctness authority. The legacy `mxx-bgg::noise_refresh` implementation remains unchanged.

### Application crates

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search simulates the frozen
encryption/decryption stages and applies the Boolean decoder interval outside the simulator.

`mxx-func-enc` and `mxx-io` currently expose compiling interface shells. Their protocol modules
remain disabled.
