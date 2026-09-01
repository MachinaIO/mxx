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
mxx-power-lut            -> mxx-bgg, mxx-dsl, mxx-ir-core, mxx-noise-simulator, mxx-primitives
mxx-func-enc/io          -> lower layers when their application modules are enabled
```

Application crates never depend on one another. `mxx-noise-simulator` is a lower-layer primitive
that may be consumed by an application such as Power-LUT, but it never depends on Power-LUT (or
any other application). Diamond WE is active in `mxx-we`; functional encryption and iO protocol
modules remain disabled.

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
the pre-existing BGG modules). It does not know Power-LUT roles, setup secret identities, RHS
packages, Fuse, ClearCoeff, or refresh semantics. `mxx-power-lut` owns import-time setup identity
validation, fixed-secret automorphism orchestration, LUT/sparse-LWR evaluation, the public-key-only
projection compiler, and the manuscript §7 refresh. Evaluator values remain the plain BGG wire
types. Its dependencies are one-way: `mxx-power-lut` depends on `mxx-bgg` and
`mxx-noise-simulator` (and their lower layers); neither lower-layer crate depends on Power-LUT.
Power-LUT owns structural declaration and linkage validation for manuscript §7, plus a small
application-specific exact `BigUint` simulator for its public `Unary` and `Binary` lowerings,
the sparse-PRF-specific monomial `OneHot` path, and the CRT refresh threshold. The simulator
reuses pure arithmetic helpers from `mxx-noise-simulator` but does not reinterpret generic IR or
maintain a second graph evaluator. Generic `OneHot` public-value norms are not assumed: the
generic program API rejects them, while the sparse-PRF path supplies validated `X^a` monomials
and active non-padding counts from the public PBC layout. The generic `mxx-noise-simulator`
remains protocol-agnostic; its refresh acceptance is not the Power-LUT application authority.
The legacy `mxx-bgg::noise_refresh` implementation is unrelated and remains unchanged.

For the §7 refresh, the secret dimension is exactly `2`; it is distinct from the BGG public-key
component-column count `2*ell_beta`. Each independently exposed mask or fresh-error digit group
covers those `2*ell_beta` columns. The mask and fresh-error digit counts are separate parameters
`d_m` and `d_e`, and both are distinct from the gadget decomposition digit count `ell_beta`.
The imported decoder anchor is noisy, `b = sB + e_B`; the noiseless relation `b = sB` is not an
operational-noise bound. Power-LUT integration acceptance is determined by this application-
specific simulator; the generic simulator may still perform structural/request validation.

### Application crates

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search simulates the frozen
encryption/decryption stages and applies the Boolean decoder interval outside the simulator.

`mxx-func-enc` and `mxx-io` currently expose compiling interface shells. Their protocol modules
remain disabled.
