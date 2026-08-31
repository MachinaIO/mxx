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
mxx-power-lut            -> mxx-bgg, mxx-dsl, mxx-ir-core
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
decidable compile-parameter conditions consumed by concrete validation and operational checking. Sampler
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

The library validates linked workflow declarations and evaluates their operational noise bounds
with the Rust checker. Protocol declarations remain crate-owned; there is no central protocol
registry or parameter-check executable. Protocol-specific declarations and replay payloads do not
live in this crate or in `mxx-ir-core`.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns only the generic BGG+ key, encoding, sampler, and safe known-value operations (plus
the pre-existing BGG modules). It does not know Power-LUT roles, setup secret identities, RHS
packages, Fuse, ClearCoeff, or refresh semantics. `mxx-power-lut` owns import-time setup identity
validation, fixed-secret automorphism orchestration, LUT/sparse-LWR evaluation, the public-key-only
projection compiler, and the manuscript §7 refresh. Evaluator values remain the plain BGG wire
types. Its dependency is one-way:
`mxx-power-lut -> mxx-bgg -> lower layers`.
Power-LUT owns only the structural declaration and linkage validation for manuscript §7. The
generic `mxx-correctness` checker derives correctness and operational-noise bounds from the actual
lowered graph; Power-LUT does not maintain a replacement-bound replay or a parallel Lean claim.
The legacy `mxx-bgg::noise_refresh` implementation is unrelated and remains unchanged.

### Application crates

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search uses deterministic
worst-case bounds and accepts a candidate only after the generic Rust operational checker accepts
the frozen workflow and concrete parameter environment.

`mxx-func-enc` and `mxx-io` currently expose compiling interface shells. Their protocol modules
remain disabled. Diamond iO must migrate to the Rust operational checker before it is enabled.
