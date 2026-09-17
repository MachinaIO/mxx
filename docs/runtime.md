# Graph runtime

`mxx-runtime` executes a validated `mxx-ir-core` graph. The GPU backend is
prepared-only and has one explicit compile, warmup, and execute lifecycle. It
does not reinterpret mutable builders or turn a production request into a
discovery run or CPU fallback.

## Compile, warmup, execute

Compilation validates the graph, concrete parameters, input contracts,
liveness, and control-flow bounds. It lowers supported GPU operations to a
fixed command topology and typed recipe; it does not allocate device storage
or submit work.

Warmup resolves the complete recipe and records exact matrix owners, CRT basis
and level, representation, device/context, layouts, fixed preimage lanes,
native descriptors, and output lifetimes. One transaction commits all retained
owners, workspaces, and reusable execution slots. It either publishes a
`PreparedGpuProgram` or releases every partial reservation. Native binds
consume saved descriptors from this recipe; they do not reconstruct layouts.
`PreparedPlanLayout` is the sole native operation-layout authority: prepared
wrappers retain the planned descriptor and pass it to native binds, with no
competing dimension, live-handle, or convenience-constructor layout source.
Evaluation-domain `PolynomialValues` uses the direct source-owner path rather
than implicit coefficient staging and an inverse-transform detour.

Execution accepts only the published program. It validates each runtime value,
acquires one exact reusable slot, submits the fixed replay sequence, publishes
outputs, and retires readers and writers using terminal GPU events. Host-handle
drop is not completion. Missing or mismatched preparation returns
`NotPrepared` before slot acquisition or GPU submission.

The production entry point is
`PreparedGpuProgram::run_with_runtime_bindings`. Convenience evaluators and
dynamic input paths are not part of the backend API. CPU execution remains a
separate backend.

## Input and resource contracts

Inputs must match the prepared contract exactly: matrix rows and columns,
ordered CRT basis and parameters, level, coefficient/evaluation format,
device/context, owner layout, and family shape. Trapdoors additionally match
sigma, gadget base, digit count, mode, and preimage bound. Structural drift is
rejected before a slot is acquired; it cannot overwrite a fixed matrix type or
replace a mismatched owner.

Scalar inputs use arbitrary-precision `BigInt` values with fixed warmup-derived
projections. Each execution claims one exact reusable slot and copies values into
the preallocated scalar backing. A value that exceeds its fixed projection is
rejected explicitly; runtime execution never grows device or pinned scalar
storage, replans geometry, or fabricates a width. The limits
`max_parallel_instances` and `max_live_gpu_executions` are accounted for
independently.

Input, capacity, and pool-exhaustion errors are recoverable. Native submission
or completion failure poisons the affected slot, which is retired instead of
returned to the pool. A prepared pool reports `Busy` immediately when every
slot is occupied; it does not poll, wait, or synchronize on the caller's behalf.
The caller may retry or retain the live output until its terminal event permits
reclamation. The ordinary executor's `ExecutionConfig::release_fence_interval`
does not change this prepared-pool admission rule.

## Sampling and replay

Preimage sampling uses fixed prepared lanes and descriptors. `Fresh` and
`Record` receive fresh submit-time randomness. `Record` waits for the accepted
native payload and stores that actual payload in the transcript. `Replay`
validates the recorded payload against the prepared shape, CRT parameters,
lane, and schema, then uploads it to fixed staging; it never resamples.

Persisted trapdoor state is only the two matrices `r` and `e`. Gram matrices,
covariance data, coefficient workspaces, and transform metadata are derived once
per preimage operation at its execution boundary and retained only for that
operation. The prepared transcript has two parts: a draw-site entry identifies
the value, and the recorded payload carries the complete public or secret bytes;
it is not a five-part trapdoor metadata envelope.

Artifacts and transcripts are complete payload contracts and must use matching
codecs, schemas, parameters, and integrity metadata. Serialized data is not
treated as a general untrusted-data parser.

## Validation

Use `cargo +nightly fmt --all`,
`cargo check -p mxx-runtime --features gpu --lib`,
`cargo test -p mxx-primitives --features gpu --lib --no-run`, and
`cargo test -p mxx-runtime --features gpu --lib --no-run` for the narrow checks.
GPU and multi-device behavior require applicable hardware tests; integration
tests require explicit approval. Source-policy tests keep prepared production
modules free of legacy constructors, dynamic resource decisions, and alternate
evaluators.

## Hash-tag encoding

Hash samples preserve insertion order. Byte strings and decimal integers use
length framing, and integer representations carry type markers, so `(1, 23)`
and `(12, 3)` remain distinct tags. Rebuild serialized graphs and hash-derived
preprocessing artifacts together when this encoding changes.
