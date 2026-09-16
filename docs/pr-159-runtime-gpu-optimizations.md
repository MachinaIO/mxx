# PR #159: prepared runtime and GPU architecture

This document records the final runtime/GPU design associated with PR #159.
The authoritative implementation is in the current workspace; historical
review plans and removed legacy APIs are not part of the contract.

## Scope

The DSL builds an immutable typed graph. `mxx-ir-core` validates that graph and
the runtime executes it on CPU or through the prepared GPU backend. Polynomial
matrices may use several CRT limbs and coefficient or evaluation layouts.
Artifacts and sampling transcripts carry complete typed payloads between
executions.

## Prepared GPU lifecycle

The backend has one explicit lifecycle:

1. **Compile** validates parameters, concrete wire types, liveness, loop
   bounds, and sampler contracts, then creates a fixed command topology.
2. **Warmup** resolves every native operation descriptor, owner layout, fixed
   preimage lane, output lifetime, and reusable slot. A single resolver
   transaction reserves all retained owners and workspaces and publishes a
   `PreparedGpuProgram` only after the complete transaction succeeds.
3. **Execute** checks exact runtime bindings and replays the published command
   sequence. It acquires one prepared slot, submits work, publishes outputs,
   and retires the slot after terminal reader/writer events.

The production entry point is
`PreparedGpuProgram::run_with_runtime_bindings`. An unprepared or mismatched
request returns `NotPrepared` before slot acquisition or GPU submission. There
is no implicit warmup, dynamic graph traversal, runtime resource decision,
dynamic scheduler, fallback runner, or CPU fallback in this path.

The main implementation files are:

- `crates/runtime/src/backend/poly_gpu/gpu_prepared.rs`: public prepared
  program boundary, binding checks, and execution lifecycle.
- `crates/runtime/src/backend/poly_gpu/gpu_prepared_lowering.rs`: fixed
  commands, replay steps, and resolver-owned descriptors.
- `crates/runtime/src/backend/poly_gpu/gpu_prepared_scope.rs`: compiled
  structural replay for subgraphs and bounded loops.
- `crates/runtime/src/backend/poly_gpu/gpu_inventory.rs`: typed capacity
  planning and one-shot warmup provisioning.
- `crates/primitives/src/matrix/gpu_prepared.rs`: low-level saved-layout
  native bind primitives.

## Ownership and capacity

Warmup records exact rows, columns, ordered CRT basis and parameters, level,
representation, device/context, family shape, and owner layout. Production
binds use those saved descriptors directly. Structural drift is rejected before
capacity is acquired and cannot replace a fixed owner.

Prepared slots are independent of graph loop cardinality. The limits
`max_parallel_instances` and `max_live_gpu_executions` bound sibling waves and
reusable execution instances respectively. Input, capacity, and pool
exhaustion errors are recoverable. A native submission or completion failure
poisons its slot; poisoned slots are retired and never reused. Waiting for an
exhausted pool observes existing terminal events and does not reserve the same
execution twice.

Runtime integer values are arbitrary-precision `BigInt`s. The warmup ledger
reserves enough words for the declared capacity. A larger value fails at the
input boundary; it does not trigger geometry growth or a second planning pass.

## Native operation and sampling rules

Every production native bind consumes a resolver-provided saved descriptor,
including NTT, transpose, input-copy, arithmetic, sampling, conversion, RNS,
rebase, CRT, gadget, compact, serialization, small-RHS, trapdoor, and preimage
operations. A convenience constructor is a low-level primitive only when it
is called by a current standalone primitive API; prepared production code does
not use it to rediscover a layout.

Preimage sampling is compiled into fixed lanes and fixed resource claims.
`Fresh` and `Record` receive fresh submit-time randomness. `Record` stores the
actual payload accepted by the native sampler after completion. `Replay`
validates the recorded payload against the prepared contract and uploads it to
fixed staging; it never samples again.

## Estimator

`mxx-bench-estimator` measures prepared mini-programs, not an application graph
and not an unprepared operation. Representative classes are keyed by concrete
shape, format, placement, input sharing, and preparation metadata. Explicit
setup measures missing classes and freezes the table. Report generation then
performs CPU-only lookup and applies graph multiplicity.

The report keeps distinct:

- aggregate device work from CUDA event spans;
- cumulative prepared-wave wall time;
- ideal dependency latency;
- prepared workspace observations; and
- dataflow/materialization and executor-dispatch costs.

No missing class is silently replaced by a singleton estimate. Synthetic
representatives describe the stated placement scenario and are not a physical
VRAM guarantee.

## Validation and review requirements

The minimum checks for this area are:

```text
cargo +nightly fmt --all
git diff --check
cargo check -p mxx-runtime --features gpu --lib
cargo test -p mxx-primitives --features gpu --lib --no-run
cargo test -p mxx-runtime --features gpu --lib --no-run
```

Targeted hardware tests are required for GPU submission, event retirement,
multi-device ownership, fixed-lane replay, and poisoned-slot behavior.
Structural-drift tests must prove zero slot acquisition and zero GPU submission.
Source-policy tests must prove prepared production modules do not call removed
legacy constructors or alternate evaluators. Integration tests require
explicit approval.
