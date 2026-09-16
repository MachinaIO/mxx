# Benchmark estimator

`mxx-bench-estimator` estimates a validated graph from prepared mini-programs.
It keeps device work, prepared-wave time, dependency latency, workspace, and
dataflow costs separate. It does not execute the application graph to discover
resources.

## Measurement boundary

The estimator first collects concrete operation classes from the validated IR.
A class includes its operation semantics, concrete parameter and shape types,
CRT level and format, input-owner sharing, device placement, and preparation
metadata. Repeated graph nodes reuse a class and contribute multiplicity;
different concrete contracts require different classes.

For each missing class, explicit setup builds a prepared mini-program using
the same compile and warmup boundary as runtime. Warmup resolves exact native
descriptors, fixed owners, output lifetimes, preimage lanes, and typed capacity
claims in one transaction. Measurement then runs only the published prepared
program. Report generation freezes the measurement table and performs CPU-only
lookup; it cannot trigger a second warmup or substitute an unprepared class.

Synthetic operands describe the declared placement scenario. They are useful
for comparing classes but are not a physical VRAM guarantee and do not certify
an application's artifact representation or ownership. A missing class is an
explicit estimator error.

## Report quantities

For class `k`, with multiplicity `n_k`, fleet wall time `L_k`, and device spans
`D_ki`, the adapter reports:

```text
work_seconds = sum_k(n_k * sum_i(D_ki))
cumulative_wave_seconds = sum_k(n_k * L_k)
independent_wave_count = sum_k(n_k)
latency_seconds = max_k(L_k)
```

`work_seconds` is aggregate device time from CUDA event spans. It is not host
elapsed time and is never divided by the device count. `cumulative_wave_seconds`
includes the prepared wave wall boundary, including coordinated submission and
completion. For independent classes, their latency is the maximum class span;
graph dependencies are added to form `CostReport::critical_path_seconds`. This
is not a promise about a physically contended fleet.

`CostReport::total_time_seconds` adds separately owned dataflow/materialization
and executor-dispatch costs. GPU-resident edges have no extra host-transfer
charge. `measured_wave_workspace_bytes` describes the prepared measurement's
incremental allocation, including one wave's outputs when the representative
allocates them; it is not a whole-graph physical peak or a provisioning
certificate.

## Dataflow and artifacts

Artifact encoding, decoding, host staging, and reload are measured through the
production codec/store boundary and reported as `TransferCost` with their
ownership. Fixture construction is outside primitive timing. Compact and raw
RNS representations retain their actual declared format; the estimator does
not insert a level-conversion trial or a placeholder operation.

The dataflow model follows graph liveness, captures, subgraphs, sequential
loops, and bounded parallel loops. It applies multiplicity to prepared classes
and retains dependency barriers. It may use CPU metadata analysis, but it does
not allocate GPU owners or submit work while producing a report.

## Sampling and integer capacity

Preimage classes include fixed prepared lanes and the complete saved resource
descriptor. Synthetic measurement uses the same fixed-lane replay contract as
production. Record/Replay payloads are actual accepted payloads, not inferred
samples.

Scalar classes use arbitrary-precision `BigInt` values. The prepared capacity
ledger accounts for the words required by each declared range. Values beyond
that ledger are reported as capacity errors; the estimator does not resize a
prepared command or silently replan it.

## Runtime agreement

The production GPU entry point is
`PreparedGpuProgram::run_with_runtime_bindings`, after explicit warmup. An
unprepared request returns `NotPrepared` before slot acquisition or GPU work.
Estimator setup follows that same contract. It does not model a fallback
executor, a dynamic resource decision, or a second reservation for an
exhausted prepared slot.

Native failures poison the affected prepared slot. Input, capacity, and pool
exhaustion failures remain distinguishable and recoverable. Terminal reader and
writer events, rather than host-handle lifetime, determine slot reuse.

## Validation

Use the following narrow checks for estimator or runtime changes:

```text
cargo +nightly fmt --all
git diff --check
cargo check -p mxx-bench-estimator --lib
cargo test -p mxx-bench-estimator --lib --no-run
cargo check -p mxx-runtime --features gpu --lib
```

Run GPU timing and multi-device tests on the applicable hardware. Integration
tests require explicit approval. Source-policy tests should prove estimator
production code constructs prepared mini-programs and does not call removed
legacy constructors or alternate evaluators.
