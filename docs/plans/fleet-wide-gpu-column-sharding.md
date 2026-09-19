# Fleet-wide GPU column sharding and dynamic wave calibration

## Status and scope

This document is the normative design for executing and estimating GPU matrix
operations across every configured GPU. It is self-contained: it defines the
runtime representation, the estimator contract, dynamic per-device wave widths,
calibration reuse, memory accounting, synchronization rules, supported primitive
classes, logging, implementation stages, and acceptance gates.

The design applies to all GPU matrix operations whose work can be partitioned by
output column, including compact small-coefficient RHS multiplication and
preimage sampling. It supersedes the column-width, GPU scheduling, and VRAM
percentage parts of `docs/plans/small-rhs-multiplication.md`. The bounded
`SmallMatrix` and `Preimage` types, compact coefficient representation, canonical
artifact format, inclusive coefficient bounds, and mathematical operation
semantics defined there remain unchanged.

The implementation may make large internal API changes. Backward compatibility
for removed scheduling APIs and environment variables is not required. Noise
simulator changes are outside this plan.

## Required outcome

For one logical primitive invocation, all configured GPUs execute that same
primitive concurrently on disjoint column ranges. GPUs are not used as workers
for unrelated estimator requests while such an invocation is running.

The implementation shall:

1. keep a separate calibrated column capacity for GPU 0 and for nonzero GPUs;
2. derive both capacities dynamically from a measured one-column or small-pilot
   VRAM high-water mark and a percentage of physical VRAM;
3. use only `MXX_GPU_VRAM_PERCENT` to configure that percentage, with a default
   of 80;
4. reuse estimator calibration in runtime when the calibration signature is an
   exact match;
5. use the same production backend functions in calibration, estimation, and
   runtime execution;
6. preserve all CRT limbs of one matrix shard on one device;
7. never materialize a forbidden full-DCRT small RHS merely to shard it; and
8. remain asynchronous in production: no device-wide synchronization and no
   host wait in a wrapper that only launches device work.

## Assumptions and terminology

All configured GPUs are assumed to be the same model and to have the same total
VRAM. GPU 0 may contain additional persistent setup data. GPU 1 is therefore the
calibration representative for every nonzero GPU; GPU 2 and later reuse GPU 1's
nonzero-role calibration. A single-GPU process has only the GPU-0 role.

Definitions:

- `C` is the logical output column count of an invocation.
- `G` is the number of configured GPUs.
- `W_gpu0` is the calibrated per-wave capacity of GPU 0 for this primitive
  signature and current planned baseline.
- `W_nonzero` is the calibrated per-wave capacity of each GPU with index greater
  than zero.
- A **device shard** is a contiguous global column interval owned by one GPU.
- A **fleet wave** is one concurrent invocation on every GPU that receives a
  nonempty device shard.
- A **primitive signature** contains every value that can change the allocation
  shape or kernel path, except the logical output-column count when the operation
  is column-separable.
- **Baseline bytes** are bytes already resident for the execution stage before
  allocating the column-scaled inputs, output, and workspace of the measured
  operation. Shared fixed operands that the operation requires are staged before
  taking this baseline.
- **Incremental peak bytes** are the memory-pool used-memory high-water above that
  baseline during the pilot operation.

No limb-level distribution is permitted. A shard contains every active CRT limb
for its rows and column interval.

## Configuration contract

There is exactly one environment variable that controls automatic GPU wave
capacity:

```text
MXX_GPU_VRAM_PERCENT
```

It is an integer in `1..=100` and defaults to `80`. Missing input selects the
default. Zero, a negative value, a value above 100, and malformed input are
errors. There are no aliases.

The percentage is parsed once while the GPU contexts are configured. Each
context records the percentage, physical total bytes for each configured device,
and this checked budget:

```text
device_budget_bytes = floor(device_total_bytes * vram_percent / 100)
```

The multiplication must be overflow-safe. Context identity and cache equality
include the percentage so a context made under one budget policy cannot be reused
under another.

Delete the following controls rather than retaining compatibility behavior:

```text
MXX_GPU_SMALL_MATRIX_VRAM_PERCENT
MXX_MUL_SMALL_RHS_TILE_COLUMNS
MXX_GPU_COLUMN_CHUNK_COLUMNS
```

Do not add byte-budget, allocator-headroom, column-width, K-tile, limb-wave, or
GPU-role-specific environment variables. Small-RHS multiplication has no
internal K or limb subdivision; runtime W is its only wave width.

Controls for benchmark repetitions, outer-loop instance counts, stream-pool
size, CUDA memory-pool release policy, and bounded preimage attempts have distinct
meanings and remain independent of this percentage.

## Ownership and crate boundary

`mxx-runtime` owns the fleet matrix representation, fleet scheduler, calibration
signature and profile types, and the reusable calibration registry.
`mxx-bench-estimator` depends on `mxx-runtime`, invokes these production APIs,
and returns or shares the resulting registry. This preserves the dependency
direction in `docs/architecture.md`; runtime must not depend on the estimator.

Native memory-pool high-water queries and reset operations belong to
`mxx-primitives` and its CUDA implementation. CUDA headers expose only the
cross-file or Rust-facing declarations; implementation bodies remain under
`crates/primitives/cuda/src/`.

## Fleet matrix representation

The GPU backend uses one logical owner composed of device-local column shards:

```text
GpuFleetMatrix
  global_rows
  global_columns
  shards: [GpuColumnShard<GpuDCRTPolyMatrix>]

GpuFleetSmallMatrix
  global_rows
  global_columns
  shards: [GpuColumnShard<GpuSmallMatrix>]

GpuColumnShard<T>
  device_id
  global_column_start
  local_columns
  value: T
```

Shard intervals are disjoint, ordered, and cover the complete global column
range exactly once. A one-GPU value is represented by one shard; there is no
separate unsharded GPU path. Every shard owner carries its ordinary producer and
release events.

Column ownership is preserved across consecutive column-separable nodes.
Operations may use different `W_gpu0` and `W_nonzero` values without physically
resharding an existing logical value: a device processes its owned range in
successive zero-copy local views. This preserves the memory and transfer
contract. If the producer and consumer have different GPU-0/nonzero width
ratios, an owner can finish before another owner, so the realized fleet width can
be below the nominal sum for later local waves. The initial implementation
accepts this honest-environment scheduling heuristic instead of introducing
producer-boundary peer copies or graph-wide compact-layout planning.

Full-evaluation fixed operands that are independent of the output column are
replicated to every participating GPU once per active invocation and weakly
cached by parameter/context and value identity while an owner remains live.
Examples include a matrix-multiplication LHS, the LHS of
`mul_small_rhs`, a preimage public matrix and trapdoor, and transform constants.
They are not copied inside a wave loop. Compact small matrices remain compact
when copied or sharded.

## Primitive calibration

### Signature

The calibration registry is keyed by a canonical primitive signature containing
all allocation- and kernel-path-relevant data, including as applicable:

- primitive kind and variant;
- matrix rows, inner dimension, and scalar-operand position;
- ring dimension, active CRT moduli/depth, level, and representation format;
- gadget base, digit count, and decomposition mode;
- compact coefficient width and inclusive bound;
- preimage distribution parameters and cutoff;
- constant-matrix variant; and
- CUDA/backend implementation identity needed to reject stale profiles.

Node IDs, graph paths, hash-tag contents, loop indices that do not change a
concrete shape, and global column offsets are excluded. If a loop binding changes
any concrete field above, it necessarily creates a different signature.

The profile records separate role measurements rather than a common minimum:

```text
GpuPrimitiveCalibration
  signature
  vram_percent
  gpu0: DeviceRoleCalibration
  nonzero: optional DeviceRoleCalibration

DeviceRoleCalibration
  pilot_columns
  incremental_peak_bytes
  bytes_per_column
```

The role baseline and pilot latency are logged with each measurement rather than
stored in the reusable slope profile. Runtime always applies the cached
`bytes_per_column` to its newly observed resident baseline.

`W_gpu0` and `W_nonzero` are derived values, not serialized configuration.

### Pilot preparation

Calibration uses the production backend entry point with production dataflow.
Before taking the baseline, it creates or stages every column-independent input
that will remain resident during the real operation. The pilot then allocates
the same column-dependent input representation, output, temporary workspace,
event storage, and stream metadata used in production. A compact-RHS pilot must
not expand columns outside its pilot range.

For estimator measurements, this pilot is the primitive's memory-calibration
warm-up. Timing warm-ups and measured iterations run only after the role widths
have been derived. The calibration result is therefore not inferred from a
later timing run whose allocator state may already have changed.

Start with one pilot column. An operation whose valid structure requires a larger
minimum range uses the smallest valid pilot and records that count. The pilot
uses dummy values and an independent sampler state; it must not mutate runtime
transcripts, artifacts, preimage progress, or production random state.

GPU 0 is measured after GPU-0-only setup has been staged. If `G > 1`, GPU 1 is
measured after the ordinary nonzero-device setup has been staged. The GPU 1
measurement is reused for every later nonzero GPU under the identical-hardware
assumption.

### Memory-pool high-water measurement

Polling `cudaMemGetInfo` is not the calibration oracle. CUDA async pools can
retain pages after warm-up and a polling interval can miss short-lived peaks.
Calibration instead uses the context's CUDA memory pool:

1. finish only the calibration preparation stream using its completion event;
2. read `cudaMemPoolAttrUsedMemCurrent` as `baseline_bytes`;
3. reset `cudaMemPoolAttrUsedMemHigh` according to the CUDA reset contract;
4. enqueue one pilot through the production entry point;
5. wait only for the pilot completion event;
6. read `cudaMemPoolAttrUsedMemHigh`; and
7. compute `incremental_peak_bytes = high_water_bytes - baseline_bytes` with
   checked subtraction.

These host waits are permitted only in the explicit pre-execution calibration
or benchmark boundary. Production operation wrappers remain nonblocking. No
step uses `cudaDeviceSynchronize` or `cudaStreamSynchronize`.

The calibration owner must keep the pilot result alive until after the
high-water query so persistent output storage is included. It then queues normal
event-ordered destruction. The query/reset API must preserve the caller's CUDA
device and must not mutate another context's pool.

The current CUDA backend allocates from the device default asynchronous pool,
whose high-water attribute is shared by all contexts in the process. Calibration
therefore resets and interprets that attribute only while exactly one live mxx
context owns the device. With multiple live contexts the measurement cannot be
attributed to this execution, so runtime and estimator fail with an explicit
calibration error without caching a profile or width. Supporting concurrent
contexts requires a future context-private allocation pool.

The role baseline is allocator-aware physical residency, not only
`UsedMemCurrent`:

```text
physical_used = total_bytes - free_bytes
resident_bytes = physical_used - ReservedMemCurrent + UsedMemCurrent
```

An inconsistent or overflowing snapshot is treated as fully resident. This includes
persistent legacy `cudaMalloc` allocations such as context-local NTT tables,
while excluding unused pages retained by the default pool because those pages
remain reusable by `cudaMallocAsync`.

Calculate the conservative linear cost as:

```text
bytes_per_column = ceil(incremental_peak_bytes / pilot_columns)
```

The simple division deliberately charges fixed pilot overhead to each column.
If the incremental peak is zero, treat calibration as failed rather than infer
an arbitrarily large capacity. If the pilot itself exceeds the role budget,
return a resource error. Integer calculations are checked and saturating values
must not be interpreted as a valid plan.

An asynchronous release from an earlier operation can otherwise complete after
the baseline and make the pilot appear to consume zero or negative incremental
memory. Runtime fences only the context release streams before the baseline. If
the final `UsedMemCurrent` nevertheless fell below the baseline while
`UsedMemHigh` did not exceed it, the interval is objectively contaminated: drop
the pilot outputs, fence/reset the pool counters, and retry the isolated pilot a
bounded number of times. A stable zero increment is not contamination and still
fails closed. This calibration-only retry never turns a production wave into a
pilot and never adds a device-wide or compute-stream synchronization.

### Role-specific width derivation

At execution-plan construction, determine the planned role baseline after
session setup and required fixed operands are resident. Calibration data supplies
the incremental bytes per column; the current plan supplies the baseline. The
widths are calculated independently:

```text
available_gpu0 = gpu0_budget_bytes - planned_gpu0_baseline_bytes
W_gpu0 = floor(available_gpu0 / gpu0_bytes_per_column)

available_nonzero = nonzero_budget_bytes - planned_nonzero_baseline_bytes
W_nonzero = floor(available_nonzero / nonzero_bytes_per_column)
```

Neither width is replaced by `min(W_gpu0, W_nonzero)`. Both are retained in the
execution plan and logs. Each width is capped by the remaining logical columns
when a concrete shard is assigned. A zero derived width is a resource error; it
does not cause CPU fallback. When another honest mxx context shares the
device-global default pool, exclusive high-water calibration is unavailable;
the implementation fails closed rather than resetting another context's counter
or assuming that one column fits.

Runtime samples the allocator-aware resident baseline after staging the
operation's fixed owners. Deferred frees are considered resident until their
release event. If the live set changes, it recomputes the width from the cached
bytes-per-column and the new baseline; it does not rerun the pilot unless the
primitive signature itself missed the cache.

## Fleet-wave scheduling

For `G >= 1`, define the full fleet capacity:

```text
fleet_wave_columns = W_gpu0 + W_nonzero * (G - 1)  // G > 1
fleet_wave_columns = W_gpu0                         // G = 1
chunk_count = ceil(C / fleet_wave_columns)
```

This is the exact schedule for newly produced column ranges and for consumers
whose input ownership has the same role-width ratio. A compact RHS produced
under a different ratio stays on its owner device and is consumed through
zero-copy local views. In that case `fleet_wave_columns` is the nominal
estimator throughput and runtime may need additional owner-local waves. This
keeps peak VRAM and implementation complexity bounded; it may conservatively
reduce realized parallelism but does not change results.

All arithmetic is checked. `chunk_count` is zero only when the validated logical
operation has zero columns and its semantics permit an empty matrix.

Within each wave, assign contiguous ranges in device order:

```text
GPU 0:     up to W_gpu0 columns
GPU 1..G:  each up to W_nonzero columns
```

The last device and final wave may receive fewer columns. Devices with an empty
range do not launch work and do not count as active for that wave.

Host-side per-device enqueue calls run concurrently, using one long-lived worker
or equivalent host task per device. Each worker uses the device-local backend and
stream pool. A fleet operation returns a set of completion events, one per active
shard. A downstream column-separable operation waits on the producer event of the
corresponding shard only. The runtime does not insert a fleet-wide host barrier
between ordinary nodes. A semantic redistribution or host-visible return may
join the relevant device events explicitly.

GPU 0 may therefore use fewer columns than the other devices without forcing the
other widths down. The next fleet wave begins on a device when its local stream
dependency permits it; logical result publication still covers all shard events.

## Estimator contract

The estimator first collects unique primitive signatures. For each signature it
calibrates or looks up the GPU-0 and nonzero roles, derives the two widths, and
then measures one fleet wave of that same primitive concurrently on all active
GPUs. It no longer distributes unrelated measurement requests among GPUs.

For a full representative wave, GPU 0 receives `W_gpu0` columns and every
nonzero GPU receives `W_nonzero` columns. If `C` is smaller than full fleet
capacity, the estimator measures the actual one-wave assignment. It does not
measure a separate remainder shape. Let `L_i` be the measured device elapsed
time and `L_fleet` the wall time from fleet enqueue through completion of all
active device events. Then:

```text
estimated_latency_seconds = L_fleet * chunk_count
estimated_work_seconds = sum(L_i for active devices) * chunk_count
workspace_bytes = sum(role-local incremental peaks for the active fleet wave)
```

Latency is wall-clock critical-path time; work is aggregate GPU-seconds. Both raw
values are logged. Peak VRAM is reported per role and per device, not only as a
fleet sum. The aggregate byte sum may be logged for capacity planning but must
not conceal a per-device budget violation.

A fleet-wide primitive already occupies every active GPU. Estimator outer-loop
logic must set its per-instance occupancy to that active GPU count and must not
divide the primitive latency by `device_pool_size` again. Independent loop
instances may share the fleet only when the scheduler explicitly partitions the
available devices between instances; the default fleet plan gives the whole
fleet to one instance.

The measurement cache key includes the canonical primitive signature, role
profile, percentage, device count, and derived role widths. It does not include a
node identity. Repeated shapes reuse one fleet measurement.

## Calibration reuse between estimator and runtime

The estimator exposes its `GpuCalibrationRegistry` and a runtime constructed in
the same process accepts that optional registry. The owning application passes
the frozen registry when estimator-to-runtime reuse is desired. An exact
signature hit reuses `bytes_per_column` and the pilot metadata;
runtime recomputes `W_gpu0` and `W_nonzero` from its current planned baselines.
This is required because runtime may have more GPU-0-only setup resident than the
benchmark did.

A profile hit is valid only when all of the following match:

- primitive signature;
- CUDA/backend implementation identity;
- device model and total VRAM;
- ring/CRT parameters and representation format;
- configured GPU role and count assumptions; and
- `MXX_GPU_VRAM_PERCENT`.

On a miss, runtime performs the explicit pre-execution pilot for that signature
on GPU 0 and, when present, GPU 1. The executor selects and arms the signature
before dispatching the node. The selected primitive entry then stages its fixed
inputs, runs one isolated pilot wave before any semantic production wave,
discards every pilot output, derives the widths, and restarts production at the
original global column. Thus the pilot shares the production entry point and
allocation path without reusing a real output wave or advancing a transcript,
artifact, preimage attempt, or production sampler state. Persistent cross-process calibration files are not part of the
initial implementation because allocator, driver, setup, and implementation
changes can make them stale.

## Primitive classification

### Direct column-separable operations

The following operations shard their output columns and use the common fleet
schedule directly:

- uniform-residue, uniform-interval, Gaussian, and hash sampling;
- matrix negate and integer scale;
- matrix add and subtract after the existing format-alignment rule;
- gadget decomposition and compact hash decomposition;
- ordinary matrix multiplication by sharding the matrix-valued RHS and output;
- scalar-by-matrix and matrix-by-scalar multiplication by sharding the
  matrix-valued operand and output;
- matrix multiply-accumulate by sharding each non-scalar RHS, bias, and output;
- `mul_small_rhs` by sharding the compact RHS and full output;
- preimage sampling by target/output column, keeping all coupled sampler rows on
  the same device;
- CRT recomposition by sharding the matching column interval at every level;
- row concatenation by sharding the same column interval in every input;
- row/column slices when the global range is mapped to each local interval.

Coefficient extraction produces a scalar and decoding is a host-visible
whole-value boundary in the current IR, so neither is column-extrapolated.

For matrix multiplication, the LHS is a fixed operand and is replicated once.
For preimage sampling, public/trapdoor data are fixed operands and are replicated
once. Preimage row candidates are never split across devices because the
preimage equation couples those rows.

### `ConstantMatrix`

Constant construction participates in the same dynamic calibration and fleet
schedule when its logical output has multiple columns. It must use a
range-aware production API rather than construct a smaller matrix with changed
semantics:

```text
constant_matrix_column_range(
    complete_concrete_type,
    constant_variant,
    global_column_start,
    local_column_count,
    bindings,
)
```

Each kernel derives its values from the global row and global column. Variant
handling is:

- `Zero`: generate the requested range directly.
- `Identity`: retain the original row count and full logical dimensions; a
  local element is one exactly when its global row equals its global column.
- `UnitRow`: evaluate the global nonzero index once and translate it into each
  shard's local range. A range containing the nonzero and an all-zero range use
  the same kernel class; calibration uses the containing range when one exists.
- `Gadget`: derive block and digit position from the global column. Never replace
  the original gadget with a smaller gadget having a different digit layout.
- `UnitColumn`, `PowerOfBase`, `Rotation`, and `Polynomial`: these validated
  forms have one logical column and therefore use one active device without
  column extrapolation.

Index-dependent value differences do not create separate cost classes when the
kernel, allocation shape, and asymptotic work are identical. If a variant takes
a genuinely different kernel path, its variant remains in the signature.

### Factorable or redistribution operations

Tensor output columns are column-parallel, but a representative range must map
global output columns back to the exact left/right indices. Use a range-aware
tensor kernel; do not choose arbitrary smaller factor dimensions merely to
produce `W` columns.

Column and diagonal concatenation require a range mapper that can intersect one
output interval with multiple input intervals. The device worker launches or
batches the exact intersections for its range. The calibration pilot must use a
range that exercises the worst valid intersection class when allocation or
launch count differs by location.

Transpose changes the ownership axis. Implement it as an explicit device-local
transpose plus peer-to-peer redistribution of the resulting column ranges. Its
calibration and estimate include transfer buffers and peer-copy time. If peer
access is unavailable, the GPU backend returns an unsupported-placement error;
it does not silently stage through ordinary CPU computation.

Operations with a validated single-column result execute on one device and have
`chunk_count = 1`. They are still represented by the fleet owner with one shard.

## Small-RHS and preimage memory invariants

The compact small-RHS representation and bounds from
`docs/plans/small-rhs-multiplication.md` remain mandatory. Fleet sharding changes
which device owns a column range; it does not permit a full-DCRT RHS.

For each device shard, `mul_small_rhs` keeps the fixed full-evaluation LHS,
compact RHS shard, full output shard, and exactly one expanded
`L * K * local_columns` workspace. Runtime calibration chooses the only width
`W`; the primitive always processes all local columns, K rows, and CRT limbs in
one call. Preimage sampling
keeps all rows required by one target-column tile together and packs accepted
columns immediately into the compact destination.

For every role and primitive, the following invariant is checked before launch:

```text
planned_baseline_bytes
  + column_dependent_resident_bytes
  + operation_workspace_bytes
  <= device_budget_bytes
```

No allocation may be justified by aggregate free memory on other GPUs. The
fleet plan fails closed if either active role cannot process its assigned minimum
valid range. There is no CPU fallback and no retry with an unmeasured expanded
representation.

## Artifact and host-boundary behavior

The canonical artifact formats do not change. Serialization walks shards in
global-column order and writes the existing canonical row-major coefficient
order. It does not first join shards into a full matrix on one GPU. A bounded
matrix remains in the canonical compact coefficient format for device-to-host
and host-to-device transfer.

Import validates the complete expected concrete schema and semantic kind once,
then decodes each global column interval directly into its destination shard.
Artifact export is a host-visible operation and may wait for the relevant shard
events. Device-only copies and imports remain asynchronous. Content hashes cover
the same canonical bytes regardless of the number of devices or shard widths;
calibration data and device placement never enter an artifact or transcript.

## Synchronization and lifetime invariants

- Never call `cudaDeviceSynchronize`.
- Successful production and calibration paths never call
  `cudaStreamSynchronize`. Error unwinding normally fences a freshly recorded
  completion event as well. If CUDA cannot create or record that safety event,
  a stream fence is the last-resort ownership-safety fallback before temporary
  storage can be released; it is not part of scheduling, timing, or normal
  execution.
- Production wrappers that only enqueue device work do not block the host.
- Calibration and benchmark code may wait on the specific completion events
  whose results or high-water marks it must observe.
- Host workers enqueue device shards concurrently; they do not serialize the
  same primitive across GPUs.
- Producer events are recorded after every launch and allocation relevant to a
  shard. Consumers wait on those events in their own streams.
- Fixed replicated operands remain alive until every consuming device event has
  completed. Destruction is queued on the existing release streams.
- A fleet owner is ready only when all of its shard events are ready, but a
  device-local successor may consume its corresponding ready shard without a
  host-wide join.
- CUDA memory-pool attribute calls are confined to context setup, explicit
  calibration, and measurement boundaries; they are not inserted into the hot
  per-wave loop.

## Required diagnostics

Each calibrated and measured primitive logs explicit values rather than values
that must be reconstructed:

```text
primitive_signature
vram_percent
gpu_count
gpu0_total_bytes
gpu0_budget_bytes
gpu0_baseline_bytes
gpu0_pilot_columns
gpu0_incremental_peak_bytes
gpu0_bytes_per_column
W_gpu0
nonzero_total_bytes
nonzero_budget_bytes
nonzero_baseline_bytes
nonzero_pilot_columns
nonzero_incremental_peak_bytes
nonzero_bytes_per_column
W_nonzero
fleet_wave_columns
active_gpu_count
chunk_count
device_wave_latency_seconds[]
fleet_wave_latency_seconds
estimated_latency_seconds
estimated_work_seconds
per_device_peak_bytes[]
```

The nonzero fields are absent rather than fabricated on a one-GPU system. Logs
also report compact bytes, full-output bytes, and exact `L*K*W*N` workspace
bytes where applicable. There are no internal small-RHS K-tile or limb-wave
diagnostics.

## Implementation stages

1. **Configuration and CUDA probes**
   - Replace the small-matrix percentage variable with
     `MXX_GPU_VRAM_PERCENT` in `crates/primitives/src/env.rs`.
   - Cache checked per-device total and budget bytes in GPU contexts.
   - Add context-local CUDA memory-pool current/high-water query and reset APIs.
   - Remove every column-width environment control and compatibility alias.

2. **Runtime calibration foundation**
   - Add canonical primitive signatures, role profiles, and the shared registry
     under GPU-specific runtime modules.
   - Add production-entry-point pilots with isolated sampler state.
   - Derive and retain distinct `W_gpu0` and `W_nonzero` values from planned
     role baselines.

3. **Fleet owners and scheduler**
   - Replace the GPU backend's single active placement model for matrix values
     with logical fleet owners and device-local backends.
   - Add concurrent host enqueue workers, shard event aggregation, liveness
     accounting, fixed-operand replication, and peer-transfer support.
   - Keep the one-device path as a one-shard instance of the same code.

4. **Direct column-separable primitives**
   - Migrate sampling, arithmetic, matrix multiplication/accumulation,
     decomposition, `mul_small_rhs`, preimage, recomposition, row concat, and
     range-preserving slices.
   - Verify each entry point accepts a global range and does not create a full
     unsharded temporary.

5. **Range-aware and redistribution primitives**
   - Add range-aware `ConstantMatrix`, tensor, column/diagonal concat, and
     transpose/redistribution implementations.
   - Include peer buffers and communication latency in calibration profiles.

6. **Estimator integration and reuse**
   - Replace the unrelated-request worker queue and configured
     `column_wave_size` with registry-backed fleet measurement.
   - Remove separate remainder measurements and floating column scale factors.
   - Return/share the registry with runtime and prevent outer-loop GPU-count
     speedup from being applied twice.

7. **Artifacts, diagnostics, and cleanup**
   - Stream canonical artifact bytes across ordered shards.
   - Add the required structured logs.
   - Delete old APIs, controls, documentation, and tests that assume a configured
     fixed column wave.

Stages that edit disjoint crates may be developed in parallel, but fleet owner
interfaces and calibration signatures must be fixed before dependent stages are
merged.

## Validation and acceptance gates

Unit tests, not application integration tests, are the default validation scope.
GPU tests run outside the sandbox on the local GPUs and use normal multithreaded
test execution unless an existing `serial_test` guard is semantically required.

Required focused tests include:

- parsing default, minimum, maximum, malformed, zero, and above-100
  `MXX_GPU_VRAM_PERCENT` values;
- checked per-device budget arithmetic and context-cache identity;
- context-local memory-pool high-water reset/query behavior without changing the
  caller's current device;
- distinct GPU-0 and nonzero baselines producing distinct retained widths;
- GPU 1 calibration reuse on GPU 2 and later;
- fail-closed zero incremental-peak handling and failure when a minimum pilot
  exceeds the role budget;
- exact fleet range coverage with unequal widths, fewer columns than GPUs,
  partial final waves, and checked-overflow boundaries;
- concurrent same-primitive execution on all active GPUs, demonstrated with
  per-device events and wall time rather than unrelated request concurrency;
- estimator formulas for fleet latency, aggregate GPU work, workspace, active
  occupancy, and absence of double GPU-count scaling;
- exact calibration-cache hit/miss behavior and runtime width recomputation after
  a larger GPU-0 baseline;
- range-aware Identity, UnitRow, and Gadget values at shard boundaries;
- all direct column-separable primitives against their existing trusted
  single-device results;
- transpose and concat redistribution across device boundaries;
- compact `mul_small_rhs` and preimage equivalence with no expanded all-column
  DCRT RHS;
- shard-preserving artifact round trips and device-count-independent canonical
  hashes; and
- event-safe early drops and repeated execution without implicit synchronization.

After focused tests, run CPU and GPU workspace unit-test builds with no new
warnings attributable to this change and the full allowed `mxx-primitives` and
`mxx-runtime` unit suites. Run the built GPU
test binaries repeatedly with identical multithreaded commands as required by
`GPU.md`. Do not use synchronization to hide intermittent failures.

Performance acceptance compares the previous single-device or fixed-column path
with the fleet implementation on identical parameters. Report raw per-device
pilot peaks, both widths, fleet wave width, chunk count, device and fleet
latencies, total estimate, runtime latency, and per-device peak VRAM. Estimator
and runtime must agree on widths and chunk count when given the same registry and
planned baselines.

Final source and documentation searches must prove that no references remain to
the three removed environment variables, configured `column_wave_size`, the
unrelated-request GPU measurement queue, or a compatibility scheduling path.
Independent review must additionally verify the synchronization and per-device
VRAM invariants against `GPU.md`.

## Completion criteria

This design is complete only when runtime and estimator both use the same
fleet-wide primitive schedule; GPU 0 and nonzero GPUs retain independent dynamic
widths; one percentage variable is the only wave-capacity control; exact
calibration profiles are reused; all primitive classes above either shard
correctly or perform an explicitly measured redistribution; artifacts remain
canonical; per-device peaks remain within budget; no forbidden small-RHS
expansion or implicit host/device synchronization remains; and the required
tests and before/after measurements pass.
