# Fleet-wide GPU column sharding and dynamic wave calibration

## Status, authority, and scope

This is a revised implementation plan, not a report of completed implementation
or validation. Its source baseline is commit `51bfdbd1e`. Configuration parsing,
fleet owners, calibration registries, column operations, artifact codecs, and
estimator measurements already exist. The work below repairs and extends those
implementations; it does not restart their original implementation stages.

Implementation was authorized, then explicitly stopped by the user on 2026-09-12.
Do not resume without a new instruction. The
[stop-time handoff](fleet-wide-gpu-handoff-2026-09-12.md) records the current
implementation, two unresolved GPU-enabled unit failures, remaining work, and
later user overrides for test repetitions and deferred physical multi-GPU checks.
Historical implementation and validation
evidence is tracked in `docs/plans/fleet-wide-gpu-implementation-status.md`;
this document remains the required contract, not an acceptance report. The
[prepared-storage accounting amendment](gpu-prepared-storage-accounting-amendment.md)
is accepted and integrated below; its acceptance does not establish that the
implementation satisfies it. Application integration tests still require separate
authorization.

This document supersedes the GPU column scheduling and VRAM policy sections of
`docs/plans/small-rhs-multiplication.md` and the earlier version of this plan.
It preserves mathematical operations, inclusive coefficient bounds, bounded
`SmallMatrix`/`Preimage` types, compact coefficients, canonical artifacts,
transcripts, and the separation between runtime and estimator crates. Noise
simulator changes and application-specific scheduling are outside scope.

Existing measurement fields described in `docs/benchmark-estimator.md` retain
their existing meaning until the coordinated migration specified below. The
revised plan distinguishes the approved contract from the current implementation.
Removed APIs and environment variables need no compatibility aliases.

The user deferred physical multi-GPU validation on 2026-09-11. Continue the
implementation and local unit validation; report the deferred hardware coverage
separately instead of treating it as an executed acceptance gate.

## Required outcome and operating assumptions

A logical invocation uses all configured devices that receive nonempty ranges
of that same primitive. It does not run unrelated estimator requests on idle
members of that fleet. Small results and preserved ownership can leave some
devices idle; using every device is not a condition for a valid invocation.
Let `C` be the logical output column count and `G` the configured device count.

Enumerate devices through `detected_gpu_device_ids` and retain the configured
ordered subset. Role 0 means the first configured device, not necessarily CUDA
ordinal 0. All role indices below refer to positions in that configured order.
Validate unique device IDs, matching models, compute capabilities,
physical VRAM, and relevant execution configuration. Reject a heterogeneous
fleet in this implementation. Equal hardware does not imply equal residency.

Role 0 has its own measured allocation profile. Configured device 1 is the
representative for the nonzero role. Its measured profile may be reused on
later identical devices; its free-memory snapshot may not. A one-device fleet
has no nonzero profile or fabricated nonzero diagnostics.

The memory guarantee begins after accepted finite-storage setup and covers the
complete planned mxx managed live set and native logical capacity. Opaque CUDA
consumption is outside that bound. Under the
[accepted production VRAM policy](gpu-driver-memory-contract-decision.md), observed
physical or allocator-adjusted budget excess during computation does not stop
otherwise valid work. Initial setup still requires both residency checks below;
these checks do not establish a bound during provisioning itself.
Device loss and allocator/driver failures remain errors; they never select a CPU
fallback, materialize a forbidden full-DCRT RHS, or silently change the accepted
plan.

Retain the default asynchronous CUDA pool. Exclusive calibration requires one
live mxx execution-context owner on each measured device. Allocator exclusivity
must cover unrelated default-pool and outside-pool device allocations over the
claimed production interval; this owner cannot control external growth. Related
parameter views may share that execution owner; equality of ring parameters
alone does not establish shared ownership. Check the existing execution identity
and context-generation mechanisms. Independent contexts or unattributable
allocation activity cause an explicit calibration/admission error. A private
CUDA pool and cross-process calibration persistence remain outside this plan.

## Configuration and boundaries

`MXX_GPU_VRAM_PERCENT` is the only automatic wave-capacity control. Parse it once
at context configuration as an integer in `1..=100`, default `80`. Malformed,
non-Unicode, zero, negative, and above-100 values are errors. Store the checked
percentage and each device's physical total in its context:

```text
B_i = floor(total_physical_bytes_i * vram_percent / 100)
```

Use checked, overflow-safe integer arithmetic. Include the policy in context
and calibration identity. Do not reread the environment during dispatch.

Remove active use of these retired controls, with no aliases:

```text
MXX_GPU_SMALL_MATRIX_VRAM_PERCENT
MXX_MUL_SMALL_RHS_TILE_COLUMNS
MXX_GPU_COLUMN_CHUNK_COLUMNS
```

Do not introduce byte-budget, allocator-headroom, K-tile, limb-wave, or device-role
width environment variables. Existing benchmark repetitions, outer-instance
batch counts, stream-pool size, memory-pool release policy, and bounded preimage
attempts keep their distinct meanings. Configuration that changes allocation or
execution paths must participate in profile validity.

`mxx-runtime` owns fleet values, invocation planning, scheduling, reservations,
calibration identities, and the registry. `mxx-bench-estimator` consumes these
production APIs. `mxx-primitives` owns allocation requirements, device-local
execution, event ownership, and native memory probes. Runtime must not depend
on the estimator. Native headers declare cross-file/FFI APIs; CUDA bodies stay
under `crates/primitives/cuda/src/`. New GPU-only Rust modules have `gpu` in their
filenames. Follow `docs/architecture.md` for dependency direction.

## Existing implementation and corrections

| Area | Existing owner | Required correction |
| --- | --- | --- |
| Percentage and GPU contexts | `crates/primitives/src/env.rs`, `crates/primitives/src/poly/dcrt/gpu.rs`, `crates/primitives/cuda/src/Runtime.cu` | Retain checked policy and execution identities; audit allocation/probe coverage. |
| Role profiles and capacity arithmetic | `crates/runtime/src/gpu_calibration.rs` | Separate reusable measurement from all-device admission and width-class bounds. |
| Owners, pilots, ranges, and dispatch | `crates/runtime/src/backend/poly_gpu/fleet.rs` | Account for retained outputs and pending allocations; preserve ownership; bound concurrent scratch. |
| Device-local operations | `crates/runtime/src/backend/poly_gpu.rs`, `crates/primitives/src/matrix/gpu_dcrt_poly.rs`, `crates/primitives/src/matrix/gpu_rns.rs` | Share allocation requirements and destination/view contracts with the planner. |
| CUDA allocation and dependencies | `crates/primitives/cuda/src/matrix/MatrixData.cu`, `crates/primitives/cuda/src/matrix/MatrixUtils.cu` | Preserve shared completion events and reader/release ownership while enabling planned storage reuse. |
| Executor boundaries | `crates/runtime/src/executor.rs`, `crates/runtime/src/executor/gpu_plan.rs` | Make calibration an explicit preflight boundary and transfer reservation ownership with runtime values. |
| GPU measurement | `crates/bench-estimator/src/gpu.rs`, `crates/bench-estimator/src/harness.rs` | Eliminate blocking barriers inside Rayon jobs; measure shared production schedules and explicit timing units. |
| Aggregation and transfers | `crates/bench-estimator/src/lib.rs`, `crates/bench-estimator/src/dataflow.rs`, `crates/bench-estimator/src/dataflow_gpu.rs` | Keep ideal-DAG and fixed-fleet metrics distinct; charge each transfer once. |

## Logical owners, storage, and range schedules

Retain `GpuFleetMatrix`, `GpuFleetSmallMatrix`, and `GpuColumnShard<T>`. A logical
owner has global rows/columns and an ordered set of device-local intervals.
Intervals are disjoint and cover the nonempty global column range exactly once.
Each interval contains all active CRT limbs for all its rows. A device may own
multiple intervals or stored chunks; one device still uses the same fleet owner
abstraction and is not required to have exactly one physical allocation.

A stored interval and a compute wave are different objects. Splitting a stored
interval into compute views must not allocate another full payload. Views retain
the storage owner, format, parameter identity, producer events, and reader/release
dependencies. Format alignment remains explicit; ordinary matrices default to
evaluation format. Bounded values remain compact across views and transfers.

Extend the existing planning types with the smallest invocation-plan structure
needed to record:

- canonical operation and allocation-class identity;
- input ownership, representation, and materialization mode;
- output ownership and exact destination allocation layout;
- fixed replicas, aliases, and retained-output allocation requirements;
- per-device capacity, checked reservations, and reusable scratch slots;
- ordered local range jobs, explicit redistribution, and event dependencies;
- a structural schedule identity usable by the estimator.

The plan contains no node IDs or sample contents in its reusable cost identity.
An executable instance separately owns the actual value handles and reservation
leases. Different owners with identical relevant layouts may reuse a measurement.
A plan is frozen before production submission; a failed preflight publishes no
production output. No production wave is relabeled as a calibration pilot.

### Ownership policy

For unary column operations, preserve input device ownership. For compatible
multi-input column operations, intersect their interval boundaries and execute
on their common owner. Preserve the matrix-valued operand for scalar products
and the RHS ownership for ordinary and compact matrix products.

If required operands have incompatible ownership, plan a redistribution to a
specified destination layout before the operation. Its peer buffers, copies,
events, and lifetimes are part of admission and measurement. Do not silently
reshard each requested range inside a supposedly local operation. Transpose is
an explicit ownership-axis change. Tensor/concat use exact global index maps.

Fresh results without inherited ownership use the deterministic capacity-capped
assignment below. Balance assigned columns across eligible devices, subject to
retained-output and minimum-workspace capacity. Equal remaining capacities use
configured device order as the tie-breaker. Do not force ownership to follow the
ratio of temporary compute widths. This avoids a circular definition between
output reservation and the width derived after that reservation.

## Complete invocation memory admission

A wave limit alone is insufficient: completed output chunks remain live, and
temporary spans from earlier jobs may not yet be reusable. The following
accounting covers the complete invocation, including event-ordered future
claims, without requiring a host wait at each wave.

### Accepted setup and finite resource seal

At an explicit initial context setup boundary, provision finite backing and
finite event, stream and pinned resources. Include ordinary and compact owners,
scratch, caches, parameter tables and related contexts. Then establish coherent
per-device observations:

```text
P_i = total_i - free_i
E_i = P_i - pool_reserved_i + pool_used_i
P_i = E_i + (pool_reserved_i - pool_used_i)
```

`P_i` is physical residency; `E_i` is allocator-adjusted live residency. Unused
pool reservation still occupies physical memory. Accept setup only when both
`P_i <= B_i` and `E_i <= B_i` hold. Unused pool reservation may be reclaimed at
this explicit boundary before repeating the coherent observation. Failed or
oversized setup publishes no executable permit. These acceptance checks do not
prove that provisioning calls themselves never exceeded `B_i`; that stronger
claim would need separately justified allocator/resource bounds, which these
queries do not provide.

Use checked arithmetic and require a quiescent allocation epoch: no pending
allocation/free activity and no unrelated owner activity while physical and
pool counters are read. The separate CUDA queries are not an atomic snapshot;
arithmetic success does not establish coherence. Retry an unverified initial
observation at setup or reject it. Preserve the caller's CUDA device during
probes. Complete native activity instrumentation, private-stream retirement
coverage and a source audit before enabling coherent epoch certification.

A seal covers the supported mxx-managed allocation sites, backing spans, typed
resources, execution contexts and worker configuration. Provision the required
managed launch metadata, reader/release/timing events and streams, or include
independently justified remaining managed demand in admission. Load reachable
kernels at setup, but do not claim that loading or stable counters bound future
opaque CUDA consumption. An unsupported managed allocation/resource class cannot
obtain a permit. Validate first production use on every permitted worker and
covered kernel/launch class.

Once sealed, production may consume only native-identified prepared spans and
resource slots, plus explicitly bounded remaining managed demand. Unsupported
requests, resource exhaustion, missing permits and unexpected managed allocation
sites fail before submitting their allocation. Unplanned backing growth or lazy
managed resource creation is prohibited. Opaque CUDA growth may exceed the
configured budget without stopping otherwise valid work.

### Managed charges, logical reservations and observed residency

Maintain independent managed accounting and native logical reservations:

- Let `E_i^setup` and `P_i^setup` be the accepted initial residency. The managed
  charge is `R_i = E_i^setup + live_managed_reservation_bytes_i`. The baseline is
  fixed for the accepted inventory lifetime; each additional reservation remains
  charged through its reserved, submitted, resident and release-pending states.
  Independently bounded remaining managed demand must fit `B_i - R_i`.
- Native logical reservations identify finite spans and typed resource slots.
  They cover fixed preparation, complete retained outputs and simultaneous
  scratch. Prepared backing included at setup is not charged again when an output
  or temporary claims a slice. Logical reservations and actual occupancy claims
  are distinct, with native backing/span/resource identities tracking aliases.

Moving a logical reservation to an occupied span does not change its managed
backing charge. A queued release returns no capacity until its producer/reader
release dependencies establish safe reuse; event-ordered prepared handoff needs
no host wait. Recycling a prepared slice never retires its baseline backing
charge. A separately charged native allocation returns managed capacity after
its release completes, even if CUDA retains the physical pool pages. Pinned host
capacity is reusable only when CPU writes cannot race preceding DMA, or through
an explicitly ordered host-fill callback and following transfer. Reservation
tokens may move between workers; activation is thread-bound, and existing Rayon
subjobs require explicit child permits.

Later physical/pool readings update diagnostic observations only. Neither high
nor low samples absorb or retire managed allocation IDs, change the accepted
baseline, or infer free logical slots. Observe at existing coherent boundaries;
do not wait for production just to refresh counters. Observed `P_i > B_i` or
`E_i > B_i` does not reject otherwise valid admission or cancel running work.
Actual allocation or device failure remains an error. Counter shape, execution
identity and coherence checks remain required for publishing observations.

Admission and reservation publication are one ordered operation in the owning
fleet dispatcher; concurrent callers cannot spend the same capacity twice.

For each device, define:

- `S_i`: sealed native backing spans, layouts, alignment and typed resource slots;
- `L_i`: earlier live/reserved logical claims, including inputs, fixed preparation
  and outstanding producer/reader/release lifetimes, each counted once;
- `O_i`: additional logical demand for the complete retained outputs, including
  destination descriptors; aliases add no duplicate payload claim;
- `T_i(w, class)`: a conservative requirement for the simultaneous temporary live
  set of a local job, including staged inputs, peer buffers, expanded workspace,
  parameter conversion and metadata absent from `L_i` and `O_i`.

Admission requires every participating device to satisfy the managed budget
and the native fit predicate:

```text
R_i + independently_bounded_remaining_managed_demand_i <= B_i
Fit_i(L_i, O_i, reserved_concurrent_temporary_i; S_i)
```

The remaining-demand term excludes charges already in `R_i`; fully
prepared classes have zero additional managed allocation demand. `Fit_i` checks concrete
available spans, layout, alignment, typed slots, all retained outputs and
simultaneous scratch. A scalar sum of requested bytes is insufficient. Temporary
requirements follow actual dependency lifetimes, not the sum of sequential jobs
or the most recent allocation. Views retain the existing input claim; imported
inputs remain temporary until ownership transfers.

Fixed replicas and planning resources also require logical reservations before
claiming storage. Before fresh ownership is known, compute each eligible device's
hypothetical fixed/preparation requirements without claiming replicas. Include
these requirements in its output-cap calculation. After assignment, reserve and
stage fixed data only on participating devices. Pilot resources on a representative
device need separate reservations even when it owns no output. Pilot and fixed
preparation requirements must fit together with already retained claims. Missing
requirements are a planning error, not permission for an unbounded pilot.

### Output assignment before compute-width selection

For fresh output, calculate a per-device output-column cap `A_i` using the known
output allocation layout and the smallest supported local job. In this step,
`L_i` includes the conservatively reserved hypothetical fixed/preparation live
set just described; it does not assume replicas have already claimed storage.
Require a conservative monotone bound across possible global starts and ownership
boundaries when a cap is computed before those boundaries are assigned:

```text
A_i = largest supported c such that the managed budget check holds and
      Fit_i(L_i, O_i_bound(c), T_i_bound(minimum_valid_job); S_i)
```

Do not assume each source operand can be arbitrarily narrowed: fixed shapes,
gadget layout, tensor factors, and preimage coupled rows stay intact. A primitive
whose exact minimum depends on a layout must supply a checked bound or reject
this fresh-output assignment mode. A zero-cap device receives no output.

If the checked sum of caps is smaller than `C`, reject before production
allocation. Otherwise use deterministic capped water filling to assign counts
`C_i <= A_i`, with `sum C_i = C`. Give fresh owners contiguous global intervals
in device order. Claim one destination per planned interval where the native
layout permits it; otherwise account for every required chunk and descriptor in
`O_i_bound`. Before width selection, bound this metadata using the maximum chunk
count permitted by the minimum valid range, not a width that has yet to be
derived. Confirm the concrete layout against its bound before reservation.
The final counts, starts and tails must decompose into valid primitive ranges;
water filling alone does not establish this. Reject an unsupported decomposition
before allocation rather than pad, truncate or alter coupled dimensions.

Inherited ownership fixes `C_i` and its interval list instead. Admit its complete
output layout on those owners. If it cannot fit, return a resource error rather
than silently moving ownership; a redistribution must be a separately planned
operation. This can conservatively reject a feasible alternative placement.
Global placement optimization is not a completion requirement.

The resident-returning primitive API does not support a result larger than its
admitted aggregate/device-local capacity. Existing outer family/artifact staging
remains available at its declared boundaries, but it does not excuse an
unbounded inner result. A future out-of-core primitive would require an explicit
producer/consumer storage contract and is outside this plan.

### Scratch reuse and overlapping invocations

For a given local workspace class, reserve reusable scratch spans whose reuse
waits on all previous consumers via stream events. This introduces no host wait.
Independent claims/jobs may overlap only if their combined logical live set has
a separate checked reservation. Preserve current useful overlap when it fits; do not fix
a race by serializing independent work or adding a global
mutex. When overlap cannot fit, choose an explicit bounded device dependency or
reject before submission. Any dependency required for capacity is represented
in the plan and accounted for in measurement.

Claim final output storage once where possible and write destination views;
otherwise reserve all retained chunks up front and include their metadata.
Native temporary claims must honor the prepared span/dependency lifetime.
Production must not grow backing through `cudaMallocAsync` calls or queue an
unbounded live set on unrelated streams under the name of sequential waves.

Across invocations, reservation leases transfer to returned owners. The executor
retires their logical claims at last use, respecting outstanding readers, while
the backing remains physically charged. Downstream preflight accounts for all
still-resident upstream owners. Plan mismatch is an error; do not perform a
mid-production replan, sampler retry, CPU fallback, or implicit
artifact spill to recover capacity.

## Allocation bounds, profiles, and pilots

The primitive layer supplies logical span and resource requirements from the
same size/layout calculations used by production. Do not duplicate native kernel
formulas in the estimator. Requirements include alignment, descriptors, scratch,
compact widths and typed resource classes. Prepared backing belongs to accepted
setup; remaining managed allocations require independently justified bounds.
Opaque CUDA consumption is observed separately, without a runtime budget cap.
Reject unsupported additional managed resource demand.

A one-column measurement is a calibration sample, not a proof of linear growth.
In particular, current matrix allocation changes event/stream behavior at
`rows <= 4 && columns <= 4`. Larger width classes may change kernels, temporary
shapes, or the number of intersections. Each production width must belong to a
supported allocation class with an explicit upper envelope `T_i(w, class)`.
Use monotone bounds within a class for capacity search; enumerate supported
classes rather than assuming the whole implementation is one linear function.
Unknown classes fail closed until their requirements and validation exist.

### Profile identity and data

Extend `GpuDeviceCalibration`, `GpuCalibrationProfile`, and the existing registry
rather than adding a competing registry. The canonical key contains, as relevant:

- primitive variant, fused layout, row count, inner dimension, scalar position;
- ordered native CRT bases, active level, ring dimension, gadget layout;
- input/output formats and compact widths with inclusive coefficient bounds;
- distribution/cutoff parameters and preimage attempt policy;
- allocation class and resident/view/imported operand preparation mode;
- global intersection structure when it changes the executed path;
- backend/CUDA build identity, device capabilities, and execution configuration;
- memory metric identity, prepared-storage configuration and native layout/bound identity;
- active fleet role assumptions and the parsed VRAM policy.

Exclude node IDs, graph paths, sample contents, hash tags, and offsets only when
they provably do not alter the class. Normalize logical columns only within a
class whose range mapping and allocation bound justify it. Cost-insensitive loop
indices do not enter the key; bindings that change a concrete class do.

Store the pilot column count, logical occupied-byte incremental high-water, its
metric identity, allocation class/bound identity and measured bytes-per-column
hint. Do not store old slot availability or executable owners in a reusable profile. Baselines and
pilot latency are observation records. No cross-process profile files are added.
A profile hit still requires invocation admission against every device's current
managed charges and available native logical capacity.

### Explicit preflight and sampling isolation

Move pilot waits into an explicit preflight method invoked by the executor or
benchmark caller before production dispatch. A device-only production wrapper
requires an admitted plan and never initiates a blocking cache-miss pilot.
Preflight may depend on the actual inputs of that invocation; it is not required
to execute every graph pilot before any graph input exists.

Prepare representative fixed inputs and the smallest valid pilot using the
production allocation and execution entry points. Dummy contents and separate
randomness domains isolate pilots from production samplers, transcript records,
artifacts, progress counters, and preimage attempts. Do not call a side-effecting
production column-source loader merely to produce a pilot. Use a schema-equivalent
isolated source with the same declared preparation mode. Preserve global indices
in production hash/preimage sampling regardless of plan widths or device count.

Measure role 0 and, when a nonzero role is required, configured device 1. If the
representative cannot admit its minimum pilot, return an explicit calibration
error; do not borrow another device's capacity or cache an invented role profile.
Do not calibrate nonzero roles for an invocation that uses only role 0.

The pilot procedure is:

1. Verify the supported seal, preparation/pilot requirements and native fit, then
   obtain reservations that include all existing retained claims.
2. Stage fixed inputs and complete their relevant preparation events; fence only
   this execution owner's earlier release streams at this calibration boundary.
3. Verify allocation exclusivity and context generations. Record the fixed-input
   logical occupancy baseline and begin a native actual-claim high-water interval,
   distinct from reservation counters. Physical/pool counters are diagnostics.
4. Claim prepared output/scratch spans and execute one isolated pilot through the
   production range runner, retaining all output owners.
5. Wait on pilot completion events; record actual logical occupied-byte high-water
   and verify owner/generation stability and ledger/claim consistency. Count aliases
   and reused spans once, and exclude fixed preparation from the incremental peak.
6. Check actual claims against their reserved layouts/resources and managed
   capacity. Only a successful, uncontaminated interval produces a profile.
   Physical-budget excess alone does not invalidate an otherwise valid pilot.
7. Drop pilot outputs through normal event-ordered release; reconcile logical
   reservations before production. A calibration-boundary release fence is allowed.
   Keep backing charged and restart production at its original global range.

An unexplained logical occupancy drop below the fixed baseline, unexpected
allocation/free activity, or changed owner/generation contaminates the interval.
Drop the isolated pilot and retry at most three complete attempts under the same
conditions. Failure is an error, not a cached miss that permanently poisons a
recoverable signature. Prepared allocating classes use a logical-occupancy metric
with an identity distinct from default-pool bytes: zero default-pool growth is
valid when their logical output/scratch claims are nonzero. A zero unexplained
logical increment is still an error, and these operations remain allocating
classes. A declared allocation-free view/no-op accounts for any required metadata
and uses its proved requirements directly, without dividing by zero or launching
a fake pilot.

### Width derivation and all-device checks

Reserve the complete retained output demand before searching scratch widths.
For an allocating class with nonzero logical pilot pressure, use this hint:

```text
s_role = ceil(pilot_logical_incremental_peak_bytes / pilot_columns)
H_i = available logical byte capacity after L_i and O_i reservations
trial_i = floor(H_i / s_role)
```

`H_i` is the candidate's internal capacity from the native span inventory, not
headroom in the separately managed allocation charge `B_i - R_i`. A fully charged physical arena
can still have free logical slots. Typed resource limits, fragmentation and
alignment are checked by `Fit_i`, not inferred from `H_i`. If classes use multiple
backing/layout kinds, use their native candidate-capacity mapping and retain all
of those dimensions in the subsequent fit check.

The measured hint is neither a reservation nor a memory proof. The pilot may
include output claims later reserved in `O_i`; this can make its candidate
conservative, but must not charge that output twice. Once ownership and `O_i`
are fixed, choose the largest class-valid width no greater than `trial_i` and
remaining local columns for which `Fit_i(L_i, O_i, T_i(w, class); S_i)` and the
managed budget check hold. Revalidate class boundaries. An allocation-free
class derives its candidate from logical ranges and its proved requirements.
A candidate below the minimum job is a resource error on an active owner.

Retain two reported capacities: `W_gpu0` and `W_nonzero`. First derive the safe
capacity `w_i` on every active owner using its own native capacity/reservations
and the appropriate role profile. Then take:

```text
W_gpu0 = w_0                              // when role 0 is active
W_nonzero = min(w_i for active i > 0)      // when a nonzero role is active
```

Do not replace both roles by their common minimum. Every local job is also
capped by its remaining interval and class constraints. Inactive devices consume
no temporary reservation and do not make width zero an active-role error.
Absent-role fields are omitted in logs. If one invocation needs several classes,
report this pair per class; no single width silently crosses a class boundary.

Before dispatch, publish a consistent reservation transaction covering all active
devices. Roll back unsubmitted reservations on error. Concurrent mxx invocations
use the same ledger; asynchronous frees, preparation, and output retention cannot
invalidate an already admitted upper bound by spending its capacity again.
External memory changes still fall under the operating assumptions above.

## Local jobs, logical fleet waves, and host workers

Build ordered local jobs by intersecting output ownership, input/source chunk
boundaries, primitive range constraints, and the admitted width classes. A job
contains its exact global range and device-local views or measured transfers.
It must not reconstruct an all-column operand to process a smaller range.

For logical accounting, wave `j` groups job `j` from each device that has one.
Consequently:

```text
actual_wave_count = max(number_of_local_jobs_i)
nominal_capacity = W_gpu0 + (G - 1) * W_nonzero
nominal_wave_count = ceil(C / nominal_capacity)
```

The nominal expressions apply only to a single class with both roles present
(use `W_gpu0` for a one-device fleet). They are optional throughput summaries,
not the schedule. Equality with `actual_wave_count` requires the concrete range
assignment to realize that packing; matching percentages/profiles alone is not
sufficient. Keep counts checked. Zero-column operations either use their
validated empty-value semantics without allocations/jobs or reject at validation.

Example: owner counts `(90, 10)` and consumer widths `(10, 90)` require at least
nine local-job groups, although nominal packing of 100 columns suggests one.
Stored chunk boundaries may require still more jobs; total columns per device
alone does not determine the exact count. Runtime and estimator consume the same
range descriptors rather than separately reconstructing either formula.

Use one long-lived enqueue worker per configured device for multi-device work.
Do not block at a `std::sync::Barrier` inside Rayon jobs or create a replacement
Rayon pool to make such a barrier progress. GPU worker progress must be independent
of `RAYON_NUM_THREADS`, including one Rayon worker with multiple GPUs. Keep the
existing single-device caller-thread fast path where it has equivalent semantics.

Workers submit bounded commands and acknowledge submission, not device
completion. The host may join submission results; a device-only wrapper must not
wait for kernel completion or queue backpressure caused by GPU execution. Use
reusable per-device scratch and stream-event dependencies to bound device work;
keep command descriptors compact/lazy so buffering does not scale with all
application slots. Do not enqueue an unbounded list of independent workspaces.

There is no device-completion barrier between logical fleet waves. A device can
submit its next local job when the plan's local dependency allows it. Independent
devices and admitted independent resources remain concurrent. Sampling rows
coupled by an equation stay together on one device.

A launch failure cancels unsubmitted commands, reports one invocation error,
and retains allocations referenced by already submitted work through their
completion/release protocol. Every worker must acknowledge success or failure;
no participant waits forever for a worker that returned an error. Do not publish
a successful result or artifact manifest for a partially failed invocation.

## Primitive coverage and redistribution

Create the implementation coverage table from the current IR classification,
including fused operations, rather than treating the following list as exhaustive.
Each accepted variant needs a range mapper, allocation requirements, pilot mode,
production owner rule, and an estimator measurement class.

| Primitive class | Required execution contract |
| --- | --- |
| Uniform/Gaussian/hash sampling | Fresh output ownership; isolated pilots; production global-index/randomness contracts preserved. |
| Negate, scale, automorphism, row sums | Preserve ordinary input ownership and use local views/destinations. |
| Add/subtract, fused row/block arithmetic | Preserve compatible layouts; plan incompatible-layout transfers explicitly. |
| Ordinary multiplication and multiply-accumulate | Preserve the matrix-valued RHS/output axis; handle scalar operand position and bias correctly; stage fixed operands once. |
| Gadget/compact decomposition and compact hash | Preserve bounded kinds and input ownership; no all-column full-DCRT digit expansion. |
| `mul_small_rhs`, including fused row-block paths | Compact RHS views, replicated fixed LHS, planned full output, one expanded workspace per admitted active slot. |
| Preimage sampling | Stage public/trapdoor owners once; load only target job ranges; retain coupled rows; pack accepted columns into compact destinations. |
| Modulus switch/reduce, centered rebase/extend, block switching, RNS ModUp/ModDown, CRT recomposition | Preserve exact operation-specific equations, bases, format and row-layout changes; account for both source/destination parameters and every temporary. |
| Constants | Generate exact requested global ranges without changing full matrix/gadget semantics. |
| Slices and row concat | Preserve compatible ownership; account for view descriptors and explicit incompatible-layout copies. |
| Tensor, column/diagonal concat | Map exact global indices/intersections; bound segment metadata and transfers; never substitute arbitrary smaller tensor factors. |
| Transpose | Explicit ownership-axis mapping and peer redistribution, with complete buffer and transfer accounting. |
| Scalar extraction, decoding, or unsupported independent ranges | Use the complete supported atomic/host-boundary operation; do not extrapolate by columns. |

Identity uses global row/column equality. UnitRow translates the global selected
column into the local interval. Gadget construction retains full block/digit
layout. One-column UnitColumn/PowerOfBase/Rotation/Polynomial forms remain
single-owner jobs. If location changes an executed cost class, retain that
intersection/location class instead of erasing it from calibration identity.

Fixed full-evaluation operands are replicated once per participating device and
kept live through their final consumers. Reuse existing weak caches by value,
parameter, context and layout identity where applicable. Compact operands stay
compact when transported. Missing peer access yields an explicit unsupported
placement error; there is no hidden host reconstruction or CPU computation.

For compact multiplication with ring dimension `N`, `L` active CRT limbs, and
inner dimension `K`, the expanded coefficient workspace has
`L * K * local_columns * N` entries; its byte size uses the actual native element
width and checked layout/alignment requirements. It is one workspace over all
local columns, all `K` rows and all active limbs. Do not add K- or limb-tiling
controls to repair a failed admission. Output, compact input and metadata are
separate accounted objects. Preserve preimage inclusive cutoffs and bounded
attempts without changing the preimage equation.

## Estimator contract and calibration reuse

The estimator accepts the same invocation-plan inputs as runtime: concrete
primitive types, preparation modes, ownership/source intervals, current planned
baselines/reservations, device configuration, and optional frozen registry.
Construct the plan with runtime APIs. Synthesized operand values may differ,
but their representations, allocations, range mappings, and dependency schedule
must match the plan. Measure each primitive using that plan across its active
fleet; do not dispatch unrelated requests on different GPUs.

A shape-only estimate without input ownership/baselines is explicitly a nominal
scenario with stated synthetic assumptions. It must not claim runtime schedule
agreement or a physical memory feasibility certificate. Exact registry equality
alone does not convert that scenario into an actual runtime plan.

Registry sharing is in-process and optional. Runtime reuses valid role profiles,
recomputes all-device admission, and owns any new profiles produced on misses.
It does not retain estimator values or baselines. The measurement-cache key also
includes the concrete schedule/layout class, active devices, derived capacities,
preparation/transfer modes and timing contract. Source offsets may be normalized
only when doing so preserves all these properties.

### Timing and wave classes

Measure coordinated production local jobs, including planned redistributions.
Record two distinct clocks:

- `device_elapsed_seconds[i]`: CUDA-event elapsed span covering that device's
  planned operation and transfers, ending after every participating stream's
  completion dependency. This is a device timeline span, not a sum of kernel
  times or a utilization measurement; internal idle gaps may be included.
- `fleet_wave_wall_seconds`: host monotonic time from coordinated submission
  through completion of all active device events. Host dispatch is included.
  Optional per-device host service times must be labeled separately.

Timing start/join/stop events are benchmark-only and use the same operation
entry points. Do not remove production ownership waits or move transfers out of
the declared timer to create an apparent speedup. Idle devices contribute zero
work. Finish warmups and reconcile their releases before memory measurements.
Use the same setup/load/store boundary for baseline and candidate comparisons.

Measure one representative per distinct scheduled wave class and record its
multiplicity. A class includes the vector of active devices/local widths and
range/transfer behavior, not just the primitive name. The final partial wave is
charged at its measured class unless an explicitly documented larger class is
used as an estimate with the same ownership and accounting. Do not assume timing
monotonicity across different kernels or active-device sets to avoid measuring
an otherwise distinct class.

For class `k`, let `n_k` be its count, `L_k` its measured fleet wall time, and
`D_ki` each active device's measured elapsed span:

```text
cumulative_wave_seconds = sum_k(n_k * L_k)
work_seconds = sum_k(n_k * sum_i(D_ki))
actual_wave_count = sum_k(n_k)
```

For one repeated class this reduces to one-wave latency times chunk count, and
outer invocation/slot multiplicities multiply that cumulative total. This is a
measured-wave cost model. Production permits device-local overlap between logical
waves, so cumulative wave time is not asserted to equal end-to-end runtime wall
time or to be a universal bound under interference. Report measured whole-invocation
runtime separately and explain overlap/calibration/materialization differences.

### Preserve metric meanings and account for transfers once

| Field or report | Meaning after coordinated migration |
| --- | --- |
| `NodeMeasurement::latency_seconds` | Ideal independent-wave dependency latency: one class's wave time, or the maximum scheduled class time when several independent classes exist. Atomic operations retain full atomic latency. |
| `cumulative_wave_seconds` | Sum of planned production wave costs with actual class multiplicities, never divided by device count. |
| `work_seconds` | Aggregate device elapsed seconds across every wave, using the explicit CUDA-event timing contract above. Existing host-timed results must be regenerated. |
| `independent_wave_count` | Number of scheduled independent wave groups. Non-independent operations remain atomic for ideal-DAG aggregation. |
| `measured_wave_workspace_bytes` | Maximum native logical occupied-byte increment per wave after fixed inputs are prepared, with its metric identity. State whether output claims are included; destinations already claimed before the interval are excluded and reported separately. Zero default-pool growth does not imply zero workspace. |
| `workspace_bytes`, `workspace_high_water_bytes` | Hypothetical simultaneous independent-wave/DAG resources, with their stated sharing assumptions; not device admission or physical peak. |
| Per-device admission/peak diagnostics | Accepted setup `P_i/E_i`, current managed `R_i`, sealed managed resources, logical retained/output/scratch reservations and fit result, plus observed pool/physical peaks and budget excess. Samples are diagnostics; excess does not stop valid work. |
| `CostReport::critical_path_seconds` | Existing ideal dependency schedule, not completion time on a fixed number of GPUs. |
| `CostReport::total_time_seconds` | Nested invocation-weighted cumulative wave time plus separately owned dataflow/dispatch costs. |

Do not silently replace the ideal-DAG fields with fixed-fleet invocation time or
label hypothetical simultaneous workspace as measured VRAM. Preserve the max-over
stages rule for reported parallelism. For waves that are not independent in
operation semantics, use an atomic dependency measurement rather than pretending
capacity scheduling made them mathematically independent.

Record a cost owner for each transfer/dispatch boundary. Primitive timing owns
its declared local/peer transfers; artifact/RAM materialization outside that
boundary belongs to `dataflow`. Generic executor dispatch excludes host submission
already in `fleet_wave_wall_seconds`. If code moves a transfer across the timing
boundary, update its dataflow accounting in the same change. A fixed fleet is
not divided by GPU count again. Independent outer instances may share devices
only when the production plan explicitly partitions those devices and accounts
for simultaneous memory.

## Artifacts and synchronization

Artifact formats and content hashes do not depend on device count, calibration,
role width or placement. Serialization traverses exact global row-major
coefficient order across the ordered intervals; shard order alone is insufficient
when rows interleave. Stream into bounded host buffers or existing artifact-store
interfaces without gathering a complete value on one GPU. Validate the full schema
and semantic kind at import, then admit destination storage before decoding
ranges. Imported bounded payloads remain compact. Temporary family staging and
streamed scalar exports retain their current declared runtime boundaries.

Successful device-only production paths contain no host device-completion wait,
`cudaDeviceSynchronize`, or `cudaStreamSynchronize`. Explicit preflight,
measurement and host-visible artifact/decoder boundaries may wait on their own
completion events. Error cleanup uses event-ordered release; if completion cannot
be established, retain/quarantine storage according to the existing failure
policy instead of recycling possibly in-use buffers. Do not introduce a device
reset or a CPU fallback.

Use existing shared completion owners and reader events; never assume every limb
owns an independent non-null event. Record allocation and producer completion,
wait on source and scratch consumers in the relevant device streams, and transfer
release ownership with every view/output. A fleet owner aggregates readiness;
a local successor needs only its relevant producer dependencies. Admission
reservations are released by the same lifetime protocol, not by host enqueue
completion. Pool queries occur at preflight/measurement boundaries, not once per
hot-path wave.

## Diagnostics

Every accepted production invocation records its available plan/admission values:

- operation/profile/allocation-class and structural schedule identities;
- profile hit/miss and pilot attempts;
- configured and active device IDs, hardware identity and VRAM percentage;
- per-device total/budget, accepted `P_i/E_i`, setup epoch, current managed `R_i`, and later observed `P_i/E_i` and budget excess;
- seal/configuration identity, native span/resource capacities and fit result, with
  fixed, complete retained-output and simultaneous temporary logical reservations;
- per-role pilot columns, memory metric, logical incremental peak and bytes-per-column hint;
- per-device safe capacity and selected `W_gpu0`/`W_nonzero` per class;
- assigned intervals, local job counts, actual wave count, and any nominal count
  clearly labeled as a scenario rather than a schedule;
- compact bytes, full-output bytes, expanded workspace bytes, peer/staging bytes.

Instrumented measurement runs additionally record the timing contract, measured
per-device elapsed spans, fleet wave wall time, class multiplicities, cumulative
time, aggregate work, observed per-device peaks and whole-invocation wall time.
Ordinary production does not add timing events or completion waits to fill these
fields. Any cached timing/capacity observations are labeled with their originating
measurement; a reused estimate is not a new runtime observation. Rolling memory
samples carry their observation status and do not replace the charged ledger.

Missing roles are absent, not zero-filled invented measurements. Overflow is an
error in planning. Saturating sentinels retained for legacy ideal reports are
never accepted as feasible physical reservations. Logs need enough identities
and raw values to reproduce accounting without exposing private sampled values.

## Ordered implementation and acceptance gates

Each stage changes existing owners and their callers together. Preserve current
mathematical/reference tests and main's resident-dispatch, compact multiplication,
shared-event and artifact optimizations. Do not retain an unchecked production
width override: measurement-internal requested widths still pass admission;
synthetic schedule tests can use pure planning inputs without allocating GPUs.

1. **Freeze plan contracts and executable inventory.** Map every current GPU IR
   variant, including fused row/RNS operations, to logical span/resource and range
   requirements. Inventory every supported kernel/launch, context and worker class,
   preparation mode and output lifetime. Gate: no accepted primitive lacks a complete
   resource argument, metric identity or supported minimum range.
2. **Implement accepted setup, the resource seal and both ledger dimensions.**
   Provision finite backing/resources, audit native activity and retirement coverage,
   and accept coherent setup only with `P_i <= B_i` and `E_i <= B_i`. Add native
   identities, transferable reservations, thread-bound activation and child permits.
   Extend destination/views and resource accounting, then fresh-output caps and
   per-device native fit. Gate: unsupported classes cannot seal; complete retained
   outputs and overlapping scratch fit or fail before an allocation is submitted.
3. **Repair calibration and cache validity.** Make preflight explicit; admit pilots,
   isolate their sampler/source effects, measure native logical occupancy for
   class-valid roles, and validate every device on hits and misses. Remove unchecked
   manual-width production bypasses. Gate: matching profiles reuse measurement but never
   old slot availability;
   misses leave production values/transcripts/progress untouched.
4. **Unify ownership-aware execution and workers.** Share the plan across ordinary,
   compact, preimage and redistribution paths. Add independent per-device workers
   and scratch/event reuse with bounded simultaneous reservations. Load staged
   preimage ranges on their assigned worker device; every import consumes an admitted
   plan, and pilots share the production range runner with isolated bindings. Gate:
   no peer copy on a declared owner-local path, no blocking Rayon barrier, and no unbounded
   asynchronous workspace accumulation or lost reader/release protection.
5. **Integrate estimator schedules and metrics.** Consume the production plan,
   measure wave classes/device spans and update aggregation, dataflow accounting,
   caches and `docs/benchmark-estimator.md` together. Gate: identical inputs produce
   identical ranges, classes and reservations; logical pressure, physical residency
   and hypothetical metrics stay distinct, and transfers are counted once.
6. **Complete artifacts, diagnostics, and regression validation.** Cover every
   operation class and metadata boundary; audit all removed controls. Gate:
   canonical artifacts, exact arithmetic, liveness and repeated execution pass
   the tests below, with warning-free CPU/GPU unit builds.
7. **Measure performance and obtain independent review.** Compare the preserved
   baseline with the candidate on identical parameters/hardware/timing boundaries.
   Investigate regressions, retain raw results and require review of per-device
   memory, synchronization and estimator correspondence. Gate: correctness and
   resource guarantees pass independently of whether a speedup is observed;
   report speedups only where supported by matched measurements.

Stages 1-5 depend on the preceding contracts. Disjoint primitive adaptations may
proceed in parallel only after their shared plan and ownership APIs are fixed.
A failure returns to the responsible stage; it never weakens a mathematical test,
adds a synchronization workaround or replaces a GPU computation with CPU work.

## Validation matrix

Default scope is unit tests. Do not run application integration tests unless the
user authorizes them in the implementation task. Record executed checks and
their exact source/hardware scope in the implementation status document.

| Case | Required evidence |
| --- | --- |
| Configuration and identities | Default/min/max/invalid percentage; checked budget arithmetic; context/role/class/bound identity mismatch rejection; related parameter views versus independent contexts. |
| Three or more devices | Reuse device 1's profile while increasing retained logical pressure or separately bounded residency only on device 2; retain independent role-0 width and safely reduce/reject nonzero capacity. |
| Complete-result admission | Multiple waves near logical capacity; reserve every retained output and outstanding release before scratch-width selection, including full-output scratch reduction; oversized resident results reject before submission. |
| Accepted setup and resource seal | Require both `P_i <= B_i` and `E_i <= B_i`; oversized/failed setup publishes no permit. Exercise first production use on every permitted worker/kernel/launch class and unsupported-class rejection. Do not infer provisioning-time safety or resource completeness from sampled peaks. |
| Independent capacity dimensions | Fully charged physical backing with free logical slots remains usable; logical exhaustion despite physical headroom rejects. Wrong layouts/alignment, stale tokens, missing permits and cross-device rollback fail before their allocation. |
| Snapshot/ledger coherence | Interleave physical/pool reads with admitted allocation/free progress, including reserved/used growth after a physical sample; valid-looking mixed counters must not remove a live charge. Exercise coherent observation refresh and unverified observation rejection without production waits; high/low observations never alter managed charges, and budget excess does not stop valid admission. |
| Scratch and overlap | Delayed readers, different streams/workers, consecutive waves and overlapping invocations; native combined logical claims fit reservations, cross-thread reuse retains dependencies, and recycling never drops physical backing charges. |
| Allocation classes | Boundary widths including 1, 4 and 5, multiple CRT depths and range intersections; actual logical claims fit typed reservations; unknown/overflow/zero-unexplained logical profiles reject. Zero default-pool growth with nonzero allocating-pilot logical pressure is valid. |
| Ownership and schedule | Unequal widths, `(90,10)` ownership with `(10,90)` capacities, fragmented intervals, fewer columns than devices, empty semantics, and incompatible layouts; exact coverage and shared actual job counts. |
| Worker liveness | At least two GPUs with `RAYON_NUM_THREADS=1`, normal Rayon settings, and injected worker failure; bounded completion/error with every worker accounted for, using a test watchdog rather than hanging the suite. |
| Pilot isolation | Cache hit/miss and contaminated retry produce the same seeded production replay/artifacts; pilot sources cannot advance production progress or attempts. Generate the seed randomly per test and reuse it only for the paired comparison. |
| Primitive correctness | All accepted direct/range/redistribution variants compared with existing trusted primitives or existing round trips, including global constant/gadget positions and exact CRT operations. |
| Compact/preimage | No forbidden full-DCRT RHS; inclusive bounds and preimage equation preserved; global target offsets and admitted scratch sizes correct. |
| Timing and aggregation | Actual wave classes and multiplicities, active devices, no second GPU-count division, ideal-DAG versus cumulative fields, and one charge per dispatch/transfer boundary; production logs require no timing events/waits and label reused observations. |
| Artifacts and lifetimes | Device-count/layout-independent canonical bytes/hashes; last-owner drops, shared completion aliases, concurrent readers, and failure paths preserve ownership. |

Use narrow tests during each implementation stage. The final build gates are:

```bash
cargo +nightly fmt --all
cargo test -r --workspace --lib --no-run
cargo test -r --workspace --lib --features gpu --no-run
```

Both builds must be warning-free. Run the applicable full `mxx-primitives` and
`mxx-runtime` unit suites plus changed estimator/DSL/IR/FHE unit tests. Execute
GPU binaries outside the sandbox. For synchronization changes, compile once and
run the relevant built binaries with the identical command 300 times, completing
all repetitions even if some fail. Other GPU round-trip smoke checks use 3-5
repetitions. Preserve normal multithreaded test execution; existing semantic
`serial_test` guards may remain. Separate the deliberate one-Rayon-worker liveness
case from the normal repeated command and report both.

Single-device tests cannot establish multi-device admission or concurrency.
Physical two-device tests must exercise same-primitive overlap, ownership and
peer transport. Physical three-device coverage is required for the nonzero-role
residency/reuse invariant; pure planner tests supplement it. Obtain hardware
through the authorized remote GPU workflow when local devices are insufficient;
record exact source commit, devices, commands, logs and any uncompleted gate.

Performance experiments compare the source-baseline commit and candidate on the
same GPU model/count with identical ring/CRT parameters, bounds, layouts,
prepared baselines and timing boundaries. Alternate paired runs and report raw
samples/distributions, role pilots, assignments, actual waves, cumulative model,
whole-invocation wall time and per-device peaks. Include small resident and
multi-wave cases. Do not infer kernel speedups from changed serialization or
validation boundaries. Retain prior accepted optimizations; regressions need an
identified cause and disposition before performance acceptance.

Final source searches reject executable aliases and unchecked scheduling
bypasses. Retired variable names may remain in this migration/removal checklist,
but not in active configuration examples or implementations. No `references`
directory is edited. Update affected public documentation in the implementation
stage without adding application-specific or unpublished material.

## Completion criteria

Plan completion means the contracts above have no unresolved design choices
required by the enumerated implementation stages. It does not assert that code
already satisfies them. Implementation completion additionally requires shared
runtime/estimator plans, accepted finite-resource seals and complete-invocation
admission in managed and native logical capacity on every device, with observed
physical-budget excess allowed during computation,
valid metric/class-specific calibration reuse, ownership-correct GPU execution,
canonical artifacts, unambiguous metrics, warning-free builds, the required
physical-device repeated tests, matched performance evidence and independent
review. Unavailable hardware or an unrun gate is reported as incomplete.
