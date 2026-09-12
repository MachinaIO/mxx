# Fleet-wide GPU implementation handoff — stopped on 2026-09-12

## 1. Read this first

### Update after the same-day resumption

The user resumed work with an explicit instruction to complete this plan.
All steps of section 8 (0 through 7) are complete and verified within the
authorized scope; the acceptance summary is
`docs/plans/fleet-wide-gpu-acceptance-2026-09-12.md` and the chronological
detail is in the status document. Physical multi-GPU checks remain deferred
by the plan itself. The stop-time text below is retained for history; where it
conflicts with this update, this update wins.

| Item | State after resumption |
|---|---|
| Two primitive failures (Step 0) | Explained (disjoint serial guard groups) and resolved with child-process isolation; assertions unchanged |
| Compact hash (Step 1) | `Decomposed`/`SmallDecomposed` hash admitted through compiled `Decompose` with a seeded COEFF source per range |
| Small centered extension (Step 1) | `CenteredExtendCompact` with compact input operands and native `gpu_small_matrix_copy_range` |
| Compact multiplication (Step 2) | `MultiplyCompact` with fixed EVAL LHS replica, compact RHS ownership, width-scaled expansion workspace; row blocks admitted per block |
| Estimator plan consumption (Step 6) | Exact wave classes from schedules, serializable admitted-plan summaries, `AdmittedInvocationPlan` scenario, and `admit_plans_from_log` fed from a prepared execution's admitted-plan log |
| Trapdoor/preimage (Step 3) | Admitted through traced native claim plans, ordered dispatch extensions and per-attempt claim boundaries, with deterministic consumer events scoped to traced steps; single-device (refused before sealing otherwise), widths {1, columns}; resident-target download and replayed trapdoor import remain outside admission |
| Imports/exports (Step 4) | `ImportMatrix`, `ImportStaging`, `ImportCompact` admitted with codec claims; exports are explicit prepared readback boundaries; `matrix_to_bytes` returns a result |
| Default graph inventory (Step 5) | `prepare_graph_admission` derives the complete inventory from the validated root scope, including traced scalar polynomial readback; enabled by `ExecutionConfig::prepared_gpu_admission`; unsupported kinds fail before sealing |
| Review (Step 7) | Independent review complete, all should-fix findings corrected and re-validated (GPU 1011/0, CPU 610/0, third pass); matched single-node performance evidence recorded with its caveat; physical multi-GPU checks remain deferred |

Later evidence directories: `imports-exports-admission/`, `graph-admission/`,
`preimage-admission/` (Steps 3–6), and `final-2026-09-12/` (Step 7: three
validation passes, repeated fixtures, matched measurement, hashes, manifest).
Earlier evidence: `test_data/fleet-sharding-implementation/resumed-2026-09-12-final/`
(complete CPU and GPU-enabled workspace library suites, warning-free builds,
source manifest, executable hashes, and the repeated runs of the four new
fixtures). New fixtures, each three consecutive runs plus one ring-dimension
512 run on the same binary:
`test_gpu_admitted_compact_hash_matches_column_runner_and_reuses_outputs`,
`test_gpu_admitted_compact_centered_extension_preserves_payload_and_shards`,
`test_gpu_admitted_compact_multiplication_matches_cpu_over_partial_waves`, and
the child-process-isolated primitive context-count tests. Physical multi-GPU
checks, integration tests and the independent review were not performed.


**Implementation is stopped at the user's explicit request. The project is not
complete, and the current GPU-enabled unit suite has two unresolved failures.**
This document records the requirements, implementation, evidence, and remaining
work so that a newcomer can continue later without the conversation history.
The future-work sections are a handoff, not authorization to resume.

At the stop boundary, no build, test, or subordinate agent was still running.
The handoff work did not change implementation files or rerun tests. There has
been no final independent review, physical multi-GPU acceptance, or complete
performance acceptance. Physical multi-GPU testing is explicitly deferred.

The latest evidence is:

| Validation | Passed | Failed | Ignored | Meaning |
|---|---:|---:|---:|---|
| CPU workspace library unit tests | 610 | 0 | 25 | All 11 executables reuse passing evidence after identical SHA-256 verification |
| GPU-enabled workspace library unit tests | 999 | 2 | 27 | Seven changed executables ran once; four identical executables reuse evidence |
| New admitted compact decomposition lifetime fixture | 3 runs | 0 | — | Same final executable; also one successful N=512 run |
| New admitted compact correction/tail fixture | 3 runs | 0 | — | Same final executable; no N=512 run for this fixture |
| CPU and GPU workspace library builds | Both successful | 0 | — | Both final build logs are warning-free |

GPU-enabled unit counts include CPU tests compiled with the feature enabled;
they are not counts of GPU kernels or application integration tests. Passing
the new fixtures does not resolve the two failures elsewhere in the suite.

This snapshot supersedes earlier checkpoint prose claiming 999 GPU-enabled
passes with no failures. That result belonged to an earlier executable set.
The earlier plan remains the intended design, subject to the user decisions
restated here and the current stop instruction.

## 2. Checkout and preservation requirements

| Item | Stop-time value |
|---|---|
| Branch | `codex/public-runtime-gpu-optimizations` |
| HEAD | `51bfdbd1ebbd96d5b968cccb8cbbf33a085681c5` |
| HEAD subject | `Integrate generic runtime, GPU transfer, and CRT improvements` |
| Locally recorded `origin/main` | `f3fe71efa301778f5ba178c985af919cfefa42c7` |
| Main ancestry | The locally recorded `origin/main` is an ancestor of HEAD |
| Working tree before this handoff | 60 modified tracked files and 19 untracked files |
| Tracked implementation/documentation diff | 14,267 insertions and 6,056 deletions; excludes untracked contents |
| Changed or untracked crate files | 73; all match the latest source manifest |

No remote fetch or pull was performed for this handoff. The ancestry statement
does not assert that the local remote-tracking ref is still the newest remote
commit. The earlier work integrated generic runtime/GPU optimizations onto a
main-derived branch while keeping unpublished application material excluded.
That publication boundary remains binding; no push is authorized by this
handoff. The workspace member list and architecture document contain no such
private crate, but a complete reachable-history publication audit has not been
performed at this stop boundary.

**HEAD alone does not contain the implementation described below.** Many core
new modules are untracked. Preserve the working tree, untracked files, and local
test evidence before any future checkout, clean, reset, rebase, or packaging.
Do not treat the stage-specific patch as a complete backup of the branch.
Adding this handoff changes the documentation inventory above, not the recorded
implementation snapshot. Evidence under `test_data/` must be copied explicitly
when transferring the work; do not assume it is included in a commit.

## 3. Objective and terminology

The intended result is a common execution path that distributes lattice matrix
operations by output columns over the configured homogeneous GPU fleet. It
must retain exact arithmetic, compact coefficient types, stable artifacts, and
correct asynchronous memory lifetimes. Runtime admission and benchmark
estimation must use the same concrete scheduling and resource contracts.

The repository is a virtual Rust workspace, with no root facade crate. Its 11
members are `ir-core`, `dsl`, `runtime`, `bench-estimator`, `bgg`, `primitives`,
`gadgets`, `func-enc`, `we`, `io`, and `fhe`. The authoritative dependency rules
are in [the architecture document](../architecture.md). Native CUDA and its
Rust wrappers belong to `primitives`; execution and ownership belong to
`runtime`; measurement and aggregation belong to `bench-estimator`. Runtime
must not depend on the estimator, and application crates must not depend on
one another.

| Term | Meaning in this work |
|---|---|
| N, L, K, C | Ring dimension, number of active CRT limbs, multiplication inner dimension, and output columns |
| Ordinary matrix | Full CRT representation, normally in evaluation/NTT format; coefficient conversion is explicit |
| Compact matrix | Shared bounded signed coefficients with an inclusive coefficient bound, without expanding every digit into a full CRT matrix |
| Shard | A device-owned global column interval containing all relevant rows and all active CRT limbs |
| Wave or local job | A compute range within ownership boundaries; its width can be smaller than a retained output shard |
| Retained output | The complete result allocation that survives individual waves; all output shards must fit |
| Prepared inventory | Finite native backing, typed device and pinned spans, streams, events, and prepared parameter/cache resources |
| Reservation and claim | Admission first reserves an exact resource layout; execution then claims the corresponding native resources |
| Pilot | An isolated calibration invocation using the production range runner, without consuming production sampling or source effects |
| Role 0 / role 1 | The first configured device / the representative timing role for later homogeneous devices; each device still has its own capacity check |

No change to mathematical operations, inclusive bounds, transcripts, noise
semantics, or canonical artifact bytes is intended. Implicit CPU fallback,
full-CRT expansion used to evade compact ownership, and implicit out-of-core
primitive execution are outside the design.

## 4. Accepted requirements and user overrides

### 4.1 Memory budget and initial preparation

`MXX_GPU_VRAM_PERCENT` is an integer from 1 through 100, defaulting to 80, read
once for the relevant configuration. For device i:

```text
B_i = floor(total_device_bytes_i * percent / 100)
P_i = total_device_bytes_i - free_device_bytes_i
E_i = P_i - default_pool_reserved_i + default_pool_used_i
P_i = E_i + (default_pool_reserved_i - default_pool_used_i)
```

Use checked arithmetic. P is physical residency; E removes unused retained
default-pool pages while including live pool allocations. Initial provisioning
may temporarily exceed B. Accept the completed setup only when a coherent
native observation establishes both P <= B and E <= B. Native activity,
execution ownership, and storage generations must agree with that observation;
an unverified setup must not yield an executable admission permit.

The default CUDA asynchronous pool remains in use. A private pool is not the
agreed replacement. The retired controls
`MXX_GPU_SMALL_MATRIX_VRAM_PERCENT`, `MXX_MUL_SMALL_RHS_TILE_COLUMNS`, and
`MXX_GPU_COLUMN_CHUNK_COLUMNS` must not return as active aliases. Do not add
K-axis or CRT-limb tiling, or a replacement budget knob, to bypass this contract.

### 4.2 Production memory policy — already approved

After accepted setup, the managed ledger is conceptually:

```text
R_i = E_i_at_accepted_setup + live_separately_managed_reservation_bytes_i
```

Prepared backing is already charged in the accepted baseline. Claiming a slot
must not charge that backing again. Nevertheless, having free budget bytes is
insufficient: native admission must also establish an exact multidimensional
fit for live inputs, all retained outputs, concurrent scratch, allocation
classes, alignments, event/stream counts, and their lifetimes within the sealed
inventory. Additional separately managed demand must fit the remaining managed
budget. Release such charges only on actual native retirement.

**During computation, actual physical VRAM usage may exceed B. Continue valid
admitted work while resources are available and CUDA operations succeed.**
The user expects the program normally to own the device and explicitly chose
continued processing over stopping submissions solely because of an observed
physical or adjusted-usage excess. This is settled policy, not an open question.

Later memory observations are diagnostic. They must not create extra capacity,
retire live charges, infer that a prepared slot is free, or trigger per-wave
completion waits merely to refresh the budget. Opaque CUDA driver growth is
not prospectively bounded by the managed ledger. Actual allocation or CUDA
errors remain errors, and must preserve ownership during cleanup; they do not
authorize CPU fallback. Retained unused pool pages do not themselves prohibit
reuse of correctly retired managed resources.

See the [accounting amendment](gpu-prepared-storage-accounting-amendment.md) and
[accepted driver-memory decision](gpu-driver-memory-contract-decision.md) for
the underlying decisions.

### 4.3 Ownership, placement, and scheduling

- Enumerate configured devices through `detected_gpu_device_ids`, preserve
  configured order, and reject unsupported heterogeneous fleets.
- Unary operations inherit input column ownership. Multi-input operations
  intersect compatible boundaries. Scalars preserve their matrix operand's
  ownership; ordinary and compact products preserve RHS column ownership.
- Prepare fixed LHS data once per participating device. Incompatible layouts
  require explicit planned redistribution, not hidden copies inside each wave.
- Fresh output placement uses deterministic capped water filling, with ties
  resolved by configured order. Capacity must include hypothetical fixed
  preparation and minimum scratch. Reserve complete outputs before searching
  for a working wave width.
- If the complete resident result cannot fit, fail before allocating it. A
  smaller scratch wave cannot solve insufficient retained-output capacity.
- Use exact native workspace queries. Allocation classes can change at width
  boundaries, including 1, 4, and 5; a bytes-per-column guess is not a fit proof.
- Intersect local ranges with input/output boundaries, valid allocation
  classes, and admitted widths. Actual wave count is the maximum number of
  local jobs, not a nominal fleet-wide division. Owners with 90 and 10 columns
  and respective widths 10 and 90 still require at least nine waves.
- Use persistent submission workers per device and a caller fast path for one
  GPU. Submission acknowledgement is not GPU completion. Do not add a barrier
  between waves, a new Rayon pool, or blocking worker coordination that would
  deadlock with one Rayon thread.
- On failure, cancel unsubmitted jobs and retain or quarantine in-flight
  resources until safe retirement. Producer/reader events govern reuse; pinned
  host writes must not race DMA reads.

CPU loops naturally admitting parallelism should use Rayon. Native ordered
reservation and claim operations remain on their designated dispatch path;
parallelizing them blindly would violate admission ordering.

### 4.4 Calibration, estimation, and artifacts

A profile identifies the operation variant, parameters, bases, format, compact
bound, layout/preparation, hardware/build/device configuration, metric, and
storage class. Irrelevant graph node IDs and sampled payload bytes are not
profile identity. A cache hit still revalidates current capacity.

Pilots use the production native range runner, with isolated randomness,
transcript, and source-loading effects. Fixed prepared baseline storage is
excluded from the logical pilot peak; actual claims are distinguished from
reservations. Contaminated pilots may retry at the explicit calibration
boundary, at most three times; production does not acquire new host waits.

For a role, a measured hint is:

```text
s_role = ceil(logical_pilot_peak_bytes / pilot_columns)
trial_width = floor(internal_capacity_after_complete_outputs / s_role)
```

The hint only proposes a candidate. Exact native fit reduces it to a valid
width and class. A supported prepared range with zero scratch can use its full
valid class range even when a scalar hint is zero. Compute capacity per device;
use a separate role-0 width and the minimum valid width among active later
devices, without forcing the first device into that same minimum.

The estimator must consume actual runtime ranges, owner assignments, resource
classes, and reservations. A shape-only nominal scenario is not a runtime
feasibility claim. For distinct wave classes k, multiplicities n, fleet wall
latencies L, and per-device event spans D:

```text
actual_wave_count       = sum_k n_k
cumulative_wave_seconds = sum_k n_k * L_k
work_seconds            = sum_k n_k * sum_i D_ki
```

Measure GPU spans with CUDA events covering participating streams/transfers;
measure fleet wave wall time monotonically through completion of all active
devices, including host dispatch. Do not divide aggregate work by device count
again. Sum-of-wave latency is not necessarily end-to-end invocation time when
work overlaps; report the latter separately. Preserve the meanings of ideal
DAG critical path and hypothetical workspace metrics. Maximum parallelism is
a maximum across stages, not a sum.

Assign transfer and dispatch costs once: planned peer movement belongs to the
primitive that requires it; artifact/RAM movement belongs to the owning dataflow
stage. Measurement keys, aggregation, and documentation must migrate together.
Production diagnostics should expose plan/profile IDs, capacities, owner
intervals, role measurements, accepted P/E, managed R, logical demands, and
observed excess without adding timing waits to normal production execution.

Canonical artifact bytes and hashes remain global row-major and independent
of device count or shard layout. Import must reserve destinations before
decoding and use bounded host staging. Compact semantic types remain compact.
Device-only success paths must not block the host on device completion;
explicit preparation, measurement, and export may wait at their own boundaries.

### 4.5 Validation and collaboration restrictions

The user's later instructions supersede older generic repetition requirements:
ordinary unit tests need one successful execution; GPU memory-lifetime or
synchronization-sensitive cases use consecutive stability checks, with three
runs used for the current fixtures. Do not restore the older 300-run gate by
default. Physical multi-GPU checks are deferred. Integration tests require
separate explicit authorization. Subagents are permitted only for the final
independent review, not further implementation. The latest stop instruction
currently prohibits all implementation and test continuation.

## 5. What is implemented locally

### 5.1 Source map and execution lifecycle

All paths in this section are relative to the repository root.

| Area | Principal files | Current responsibility |
|---|---|---|
| Native ownership and accounting | `crates/primitives/cuda/src/Runtime.cu`, `crates/primitives/cuda/include/Runtime.cuh` | Activity epochs, related execution owners, setup constants, streams/events, reclamation and retirement |
| Finite native inventory | `crates/primitives/cuda/src/gpu_admission.cu`, `crates/primitives/cuda/include/gpu_admission.cuh` | Typed device/pinned/resource slots, exact claims, rearming, child ownership, occupancy and closure |
| Rust prepared storage | `crates/primitives/src/matrix/gpu_admission.rs` | `GpuPreparedStorage`, requests, slot identities, reservations, workspace layouts |
| Retained range kernels | `crates/primitives/src/matrix/gpu_view.rs`, `gpu_transform.rs` in the same directory | Borrowed rectangles, retained destinations, compact operations and exact workspace queries |
| Transfers and RNS | `crates/primitives/src/matrix/gpu_staging.rs`, `gpu_rns.rs` | Bounded transfer/codec and transform plumbing |
| Runtime memory and scheduling | `crates/runtime/src/gpu_memory.rs`, `gpu_schedule.rs`, `gpu_enqueue.rs` | Managed ledger, fit/leases, column intervals, waves, persistent dispatch and failure acknowledgements |
| Invocation description | `crates/runtime/src/gpu_invocation.rs` | Concrete typed requests and staged target layouts without production loader consumption |
| Prepared inputs and outputs | `crates/runtime/src/backend/poly_gpu/gpu_prepare.rs` | Source fragments, fixed replicas, and typed ordinary/compact retained results |
| Admission and compilation | `crates/runtime/src/backend/poly_gpu/gpu_admit.rs`, `gpu_compiled.rs`, `gpu_preflight.rs` | Inventory selection, isolated pilot, complete reservations, structural validation and shared runners |
| Fleet/backend integration | `crates/runtime/src/backend/poly_gpu/fleet.rs`, `crates/runtime/src/executor.rs`, `crates/runtime/src/executor/gpu_plan.rs` | Fleet values, operation routing, explicit preflight and family/artifact boundaries; legacy paths remain |
| Calibration | `crates/runtime/src/gpu_calibration.rs` | Profile identity, registry, measurement and width selection |
| Estimation | `crates/bench-estimator/src/dataflow.rs`, `dataflow_gpu.rs`, `gpu.rs`, `harness.rs`, `lib.rs` | Partially migrated measurement/dataflow; full admitted-plan consumption remains unfinished |

The explicit prepared route is approximately:

```text
caller supplies complete inventory
  -> prepare_memory: prepare related resources, close native domains,
     verify initial observations, install ledger and prepared_required
  -> concrete invocation preflight
  -> select all fixed inputs, retained outputs, and scratch layouts
  -> prepare inputs and calibrate or validate a cached profile
  -> reserve the complete admitted batch
  -> compile the ordered invocation queue
  -> matching backend call consumes that queue
  -> return a typed retained owner; events govern later reuse/retirement
```

**This is not yet automatic graph-wide provisioning.** The caller currently
must provide a complete inventory to `prepare_memory`, and explicit prepared
coverage does not extend to every production operation. Older diagnostic and
manual routes still exist. Do not equate closure of native allocation domains
at this explicit boundary with complete admission for an arbitrary graph.

The compiled operation representation currently covers constants, ordinary
sampling and polynomial values, plain hash sampling, typed modulus conversion,
CRT recomposition, RNS conversion, centered rebase, decomposition, negate,
transpose, tensor, slice, row sum, addition, row/column/diagonal concatenation,
row-block addition, subtraction, scaling, automorphism, ordinary multiplication
including scalar placement, and accumulation. Each entry has restrictions
checked during structural argument validation; this list is not a claim that
every IR variant or operand combination is admitted. Compact hash sampling,
compact multiplication integration, small centered extension, and preimage
coverage remain substantial gaps.

### 5.2 Existing native and earlier-stage changes worth preserving

- Mixed narrow/wide compact RHS multiplication uses range-sized expanded
  workspace, processes nonempty narrow then wide portions, and preserves full
  physical pitches and reader dependencies. Workspace scales with L, K, local
  columns, N, and actual word width rather than total result width.
- Borrowed compact decomposition can directly read COEFF data without an
  approximation correction; other cases use a range-sized copy, batched INTT,
  and correction. CPU preparation uses Rayon where appropriate, while native
  allocation/claim ordering stays on the dispatch thread.
- Retained RNS conversion passes metadata by value when the source/target limb
  product is at most 64, otherwise uses typed device transform workspace.
  Legacy whole-matrix paths have a separate contract.
- CRT recomposition uses by-value metadata for one/two levels and fixed transform
  workspace above that. Setup prepares Garner tables, with 8*L^2 bytes per
  ring/device, instead of hidden per-wave host metadata construction.
- Ordinary modulus conversion writes retained rectangles with resident metadata
  and preserves its exact arithmetic contract.
- Setup/reclaimer accounting distinguishes device activity from foreground
  execution-owner activity; coherent epochs include worker-idle checks.
- Zero-row gadget validation exists on both CPU and GPU paths to avoid division
  by zero. An earlier artifact-lock fix explicitly unlocks `flock` even when a
  duplicated descriptor survives; retain its ownership regression coverage.

These are local implementation facts, not a final review of every affected
application file. Consult the complete dirty diff and the
[allocation audit](gpu-managed-allocation-audit.md) before packaging changes.

## 6. Latest edited stage: admitted compact decomposition

This stage implemented a typed admitted decomposition path and passed its new
targeted fixtures, but its complete GPU-enabled unit run failed elsewhere.

The production changes are:

1. Native `gpu_small_matrix_create` takes `initialize_zero`. When true it queues
   zeroing before the existing write-completion event, without an extra event
   or a host upload. Public Rust `GpuSmallMatrix::new_zero` uses this safe
   initialization; checked allocation sizing is shared through
   `allocation_bytes`. Internal empty allocation remains distinct. FFI callers,
   clones, and private sampler allocation calls were updated consistently.
2. `PreparedMatrixValue::{Matrix, Compact}` and `PreparedFleetOutput` live in
   `gpu_prepare.rs`. Ordinary and compact fleet outputs remain separate public
   types. Generic admitted execution checks the expected result kind before
   consuming its queued invocation.
3. Decomposition identity records input rows, digit count, mode, and bound.
   It does not compare live compact GPU matrices or download canonical bytes
   merely to establish invocation identity.
4. Admission selects exact `CompactPayload` retained output slots, initializes
   the complete output once, and prepares ordinary inputs in COEFF format.
   Approximation scratch uses actual source row counts and rearmed column
   ranges; correction workspace receives explicit fixed claims.
5. The native correction workspace query decides whether external metadata is
   needed using one-column auxiliary capacity. If one column cannot contain
   the metadata, wider ranges use the fixed external span too. A short tail
   therefore does not silently switch allocation class. Query and execution
   share this decision; capacity checks were not relaxed.
6. Fleet gadget decomposition and its row-block route consume the compiled
   prepared path and return typed compact owners.

Principal files are `MatrixSmallRhs.cu` and its header, `MatrixDecompose.cu`,
`crates/primitives/src/poly/dcrt/gpu.rs`, `matrix/gpu_dcrt_poly.rs`,
`matrix/gpu_admission.rs`, `sampler/trapdoor/gpu.rs`, and runtime
`gpu_compiled.rs`, `gpu_prepare.rs`, `gpu_admit.rs`, and `fleet.rs` at the paths
listed above.

The first new test,
`test_gpu_admitted_compact_decomposition_preserves_inputs_and_reuses_outputs`,
uses small defaults, five columns, differently sharded one/two-row blocks,
mixed COEFF/EVAL inputs, mixed 17/54-bit CRT limbs, small/regular decomposition,
and dropped-limb cases. It checks an existing CPU primitive oracle, initial
zero bytes, invalid-digit preflight rejection, wrong-invocation rejection,
retained-output reuse over multiple invocations, inclusive bounds, canonical
bytes, and unchanged source formats. It passed three times on the final binary
and once with ring dimension 512.

The second,
`test_gpu_admitted_compact_correction_workspace_reuses_tail`, uses N=16 by
default, three columns, 64 30-bit CRT limbs, and 32 dropped limbs. It exercises
8,448 bytes of correction metadata with width two and a one-column tail, then
reuses the workspace/output over two cycles against the existing CPU oracle.
Its test inventory has six ordinary width-two slots. It passed three times on
the final binary. The larger-N run above does not cover this 64-limb fixture.

Earlier failed stage attempts remain in evidence: missing FFI boolean arguments,
fixture type mismatches, incorrect fixture parameter placement causing a foreign
execution owner, and a too-small test inventory selecting width one instead of
the expected width two. These were repaired before the final targeted runs.
Their prefixed logs are historical; the unprefixed GPU suite failure is current.

## 7. Exact validation evidence and unresolved failures

The latest evidence directory is
[admitted-compact-decomposition](../../test_data/fleet-sharding-implementation/admitted-compact-decomposition/).
Its key records are `build-summary.json`, `cpu-summary.json`, `gpu-summary.json`,
`target-summary.json`, `source-manifest.json`, `stage.patch`, and `before.json`.
Raw build, suite, repeated-target, and N=512 logs accompany them.

Both final build commands succeeded without warnings:

```bash
cargo test -r --workspace --lib --no-run
cargo test -r --workspace --lib --features gpu --no-run
```

The recorded CPU build was cached (about 0.067 seconds); the final GPU build
took about 50.60 seconds. Formatting had run with `cargo +nightly fmt --all`
before the frozen build/test sequence. None of these commands was rerun for
this handoff.

| GPU-enabled crate | Passed | Failed | Ignored | Latest evidence |
|---|---:|---:|---:|---|
| runtime | 239 | 0 | 4 | Executed once |
| bench-estimator | 59 | 0 | 0 | Executed once |
| bgg | 69 | 0 | 0 | Executed once |
| dsl | 73 | 0 | 0 | Identical executable reused |
| fhe | 21 | 0 | 1 | Executed once |
| func-enc | 0 | 0 | 0 | Identical executable reused |
| gadgets | 110 | 0 | 20 | Executed once |
| io | 0 | 0 | 0 | Identical executable reused |
| ir-core | 116 | 0 | 0 | Identical executable reused |
| primitives | 297 | 2 | 0 | Executed once; exit 101 |
| we | 15 | 0 | 2 | Executed once |

All 73 source-manifest entries and all 22 recorded CPU/GPU executable hashes
were verified against the stop-time files. Reused evidence is explicitly marked
in the summary JSON; it is not a claim that all those tests ran again in this
stage. The recorded device was one RTX 4080 SUPER; hardware was not reprobed
for the handoff. No physical multi-GPU or integration test result is implied.

Important SHA-256 identities:

```text
Final GPU runtime:
600cedbd35d1f0baf6608fdc34acc32a21379b1e153870c895e2ef9f771a40f2
Final GPU primitives:
39a8ab6aa423159f67e03e1b5746dc1a74e92b08dfa446ad69b4822b86df74f5
Final CPU runtime:
275704a8c82aa563bf896f053306a16cf9e6d85873f3eaa8492d6d6baf098618
Saved previous GPU runtime used for the reference fixture:
0e613b3a1f37208e83fe8e2a5892dcc6aecd4b0d7966c5d0644c2a4be5d163d3
```

### 7.1 Current failure A: exclusive default-pool reset

Source: `crates/primitives/src/poly/dcrt/gpu.rs`,
`test_gpu_default_mempool_usage_and_high_water_reset` (around line 2810).
The failure is at the high-water reset, before the subsequent usage assertions:

```text
reset default mempool high-water:
"default mempool high-water reset requires exactly one live mxx context"
```

### 7.2 Current failure B: related-ring context count

Source: the same file,
`test_gpu_related_rings_share_execution_and_preserve_async_lifetimes`
(around line 2706). At approximately line 2720 it asserts that constructing
the source adds one context to the preceding global observation:

```text
assertion left == right failed
left: 1
right: 2
```

The assertion is `source_state.live_contexts == before.live_contexts + 1`.
Both failures are retained in
`gpu-mxx_primitives-203401fd49222996.log` in the latest evidence directory.

### 7.3 What is known and what is only a hypothesis

Both failing tests use `#[sequential]`, imported from the unnamed
`serial_test::serial` guard. Many prepared-storage/view tests use the different
named `#[serial_test::serial(gpu_context)]` guard. Their observations concern
process-global context counts. Those distinct guards do not mutually exclude
creation or destruction of contexts by the other group.

**Interference between test groups is a plausible explanation, not a confirmed
root cause.** The precise overlapping test and context-lifetime trace have not
been captured. Neither an isolated reproduction nor a post-failure stability
run has been performed. The handoff does not clear production lifetime code of
responsibility, and no counter assertion or reset precondition was weakened.

If implementation is authorized later, preserve the current failing evidence
first. Compare isolated runs with ordinary concurrent suite execution and
trace context ownership before choosing a repair. An existing precedent is the
child-process isolation in
`test_gpu_prepared_pinned_pending_upload_all_owners_drop` in
`crates/primitives/src/matrix/gpu_admission.rs`, with its marker declared in
`crates/primitives/src/env.rs`. That pattern retains assertions and checks that
exactly one child test actually passed. It may be appropriate if global test
state is confirmed as the cause; it is not yet the selected fix. Do not add a
global mutex, reduce production parallelism, set suite-wide single-threaded
execution, or weaken lifetime assertions just to obtain a green summary.

### 7.4 Limits of the performance and historical evidence

The unchanged reference fixture
`test_gpu_fleet_compact_row_blocks_preserve_partial_waves` took approximately
0.244112 seconds with the saved previous runtime and 0.233863 seconds with the
final runtime. These are whole-process fixture timings of an existing route,
not a matched benchmark of the new admitted production path. They establish
neither a kernel speedup nor complete performance acceptance.

The preceding `retained-compact-decomposition` evidence directory had 999 GPU
passes, zero failures, and 27 ignored, with runtime 237 and primitives 299
passing. The new stage adds two runtime tests but currently loses two primitive
passes. The old result must not replace the latest failing result. Earlier
stages and repaired attempts remain useful regression history, not acceptance
of the final source tree.

## 8. Remaining work, in dependency order

The original plan proceeds through native inventory, sealed accounting,
isolated calibration, unified operation dispatch, estimator integration,
artifact/diagnostic regression coverage, and performance/review acceptance.
Implementation has progressed across several of these areas, but none of the
following gaps can be closed merely by making the current two tests pass.
There is no defensible completion percentage or short completion estimate.

### Step 0 — Resolve the current validation failure after authorization

**Files:** the two primitive tests above, their actual context-owner/native
dependencies, and the prepared-storage test isolation precedent if relevant.
Preserve source hashes and failed logs, establish the cause, make a narrow
repair, and rerun the affected executables. Use consecutive runs for a changed
lifetime/synchronization path. Finish with warning-free CPU/GPU library builds
and complete non-ignored unit evidence for the resulting binaries, respecting
the user's one-run rule elsewhere. Completion means both failures are explained
and resolved without hiding them through weaker assertions or reduced execution
parallelism. This step alone does not complete the overall implementation.

### Step 1 — Complete compact hash and small centered extension

**Files:** runtime `gpu_compiled.rs`, `gpu_prepare.rs`, `gpu_admit.rs`,
`gpu_preflight.rs`, and `fleet.rs`; primitive sampler and retained-view APIs.
`CompiledMatrixInvocation::arguments` currently accepts plain hash sampling,
not `HashVariant::Decomposed` or `SmallDecomposed`. Existing fleet decomposed
hash routes remain outside complete compiled admission.

Reuse `decomposed_hash_column_runner` and
`GpuDCRTPolyHashSampler::sample_hash_gadget_source_columns`: they validate digit
count and row divisibility, generate the required COEFF source with global
column offsets, and invoke decomposition. Do not introduce an unnecessary hash
kernel or NTT conversion. Add retained compact destinations, source/correction
scratch claims, structural identity, and isolated pilot randomness without
consuming production tags or loaders. Small centered extension also needs
retained rectangles and complete admission across related contexts rather than
only its existing whole-copy primitive. Completion requires exact existing
oracle/transcript checks, preserved compact bounds, and reusable admitted tails.

### Step 2 — Integrate compact multiplication into compiled admission

**Files:** runtime typed arguments/preparation/admission and primitive
`gpu_view.rs`, `gpu_transform.rs`, and `MatrixSmallRhs.cu` interfaces.
Native borrowed-range kernels and small-RHS workspace queries already exist.
The compiler still needs genuine compact input ownership: preserve RHS shards,
prepare full EVAL LHS replicas once, reserve full ordinary outputs, and claim
the narrow/wide expanded workspace for each admitted width. Avoid representing
compact inputs as fake ordinary matrices or downloading them for identity.
Completion requires exact mixed-width results, source-format preservation,
partial tails, output reuse, and production/pilot use of the same native runner.

### Step 3 — Close trapdoor and preimage resource admission

**Files:** `crates/primitives/src/sampler/trapdoor/gpu.rs`, native trapdoor
kernels, runtime preflight/admission/compiler and target-loading integration.
Inventory fixed public/trapdoor owners, related parameters, finite caches,
scratch, streams/events, and bounded sampling attempts before sealing setup.
Retain the full-CRT relation, inclusive cutoff, and typed compact output.
Per-range target loading must preserve global offsets and deterministic replay
while calibration avoids consuming the production loader. Completion requires
resource/lifetime checks and existing trusted primitive or round-trip evidence;
do not substitute a new mathematical sampler or loosen a bound.

### Step 4 — Finish imports, exports, and remaining operation boundaries

**Files:** native serde/workspace paths, primitive staging, runtime artifacts,
executor GPU planning, and uncovered fleet operation cases.
Admit destinations before decoding, bound pinned buffers and metadata, preserve
canonical global row-major bytes and schema semantics, and connect artifact
liveness to retained owners. Audit scalar, atomic/fused, and mixed-context
reduction cases, including redistribution and global sampling indices.
Completion requires exact artifacts independent of layout and correct cleanup
when loading, submission, or export fails. Device-only success paths must stay
asynchronous; host reconstruction is not a replacement for missing GPU support.

### Step 5 — Make complete preparation the default graph path

**Files:** `crates/runtime/src/executor.rs`, `executor/gpu_plan.rs`,
`gpu_invocation.rs`, `gpu_memory.rs`, and backend preparation/admission modules.
Derive finite inventory from concrete IR operations, fixed preparation, caches,
transfers, worker/launch classes, liveness, and bounded concurrent demand.
Prepare every related context and required first-use resource before setup is
sealed. Enforce the accepted initial P/E checks and production-excess policy.
Unsupported classes must fail before partial execution or externally visible
results. Completion means ordinary graph execution obtains complete admission
without a caller hand-authoring an inventory. Remove legacy unchecked production
bypasses only when their full operation coverage is replaced.

### Step 6 — Finish estimator consumption of actual admitted plans

**Files:** `crates/bench-estimator/src/dataflow.rs`, `dataflow_gpu.rs`, `gpu.rs`,
`harness.rs`, `lib.rs`, runtime schedule/calibration interfaces, and
`docs/benchmark-estimator.md`.
Represent actual owner intervals, local job counts, wave classes and their
multiplicities, complete outputs, and native resource classes. Separate measured
device work, fleet wall latency, cumulative waves, and whole invocation time.
Revalidate cache capacity, migrate measurement keys, and account for transfers
and host dispatch once. Completion includes asymmetric schedule regressions
and a traceable match from runtime plan to estimator output; a nominal shape
estimate must be labeled as such.

### Step 7 — Finish acceptance evidence and independent review

Complete the allocation/range/metric audit across every supported GPU IR and
fused variant, including first use of each worker and native allocation class.
Run only the authorized validation scope, preserve old reference tests, and
collect matched baseline/candidate performance on the actual admitted path with
the same parameters, lifecycle, transfers, and timing boundary. Review the full
dirty change set, not merely the latest stage patch. A final independent reviewer
must check `GPU.md`, arithmetic, ownership, parallelism, admission, and estimator
semantics. No such final review has happened.

Physical two/three-GPU checks, peer transfer/lifetime coverage, role-1 pressure,
generation coherence across devices, and multi-device execution with
`RAYON_NUM_THREADS=1` remain explicitly deferred. Local single-device planning
tests are useful but do not establish these properties on physical hardware.
Do not launch remote work until the user reopens that scope.

## 9. Acceptance checklist for a future continuation

Treat these as requirements to verify, not claims that every item already has a
passing final test. Existing evidence should be reused only when its source and
executable identity remain applicable.

| Requirement | Required evidence or completion criterion |
|---|---|
| Configuration and identity | Device order/homogeneity, budget parsing, related-ring ownership, profile invalidation |
| Coherent initial setup | Initial P/E acceptance, foreign activity/generation rejection, complete native domains |
| Prepared inventory vs byte budget | Fully charged but logically free slots remain usable; logical exhaustion rejects despite physical headroom |
| Complete output capacity | All retained outputs reserved before scratch widths; failure without partial output allocation |
| Allocation classes | Exact native query/runner agreement, widths 1/4/5, partial tails and zero-scratch cases |
| Claim integrity | Wrong kind, stale generation, foreign owner, failed batch and rollback preserve resource state |
| Async lifetime | Delayed readers, thread handoff, pinned DMA reuse, owner drop and failure quarantine |
| Calibration | Same production runner, isolated sampling/source effects, actual claim peak, valid cache reuse |
| Arithmetic and compactness | Exact RNS/CRT/modulus operations, inclusive compact/preimage bounds, unchanged formats/transcripts |
| Artifacts | Canonical bytes/hashes across layouts, bounded import/export staging and correct liveness |
| Estimation | Actual wave multiplicities, correct GPU-work/wall-time separation, no transfer/dispatch double count |
| Whole current suite | Resolve the two recorded primitive failures; warning-free builds and all non-ignored unit evidence |
| Performance | Matched measurements of the actual admitted path; whole-fixture timing alone is insufficient |
| Physical multi-GPU | Deferred by user; retain as an explicit unverified limitation |
| Independent review | Not performed; final full-change review required after authorized implementation |

## 10. Safe resumption and evidence handling

Nothing in this section should be executed while the stop instruction remains
in force. Once the user explicitly resumes work:

1. Read this handoff, `BUILDER.md`, `GPU.md`, and the required design documents.
   Use `REVIEWER.md` for the eventual independent review. Confirm the actual
   worktree/branch and preserve dirty and untracked files before changing state.
2. Preserve the latest failure logs and source manifest. `stage.patch` and
   `before.json` describe only the admitted compact stage, not all changes since
   HEAD. Historical prefixed attempts must not overwrite the final-stage files.
3. Verify whether current source still matches the manifest before attributing
   its evidence to a new checkout. Build commands below are library-only and do
   not authorize integration tests. GPU execution must use an environment where
   GPU access is available, outside the sandbox as required by the repository.
4. Investigate the two existing failures before broadening implementation.
   After a repair, record exact commands, environment, source/executable hashes,
   counts, failures, hardware, and logs. Do not report a filtered command as
   passing unless it actually executed the intended test.
5. Continue the dependency-ordered work above; maintain a current acceptance
   table and leave the physical multi-GPU rows deferred until scope changes.

Useful read-only commands from the repository root:

```bash
git status --short
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor origin/main HEAD
git diff --stat
git diff --check
```

A source-only identity check, without compiling or running tests:

```bash
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
evidence = Path('test_data/fleet-sharding-implementation/admitted-compact-decomposition')
manifest = json.loads((evidence / 'source-manifest.json').read_text())
mismatches = [name for name, expected in manifest.items()
              if not Path(name).is_file()
              or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected]
print({'checked': len(manifest), 'mismatches': mismatches})
raise SystemExit(bool(mismatches))
PY
```

After an authorized implementation change, the established build commands are:

```bash
cargo +nightly fmt --all
cargo test -r --workspace --lib --no-run
cargo test -r --workspace --lib --features gpu --no-run
```

The current test summaries identify the exact executable paths; rebuilding can
change them. List the intended binary's tests before using `--exact`, because
Rust test names include module prefixes. For example, the current primitive
binary can list names using:

```bash
target/release/deps/mxx_primitives-203401fd49222996 --list
```

Select the listed full test name for each of the two failures, capture separate
logs, and compare with normal concurrent suite execution. For a changed
lifetime path, compile once and use the identical direct-binary command for
the required repeated runs. Reuse unchanged-binary evidence explicitly; do not
describe reused runs as freshly executed. Temporary edit/build helper scripts
and the saved previous runtime binary are not durable handoff dependencies and
must not be blindly rerun against the already edited tree.

## 11. Reading map

This document is the stop-time entry point. Supporting documents provide deeper
details, but their earlier success claims and authorization language must be
read in the context of this stop and the current failed suite.

- [Required fleet design](fleet-wide-gpu-column-sharding.md): placement,
  invocation coverage, native resources, scheduling, metrics, and original gates.
- [Prepared-storage accounting amendment](gpu-prepared-storage-accounting-amendment.md):
  accepted setup accounting and prepared capacity semantics.
- [Driver-memory decision](gpu-driver-memory-contract-decision.md): accepted
  production physical-budget excess and continued processing.
- [Managed allocation audit](gpu-managed-allocation-audit.md): native allocation
  domains, closure boundaries, and remaining inventory obligations.
- [Chronological implementation status](fleet-wide-gpu-implementation-status.md):
  earlier changes, failed attempts, regressions, and evidence history. Its older
  checkpoints are not a substitute for the current results in section 7.
- [Latest evidence directory](../../test_data/fleet-sharding-implementation/admitted-compact-decomposition/):
  current source identities, builds, two failed unit tests, and passing targets.
- [Architecture](../architecture.md), [builder rules](../../BUILDER.md), and
  [GPU principles](../../GPU.md): layering, source ownership, validation, and
  asynchronous/performance constraints for any authorized continuation.

The next implementation action requires a new user instruction. Until then,
preserve this incomplete but inspectable working tree and its failed evidence.
