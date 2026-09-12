# Managed CUDA allocation coverage

This audit supports the setup transition in
`gpu_prepared_storages_finish_setup`. It covers mxx-managed allocations and
resources under the [approved VRAM policy](gpu-driver-memory-contract-decision.md).
It does not bound opaque CUDA allocation demand or prove completion of the
fleet-wide invocation planner.

## Audited allocation surface

A source search for `cudaMalloc`, `cudaHostAlloc`, `cudaMallocHost`,
`cudaEventCreate`, `cudaStreamCreate`, and `cuMemAlloc` across
`crates/primitives/cuda/` identifies the following managed allocation families.
CUDA function/module materialization and implementation-private launch demand
are outside this managed-resource contract.

| Source and entry points | Allocation and post-setup enforcement |
| --- | --- |
| `crates/primitives/cuda/src/Runtime.cu`: `gpu_context_create`, constant/twiddle upload helpers | Device parameter tables (NTT constants and the immutable full-basis Garner inverse table), owner compute/release/timing streams, and initial completion/timing events are created during context setup. Related context creation is rejected after any admission domain closes. Construction activity begins before that check and survives catch-path cleanup. |
| `crates/primitives/cuda/src/Runtime.cu`: `gpu_pinned_alloc` | Nonzero pinned requests enter tracked activity before inspecting the prepared domain. After closure, the exact next pinned slot supplies storage; a missing or mismatched claim fails before the cache or `cudaHostAlloc` path. |
| `crates/primitives/cuda/src/gpu_admission.cu`: `GpuCudaResource::acquire` | Completion events and private submission streams consume typed resource slots after resource-domain closure. Standalone event/stream creation is unavailable in this mode. |
| `crates/primitives/cuda/src/gpu_admission.cu`: `GpuDeviceWorkspace::acquire` | Batch, transform, compact, sampler and transfer device spans consume exact typed slots after their corresponding domain closes. Zero-byte requests allocate no device memory. |
| `crates/primitives/cuda/src/gpu_admission.cu`: `gpu_prepared_storage_create` | Backing completion/reuse events, typed device/pinned workspaces and resource handles are created before admission. Activity begins before checking the setup-only guard. A closed owner cannot grow this inventory. |
| `crates/primitives/cuda/src/matrix/MatrixData.cu`: `gpu_matrix_create` | Data, auxiliary backing, writer events and allocation-ready events belong to the standalone branch. After closure, `gpu_prepared_matrix_claim` supplies the exact admitted matrix. Empty matrix descriptors allocate host metadata only, and require a matching dispatch context. |
| `crates/primitives/cuda/src/matrix/MatrixSmallRhs.cu`: `gpu_small_matrix_create`, hard-cutoff preparation | Nonempty compact payloads and cutoff workspaces use the typed workspace helpers; their prepared reuse events also serve as completion events. An empty compact payload consumes a separate typed completion-event claim instead of creating an event outside admission. The cutoff decision workspace is nonzero (`sizeof(int)`), so its prepared completion event exists. |
| `crates/primitives/cuda/src/matrix/MatrixTrapdoor.cu`: consumer-event helper and covariance-cache construction | Consumer events use the common typed resource helper after closure. Persistent covariance data, its private stream and ready event must be created before ordinary admission; late cache construction is rejected. |
| `crates/primitives/cuda/src/matrix/MatrixUtils.cu`: consumer-event helper and missing writer-event branches | Thread-local consumer events are bypassed in favor of typed resources after closure. Missing matrix writer events fail instead of allocating. The standalone lazy branches now enter allocation activity before the guard and event creation. |

The Garner table requests `8 * L * L` device bytes per registered ring and
physical device, where `L` is that ring's full CRT depth. It is uploaded by the
existing setup-only ring-constant allocator and freed on the same owner release
stream as the NTT tables. Actual physical residency, including allocator
rounding, is covered by the initial native receipt rather than equated with this
requested byte count. Retained ordinary modulus-conversion ranges pass bounded
launch metadata referencing that immutable table: they allocate no per-wave
pinned buffer, transfer workspace, or completion event. The ordinary whole-matrix
workspace query still accounts for its staged metadata path.

Retained RNS ModUp/ModDown ranges use bounded launch arguments when the
source/target limb product is at most 64. Larger products claim one exact
`TransformWorkspace` span, even when the retained matrix's auxiliary arena could
hold it. This keeps calibration and production requirements independent of the
physical output width. A GPU setup kernel writes that metadata before conversion;
no pinned metadata buffer is involved. The same event-ordered workspace release
allows consecutive waves to reuse the span. Source readers reuse the output's
completion event, and range NTT preserves the rest of the retained output.
The whole-matrix RNS query still permits auxiliary-arena reuse.

Retained plaintext CRT recomposition also references setup-owned Garner tables.
One or two input levels fit bounded launch arguments and need no metadata span.
Larger input lists claim exactly `level_count * sizeof(CrtLevelMetadata) +
sizeof(CrtOutputMetadata)` bytes in a `TransformWorkspace`. GPU setup launches
copy at most two public records at a time, followed by the unchanged exact
reconstruction arithmetic. No pinned host metadata remains live between waves.
Whole-matrix calls retain their staged metadata contract, with smaller records
because the inverse table is already resident. The output completion event joins
all input readers; only the written output row range undergoes NTT.

Compact RHS multiplication accepts logical input/output rectangles while keeping
the original compact payload owner and its column view. The native workspace
query and production allocation use one shared calculation for the narrow and
wide expanded spans. Each nonzero span is a `CompactWorkspace` claim, in narrow-
then-wide order, covering all inner rows and the current RHS columns. Retained
output owners need no additional ordinary output allocation per range. Source
readers and compact release retain the existing output-completion dependencies.
The ordinary row-block multiplication entry point uses the same range primitive.
Runtime compact multiplication admission and automatic inventory provisioning remain
separate unfinished obligations.

Borrowed compact decomposition reads coefficient rectangles directly when no
approximate correction is needed. Otherwise it consumes range-sized ordinary
matrix copies in source order, followed by any nonzero correction metadata
workspaces in source order. Batched inverse transforms allocate no additional
payload. The correction query checks whether a one-column copy's auxiliary
arena suffices; when it does not, wider ranges keep the same fixed separate
metadata span so the allocation class also covers short tails. A fresh compact result then consumes one `CompactPayload` claim;
a retained result writes only the selected rectangle and allocates no new payload.
Source consumers reuse the compact output completion, and prior compact readers
remain ordered before subsequent writes. The owned-row-block entry point keeps
its in-place normalization. Runtime decomposition admission now selects complete
`CompactPayload` output slots, prepares coefficient inputs once, reserves the
row-shaped copies and fixed metadata spans, and binds the same range runner to
pilots and production. Compact zero initialization uses the existing stream and
payload completion, without extra resources. No ordinary output proxy is used.
The 64-limb runtime fixture covers a real additional correction span and a short
tail; automatic inventory provisioning and other compact invocation classes
remain unfinished.

All four managed domains close together: ordinary/device workspace, pinned,
transfer, and CUDA resource slots. A partial arithmetic-only reservation does
not enable coherent epoch certification. `allocation_tracking_complete` becomes
true only through the explicit complete-inventory transition, not by observing
stable CUDA counters or by a successful pilot.

## Transition and lifetime checks

The transition validates a nonempty, distinct, complete native storage inventory
on one device/execution owner, including related parameter contexts. It rejects
live reservations, leased/quarantined slots, foreground host allocation calls,
measurement intervals and uncertain releases. Pending setup uploads and tracked
asynchronous host reclamation may continue across closure. Retirement uses the
existing device activity counters and reclaimer-idle checks; it does not enter
the foreground owner-allocation counter. It closes all managed domains
before enabling epoch observation. It does not itself issue a residency receipt
or wait for device completion.

The subsequent initial allocation epoch fences the owner's existing streams and
pinned reclamation when needed. Concurrent allocation activity or changed
owner/device/context revisions prevents verification. Runtime admission validates
those native receipts again before publishing the ledger and applies both initial
physical and adjusted residency checks. No permit is published on setup failure.
The accepted inventory is retained by the ledger for its lifetime.

Private compact serialization streams join their completion back to matrix
producer streams before their resource handles return. Covariance-cache setup
joins all input readers after its temporary free; cache destruction joins frees
to the owner release stream. Prepared matrix and workspace release paths retain
producer/reader dependencies on the owner release streams. Deferred pinned
reclamation remains part of the epoch readiness check. An uncertain release
quarantines its execution instead of manufacturing available capacity.

Subsequent observations are nonblocking diagnostics. They cannot absorb or
retire managed reservations, and exceeding the configured physical budget does
not stop otherwise valid admitted work.

## Validation scope

The managed-setup regression rejects missing/duplicate inventories and active
leases, checks the empty-compact event claim, and confirms that unreserved
ordinary allocation remains unavailable after closure. The runtime regression
uses actual initial residency receipts to install a ledger and then exercises
ordinary negate/add/subtract calls through the same prepared preflight fixture
as the existing synthetic-pressure test. Existing mathematical assertions and
synthetic budget-excess coverage remain intact.

Build and execution evidence is recorded under
`test_data/fleet-sharding-implementation/managed-setup/`. Passing these checks
establishes this setup/admission path, not default graph-wide provisioning,
remaining operation families, estimator integration, or physical multi-GPU
acceptance. Those remain separate completion gates.

## Claim tracing and dispatch extensions

Closed domains satisfy every allocation from the next reserved claim in order.
Two mechanisms make host-driven samplers admissible without enumerating their
kernels by hand:

- `gpu_claim_trace_begin`/`gpu_claim_trace_end` (Rust `trace_native_claims`)
  record, on an open domain and on the calling thread, the ordered claims an
  operation would make: matrix owners (shape, level, format), device and pinned
  workspaces (kind, bytes, alignment) and CUDA resources. Paths that acquire a
  resource only under sealed admission (thread-local consumer and owner-link
  events) record the same claim on open domains. Claim counts must not depend
  on stream coincidences; the consumer tracker therefore claims its event on the
  same-stream fast path as well.
- `gpu_matrix_dispatch_extend`/`gpu_matrix_dispatch_retract` append ordered
  reservations to the thread's active permit once its earlier claims are
  consumed, and remove them after the step, requiring every extension claim to
  have been consumed. `PreparedClaimBroker` reserves smallest-fitting slots from
  the accepted inventory for a recorded claim list, enters or extends, runs the
  step and finishes.

Traced plans are recorded before sealing at the widths admission will use and
are keyed by the operation's shape/parameter identity, never by payload values.
