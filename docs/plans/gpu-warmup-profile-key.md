# Minimal GPU warmup timing keys using existing implementations

Status: implementation plan; production code is not changed by this document.

## 1. Purpose and fixed decisions

Warmup predicts a validated graph's production time and checks whether its schedule
fits available resources. Equivalent operations should share measured time across
graph sites and identical GPUs. Data ownership and resource checks remain exact.

Use existing files, types, dispatch, measurement harness, and scheduling code.
Do not add `gpu_profile.rs`, a parallel profile hierarchy, a kernel-ID registry,
or new public types merely to rename existing concepts. Extend existing private
serialization records and extract small helpers only when necessary.

The agreed policy is:

* Measure existing representative candidate widths and required exact tails or
  mapped fragments, not every integer width or every primitive.
* Reuse a point when its normalized key and coordinate match.
* Add sequential job times; compose parallel jobs through existing waves.
* Do not add interpolation or extrapolation. Selected jobs need exact measured
  points or existing exact no-work accounting.
* Assume all GPUs are identical and have equal compute performance. GPU model,
  UUID, and ordinal do not distinguish timing. Free VRAM, resident data, and
  transfer routes still differ per GPU.
* Measure one preimage attempt including acceptance checking. Keep production
  retries, rejection behavior, and retry-dependent memory checks unchanged.
* Use warm-cache timing only when production guarantees readiness. Count cache
  construction if it occurs inside the predicted run.

For 150 columns dispatched sequentially as 64 + 64 + 22, use
`2 * time(64) + time(22)`. No measurement at 150 or proportional-time assumption
is needed. Searching alternative schedules can require additional points.

Keep the cache within one `prepare()`, shared across candidate placements.
Cross-preparation persistence, disk caching, a new scheduler, and cryptographic
semantic changes are outside this implementation scope.

## 2. Definitions

| Term | Meaning |
| --- | --- |
| Graph site | A node at a particular scope/location; different sites can perform identical work. |
| Lowered operation | Actual backend work after fusion, alias elimination, and range mapping. |
| Invocation/job | One production dispatch and all outputs it generates. |
| Width | Local output columns processed by that invocation, not total graph columns. |
| Timing key | Conditions that must match to reuse measured time. |
| Coordinate | Width for ordinary jobs, payload bytes for transfer-only jobs, using existing `GpuWarmupProfileRequest::coordinate()`. |
| Point | Measured time, repetition count, and spread for one key/coordinate. |
| Route | Actual data movement/materialization: resident data, P2P, host staging, etc. |
| Binding | Exact site, GPU, ranges, ports, data/cache owners, and resource conditions. |
| Wave | Jobs allowed to run concurrently before the next schedule dependency/synchronization boundary. |

Time sharing does not imply shared data or equal available memory.

## 3. Existing implementation to reuse

Paths are relative to the repository root.

| File / existing symbols | Change |
| --- | --- |
| `crates/backends/src/backend.rs`: `GpuWarmupProfileKey` | Reduce the existing key. |
| Same file: `GpuWarmupProfilePoint`, `GpuWarmupProfileTable`, `GpuWarmupSessionProfileCache` | Keep storage and expose time-only reuse. |
| Same file: `GpuWarmupProfileRequest`, `GpuWarmupProfile`, `GpuWarmupProfileProvider` | Keep exact requests/results; adapt existing provider methods. |
| `crates/backends/src/gpu_measurement.rs`: `profile_key`, `profile_context_words`, `profile_shape` | Consolidate key construction using existing serialization/hash helpers. |
| `crates/backends/src/gpu_warmup.rs`: `measure_profile_for_session`, `record_canonical_point` | Share time while rebuilding current-job resource evidence. |
| Same file: width collection, stage costs, wave/union functions | Preserve candidate generation and schedule composition. |
| `crates/backends/src/gpu_column_policy.rs` | Reuse typed operation variants, mapping, routes, and fused union jobs. |
| `crates/backends/src/backend/poly_gpu/fleet.rs` | Reuse dispatch, cache preparation, and resource queries. |
| `crates/backends/src/sampler/trapdoor/gpu.rs` | Share existing attempt body between measurement and production retries. |

Today planner and provider construct keys independently. Keys contain graph,
device, range, port, retry/cache, and liveness information. Also, `exact_profile()`
reconstructs a complete old profile including device-specific memory and cache
identity. Removing device identity without changing that lookup would return
another GPU's evidence. Fix both issues in the same implementation batch.

## 4. Target key and canonicalization

Modify the existing definition in `backend.rs`; all field types already exist:

```rust
pub struct GpuWarmupProfileKey {
    pub implementation_variant: GpuWarmupEffectiveVariant,
    pub operation_identity: [u8; 32],
    pub route_descriptor: GpuExecutionRouteDescriptor,
}
```

Retain existing derives.

* `implementation_variant`: actual ordinary/fused/host/transfer implementation.
  Prefer existing typed variants. A necessary `Custom` value describes actual
  implementation behavior, never scope, liveness, or retention.
* `operation_identity`: canonical timing fingerprint of actual operation, fixed
  dimensions, native parameters, storage formats, and materialization geometry.
  It is no longer copied directly from `request.signature.operation`; that graph
  identity stays unchanged in requests and production validation.
* `route_descriptor`: normalized existing route descriptor. The exact physical
  descriptor stays separately in the request/result.

Do not add GPU model/number or a hardware-class type. Reuse existing backend/build
and context invalidation. Context reset invalidates native bindings and can clear
the session timing cache; it does not require a new timing-key field.

Extend existing `ProfileContext`, `ProfileType`, `ProfileMatrix`, and
`encoding::hash_canonical`. Rename `profile_context_words` to
`profile_context_identity`, returning the original `[u8; 32]` digest. Fold
`profile_shape` into this construction. The provider's `profile_key` becomes the
single authority; remove planner-side `canonical_profile_key_with_context`.

Normalization rules:

1. Use actual canonical domain, existing execution/fused variant, selected native
   algorithm, and lowered operands. Do not hash arbitrary display labels, graph
   scopes, wire IDs, whole graphs, complete `ParamEnv`, or input contents.
2. Resolve relevant expressions to native values. Include degree, ordered CRT
   basis/level, limb width, representation, coefficient/evaluation format,
   decomposition settings, bounds, and work-affecting scalar parameters. Reuse
   concrete types and `BackendStorageDescriptor` rather than new layout types.
3. Normalize only dimensions proven to follow local output width. In
   `(m x k) * (k x w)`, retain `m` and `k`; normalize `w`. Extend existing private
   serialization records with fixed columns and mapper-derived width dependence.
   Blanket omission of all matrix columns must not erase the inner dimension.
4. Preserve physical strides, alignment, row groups, output geometry, and relative
   fragment offsets. Rebase absolute offsets against actual buffer origins before
   removing them. A different specialized tail kernel remains distinct.
5. Reuse `normalized_route_descriptor_for_key`; move its pure definition to
   `backend.rs` if shared access requires it. Relabel devices by first-use roles,
   preserving shared-owner relationships and copy order. Do not collapse multiple
   sources into their total byte count.
6. Omit extents/staging bytes only when coordinate and retained geometry determine
   them. Serialize derivation coefficients in the existing context; retain exact
   irregular geometry. Never erase a fixed full-input transfer as width-dependent.
7. If transfer paths differ in bandwidth/topology, include the backend-reported
   transfer class in the context digest. This is not GPU-model identity. Reuse
   existing capability facts; if equivalence cannot be established, preserve the
   relevant physical endpoints for that transfer rather than share unsafely.
8. Remove port numbers from time equality only. Port-dependent dispatch, output
   geometry, or route differences must remain represented.
9. Distinguish actual measurement boundaries and host child coverage in the
   context when needed. Never merge container-inclusive and child/local timings.
10. Serialize scalar parameters deterministically: reject non-finite real values,
    canonicalize zero, preserve exact integers and stable ordering. Preserve any
    data-dependent specialized branch that changes actual work.

Audit all existing canonical/fused domains for complete mapping. This verifies
coverage; it does not measure unused domains. Unknown mappings fail preparation.

## 5. Existing profile records and time-only lookup

Keep `GpuWarmupProfilePoint` and `GpuWarmupProfile`. Do not add separate timing,
resource-evidence, or combined-cost types. A point may retain its representative's
exact observations for diagnostics, but only timing scalars may be shared.

Replace `exact_profile` / `exact_profile_for_request` at the sharing boundary:

```rust
impl GpuWarmupSessionProfileCache {
    // Tuple order: mean_seconds, spread_seconds, repetitions.
    pub fn exact_timing(
        &self,
        key: &GpuWarmupProfileKey,
        coordinate: usize,
    ) -> Result<Option<(f64, f64, usize)>, GpuWarmupProfileError>;
}
```

Reuse finite/nonnegative time/spread and positive-repetition validation. Preserve
existing exact no-work accounting. Keep `insert_profile` / `insert_point`, adapting
key-consistency checks. Validate exact request/observation consistency on insertion;
physical range/device equality is no longer a cross-job time equality condition.

Never populate another job with the representative's memory, workspace, cache ID,
residency delta, or physical route. Rebuild those fields from the current request
and existing allocation/resource helpers. Resource interpolation must not authorize
cross-job memory admission.

Change preimage metadata/evidence maps indexed by `(GpuWarmupProfileKey, width)`
to use the resolved `GpuWarmupProfileRequest` (already `Eq + Hash`) or the existing
exact native evidence context. Keep device/context, native cache, range, and
production retry settings in that evidence key. No new resource-key type is needed.

## 6. Provider interfaces and processing flow

Extend the existing trait. Retain `configure_gpu_plan_budgets` and
`register_operation`. Replace `measure` with `profile_for_request`, since a hit
returns a profile without timing execution. Do not leave parallel APIs.

```rust
pub trait GpuWarmupProfileProvider {
    // Existing budget configuration and descriptor registration remain.
    fn resolve_request(
        &mut self,
        request: &GpuWarmupProfileRequest,
    ) -> Result<(GpuWarmupProfileKey, GpuWarmupProfileRequest), GpuWarmupProfileError>;

    // cached_timing: (mean_seconds, spread_seconds, repetitions).
    fn profile_for_request(
        &mut self,
        request: &GpuWarmupProfileRequest,
        cached_timing: Option<(f64, f64, usize)>,
    ) -> Result<GpuWarmupProfile, GpuWarmupProfileError>;
}

fn measure_profile_for_session(
    provider: &mut dyn GpuWarmupProfileProvider,
    session: &mut GpuWarmupSessionProfileCache,
    request: &GpuWarmupProfileRequest,
) -> Result<GpuWarmupProfile, GpuWarmupProfileError>;
```

`resolve_request` uses the registered descriptor and actual dispatch to resolve
physical routing and required cache/binding readiness, then returns the canonical
key and exact request. It may prepare resources but does not time execution.
An unresolved provisional host-staging route cannot satisfy a timing lookup.

`profile_for_request` obtains current-job resource/residency evidence through
existing helpers. On a hit, use only the supplied timing scalars. On a miss, use
the existing harness and production entry point. Return the existing complete
per-job profile in either case.

`measure_profile_for_session` must:

1. Resolve request and key.
2. Call `exact_timing(key, resolved_request.coordinate())`.
3. Call `profile_for_request` with the optional timing.
4. Validate the result against the current request: route, cache identity,
   resource evidence, and production retry policy.
5. On a miss, insert the validated representative point; on a hit, do not insert
   a second point at that key/coordinate.
6. Return the current-job profile to existing stage-cost construction.

If actual dispatch changes resolved work, reject the stale-key sample and resolve
again with a bounded retry; never relabel it. Update trait forwarding for `&mut P`,
fake providers, and callers together. Count actual measurements separately from
resource-only profile calls.

Create one session cache at the preparation entry point and pass it through
`warmup_input_for_owner_placement` and candidate collection. Keep serial ownership;
no new cache service or locks. Select an admissible representative in deterministic
device order. Failure on one GPU is not fleet-wide infeasibility.

## 7. GPU-specific VRAM and data management

Use existing admission/liveness code for each physical GPU. Count the union of
resident inputs/caches, other live allocations, retained outputs, and concurrently
required scratch/staging. Compare the peak with that GPU's budget and available
capacity. Preserve host/pinned-memory limits; do not double-count shared storage.
A timing hit never proves that the job fits on the target GPU.

GPU 0 and GPU 1 may share `time(64)` while GPU 0 is nearly full. Reject that GPU 0
placement or choose a smaller admissible width; keep GPU 1's candidate. If data
resides only on GPU 0, GPU 1 may require a transfer and thus a different timing route.

Keep exact cache IDs, output bindings, ranges, random draw coordinates, and context
validity in production requests/plans. Fixed execution still rejects stale bindings
or invalid resource plans.

## 8. Production-only timing and preimage attempts

An ordinary sample covers one production invocation, including required
materialization, conversion, transfers, computation, and completion. Shared
transfers are counted once separately and excluded from consumer timings. Reuse
host/control child-coverage accounting and fused union mapping: do not double-count
parent/child work or charge one fused call once per output port.

Keep existing timing-scope/cache-state enums where requests and validation use
them, removing only independent key fields. Generate only scopes corresponding
to production work. Do not add device-wide synchronization or block normal async
wrappers. Do not perform a broad enum cleanup as part of this change.

For warm caches, prepare/validate the actual owner before the predicted run if
available. If an earlier graph operation creates it during the run, count cache
construction in that producer and preserve a readiness dependency to the consumer.
If neither is guaranteed, reject the unsupported warm-only request. A synthetic
warmup cache is not proof of production readiness. Keep cache bytes in VRAM checks.

Reuse the preimage body inside `bounded_retry`; extract only this private helper
in the existing primitive file for production and measurement to call:

```rust
fn attempt_preimage_tile(
    sampler: &GpuDCRTPolyTrapdoorSampler,
    params: &GpuDCRTPolyParams,
    trapdoor: &GpuDCRTTrapdoor,
    public_matrix: &GpuDCRTPolyMatrix,
    tile_target: &GpuDCRTPolyMatrix,
    destination: &mut GpuSmallMatrix,
    destination_column_start: usize,
    output_rows: usize,
    column_count: usize,
    attempt_seed: GpuRngSeed,
) -> Result<bool, SmallMatrixError>;
```

All types and `preimage_seed` already exist. Preserve candidate generation,
coefficient conversion, `try_pack_preimage_hard_cutoff_tile`, cleanup, and acceptance.
Initialize the destination with the authoritative cutoff before calling the helper.

Time normal invocation preparation/completion plus one attempt. A rejection is a
measured attempt, never valid graph output. Do not indiscriminately suppress
`AttemptExhausted`. Record outcomes and packing/cleanup differences in existing
diagnostics. Production keeps its configured retries; resource evidence uses that
production bound even when timing measures one attempt.

State in the existing report: “one attempt per preimage job; retries are not
included.” No new assumption enum is needed. Do not invent expected retries or
claim a worst-case elapsed-time bound. Harness repetitions remain controlled by
`GpuWarmupMeasurementConfig`; they are separate from sampling attempts.

## 9. Width collection, composition, and reporting

Keep `planned_width_anchors`, local candidate generation, and fragment enumeration.
Traverse only the validated graph's candidate dispatches. Deduplicate points by
`(key, request.coordinate())`. Measure a newly required tail/route before accepting
its schedule. Preserve lazy loop/wave classes instead of expanding every instance
or column.

Reuse `job_profile_keys`, `profiles_by_job`, `time_by_job`, stage costs, and wave/union
functions. Exact job-map keys retain site/range/port identity; timing references
can share keys. Do not add `GpuJobTimingRef`. Keep the existing barrier-wave rule:

```text
device time in a wave = sum of sequential job times on that device
wave time = max(device times) + separately owned wave overhead
total time = sum(wave time * wave multiplicity)
```

Count dependent transfers and host work at their actual positions. Preserve overlap
semantics. Missing exact points cause preparation errors, not fallback to another
route/port or unvalidated affine timing. Fixed execution does not measure again.

Extend existing preparation evidence/reporting with unique keys/points, actual
collection calls, hits, and resource-query counts. Distinguish harness repetitions
from point collections. Do not create a reporting framework or claim unmeasured
speedups.

## 10. Field migration

| Old key field | Treatment |
| --- | --- |
| `implementation_variant` | Keep existing enum; remove scope/live/retained label fragments. |
| `operation_identity` | Canonical timing identity; original graph identity stays in request. |
| `effective_domain`, `noninterpolated_shape`, `native_parameters` | Include necessary actual facts in context digest; remove graph scope/instance identity. |
| `route_descriptor` | Normalize existing type; keep exact request/result descriptor. |
| `route` | Derive from descriptor; remove duplicate key discriminator. |
| `device` | Production/resource context only, no model/number in timing equality. |
| `executed_range_start`, `executed_range_class`, `fragment` | Exact request metadata; actual work/layout differences remain in digest/route. |
| `binding_port` | Production binding; actual output geometry remains in timing identity. |
| `retry_cap` | Production behavior/resource evidence, not one-attempt time equality. |
| `cache_identity`, `cache_state` | Readiness and ownership checks; time follows real warm-consumer contract. |
| `timing_scope` | Keep request coverage/validation; remove independent key field. |

Remove superseded constructors and whole-profile cache-hit reconstruction, without
compatibility wrappers. If serialized frozen artifacts embed changed records,
update their existing compatibility check and reject stale records. Do not add a
new versioning framework or reinterpret old keys as new ones.

## 11. Implementation sequence

1. Audit actual dispatch/native parameters, host coverage, cache lifetime, and
   union-job behavior using existing inventories and diagnostics.
2. Change key/hash normalization, provider APIs, time-only lookup, exact resource
   map keys, point validation, and callers in one connected implementation batch.
3. Move cache ownership to preparation; connect candidate collection, per-GPU
   admission, warm readiness, one-attempt timing, and existing wave composition.
4. Update tests/docs, remove unused paths, format, and validate. Preserve old
   tests' physical-binding guarantees at the correct new boundary.

No new source module or public type is required. Existing private serialization
records and small helpers may change as needed. Preserve unrelated work; use
intermediate checks only for real API/lifetime uncertainty, not every field rename.

## 12. Acceptance and validation

Use existing fake providers and test locations.

| Case | Required result |
| --- | --- |
| Equivalent sites or identical GPUs | One collection per key/coordinate; separate exact bindings/resources. |
| GPU 0 full, GPU 1 available | Shared time if route matches; independent placement feasibility. |
| Data resident on only one GPU | Correct local/transfer route and memory accounting. |
| Width 32 vs 64 | Shared key for equivalent width-dependent work; separate points; fixed inner dimension retained. |
| Degree/CRT/format/stride/kernel/native parameter changes | Distinct keys whenever actual work differs. |
| Offset/device renumbering only | Same key only for equivalent rebased work/routes. |
| Multi-source/irregular/different-link route | Required geometry, topology, and byte differences preserved. |
| Unresolved route | Resolve before lookup; no provisional hit. |
| Fused outputs | One charge per actual invocation; exact validation for all ports. |
| 150 columns split 64 + 64 + 22 | `2*t(64)+t(22)` sequentially; no t(150), t(0), or interpolation required. |
| Parallel waves/shared transfers | Existing max/multiplicity/dependency rules, no double charge. |
| Missing point | Explicit preparation error, no approximate fallback. |
| Changed cache/context/retry limit | No reuse of another job's native resource/readiness evidence. |
| Warm cache unavailable | Real preparation/dependency or explicit error, no omitted production build. |
| Preimage rejection | Attempt timed, invalid output never escapes, production retries unchanged. |
| Unused primitive or repeated loops | No unused measurements; collection count follows unique points, not repetitions. |
| Fixed execution after prepare | No new measurements; exact bindings valid and outputs correct. |

For implementation, run targeted unit tests, `cargo +nightly fmt --all`, and the
warning-free workspace compile gates `cargo test -r --workspace --lib --no-run`
and its `--features gpu` variant. Follow `GPU.md` for execution permissions and
repetitions appropriate to native/synchronization changes. Integration tests need
an explicit implementation-task request. Compilation is not GPU validation.

Compare before/after counts and runtime on the same authorized graph, parameters,
and hardware; include resource-query costs and unverified assumptions. For this
plan-only task, check paths, symbols, interfaces, and instruction consistency;
no Rust build or GPU run is needed.

Completion requires the existing production path to share compatible time while
preserving exact data/resource correctness and composing representative widths
plus exact tails. Merely reducing key fields is insufficient.
