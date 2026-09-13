# Benchmark timing semantics

`mxx-bench-estimator` distinguishes device work, cumulative measured wave time,
and an ideal dependency schedule. All times are seconds. The GPU adapter's
`work_seconds` is aggregate device-seconds from CUDA events, never host elapsed
seconds or elapsed time divided by the device count. The generic host harness
continues to report host elapsed work seconds.

The nominal GPU measurement API receives concrete shapes and synthesizes its
inputs. `CostReport::measurement_scenario` therefore reports
`SyntheticFreshPlacement`. This scenario assumes prepared synthetic resident
operands and fresh column placement under the observed capacities. It does not
supply the runtime's retained ownership intervals, complete output reservation,
or admitted invocation plan, and cannot establish runtime schedule agreement or
physical memory feasibility. Sharing the calibration registry does not remove
this limitation.

Normal GPU reports now collect a second, CPU-only traversal after explicit
resource warmup has established bounded sibling widths and column caps. This
traversal deduplicates synthetic batch classes by operation parameters, concrete
shapes, canonical input sharing, sibling count, source descriptors, and the fleet column cap.
Explicit measurement runs each missing class through the existing GPU batch
entry points. Final estimation only reads the frozen table: missing classes are
warmup errors, including a changed configuration requiring an unmeasured tail.
It never measures a missing batch during report generation.

A joint observation replaces primitive work, cumulative time, wave count,
preimage work, benchmark-role totals, and measured wave workspace in the normal
report. It is charged once for the whole sibling group. Transfers and executor
management are added separately. Zero-cost metadata nodes contribute no GPU
waves. The ideal dependency schedule remains the nominal per-node model.

For selected matrix input fragments, the normal CPU traversal calls the same metadata-only `plan_matrix_layouts` selector
as native admission. It assigns output slots, preserves their exact context,
CRT level and format through later owner aliases, and fits each invocation's
column schedule against the eligible typed inventory. Ordinary and compact
matrix operations supported by IR lowering use this path. Preimage sampling
binds the resource plan already discovered by explicit warmup and uses the same
selector and schedule consumer. A missing Preimage plan is a warmup error, not
permission to probe. Opaque operations retain the containing model. Prepared
Trapdoor/Preimage graph execution currently requires a single device; this
restriction must be resolved before claiming full multi-GPU sampler support.

Explicit warmup now provisions a native prepared inventory for each selected
primitive fleet batch. Capacity comes from the node's typed claims;
full retained destinations and reusable scratch are allocated before dispatch.
Real synthetic operands bind to the same IR placement lowering, followed by
native reservations, compilation and the production batch/scalar entry points.
The native observer reports preparation, output initialization and column waves.
Every observed invocation schedule must equal the selected model schedule;
disagreement is an error, never a substituted nominal measurement. No protocol
graph is executed to obtain these observations.

Observation aggregation is streaming: retained records scale with the sibling
count, not the number of column waves. GPU work is aggregate execution-owner
CUDA-event time. Cumulative time sums actual wave and prerequisite timings;
ideal latency adds prerequisites to the maximum independent-wave time. Workspace
observations subtract the retained baseline from the measured prepared peak.
Concurrent hypothetical scratch sums the per-wave increments. These prepared
occupied-byte measurements are neither physical VRAM nor whole-graph peak use.

These are **native prepared measurements with synthetic operands**. Preimage's
public matrix remains a full resident owner during preparation; its covariance
cache is constructed before the seal, as in production trapdoor preparation.
No trial sampling is used to discover resources in this path. The checked
explicit-class setup boundary reopens allocation only with no unfinished native
invocations, and final estimation still performs CPU lookup only.

Selected classes carry their ordered source fragments into warmup. Inputs are
created directly at the selected CRT level, format and parameter context; shared
owners reuse one materialization. Compact payloads are generated in parallel on
CPU and uploaded directly to their selected device contexts. No intermediate
full-level matrix, level-conversion trial or compact placeholder kernel is needed.
Source layouts participate in the frozen class key.

Admission uses one fleet inventory and one prepared backend for all configured
GPUs. The same fleet column cap applies to every device, including remaining
nominal primitive classes. Source intervals and retained assignments remain
scoped to the device/context that owns them.

Artifact and staged inputs acquire typed owner claims at their first materializing
consumer, before output selection. Later uses retain that owner. The existing
transfer model charges the import/load separately, once per actual dataflow
materialization. Normal matrix consumers use the native prepared measurement path.
The `SyntheticFreshPlacement` report remains a scenario: when artifact bytes and
layout metadata are unavailable, it uses fresh balanced full-level evaluation
fragments for ordinary matrices (coefficient fragments for compact inputs). It
must not be interpreted as a measurement of an unknown artifact's actual format
or as a worst-case bound over arbitrary artifact layouts.

Heterogeneous semantic sibling classes and remaining non-matrix primitive classes
retain their existing restrictions. Multi-GPU runtime acceptance still requires
hardware validation; compilation and single-device results do not establish it.

For every active device, benchmark-only timing events join the execution owner's
compute and release streams, record a start, and gate the operation streams on
that start. The stop event joins every participating owner stream after the
ordinary production entry point. `device_elapsed_seconds` measures that device
timeline span, including internal idle gaps and transfers performed by the
primitive. It is not a sum of kernel execution times or a utilization measure.
Idle devices contribute zero work.

`fleet_wave_wall_seconds` uses a host monotonic clock from coordinated enqueue
through completion of every active device's stop event. It includes host dispatch
and is not the maximum of independently measured device spans. Prepared input
construction and prior releases precede the timer. Output owners remain alive
until the wall clock stops; their subsequent retirement is excluded. Warmup
outputs are released and reconciled before resetting the memory baseline. Atomic
operations use the same event and host clock boundaries on their single worker.
Comparisons must preserve these boundaries and the production ownership waits.

## Bounded representative measurement

Estimation never executes the protocol graph. CPU collection deduplicates requests
by operation semantics and concrete parameter/shape class before GPU measurement.
Measurement operands/results, persistent-byte accounting and transfer classes
resolve concrete wire types through ValidatedGraph::concrete_wire_type, also used
by runtime GPU operation lowering. Child invocations use their actual parameter
environment; validation's cached root types apply only to its original bindings.
This CPU resolution does not execute GPU work or add a structural validation pass.
Existing loop-index-dependent cost restrictions still apply.
Repeated nodes and loop multiplicities reuse these measurements. GPU work scales
with distinct measurement classes and harness repetitions, not node count; CPU
analysis still traverses the IR. Different shapes can legitimately require separate
representatives, so this is not a bound solely on operation opcode count.

The graph-execution measurement callback API has been removed. An optional runtime
observer remains available for explicit diagnostic benchmarks of actual execution;
it is not an estimator input or enabled by normal runtime execution.

Resource-discovery GPU work is restricted to explicit graph warmup:
`backend.prepare_graph_admission(&graph, capture_trace, &inputs, wave_bound, true)`.
Drop the returned guard after preparation and reuse the same backend for
production. Warm every graph and concrete parameter set used by a round trip,
including nested scopes. This prepares resources without executing the protocol
or publishing artifacts; synthetic samples are not production outputs.

Ordinary `execute` always passes `warm_up = false`. Polynomial-readback,
trapdoor and preimage cache misses return an explicit preparation error before
any discovery trial runs. Production neither silently warms missing classes nor
extrapolates unprepared resource plans. Operations covered entirely by native
layout queries need no discovery warmup. Real resource allocation/initialization
and sampling attempts that produce the requested result remain production work.

```rust,ignore
// Explicit warmup: use the actual validated graph, bindings and input layout.
let wave_bound = ExecutionConfig::default().max_parallel_instances;
drop(backend.prepare_graph_admission(&graph, false, &inputs, wave_bound, true)?);
// Production: same backend retains the discovery plans.
let result = execute(&graph, &mut backend, inputs, &mut store, SamplingMode::Fresh)?;
```

Representative benchmark warmups or a different backend instance do not populate
this backend's graph-resource caches. If trace capture will be used, prepare
with `capture_trace = true` as well.

Prepared runtime admission derives widths from native allocation bounds and current
reusable slots without running calibration kernels. Low-level allocating execution
requires a supplied profile or explicit `select_operation(operation, true)` warmup;
`select_operation(operation, false)` never starts a calibration pilot. The graph
executor always uses the production form. Benchmark representatives perform their
calibration during explicit setup before timed production samples.

The nominal placement uses the runtime's `GpuColumnSchedule`. Full and partial
wave classes retain their own active-device vectors, local widths, and measured
costs. A partial wave is measured independently; no timing monotonicity across
widths or device sets is assumed. Position-dependent constants, tensor ranges,
and column/diagonal concat ranges are measured at their actual scheduled offsets
instead of multiplying the first offset's cost. Their descriptors are generated
lazily. Ordinary repeated shape classes carry an explicit multiplicity.

Raw fleet-wave observations are retained separately from the aggregate node
measurement. Their keys include concrete operation shape, device jobs, per-device
batch sizes, physical devices, memory policy, operand sharing, placement scenario,
and the timing contract. Graph node identities and repetition counts do not create
new observations. Reusing an observation applies the current multiplicity to the
measured whole-wave cost.

The timing harness carries one prepared operand set per member through device and
fleet timing. Shared operands keep their existing owners; distinct operands remain
distinct. Batch size comes from this member list rather than a separate repetition
count. Explicit fleet preparation generates only missing operands, preserving
supplied fixed or scaled inputs and their sharing. Matrix binary operations,
accumulation, negate, scale, automorphism and Preimage use their production batch
APIs with each member's operands. Other primitives retain scalar dispatch per
member. A batch still shares one concrete node description and parameter environment;
heterogeneous operation parameters require their own class representation.
Standalone nominal measurement requests one member. Normal bounded reports now
request synthetic sibling batches from the frozen class table. Connecting actual
admitted placement/resource classes to the same harness remains incomplete;
scalar timings must not be divided by the admitted sibling count.

The admitted-batch adapter consumes admitted invocation summaries, including
ordinary and compact input-owner labels, and accepts a position-sensitivity flag
per invocation. Owner labels are assigned in first-use order over the whole
bounded batch. They preserve sharing between siblings without putting process-
local matrix IDs into measurement identities. These labels establish logical
operand sharing, not native slot identity or equivalence of prepared replicas.
The lookup still needs concrete operation, format, placement, and preparation
metadata; it must not use the owner labels alone as a complete measurement key.
When an active invocation depends on absolute column position, it requests each
joint wave at the ranges in the original admitted schedules. This preserves
unequal device widths, owner boundaries, and short siblings that finish early.
After all position-sensitive siblings finish, remaining equivalent waves can
again use multiplicity. Schedule classes describe contiguous runs, so their
first-wave indices and multiplicities can be expanded without changing placement.
The normal report path can consume complete whole-batch observations through
`MeasurementBackend::measure_dataflow_batch`. The dataflow walk requests one
observation per non-structural node at each actual admitted sibling boundary,
after admission and before member publication. It charges the returned joint
cost once, including nested scopes and tail batches. Complete observations
replace total primitive work, cumulative time, preimage work, tagged role costs,
wave counts and measured workspace peaks. Dataflow transfers and dispatch are
then added once. Dependency-critical-path, unlimited-resource memory, and
standalone per-subgraph figures remain their separate hypothetical models.

Backends may return no observations for the entire walk, preserving their
standalone measurement model. Partial coverage is an explicit missing-measurement
error; the estimator never mixes measured joint costs with guessed singleton
costs. `dataflow.primitives` identifies complete joint observations in the report.
The GPU backend supplies synthetic whole-batch observations through this callback.
For selected layouts, the callback consumes fleet schedules measured through
native prepared submission, with modeled source fragmentation, levels and formats.

`GpuMemoryLedger::fit_columns` exposes the production column-placement query
without acquiring resources. Its opaque `GpuColumnFit` provides the chosen
schedule and the exact per-device fixed/output/scratch requests. Its `summary()`
uses the same geometry and typed resource schema as an acquired plan's summary,
so class selection need not reconstruct widths, tails or resource sizes. Reading a fit
does not poll releases, acquire a lease, import inputs, or execute GPU work.
Resource kinds are typed: `Prepared(kind)` retains the native slot domain and
`ManagedPhysical` identifies a physical lease. Native resource summaries can
recover their exact typed setup claims without parsing diagnostic strings.
Uninitialized prepared backing consumes matrix claims with their actual CRT
level and format; byte planning and allocation use the same claims.

Production commits those same requests and rechecks availability at commit time;
a fit is not permission to execute. Both native and hypothetical inventories use
`GpuColumnFit::new` for the same placement and column-width search. The read-only
`GpuColumnInventory` interface supplies exact typed request fits; requests contain
storage IDs rather than references to native backing. The estimator scopes each
planned inventory to one ring across devices, preserving context-local slot IDs
and retained-owner exclusions. Hypothetical fits use native layout bounds, not
fabricated measured occupancy evidence. Native commit resolves IDs only against
the ledger's accepted backing, including the current enclosing wave's region
permission. `GpuDcrtBackend::matrix_column_requirements` binds type-determined IR
operations to selected output requests and eligible scratch inventories using the
same provider as native admission. Native destination and input-preparation
selection now use the same snapshot assignment as hypothetical selection,
including native region ownership and pending-reader eligibility. Native scratch
selection and hypothetical slot assignment also use maximum-cardinality matching
with the same size-class and
backing-size preferences. A flexible claim can move to another eligible slot to
avoid starving a later constrained claim. Neither path acquires rejected candidates.
It does not infer source fragments from bounds. Preimage's traced resource plan
is looked up using the same full class key as production sampling; the shared
lookup never launches discovery. For proven input fragments,
`GpuContextInventory::matrix_input_plan` calls the same source projection as
production, selects normalization/replica requests in native preparation order,
and reuses preparations explicitly supplied for that same input owner. Rejected
candidates leave the inventory unchanged. Unknown artifact contents are handled
under the explicitly reported synthetic fresh-import scenario; this is not an
observation of the artifact's actual representation. Source and resource bindings
drive normal fleet class collection. Explicit measurements consume the resulting
column jobs and create operands directly at their selected levels and formats.

For each class `k`, with multiplicity `n_k`, fleet wall time `L_k`, and device
spans `D_ki`, the adapter reports:

```text
work_seconds = sum_k(n_k * sum_i(D_ki))
cumulative_wave_seconds = sum_k(n_k * L_k)
independent_wave_count = sum_k(n_k)
latency_seconds = max_k(L_k)
```

Atomic operations retain their complete dependency latency. Sequential loops
multiply dependency latency; parallel loops preserve one iteration's ideal
latency. Graph edges remain dependency barriers. `CostReport::critical_path_seconds`
and `maximum_parallelism` describe an unlimited-resource dependency schedule,
not completion on the configured physical fleet. Parallelism is the maximum
across stages. `total_time_seconds` includes nested invocation-weighted cumulative
wave costs plus separately owned dataflow and executor costs. Logical device
work and cumulative wave costs are never divided by the GPU count again.
Production may overlap local jobs across logical waves; cumulative wave cost is
not an end-to-end runtime measurement or a universal bound under interference.

`measured_wave_workspace_bytes` is the maximum measured per-wave incremental
allocation after preparing resident inputs. The current adapter allocates output
owners inside the measurement, so this field includes outputs as well as
transient workspace. Logs state `measured_outputs_included = true`. It is neither
scratch alone nor the complete physical peak. `workspace_bytes`,
`workspace_high_water_bytes`, and `peak_memory_bytes` remain hypothetical
simultaneous-wave/DAG resource figures. Persistent inputs retain graph sharing;
replication to hypothetical additional fleets is not modeled. These figures are
not a provisioning plan. `chunk_count` counts structural primitive waves within
a scope; `per_subgraph` records nested invocation counts separately.

Transfer ownership is explicit. Local and peer transfers inside the primitive
entry point belong to its event span and fleet wall timer. Artifact/RAM
materialization outside that boundary appears in `TransferCost` with
`owner = DataflowMaterialization`. The executor dispatch proxy measures scalar
IR/liveness/map management and excludes GPU host submission already charged to
primitive wall time. Calibration input setup is excluded from primitive timing;
production artifact/staging boundaries are charged by the dataflow model.

Measurement observations are frozen within a backend instance. Request lookup
resolves to a cache identity containing the concrete shape, nominal schedules,
active devices, capacities, VRAM policy, preparation boundary, and timing
contract. An observation cache hit does not perform or certify fresh admission.
No cross-process timing files are reused. Results generated with the former
host-timed GPU work contract must be regenerated.


Transfer calibration measures the complete declared matrix or artifact through
one production staging/codec/store boundary. It does not shrink the matrix or
multiply a smaller artifact's latency. Compact fixtures are encoded on the host
using the runtime canonical encoder and imported into the configured GPU fleet;
fixture preparation is outside the measured transfer. Repeated restore-owner
releases are reconciled before each timed iteration. Executor dispatch is
calibrated even when the graph has no artifact or staging transfers.

Before constructing a transfer fixture, a capacity diagnostic compares an
allowance of four complete native matrix allocations (twelve for trapdoor
fixtures), or the larger compact payload allowance, against each device's
observed available budget. It reports allowance/target mismatch and observed
physical-budget excess without rejecting the fixture. Under the
[accepted production VRAM policy](plans/gpu-driver-memory-contract-decision.md),
actual CUDA allocation determines whether the full declared shape can run.
The diagnostic is neither a managed reservation nor a proof covering opaque CUDA
resources. Allocation failure remains an error; no smaller measurement is
substituted. Default-pool width candidates also remain at least the supported
class minimum when target headroom is exhausted; they do not grant admission.
Zero-shape
transfers currently return an explicit unsupported error because the runtime
fleet codec does not yet support empty artifacts consistently.

### Loop admission and transfer accounting

Dataflow analysis asks `MeasurementBackend::family_wave_size` before every bounded
prefix of a parallel loop, before importing its broadcasts or visiting its body.
The `LoopAdmissionRequest` carries the validated graph, owning scope, node,
bindings, total count, next member index, scope-entry inputs, argument identities,
Broadcast/Zip modes, current values, other active siblings, ancestor frames,
already-placed broadcasts and retained output ports. Its selected
input accessor inspects family representatives and overrides without loading
payloads. An admission error propagates; it is never replaced by a serial wave.
Every scope invocation revisits admission on CPU. Even identical descriptors and
ancestor frontiers can encounter different backend capacity, so scope output/cost
templates are not cached. Primitive measurements remain deduplicated separately;
this traversal performs no GPU trial.
`MeasurementBackend::dataflow_step` exposes CPU traversal boundaries: Enter for
one actual bounded scope batch, BeforeNode before any sibling publishes that node,
AfterNode after each sibling publishes its results, and Leave before the caller
stages or wraps the returned values. It borrows the concrete sibling states and
ancestor frames. Callback errors abort analysis; prepare_dataflow initializes the
next attempt. The GPU backend consumes these events for the complete active scope
stack. BeforeNode issues per-node hypothetical assignments; AfterNode binds the
published values to those assignments; Leave transfers surviving returned-owner
bindings to the parent.

Dataflow scopes advance the whole bounded sibling batch in IR order. SubgraphCall
propagates that batch; nested loops execute for each outer sibling in the same
order as the runtime. Earlier siblings have published the current node's results,
while later siblings have not. Sequential iterations carry their actual updated
input layouts and bindings. Transfer costs are accumulated from each admitted
wave instead of multiplying one representative body's transfer state. Host and
artifact families collapse uniform metadata after device identities are gone. The query then feeds
the executor's `retained_loop_output_ports` and `staged_loop_outputs` decisions. A full nested wave can retain device outputs;
a narrower wave crosses the host boundary. Root and retained sequential outputs
keep their executor-defined staging behavior. This query is CPU-only and must
not execute the protocol or discover a GPU class.

Scalar artifact broadcasts contribute one import per parent argument per loop,
outside the body-count multiplier. The loop retains that materialization; the
parent descriptor's transfer state remains unchanged for later consumers.
Zip inputs still contribute their per-member transfers.

The GPU measurement backend collects the concrete graph before dataflow analysis.
During explicit `measure_collected`, each worker discovers required resource
classes and freezes an unbacked root inventory. Production and estimation share
`GpuDcrtBackend::graph_resource_demand`: descending wave bounds and halving column
caps use native typed capacity and the setup memory budget. The final loop query
binds logical values to the active scope's derived layouts, excludes retained
parent and sibling assignments, and fits candidates on CPU. Each primitive node
matches its containing claims against then-eligible slots. Assignments retain their issuing node as well as the pooled
claim index; later reuse of that index cannot revive an expired allocation.
Expired assignment records are discarded, and a new estimate resets the scenario.
The parent does not issue an envelope for a child call or loop before entering it.
An accepted parallel wave supplies its child resource demand and initial broadcast
assignments. Subgraph and sequential entries lower their actual bounded input
batch through scope_resource_demand with discovery disabled.

Returned device values keep their established containing slot sets across scope
exit. Shared staging decisions release staged outputs before the next wave is
admitted; retained outputs survive until their caller liveness ends. Selecting a
family member preserves its logical owner. An unresolved dynamic selection keeps
a containing set of candidate owner slots rather than inventing a native address.
Scope liveness tables are derived once at entry with Rayon, and independent
ancestor/sibling retention queries run in parallel. Source binding visits changed
arguments/results instead of rescanning every immutable layout after each node.
Host-only families are skipped without enumerating their members.

Workers refit at a common reduced limit when their largest feasible widths differ;
smaller widths are not assumed feasible because staging changes resource demand.
Root families still stage under the executor's shared rule, including a full wave.
The first accepted wave records its broadcast-placement claim assignments. Later
waves exclude those same slots and reuse their loop-owned broadcast descriptors;
they neither invent fresh capacity nor charge a second import. Candidate matching
across measurement workers runs in parallel on CPU; placement state changes only
after every worker accepts the common width.

This remains an explicit synthetic fresh-root scenario, not a native reservation
or observed production ownership. Collection uses a serial transfer scenario;
final analysis performs no discovery or GPU trial. Nested parallel loops,
subgraph calls and sequential carried values use the active assignment stack;
fitting errors propagate instead of silently substituting width one. Consuming
accepted primitive classes in the timing aggregate is still outstanding. Value
layouts describe transfer state, not native owner identities or slot eligibility;
the current and ancestor caches also contain expired values.
`LoopAdmissionRequest::gpu_retained_scopes` now filters those caches with the
runtime GPU inventory's `owner_liveness` calculation, including root fusion
aliases, retained child inputs and separate ancestor/sibling identities. It includes
the current node's results for previously issued siblings and excludes them for
unissued siblings. The returned host frontier does not establish GPU completion.

`GpuContextDemand::value_claim_indices` resolves a scope-owned value to its
containing capacity claims at that frontier. It keeps sibling identities and
family member paths separate, follows reverse alias edges lazily, and preserves
child-scope references instead of expanding repeated family members. Child
alternatives with identical claim numbering and numeric lifetimes merge their
value-choice metadata recursively, rather than retaining a tree per loop index.
Different numeric assignments remain separate. The
`GpuHypotheticalWave::value_claims` adapter maps those indices through the accepted
typed assignment. These are hypothetical containing bounds, not native output
addresses; borrowed inputs still require their established source bindings.
This query alone does not complete nested admission or escaping-output ownership.
It preserves source alias metadata but does not equate independently materialized
descriptors or assign future owners to native slots. This shared liveness path alone does not
establish the remaining admission integration.

Value layouts also carry symbolic logical-value identities within one dataflow
analysis. Copies of resident values preserve identity; independent imports create
distinct identities. Identity equality proves references to the same logical value;
inequality does not prove disjoint native storage. Staging/export removes the device owner from that descriptor.
Uniform family templates bind indices only to body-local identities, preserving
captured owners across siblings. Cached costs rebase fresh result identities for
each invocation while retaining borrowed inputs. These tokens are not native IDs
or eligibility proofs. An unresolved selection receives a stable result identity
without claiming which candidate it aliases; family results have index-stable
identities under the same replication rules. Trapdoor public components retain a
projection of their opaque logical owner's identity. The sampled public output and
TrapdoorPublic views of that sampled trapdoor therefore agree, matching runtime's
shared public matrix. Equal shapes never establish identity.

`ValueLayout::gpu_inventory` binds scalar and family inputs to explicit
source resource bounds and a separately namespaced symbolic identity. It does not
invent placement or treat a symbolic ID as a native owner. The resulting
GpuInventoryValue can be passed to runtime `scope_resource_demand` as an ordered
list of concrete sibling bindings and inputs. `LoopAdmissionRequest::gpu_wave_demand`
resolves those child bindings and selected descriptors, including cold scalar
broadcast placement, without importing them. Both the public query and live
admission call the same sibling-demand fold, including shared broadcast import
resources. GpuScopeResources returns the context demands together with a separate
wire-to-GpuInventoryValue map for each candidate body. These maps preserve complete
resource bounds, including unresolved formats and alternative partitions; optional
fragments remain definite only when established by shared lowering. GpuContextDemand claims can be
planned/fitted with GpuPreparedSlotSnapshot. Accepted GpuHypotheticalWave results
retain those same per-body maps. GpuAdmissionInventory::bind_sources associates
derived bounds with the corresponding scope instance's logical values,
including resident family members; aliases preserve bindings and independently
produced values keep distinct owner keys. Missing bounds remain unbound; an
unresolved format can be consumed through the production containing bound without
inventing concrete fragments or performing a trial. The ordinary compiled runner
still receives only definite fragments and binds real operands at execution. The caller's materialization-identity contract requires consistent
layouts for aliases; no shape-based native ownership is inferred. Discovery is disabled on this
query. `LoopAdmissionRequest::gpu_admit_wave` searches a supplied immutable
GpuAdmissionInventory by descending W and halving column caps. It applies the
production sibling demand and primitive typed assignment on every required context,
returns the actual per-context assignments, and excludes all ineligible slots.
Empty/tail candidates obey the remaining and caller limits. This fit grants no
native reservation. The caller must supply complete ownership-based eligibility
and explicit source bounds; the query cannot infer them from equal shapes.
GpuContextDemand::retained_claim_indices identifies capacity claims whose owners
remain live immediately before an IR position. It excludes producers at that
position and retains values through their last reader. With the exact demand's
established context assignment, GpuContextInventory::exclude_retained maps those
claim indices to slot identities and preserves previous ineligibility. Capacity
indices are scoped to that demand, not native IDs or cross-scope identities.
Normal loop queries now construct this scenario automatically during explicit
measurement and consume it through `family_wave_size`. Nested active sibling
inventories track retained assignments and returned owners across calls and loops.
Selected primitive schedules reach the normal fleet class consumer. The scenario
still identifies synthetic operands because the estimator does not load the
protocol's actual input artifacts to discover their representations.

### Dynamic family access contract

Dataflow estimation requires every possible member of a dynamically indexed
family to have the same transfer state (resident GPU value, host staging, or
artifact), including per-member export overrides. Explicitly packed mixed-state
families should use static accesses or be normalized by the caller before
estimation. The estimator uses the representative member's state and does not
add a runtime validation pass for this contract.

Public import calibration uses a fresh artifact-store verification cache for
each iteration, including timed iterations after warmup, so each logical import
includes its first-load content-hash verification.

Packed families and artifact-family descriptors retain distinct placement semantics.
Loop placement imports packed scalar artifacts once before the bodies, leaves host
matrices for body materialization, and keeps artifact-family descriptors lazy.
Inventory lowering folds packed member metadata in parallel without equating
different symbolic owners. Uniform cold or captured families use representative
metadata rather than expanding every member; resident replicated owners require
explicit source bounds. Empty envelopes are neutral on either side of a fold.
