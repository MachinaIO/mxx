# Benchmark timing semantics

`mxx-bench-estimator` distinguishes device work, cumulative measured wave time,
and an ideal dependency schedule. All times are seconds. The GPU adapter's
`work_seconds` is aggregate device-seconds from CUDA events, never host elapsed
seconds or elapsed time divided by the device count. The generic host harness
continues to report host elapsed work seconds.

The current GPU measurement API receives concrete shapes and synthesizes its
inputs. `CostReport::measurement_scenario` therefore reports
`SyntheticFreshPlacement`. This scenario assumes prepared synthetic resident
operands and fresh column placement under the observed capacities. It does not
supply the runtime's retained ownership intervals, complete output reservation,
or admitted invocation plan, and cannot establish runtime schedule agreement or
physical memory feasibility. Sharing the calibration registry does not remove
this limitation.

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

## Admitted invocation plans

`GpuNodeMeasurementBackend::admit_invocation_plan` accepts the runtime's
`GpuAdmittedPlanSummary` for a node before measurement. The summary is produced
by `GpuColumnMemoryPlan::summary` and exposed for an admitted, unconsumed batch
through `GpuDcrtBackend::admitted_invocation_summaries`. It records the actual
owner intervals, the admitted local width per device, local job counts, the
actual wave count, the exact wave classes with their multiplicities and
representative jobs in global coordinates, and the claimed native resource
classes per device (fixed inputs, retained outputs, per-wave scratch) with their
kinds, shapes, formats, bytes and alignments.

A prepared execution (`ExecutionConfig::prepared_gpu_admission`) records every
admitted invocation in the fleet backend's `admitted_plan_log`, keyed by the
operation identity the executor selected. `admit_plans_from_log` maps that log
onto the graph's root-scope nodes through the same identity, so the estimator's
classes are exactly the runtime's admitted plans for that graph.

When a node has an admitted plan, the estimator measures one representative
fleet wave per admitted wave class instead of the nominal full/partial classes,
and multiplies by the class multiplicity. The wave count is the plan's maximum
local job count: owners with 90 and 10 columns and widths 10 and 90 yield nine
waves (one class with both devices active, one class with only the first device
active eight times), not one nominal fleet wave. Offset-sensitive nodes measure
every actual wave at its own global offsets. Observation keys include the
scenario label and the admitted classes, so nominal and admitted observations
never alias. `CostReport::measurement_scenario` reports `AdmittedInvocationPlan`
only when every measured fleet-wave node had an admitted plan; otherwise it
remains `SyntheticFreshPlacement`, and per-node logs name the scenario. Operand
values are still synthetic, and an admitted plan is not a physical-memory
certificate; it is a traceable match between the runtime schedule and the
estimator's wave classes.

The nominal placement uses the runtime's `GpuColumnSchedule`. Full and partial
wave classes retain their own active-device vectors, local widths, and measured
costs. A partial wave is measured independently; no timing monotonicity across
widths or device sets is assumed. Position-dependent constants, tensor ranges,
and column/diagonal concat ranges are measured at their actual scheduled offsets
instead of multiplying the first offset's cost. Their descriptors are generated
lazily. Ordinary repeated shape classes carry an explicit multiplicity.

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
