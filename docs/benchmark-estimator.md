# Benchmark timing semantics

`mxx-bench-estimator` distinguishes full logical work, cumulative production wave time,
and an ideal dependency schedule. `NodeMeasurement::work_seconds` includes every
production wave. `cumulative_wave_seconds` sums their measured fleet wall times;
`CostReport::total_time_seconds` additionally includes every nested invocation.
Benchmark-role totals use the same cumulative accounting.

For operations accepted by the existing type-aware GPU column-range capability,
the backend measures a coordinated fleet wave and records the exact calibrated
`independent_wave_count`. Logical work and cumulative time multiply by that count;
dependency `latency_seconds` remains one measured fleet wave. A final partial wave
retains the existing conservative full-wave cost. Fleet wall time includes coordinated
enqueue/completion and is not replaced with the maximum device timer.

All calibrated column-separable operations use this rule, including hash sampling,
matrix arithmetic, and supported ranged trapdoor/preimage operations. Inner products,
reductions, and trapdoor dependencies inside a measured wave retain their measured
cost. Graph edges remain dependency barriers. Operations without a valid independent
column range use the complete atomic measurement unchanged. Sequential loops multiply
dependency latency; parallel loops preserve one iteration's latency. Neither rule
changes production scheduling, matrix values, hash tags, or sampling parameters.

Ideal simultaneous-wave parallelism and transient workspace scale by the wave count,
with saturating resource arithmetic. These are hypothetical unlimited-resource values,
not the measured GPU fleet's physical peak. Persistent inputs retain the graph's shared
ownership model; replication and transfer of fixed operands to hypothetical additional
fleets are not modeled. Consequently these resource figures are not a provisioning plan.
`chunk_count` sums primitive waves within a scope and counts each structural node once;
nested invocation counts are separate in `per_subgraph`.

`measured_wave_workspace_bytes` retains the largest measured bounded-wave scratch
without multiplying it by wave count or loop iterations. It excludes resident
inputs and is not a whole-graph physical peak. This field is reported separately
from ideal simultaneous-wave workspace.

The GPU log `measured independent GPU fleet waves` records measured fleet latency,
wave count, complete work, cumulative wave time, and physical wave workspace explicitly.
Benchmark results must be rerun when their measurement semantics change.
