# PR #161 GPU Admission Liveness Follow-up

## Executive recommendation

Forward-port the GPU runtime and bench-estimator liveness fixes described in
this report onto the latest PR #161 head. Do not forward-port the current
working tree wholesale: it also contains Exponent-LUT Gaussian/noise,
parameter, NAND, and unrelated integration changes.

The immediate production fix is to preserve an absolute timeline anchor when
an alternative group is formed at a node. The current implementation previously
converted a node-local demand beginning at position `p` into a demand beginning
at `0`. That changes a local alternative into a scope-wide lifetime and prevents
capacity reuse at a later, disjoint node. The forward-port should also retain
the cached-output/reimport liveness fixes and the candidate replay/recipe
changes listed below.

The measured-looking TB values described here are admission-demand snapshots,
not measured runtime peak VRAM. A fresh four-A100 rerun was not completed: the
old pod disappeared and a replacement could not be started because of
Runpod inventory/account-balance failure.

## Scope and non-goals

This report covers only GPU runtime and bench-estimator admission/liveness
work that can be considered for PR #161.

It does not cover:

- Exponent-LUT Gaussian/noise distribution changes;
- NAND circuit, readout, or parameter-search changes;
- lattice-estimator changes;
- Lean fixes or ignored Lean tests;
- a Runpod driver, source archive, or benchmark-log change;
- a claim that the admission snapshot is the actual execution peak.

The existing report
[`docs/pr-161-gpu-measurement-device-ownership-follow-up.md`](pr-161-gpu-measurement-device-ownership-follow-up.md)
already covers device/context ownership, complete-fleet setup sealing, and
the unused-device placeholder. This report is complementary: it covers
liveness, alternative timelines, candidate replay, and the resulting demand
overestimate. The device-ownership changes should be forward-ported according
to the existing report rather than duplicated here.

The recommendations are divided into three buckets:

### Implemented and eligible for forward-port

The absolute alternative anchor, cached-output/reimport liveness, and semantic
candidate replay/child-scope refactor are eligible, subject to the performance
qualification below.

### Diagnosed but not implemented

The separate `TrapdoorSample` temporary claim lifetime fix and any
schedule-aware reduction of the remaining demand are unresolved follow-ups.

### Diagnostic-only

Extreme-breakdown execution, provenance fields used only for its report,
verbose category logging, and Runpod evidence are investigation aids only.

## Observed TB-scale reproduction

The temporary extreme-demand breakdown reproduced the following prepared-demand
snapshots for the multi-device admission path:

| Candidate | Reported prepared demand | Meaning |
| --- | ---: | --- |
| configured maximum wave and width | **14.967 TiB** | Sum of prepared backing requested by the admission snapshot for each reported device; not a runtime-peak measurement |
| wave `1`, width `1` | **3.806 TiB** | The same snapshot calculation after reducing both candidate dimensions; still not a runtime-peak measurement |

The decrease does not establish that the execution needs 3.806--14.967 TiB.
The diagnostic deliberately reports the slots selected by
`prepared_demand_bytes`; it does not prove that every retained owner in the
snapshot is live at the same instant. In particular, a nearly fixed TB-scale
component is evidence to inspect retention, alternative envelopes, owner
placement, and export/reimport handling before treating it as hardware
capacity demand.

### Breakdown categories

The diagnostic categories below are claim-projection categories. They partition
the prepared slots used by the snapshot and reconcile to
`prepared_demand_bytes`; they are not a proof that the corresponding slots
overlap in execution.

| Category | What it counts | Interpretation |
| --- | --- | --- |
| `matrix_ordinary_capacity` | Ordinary matrix capacity claims | Baseline matrix capacity selected for the context |
| `matrix_slot_exact` | Exact level/format claims attached to a matrix slot | Typed claim for a selected slot; it may coexist with ordinary capacity |
| `matrix_required_exact` | Required exact claims that occupy a slot again | Deliberate simultaneous typed multiplicity |
| `matrix_alternative_envelope_increment` | Extra claims introduced by the alternative envelope | Maximum multiplicity across mutually exclusive alternatives, not simultaneous execution of every branch |
| `matrix_synthetic_workspace_backing` | Minimum matrix backing inserted for a workspace-only selection | Synthetic support backing required by the prepared-slot planner |
| `compact_payload` | Compact payload slots | Bounded payload material used by family/compact paths |
| `workspace_batch`, `workspace_transform`, `workspace_compact`, `workspace_sampler`, `workspace_transfer`, `workspace_other` | Typed workspace slots | Native workspace backing classified by `GpuPreparedSlotKind` |

The diagnostic also prints context totals, slot counts, largest slot,
capacity-class samples, and bounded origin/owner samples. Lifetime envelopes
are metadata used to explain a demand; they must not be reported as an overlap
proof. If claim selection cannot be reconciled with the claim projection, the
diagnostic fails rather than assigning bytes to an invented category.

For the wave `1`, width `1` snapshot, the reported category breakdown was:

| Category | Reported demand |
| --- | ---: |
| transfer | 1.464 TiB |
| ordinary matrix | 0.945 TiB |
| required exact | 0.661 TiB |
| slot exact | 0.661 TiB |
| compact payload | 0.066 TiB |
| compact workspace | 0.008 TiB |
| **total (rounded)** | **3.806 TiB** |

The equal `required exact` and `slot exact` values do not by themselves prove
that the two categories are double-counted or that they are distinct live
allocations. They are classifications in the prepared-demand accounting; the
claim projection and slot planner must be audited before drawing an overlap or
ownership conclusion.

## Primary root cause: node-local alternatives lost their absolute anchor

`GpuContextDemand::include_alternative` merges mutually exclusive branches.
Before this change, the merged matrix and workspace demands were rewritten with
`begin = 0`, while their ends retained the existing matrix/workspace owner
lifetime. For a node-local alternative at position `p`, this changed the
intended intervals from approximately `[p, p]` and `[p, end]` to `[0, p]` and
`[0, end]`. Two alternatives at separated nodes therefore appeared to overlap
from the beginning of the scope, so the capacity planner could not reuse a
slot at the later node.

The production fix adds `GpuContextDemand::alternative_anchor`. The anchor is
the earliest absolute position available when the alternatives are merged:
existing alternative anchors and the branch demand beginnings are considered.
`include_alternative` writes that anchor back to matrix and workspace demands,
while preserving the existing `matrix_until` and `layout_until` ends. The
anchor is propagated through `simultaneous` and therefore remains consistent
when sibling demands are merged.

The required invariants are:

1. a node-local alternative keeps its node-local absolute beginning;
2. owner/`until` ends are not shortened by alternative merging;
3. a broader scope or child alternative uses the appropriate containing anchor;
4. direct `include_alternative` and recipe replay produce the same timeline;
5. later disjoint node-local alternatives can reuse matrix/workspace capacity;
6. same-node multiplicity and long-lived owners remain charged.

This is a conservative timeline correction. It does not infer overlap from a
lifetime envelope and does not make an alternative branch free.

## Diagnosed but not implemented: TrapdoorSample temporary-claim lifetime

`TrapdoorSample` traces sample, export, and import claims. These claims do not
all have the same lifetime: temporary sampling/export/import work can finish at
the node, while the public matrix and trapdoor owners can remain live for later
consumers. The current graph-demand code still has a separate lifetime hazard
when the temporary claims are charged under the maximum retained end of the
two output ports. That can keep temporary claims in the admission snapshot
longer than the operation needs and contributes to the fixed-demand diagnosis.

This issue is distinct from the `alternative_anchor` bug and is not implemented
by the current change. It is a separate future fix/unresolved follow-up. The
future change must preserve the existing owner distinction and audit the
claim-specific begin/end assignment in `trace_trapdoor_sampling` and the
`NodeKind::TrapdoorSample` branch. It must not globally shorten the
public/trapdoor owner lifetime: only temporary claims may be shortened, and
the later-consumer owner must remain live.

## Other production changes to forward-port

### Candidate replay and recipe reuse

The admission search probes wave and scratch-width candidates. The current
runtime builds a candidate-independent `AdmissionInvariant` and, for supported
root graphs, records a `ScopeDemandRecipe`. Later candidates replay the recipe
and only re-evaluate the candidate-dependent parts. This is a semantic
refactor that can avoid rebuilding the same large graph and liveness state, but
the current evidence does not show that recipe replay has removed the observed
performance regression.

In particular, for the wave `64`, width `72` recipe run, each candidate replay
took approximately **34.6--38.2 seconds** and recorded
`fixed_events=69,882`, `dynamic_events=33,507`, and `child_events=112`.
Therefore recipe introduction alone is not a performance acceptance criterion;
additional optimization and direct-versus-replay comparison are required.

Forward-port the replay contract together with its child-scope handling; do
not copy only the timing optimization. A recipe must replay the same demand
semantics as the direct graph builder, including alternatives, owner origins,
workspace claims, and output layout updates.

### Cached output and reimport liveness

The executor has two relevant lifetimes:

- `original_output_end`: the original resident producer owner's lifetime;
- `cached_value_end`: the lifetime of the cached/reimportable value used by
  later consumers.

Root frame initialization and retained-scope pruning must use
`cached_value_end` when deciding whether a cached or reimported value survives.
`original_output_end` remains correct for original producer claims and must not
be replaced indiscriminately. Child-input borrowing rules must remain intact.

The regression graph exports a non-input matrix, consumes it, performs an
intervening operation, and consumes it again. It checks both the cached value
and the production inventory's owner/source mapping at the second consumer.

## Forward-port file and symbol list

The following files/symbols are the concrete forward-port surface. Preserve
the latest PR #161 preimage operations and accepted-width replay behavior while
integrating them.

| File | Symbols/behavior to forward-port |
| --- | --- |
| `crates/runtime/src/backend/poly_gpu/gpu_inventory.rs` | `GpuContextDemand::alternative_anchor`, `same_timeline`, `simultaneous`, `include_alternative`; candidate-independent invariant/recipe integration; child-node replay |
| `crates/runtime/src/backend/poly_gpu/gpu_admission_recipe.rs` | `ScopeDemandRecipe` event recording, finish, and replay; preserve demand/owner/layout semantics across candidate replay |
| `crates/runtime/src/backend/poly_gpu/gpu_admission_child.rs` | `ChildNodePlan` and child evaluation/replay, including output-end and layout-update propagation |
| `crates/runtime/src/executor/gpu_plan.rs` | `OwnerLiveness::cached_value_end`, `original_output_end`, and the owner-liveness rules that distinguish cached/reimported values from original producer owners |
| `crates/bench-estimator/src/gpu_admission.rs` | `GpuGraphInventory::scope_until`, retained-scope pruning via `cached_value_end`, and the separated-consumer regression coverage |

The existing device-ownership report additionally covers the setup claim
representation, per-device preparation, and complete-fleet seal. Those changes
must remain device-aware and must not be replaced by the liveness forward-port.

## Regression tests

The current change set includes or requires the following focused coverage:

- `test_node_local_alternatives_keep_absolute_anchor_for_later_reuse`: two
  separated node-local alternative groups retain their absolute positions,
  reuse matrix/workspace capacity later, preserve same-node multiplicity, and
  keep long-lived owners;
- the alternative-family tests covering branch-local multiplicity, exact typed
  claims, child alternatives, and composed alternatives;
- `test_exported_non_input_matrix_survives_reimport_until_second_consumer`:
  an exported non-input matrix remains available through an intervening
  operation and second consumer, including production inventory owner/source
  mappings;
- existing candidate replay/child-scope tests and PR #161 device/context
  ownership tests.

The tests should fail if `include_alternative` is changed back to `begin = 0`,
or if cached/reimported values are pruned using only `original_output_end`.

## Validation performed

The following checks passed in the current worktree:

```bash
cargo test -p mxx-runtime --features gpu alternative --lib
cargo check -p mxx-bench-estimator --tests --features gpu
cargo test -p mxx-exponent-lut --features gpu --test gpu_nand_circuit --no-run
cargo +nightly fmt --all -- --check
git diff --check
```

The alternative-focused runtime run reported 9 passed tests. The GPU NAND
command was compile-only (`--no-run`); it is not evidence of a completed GPU
round trip. No new A100 rerun is available for this follow-up because the old
pod disappeared and the replacement attempt failed before measurement due to
Runpod inventory/account balance. Consequently, this report makes no runtime
peak-VRAM claim and no new benchmark claim.

## Diagnostic-only material that must not enter PR #161

The following are temporary investigation aids and must remain outside the
PR:

- `MXX_GPU_ADMISSION_EXTREME_BREAKDOWN=1` and its intentional early-stop path;
- verbose claim-projection provenance and owner/category logging used to
  reconcile `prepared_demand_bytes`;
- Runpod driver output, monitoring output, source archives, and ad-hoc logs.

The production forward-port should retain the sound admission/liveness fixes
without making this diagnostic flag, its verbose output, or its early-stop
behavior part of the normal measurement contract.

The `TrapdoorSample` lifetime issue and any schedule-aware peak reduction are
not diagnostic-only: they are diagnosed production follow-ups, but they are
not implemented and must not be represented as completed fixes.

## Handoff

Forward-port the runtime and bench-estimator changes onto the latest PR #161
head, preserve the existing device-ownership work from the earlier report,
then rerun the targeted tests and a fresh four-A100 measurement when a pod is
available. Do not treat the current TB-scale snapshot as measured peak VRAM,
and do not push any implementation or report changes without explicit
authorization.
