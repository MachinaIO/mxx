# Workspace architecture

This repository is a virtual Cargo workspace with no root facade crate. Consumers depend directly
on the crate that owns an abstraction.

## Dependency layers

```text
mxx-runtime              -> mxx-ir-core, mxx-primitives
mxx-bench-estimator      -> mxx-ir-core, mxx-runtime; optional mxx-primitives
mxx-dsl                  -> mxx-ir-core
mxx-gadgets              -> mxx-dsl, mxx-ir-core, mxx-primitives, mxx-runtime
mxx-bgg                  -> mxx-dsl, mxx-gadgets, mxx-ir-core, mxx-primitives
mxx-fhe                  -> mxx-dsl, mxx-ir-core, mxx-primitives
mxx-we                   -> mxx-bgg, mxx-ir-core, mxx-gadgets, mxx-runtime
mxx-func-enc/io          -> interface-only crates with no dependencies
```

Application crates never depend on one another. Diamond WE is active in `mxx-we`; functional
encryption and iO protocol implementations have been removed during the DSL migration.

## Responsibilities

### `mxx-primitives`

Owns polynomial and matrix representations, OpenFHE integration, concrete sampling, and native
CUDA. CPU Gaussian sampling resamples individual coefficients outside the authoritative integer
cutoff. CPU preimage sampling rejects a whole candidate outside its cutoff so `B * K = P` is
preserved. GPU Gaussian sampling enforces the same cutoff per coefficient in CUDA. Batched GPU
preimage sampling rejects a whole GPU-generated candidate after full-CRT centered-norm checking,
preserving both the preimage equation and the authoritative cutoff.

### `mxx-ir-core`

Owns the canonical executable graph, compile expressions, artifact metadata, parameter/type/shape
validation, execution ordering, and liveness. `derive_param_constraints` is the shared source of
decidable compile-parameter conditions consumed by concrete validation. Sampler
nodes serialize required integer coefficient cutoffs. Subgraph and parallel-loop bodies are
structural and stored once.

`protocol` owns protocol declarations, input contracts, frozen graph annotations, sampler-free
ideal/predicate specifications, and structural validation of linked workflows. These are core
graph data and checks, independent of the DSL used to construct a graph. There is no separate
correctness crate and no generic symbolic noise simulator.

The Lean exporter owns primitive execution-relation generation and application-independent linked
claim assembly. It receives explicit graph connections and endpoint semantics, not a WE protocol
implementation, and does not infer noise bounds or expand structural families into individual lanes.
`lean::protocol` converts a protocol declaration into exported roots and a linked claim;
`lean::claim` renders the final proposition. Applications supply backend bindings and decoder
semantics, while their mathematical bounds and proofs remain application-owned.

### `mxx-dsl`

Creates immutable core nodes immediately. It has no symbolic reinterpretation layer.
The constructed graphs feed core-owned `IdealSpec` and `PurePredicateSpec` validation.
Indexed `Family<T>` values preserve composite element schemas. `parallel` and `iterate` create
structural loops; lexical reads become explicit core dependencies with inferred member indexing.

### `mxx-runtime`

Executes validated schedules on CPU or GPU primitive backends and owns runtime values, sampling
transcripts, sessions, artifacts, and bounded parallel waves.
Temporary families never spill to disk. The GPU fleet admits bounded sibling waves
against available prepared storage before loading their inputs. Each accepted wave
retains a capacity reservation through output publication; nested operations borrow
that capacity. Compatible matrix operations and Preimage residuals share native
batch submission. Intermediate family results use host RAM staging when required
by the admitted wave's output policy. Column widths stay within the accepted
scratch cap and the primitive's calibrated resource limits.
Setup-time child-wave provisioning and live admission use the same simultaneous
matrix and workspace demand merge. Both retain actual IR start/end positions.
Shared batch workspace claims preserve multiplicity at each position; equal-sized
ordinary owners are never merged. Identical alternative demands preserve their
exact intervals and preparation identities; different alternatives retain a
containing bound. Escaping child resources map to the parent output lifetime,
while transient child resources end at the parent operation.
The owner-liveness calculation in executor/gpu_plan.rs is also consumed by the
GPU estimator's retained-scope metadata path. It filters expired cached values
without dropping borrowed child inputs, keeps ancestor scope identities separate,
and preserves proven wire aliases. Materialization identity and native slot
eligibility still require their own explicit bindings.
GpuDcrtBackend::scope_resource_demand exposes the existing CPU resource lowering
for an ordered sibling candidate with discovery disabled. It and live admission
share the same complete wave-demand fold, including cold broadcast imports;
live admission retains its cached lowered operations across candidate searches.
GpuScopeResources preserves complete per-body GpuInventoryValue maps alongside
the context demands. CPU planning retains both definite fragments and conservative
bounds for unresolved future-output layouts. The compiled runner projects only
definite fragments from the same maps and binds real operands at execution. GpuInventoryValue carries explicit source fragments and
native or separately namespaced symbolic identities. GpuContextDemand::claims is
the shared typed-request builder used by provisioning, live admission and CPU
planning; simultaneous merging preserves its interval and shared-input rules.
The query creates no backing or reservation and rejects missing warmup templates.
Typed-slot compatibility and maximum-cardinality assignment live in primitives.
GpuPreparedSlotSnapshot::plan builds an unbacked inventory from typed claims
using the same native capacity builder as real storage creation. It allocates
CPU metadata only. Storage ID zero isolates planned requests from real native
reservations; identities are local to one planned inventory. Setup byte demand
uses these capacities, including the minimum matrix owner required when only
new workspaces are missing.
GpuPreparedStorage::snapshot copies native slot capacities and region-relative
eligibility without GPU work or ownership tokens. GpuPreparedSlotSnapshot::assign
uses the same native layout calculation as reservation; callers supply the
scenario's eligible slots. Runtime retains pending-reader policy and performs
an atomic native reservation after hypothetical assignment. A snapshot or a
successful layout match grants no lease and may be stale before commit.
Live W/C search takes one ownership snapshot after polling releases and reuses
it throughout that search. A stale commit or selected minimum-candidate wait
starts a new search and snapshot. Primitive-step claims are grouped by backing
store into one native reservation per store; partial failures cancel the whole
reserved prefix before dispatch. Repeated simultaneous demand merges preserve
proven input preparation identities as well as their lifetimes.
Live admission merges matrix lifetimes at the existing IR positions across the
candidate siblings before fitting native slots. Native matrix batches share
normalization and replica capacity only for a proven shared input owner at the
same position and required format. Actual native owners use their IDs; a cold
scalar matrix artifact broadcast uses its parent wire to identify the one import
performed after admission. Its full owner and import scratch are reserved once
per wave, and the executor reuses that placed owner across siblings and later
waves, including repeated arguments referencing the same parent wire. Host values
and family descriptors retain their existing conservative accounting. Equal
shapes and unresolved aliases do not establish sharing. Unresolved control-flow
alternatives retain conservative containing peaks, including escaping owners.
Root and child input metadata use the same native-ownership classification.
Already resident matrix, compact and trapdoor values, including fully resident
packed families, are borrowed rather than charged as another import. A root
artifact declaration takes precedence over a supplied value, matching execution;
its fragment layout stays unknown until materialization.
Root admission and each accepted loop wave retain the exact resource-operation
lowerings used by their capacity calculation. Preflight resolves the existing
scope/node reference and concrete instance bindings against that owned record;
it does not lower the resource operation again. The backend holds only weak
references to active records, so nested waves preserve their parents' records
and completed waves do not leave reusable operation state. Direct calls without
a scope admission still lower their invocation at their ordinary preflight.
Primitive admission also records the selected original shards, normalized
fragments, shared replicas, and output context/level/format for each retained
output interval. Compilation consumes these exact bindings; it does not search
inputs or derive output placement again after preparation.
Uniform and Gaussian sampling likewise share one CPU-only distribution lowering.
Ordinary sampling nodes consume the admitted definition; only real execution
supplies fresh randomness. Replay continues to import recorded values.
Constant IR and direct invocation construction share one operation lowering.
Ordinary constant nodes pass their IR identity to preflight and consume the
accepted operation. The existing gadget/context compatibility check is shared
with CPU parameter validation and runs when that operation is constructed.
Polynomial constants reserve the production polynomial-upload resources; GPU
range constants such as Zero and Identity do not reserve an artificial upload.
Inventory recognizes type-determined GPU operations through their common
lowering rather than a duplicated list of supported arithmetic node kinds.
Artifact and host-staging import inventory consumes the production import
operations' scratch and workspace queries. Unresolved compact/RNS representations
use containing alternatives; inventory does not separately calculate codec or
pinned-upload sizes.
Input preparation uses one metadata-only source planner for both inventory and
concrete preflight. It records the initial shape/format and orders mixed-format
fragment normalization before replica assembly. Submission consumes that layout
directly. Until concrete ranges are selected, inventory conservatively covers
both full-range and owner-local alternatives using the same planner.
Root/wave admission also retains known intermediate fragment layouts by existing
IR wire reference for each accepted instance. Direct IR preflight binds these
references to real operand IDs and shares their accepted layout metadata. Rejected
candidates publish no layouts. Unresolved layouts and fused boundary operations
retain their existing conservative/concrete lowering paths.
Root/wave admission retains observed input fragments by actual immutable owner
ID. Preflight borrows these snapshots through shared metadata and uses them in
source selection. Every fleet matrix, including a newly produced value, publishes
immutable fragment metadata with its native owner. Admission shares that metadata
and later source selection consumes it without rescanning native shards. A value
produced after admission uses its own published layout through the same source
planner. Equal shapes never substitute for IDs.
GPU operation lowering has one prepared representation, `PreparedOperation`, shared
by IR admission, direct invocation preflight, and the compiled executor. Host-valued
polynomial observations and threshold decoding are boundary requests over the same
typed polynomial-readback operation; they do not create a parallel readback plan or
matrix invocation queue. Native readback claims retain their exact level and
coefficient/evaluation format through inventory demand and execution.
Direct fleet calls use this same admission path rather than accepting externally
assembled operation plans.
Direct primitive preflight includes deferred upload spans and completion events
when selecting reusable scratch capacity. It waits for selected pending resources
only after the complete minimum candidate fits, before native reservation; it
rechecks capacity when retirement completes during selection. Rejected trials
do not wait for uploads, and unrelated transfers are not drained.
Deferred upload retirement exposes both its pinned allocation and prepared
completion event as pending capacity. If no immediately available wave fits,
admission waits only for resources in a feasible minimum candidate and retries
against fresh availability; it does not drain unrelated transfers.
A host observation of a matrix's completion applies only to that event record.
When ordinary reader tracking folds new work into the producer event, it clears
the observation so subsequent reads and prepared-slot recycling wait for the
new dependency. Read-only readers retain their separate release dependencies.
CPU backends retain bounded sibling waves, including across subgraph calls. Values
are released at their final live reference, including captures and member aliases.
Root Family exports stream directly to their final artifact paths; the manifest is published
only after execution succeeds. Other exports write their RAM payloads to final storage.
Durable exported artifacts remain available for online execution.

The common estimator tracks runtime storage boundaries across Family, capture,
subgraph, and sequential-loop values. It measures raw-RNS RAM staging/reloading
and compact artifact encoding/writing/reading/decoding with the production backend
and artifact store, then adds their invocation-weighted wall times to
`total_time_seconds`. GPU-resident edges have no added transfer charge;
`total_work_seconds` remains primitive GPU work. The runtime and estimator share
the matrix-Family staging rule and artifact codecs.

`dataflow` and `transfers` report the extra costs and counts separately. CPU IR
dispatch is a measured scalar-node proxy, not a full protocol execution trace.
Artifact import calibration includes public content-hash verification (also
charged conservatively to private imports). Measurements sum boundary costs;
they do not predict overlap with unrelated operations, cold-cache storage
behavior, or contention in a complete execution. Large transfer shapes use a
VRAM-bounded representative wave and round the final partial wave upward.

Compact serialization remains the durable-export format. GPU RNS transfers expose D2H
start/completion separately, and H2D sources are owned by the existing event-based pinned
memory reclaimer. This permits a previous output store and next input load to overlap.
Fleet snapshots pipeline two shards and reuse two pinned buffers, bounding transfer
scratch independently of the number of column chunks.
Completed scalar exports are written immediately and retained as artifact handles, including
members projected from batch families.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns BGG+-specific keys, encodings, sampling, evaluation, lookup, decoding, artifacts,
slot transfer, and refresh. Both build executable graphs through `mxx-dsl`.

### Application crates

`mxx-fhe` builds Ring Regev/Ring-GSW and leveled BGV graphs, including CRT modulus
switching, hybrid RNS key switching over QP, relinearization, and rotations. BGV encrypt/decrypt exchange SIMD slots
by default, with internal encoding and zero-padding of short inputs. Cryptographic arithmetic
and sampling execute through the DSL runtime; runtime is a test-only dependency.
It tracks coefficient noise bounds per ciphertext and reuses primitive ring parameters and DSL
matrix handles. Bootstrapping is out of scope. CPU and GPU backends share the same
FHE graphs. GPU centered basis conversion uses native unsigned CRT residues and
stream-ordered INTT/lift/NTT operations without a host coefficient round trip.
Hybrid RNS ModUp/ModDown use dedicated graph nodes with an explicit ordered
source basis, checked by the runtime against registered parameters. CPU and CUDA
primitives fuse CRT accumulation between one input INTT and one output NTT per
digit, preserving the approximate centered-sum semantics and noise bounds.
FHE artifacts stay in memory or enter the protocol as direct runtime inputs.

`mxx-we` owns the implementation-independent witness-encryption declaration/runtime traits and the
Diamond protocol. A Diamond protocol fixes a layered Boolean shape but accepts gate opcodes and
previous-layer indices as public runtime families. Encryption and decryption consume the same
circuit assignment; witness bits are decryption-only inputs. Parameter search uses deterministic
worst-case bounds and accepts a candidate only after Lean checks the generated theorem for the
same frozen workflow, backend layout, and concrete parameter environment. The selected candidate
retains its checked artifact; numerical rejection and checker failures remain distinct.

`mxx-func-enc` and `mxx-io` expose only their common interface traits. The disabled AKY24 FE,
AKY24 iO, and Diamond iO modules and their exclusive BGG helpers have been removed. See the
README for the `main` branch containing the latest iO implementations. Reusable implementations
in `mxx-gadgets` remain available.

Tall's old-simulator-dependent parameter search and noisy verification modes are explicitly
unavailable pending a Tall-specific correctness implementation. The independent noiseless runtime
round-trip remains available; it is not a substitute for a proved noisy bound.

## Generated Lean artifacts

Each crate keeps its handwritten Lean modules directly under `lean/`, without a nested package-name
directory. Shared modules have crate-qualified filenames such as `PrimitivesBounds.lean` and
`RuntimeMatrixOps.lean`, avoiding collisions when several packages share one Lean search path.
Lake libraries list their module roots explicitly. The `MxxPrimitives.lean`, `MxxRuntime.lean`,
`MxxIR.lean`, `MxxGadgets.lean`, and `MxxBgg.lean` entry modules collect reusable imports; mathematical
namespaces and theorem names are independent of this file layout.

Diamond parameter search generates and checks Lean artifacts through the production library API;
the GPU integration test uses that same search. No separate example executable is required.
Crates do not contain example targets: reusable extraction fixtures live in ordinary unit-test
modules, and generated files belong under ignored `test_data` or temporary artifact directories.
