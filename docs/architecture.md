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
mxx-exponent-lut         -> mxx-bgg, mxx-dsl, mxx-ir-core, mxx-primitives, mxx-runtime
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

Artifact connections use core-owned `ArtifactCatalog` metadata through
`BuiltGraph::validate_with_catalog`; the DSL does not open stores or execute
graphs. Existing `GraphValueSchema::artifact_input` preserves composite types.
Runtime `ArtifactCheckpoint<D, I>` stores versioned application descriptors and
expected identities, borrowing execution results so callers retain ownership of
resident outputs and staged leases. Reopening requires trusted production,
identity, descriptor-digest, and canonical catalog-digest anchors and a finalized
file-store session. Checkpoint creation accepts only finalized lazy artifact
outputs whose names, descriptors, family cardinalities, and handle indices match
the manifest; zero-length families intentionally have no handles.
Producers use `execute_in_session_with_config` with an explicit fresh execution
nonce; ordinary `execute_with_config` writes artifacts but does not create the
finalized session required by this durable checkpoint contract.
It validates metadata, not private payload integrity or lattice security.
Application importers retain their parameter, key, level, and Program checks.
The estimator's `measure_graphs` collects all caller-selected stages before
measurement and reports them in order; hardware and execution remain separate.

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
The RNS-BGG PRF round-trip integration warms representative primitives and host transfers
at every actual CRT level, then measures one full preprocess/export/reopen/online execution.
`MXX_EXPONENT_LUT_BENCH_WARMUPS` counts primitive warmups, not full round trips.
For the RNS-BGG one-output PRF test, `MXX_EXPONENT_LUT_BENCH_STAGE=estimate`
runs GPU-calibrated estimates of that exact producer and production-online
graph instead of the round trip. Selector setup is charged to the single
output; the online correctness oracle is excluded. The common estimator now tracks runtime storage boundaries across
Family, capture, subgraph, and sequential-loop values. It measures raw-RNS RAM
staging/reloading and compact artifact encoding/writing/reading/decoding with the
production backend and artifact store, then adds their invocation-weighted wall
times to `total_time_seconds`. GPU-resident edges have no added transfer charge;
`total_work_seconds` remains primitive GPU work. The runtime and estimator share
the matrix-Family staging rule and artifact codecs.

`dataflow` and `transfers` report the extra costs and counts separately. CPU IR
dispatch is a measured scalar-node proxy, not a full protocol execution trace.
Artifact import calibration includes public content-hash verification (also
charged conservatively to private imports). Measurements sum boundary costs;
they do not predict overlap with unrelated operations, cold-cache storage
behavior, or contention in a complete execution. Large transfer shapes use a
VRAM-bounded representative wave and round the final partial wave upward.
`MXX_EXPONENT_LUT_BENCH_ITERATIONS` controls measured repetitions in this mode.
No protocol-specific selector-loop sampling is used.

Exponent-LUT addition uses ordinary parallel reduction trees. RNS-BGG preprocessing uses
`MXX_EXPONENT_LUT_PREPROCESS_BATCH_SIZE` (positive integer, default 256) for complete selector
and branch batches. Selector generation runs inside the consuming loop, once per active
selector; all output sums share that ciphertext before the batch is released. Output
accumulators remain in RAM across batches. Their matrix payloads use raw-RNS host
snapshots, including intermediate reduction families.
Compact serialization remains the durable-export format. GPU RNS transfers expose D2H
start/completion separately, and H2D sources are owned by the existing event-based pinned
memory reclaimer. This permits a previous output store and next input load to overlap.
Fleet snapshots pipeline two shards and reuse two pinned buffers, bounding transfer
scratch independently of the number of column chunks. Mask and terminal sampling use
matching branch batches, and public preparation finishes each batch before advancing its accumulated sum.
Completed scalar exports are written immediately and retained as artifact handles, including
members projected from batch families. Remaining live input values and output accumulators
are separate from the bounded per-batch working set.

### `mxx-gadgets` and `mxx-bgg`

`mxx-gadgets` owns BGG-independent circuits and reusable circuit gadgets.
`mxx-bgg` owns BGG+-specific keys, encodings, sampling, evaluation, lookup, decoding, artifacts,
slot transfer, and refresh. Both build executable graphs through `mxx-dsl`.

`mxx-exponent-lut` owns the native mod-Q Program traversal and the
`ExponentLUTEncodingScheme` interface. Its BGG and RNS-BGG implementations supply
concrete keys, encodings, preparation, and gate operations. Its `PublicKey` and
`Encoding` associated types are scalar graph values with any known-plaintext
metadata. The interface does not expose separate `PublicValue` or
`EncodingValue` associated types. Common Program lowering resolves captures and
indexed inputs and constructs one DSL `parallel` for the complete Family body;
primitive adapters receive only scalar values.

`RnsBggScheme::prepare_program` provides program-driven scalar setup: applications
supply native plaintext exponents, input levels, a payload secret, and a seed/domain
root. The scheme derives dataflow levels and per-instance RHS/LUT material, including
distinct RHS packages when a declaration is consumed at multiple levels. Typed
`ProgramBindings::gate_rhs` overrides apply only to root multiplication gates;
family bindings retain their structural capture/member contract. The scalar setup
API explicitly rejects structural families rather than expanding them on the host.
`RnsBggPreparedProgram` exports prepared caches and private input contributions as
one typed descriptor bundle, and imports it against an independently trusted
`RnsBggProgramExpectation`. Online evaluation receives the mask secret and constructs
encoding/helper bindings internally. Applications still own graph build/validation,
artifact runtime execution, security/noise checks, and explicit test-only oracle
outputs; no secret-key oracle is included in the production export bundle.

The `mxx-exponent-lut` crate root exposes only the typed application facade:
`NandCircuit`/`CompiledNandCircuit`, `SparseLwrPrfProgram`, phase artifacts,
`EncodingSecrets`, and `ExpectedEncoding`. Frozen binding maps, gate-instance
identifiers, family ranges, and durable artifact namespace/token details remain
available under the explicit `advanced` namespace (and their owning modules) for
backend adapters and migration tooling; ordinary callers do not need to construct
those low-level values.

`program::ProgramValue<S, F>` is the common scalar/family carrier at application
boundaries. `FamilySource` describes shared versus indexed references for both
state wires and RHS operands. Program wire, RHS, and helper family bindings
record their represented range, so packed active inputs need no padding.
DSL Family schemas preserve shared versus indexed storage per graph field.
`PreparedProgram` owns cache export/import and the generic
`PreparedProgramDescriptor` schema, including scalar/shared caches, indexed
family caches, public keys, Program identity and active ranges. Resolved setup
supplies the scheme binding; import compares it with the trusted descriptor
identity before restoring graph ports. Backend adapters define the concrete
GraphValue schemas, not the Program artifact traversal. This API does not
require a new cryptographic artifact trait.
Both RNS-BGG and BGG instantiate this transport in their file-artifact tests.
BGG cache schemas serialize structured gate identities as ordered entries, and
its public-key schema preserves the public-plaintext flag. The shared-RHS BGG
consumer imports the complete prepared Program instead of injecting a manually
reopened decomposition. BGG's production refresh setup still has separate
artifact orchestration to migrate.
Captures retain their original handles; body-local preparation leaves the loop
through explicit `Family<Cache>` outputs. Symbolic noise captures also retain
their source identity and reductions account for shared-source multiplicity.

Direct refresh schedules scalar terminal operations in the common application.
Preparation returns the online cache separately from its raw public result;
only the cache is persisted per member. The application reduces raw members and
logical terms before the final refresh ModDown. RNS-BGG material retains prepared
ciphertext lifts and imported evaluation consumes those cached fields. Generic
GraphValue schemas drive direct-slot term artifact transport. Standalone RNS-BGG
Family readout also uses the same terminal scheduler. Independent projection
is fused into its member loop; aggregate projection runs after raw reduction.
Its adapter retains output mode/range metadata, while member caches and
artifact transport use the same scalar prepared records. Shared LUT algebra lives in
`crates/exponent-lut/src/utils.rs`; RNS-BGG does not call the BGG encoding implementation.
The shared interpreter dispatches `Add`, `MulPublic`, `MulPrivate`, and explicit
`Lut` gates for both scalar and family scopes. `private_mul` emits `MulPrivate`;
LUT declarations require a table ID. BGG caches one gadget decomposition for a
direct product. RNS-BGG caches the exact low-level gadget decomposition transported as bounded
digits to the input modulus, and performs one public block ModDown and one
online row ModDown.
The ordinary Program-driven PRF route uses these direct operations.
Within a family, a shared fixed RHS uses one cached BGG decomposition or one
cached RNS-BGG bounded decomposition outside the indexed products. Artifact descriptors retain
this member-sharing distinction independently of sharing across PRF outputs.
Program preparation and binding require matching exponent layouts and ring
dimensions, plus plaintext metadata for every public input. Lowering relies
on these contracts; it does not query the backend for layout compatibility or
public-attribute availability at entry or before each multiplication.

BGG's noise adapter owns both the approximate gadget's secret-weighted
reconstruction distribution and its assignment to gate/readout residual terms.
Setup supplies public parameters; the average model preserves its explicit
independent-residue heuristic and Var(2e) convention. The common simulation
reports accept a Program-result type parameter without a backend default. The
BGG-specific snapshot and its Program/readout/refresh equations live in
`bgg/noise.rs` as `BggNoiseSnapshot`. Common simulation no longer imports a
backend; its Program error boundary carries a diagnostic string mapped by the
backend adapter. Production setup still has concrete BGG dependencies.

BGG preprocessing stores gadget decompositions as generic preimage artifacts;
RNS-BGG uses rectangular polynomial blocks, bounded preimages, and exact RNS level transitions. Their online
encoding operations consume prepared public data. The primitives, IR, DSL,
and runtime layers own executable basis conversion, not application semantics.

The sparse-LWR accumulation interpreter uses scheme-associated values and
helpers. The common PRF owns PBC active block ranges, used by both ordinary
Program bindings and cached batch preparation. `ExponentLUTReadoutScheme` exposes value readout with structural batch
carriers and prepared witnesses. RNS-BGG public preparation and artifact operations
live in `public_key.rs`; sampling and online operations live in `encoding.rs`.
Its shared relation/cache carriers and capability implementations live in
`mod.rs`, alongside `noise.rs` and `gpu.rs` as the remaining backend files.
BGG uses the same five-file layout: shared declarations and capability
implementations in `mod.rs`, public preparation and lazy artifact imports in
`public_key.rs`, and online operations in `encoding.rs`. Intermediate labels
reuse `BggEncodingFamily` and output batches are ordinary `Family<Mat>` values,
without separate PRF label/output wrappers. Each file groups a type's inherent
methods into one implementation block; trait implementations remain separate. Shared
manifest-based storage accounting belongs to `noise_refresh/simulation.rs`.
The optional
`ExponentLUTTerminalScheme` exposes raw row-output terms and finalization.
Its common `PreparedTerminal` evaluator builds an ordered balanced sum and
finalizes once, preserving the RNS-BGG aggregate-before-switch requirement.
The common readout module owns RowLutTable<C>, the public exponent-indexed
table of polynomial rows. Its block-component count is static; RNS-BGG consumes
RowLutTable<2> through its terminal samplers, with rectangular polynomial rows
inside each block when routing to a reusable state. Table identity and selected
coefficient bounds are independent of ciphertext representation. ValueReadout
also constructs weighted centered rows and their signed-domain digit bound;
backends supply the target modulus and consume the resulting public table.
RowLutTable also owns the public monomial/automorphism sieve sum over lowered
graph-value rows, preserving the structural balanced reduction and reusing
rows across branches.
RNS-BGG implements scalar and structural-family native digit readout. Fresh terminal
row targets contain `w*(f-1/2)`, and public offsets produce `w*f` in the first
native half and `w*(1-f)` in the second. The inverse of two is inside the fresh
target, so encoding errors remain unscaled, including odd errors. Public matrices
and completed encodings are never divided by two. The adapter preserves the
full native subring width `N/2^m` for stride `2^m`.
Indexed targets route base-p digits into either block, any column, and any coefficient
before sampling fresh RHS errors. RNS-BGG readout weights use the same shared/indexed
enum for runtime values and descriptor metadata; only runtime values retain
indexed graph rows. Descriptors store those rows separately as public artifacts,
with the coefficient bound belonging to the indexed variant. Scalar and family readout artifacts persist
prepared public matrices, terminal branches, and affine offsets. Scalar, shared
family, indexed family, and summed offsets are imported directly; imports do not
rebuild weight multiplications or public aggregation. Raw terminal-family mode supports combining data, mask and fresh
error families before the single outer ordinary ModDown.
`ExponentLUTRefreshScheme` refreshes the selected scheme's own `Encoding`.
The base scheme exposes encoding dimensions from public-key metadata.
`RefreshTranscriptTopology::new` uses these dimensions to count mask and
shared fresh-error PRF outputs; both schemes contribute their actual
gadget-dependent key columns. RNS-BGG resumable states have `2D` columns,
where `D` is the next level's exact per-prime radix digit count; only the
terminal chain endpoint has two polynomial columns.
`noise_refresh/refresh.rs` contains no concrete backend types; matrix routing
uses the segmented reduction in `utils.rs`, and backend adapters assemble their
own cryptographic wires. Its common online evaluator owns homogeneous slot
grouping, structural parallelism, mask/fresh-error addition before a single
switch, decoder application order, and canonical CRT recomposition order.
The witness path also uses `prepare_refresh_public_routes` for preprocessing:
it applies the prepared route witnesses, reduces the canonical ragged mask
ranges and shares the fresh PRF source across CRT slots. BGG public routing
reuses the decompositions persisted for online evaluation instead of rebuilding
and decomposing the route targets. The segmented reducer accepts a graph-value
carrier and an addition operation, preserving structural odd-width reductions.
BGG's `route_preimage_family` owns the gadget-shaped target construction and
accepts source schema metadata plus one scale family. Corrected scales already
contain digit weights and are passed with base one. Public and online aggregation
use `RefreshPrfLabelIndex` as their canonical range source; online material no
longer duplicates those dimensions in a separate coverage record. The index
retains RefreshTranscriptTopology itself plus cached mask prefixes. Setup output
counts and index construction use the topology cardinality, counting fresh
labels once across CRT slots.
Both phases call the same graph-value routing traversal. The witness capability
supplies the online row product through `route_refresh_row`; the backend builds
public digit offsets, which the common online aggregator adds after reduction.
`RefreshPrfFamilyMaterial<V>` carries the scheme's row type. The common refresh
module no longer imports concrete `Mat`, `Preimage` or `Ring` carriers. BGG's
half-offset vector construction remains in `bgg/mod.rs`.
`RefreshVerification<V>` retains the backend's `GraphValue` carrier and the
exact scalar residual operands used by threshold decoders. Relation arithmetic
and component extraction belong to the backend: BGG's builder is in `bgg/mod.rs`;
the common container can export RNS-BGG polynomial blocks without concatenating or
slicing them. This graph boundary does not itself prove the supplied relation.
BGG supplies the individual arithmetic operations in `bgg/mod.rs`;
its production refresh entrypoint uses this evaluator.
PBC trusted bits and runtime input materialization remain application-owned.
BGG selector sampling lives on `ExponentLutEncodingSampler::sample_pbc_selectors`
in `bgg/mod.rs` and returns `Family<Mat>` directly; no selector-family wrapper
or concrete sampler entrypoint is exported by the common PBC module.
The three refresh contracts are `ExponentLUTRefreshScheme` for common public
and encoding arithmetic, `ExponentLUTRefreshWitnessScheme` for witness-based
anchor and online decoder-base preparation, and `ExponentLUTRefreshDirectScheme`
for direct decoder issuance and weighted terminal/digit preparation. BGG supplies
the witness capability; RNS-BGG supplies the direct capability. Neither implements
dummy operations for the other decoder construction. The shared application
continues to own preprocessing, online preparation and evaluation scheduling.
RNS-BGG constructs the output-key-shaped decoder
`s*(A_sum-H*A_out)+e_dec`, bound to the combined masked target.
`PreparedDirectRefreshSlot` prepares and evaluates scalar or structural terminal
families through traits, sums raw terms, applies ordinary ModDown, subtracts the
decoder, and only then extracts a CRT residue. RNS-BGG rounds each masked channel at spacing `H = q_ref / prime` before
reconstructing the output with exact residue extension and public CRT
idempotents. Its data targets contain `H * p_ref * mu * t * G_ref`, so the
refreshed state can enter the next private product or LUT directly. The common
preterminal modulus retains every output prime; recomposition does not restore
consumed modulus levels. Runtime parameter registration includes
`RnsBggParameters::refresh_residue_parameters` for the selected output level.
`RefreshDigitPlan` computes mask and fresh counts independently using exact
base-p support widths. Its fixed-point search includes the digit-dependent
exposed-error bound. Fresh digit inputs are shared across CRT channels while
mask labels are channel-specific. RNS-BGG's digit sampler generates structural
fresh targets weighted by `p^j` or `H*p^j`, plus exact public distribution offsets.
`PreparedDirectRefresh` in `noise_refresh/refresh.rs` owns each slot together
with its public affine offset and connects the complete
preprocessing, online decoder preparation, digit routing, evaluation, and
recomposition sequence using these traits. `PreparedSparseLwrPrf` in
`noise_refresh/prf.rs` connects structural selection steps through
`ExponentLUTFixedRhsScheme` from the encoding interface. The application
explicitly composes the publicly weighted fixed RHS and then prepares the
private product. Both BGG and RNS-BGG delegate structural weighted-family summation
to the common graph-value utility. BGG decomposes each composed GSW family
in public preparation and keeps `Family<Preimage>` for online external products;
RNS-BGG decomposes at the output level, transports bounded digits to the input
level, and applies its prepared product followed by ModDown. Neither adapter owns the PRF
step traversal.
The capability has no PRF or one-hot-specific methods. It also exports/imports
one fixed-product cache and exposes its concrete descriptor and scheme binding.
`PreparedSparseLwrPrfDescriptor` owns the ordered step descriptors and a single
scheme binding; common PRF export/import authenticates that descriptor against
the caller's expected identity and restores each step. RNS-BGG no longer has a
separate PRF descriptor or a specialized PRF transport implementation. Public label vectors are bound in
preprocessing. RNS-BGG exports and reopens both prepared PRF batches and complete
refresh bundles through the runtime artifact store. Online evaluation performs
encoding ModDown but does not rerun public aggregation or public ModDown.
The BGG witness-based setup driver retains its backend-specific sampler and
artifact types; it uses the common Program and refresh arithmetic evaluators.
Its prepared readout uses one `GraphValue` record for three preimage families
and one scalar preimage. A backend schema constructor derives helper ordering,
port shapes and coefficient bounds from identity-bound setup geometry. Setup
checks the produced schema and uses the same schema for artifact import and
manifest validation, with one prefix and four typed ports. It no longer
reconstructs each readout preimage or stores a duplicate helper list.
`bgg/mod.rs` owns `RefreshCompiler`, its bound setup/result types and B*K decoder
linkage. The anchor constructor always builds its target product; decoder
bindings retain the exact target identity.

RNS-BGG uses distinct `RnsBggPayloadSecret` and `RnsBggMaskSecret` types. Input
preprocessing computes a private `RnsBggInputMaterial` using only the payload
secret; online input generation adds the mask-secret term and fresh error.
Ordinary/family/terminal RHS samplers take only the payload-secret type. The
public mask sampler and prepared scalar/family/Program caches are independent
of the mask secret. Online preparation generates separate `RnsBggOnlineMasks`
from the public matrices and `s`; gates sharing a bank reuse its rows.
Scalar, family, and terminal evaluation consume these helpers explicitly.
Linear Program caches retain scalar/family layout once and use the prepared
operation as the authoritative Add/MulPublic tag; both online gate callbacks
share the same lowering path.
Serialized linear operation metadata carries the RHS identity only in its
Add/MulPublic variants. Reduction has no RHS field and requires an input family
range; its output is scalar. Input/output ranges derive from the operation.
These bindings and their transport live beside Program cache artifact dispatch
in `rns_bgg/public_key.rs`. Input-material and scalar/family LUT artifact
bindings share that owner, as does the remaining direct-refresh adapter after
slot transport moved to the common refresh layer. The separate
`linear_artifacts.rs`, `artifacts.rs` and `refresh_artifacts.rs` modules are removed.
Basic block imports live beside the RNS-BGG block types and
use `GraphValueSchema` without depending on Program cache transport.
Prepared LUTs own their binding metadata; Program cache entries refer directly
to that metadata for helper preparation and online checks, rather than retaining
a second scalar/family binding copy.
Public artifact exporters persist the mask matrices and bindings, not the
online rows. Input artifact transport preserves private payload contributions
separately from public keys. Public RNS-BGG blocks use the same `GraphValue` flattened
ports as structural families; schema-based imports reopen those ports without
backend-specific row/column artifact naming. `DslContext::public_outputs` owns
ordered public field emission for scalar and structural values; RNS-BGG direct,
linear and LUT-branch exporters supply their named fields to that common path.
Scalar/family terminal and native digit readout operations follow the same
RNS-BGG public-preparation versus online-evaluation ownership split. The former
`family.rs`, `program.rs`, `program_artifacts.rs`, and `readout.rs` modules are
removed rather than retained as compatibility namespaces. Prepared terminal
families derive binding metadata from their scalar member cache rather than
keeping a second copy; this performs no polynomial computation. Terminal material
uses the common `ProgramValue` scalar/Family carrier directly. Indexed sampling
returns complete scalar material from one structural loop, including captured
mask handles and cached bounded decompositions, without a manual port adapter. Common terminal
scheduling handles both standalone readout and direct refresh. Independent
outputs finalize inside the member loop; aggregate outputs finalize once after
raw reduction. Terminal stage and batch artifacts preserve the finalized public
output, keeping public aggregation and switching in preprocessing.
`PreparedDirectRefreshSlot` owns ordered target/offset/terminal artifact transport
through generic `GraphValue` schemas. Each serialized slot keeps its level and
terms together, replacing parallel arrays. Cached terminal values reopen without
public aggregation or switching. Refresh label routing also owns structural
slicing of public and encoding digit families; schemes do not implement range
selection. The fresh group is selected once and reused across CRT channels. See
`docs/plans/exponent-lut-rns-bgg-implementation-plan.md`.

The BGG Program interpreter traces worst-case bounds and average-case variances
per wire; the existing noise-refresh application applies that interpreter to the LWR PRF and checks the
decoder and mask thresholds. This application-specific simulator does not
depend on the removed generic noise-simulator crate or replace the generic
operational checker used by other applications. RNS-BGG has a separate deterministic
noise interpreter over the same Program traversal; its parameter audits run
at setup time using public payload contracts. Both reports retain a common
`NoiseDependencyGraph`, recorded by a decorator during that same traversal.
The flat arena preserves repeated wire references, RHS/helper binding reuse,
broadcast ancestors and active family reductions without expanding members.
Binding identities do not assert sampler independence; distributions and
per-source covariance transfers still need explicit model contracts.
`RnsBggParameters::plan_refresh` takes data and PRF noise values produced with
the selected parameters. They may come from separate Program traces or an input
sampling contract. It derives their
common terminal level and bounds without separate copies in the workload request,
solves mask counts against the actual emitted-digit error recurrence, audits
fresh support and next-state scale, and reports separate distinct PRF output,
terminal emission, and artifact-storage counts. These deterministic arithmetic
checks do not prove the related-secret and correlated-mask security assumption.
The refresh plan also retains physical `NoiseMoments` for one CRT-shared fresh
distribution and each centered mask. These are conditional on independent
uniform native digits, retain the nonzero means, and do not change worst-case
acceptance. The common exact interval calculation also supplies BGG refresh
reports with physical payload moments. These remain separate from operational
error variances in `2e` coordinates. BGG snapshots derive payload moments from
the authoritative base and mask/fresh counts rather than storing mutable
variance copies; changing a digit count cannot leave stale payload statistics.

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
