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
