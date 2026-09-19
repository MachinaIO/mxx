# Exponent-LUT Branch Changes

This document explains the Exponent-LUT work in this branch to a reader who is
new to both the implementation and its history. It also calls out changes
outside `crates/exponent-lut/` that are likely to overlap or conflict with
other pull requests.

## Comparison boundary

The comparison in this document is:

```text
base: main (merge base 48117fdb2d296b1c8102c65ac89d6cb0958ff4db)
head: codex/exponent-lut-update-0917
diff: main...HEAD
```

At this boundary, `crates/exponent-lut/` is entirely new: 102 files and 47,161
added lines. The branch as a whole is much larger because it also transforms
the old root package into a multi-crate workspace and adds or moves the IR,
DSL, runtime, primitive, GPU, estimator, BGG, FHE, and gadget layers used by
Exponent-LUT. Those cross-repository changes are described separately in
[Changes outside the Exponent-LUT crate](#changes-outside-the-exponent-lut-crate).

The current code is the source of truth. Intermediate implementations that
once existed on the branch, including the legacy BGG adapter, compatibility
facades, and older monolithic Program API, are not part of the final design.

## Executive summary

This branch adds a complete application layer for evaluating programs whose
values are encoded as exponents of ring monomials. It provides:

- an immutable, scheme-independent Exponent-LUT Program;
- a single lowering interface shared by public-key preparation, encryption,
  online evaluation, and noise simulation;
- one retained cryptographic backend, RNS-BGG;
- artifact-backed `preprocess`, `encrypt`, and `online_eval` stages;
- deterministic worst-case and probabilistic average-case noise analysis;
- a scheme-independent noise-refresh application built from sparse-LWR PRFs,
  packed Boolean circuits, and CRT reconstruction;
- application adapters for NAND circuits and sparse-LWR PRFs;
- GPU-gated end-to-end tests that dynamically select parameters satisfying
  correctness and a requested lattice-security target.

The final dependency direction is:

```text
application (NAND / sparse-LWR / refresh)
        |
        v
immutable ExponentLutProgram + common stage API
        |
        v
ExponentLUTProgramLowering implementation
        |
        v
RNS-BGG graphs -> mxx-dsl -> mxx-ir-core -> mxx-runtime
                                      |
                                      v
                              CPU or GPU backend
```

The common Program, stage API, and noise-refresh application do not depend on
BGG or RNS-BGG details. RNS-BGG is an adapter implementing their contracts.

## Conceptual model

### What an Exponent-LUT value is

The application represents a logical value `x` by a monomial `X^x` in a
negacyclic polynomial ring. This changes the useful primitive operations:

- multiplying monomials adds their logical exponents;
- applying a ring automorphism transforms an exponent in a controlled way;
- a prepared private action multiplies an encrypted exponent by a fixed RHS;
- a lookup table routes an encrypted exponent through automorphism and
  switching material to produce another encoded value.

The Program describes those logical operations. It does not contain RNS-BGG
matrices, secrets, graph handles, noise values, or runtime sessions.

### Three different meanings of “input”

The implementation intentionally distinguishes these concepts:

| Input kind | Program representation | Plaintext availability |
| --- | --- | --- |
| Ordinary wire input | `WireId` with scalar/family shape and payload knowledge | Supplied to `encrypt`; a public wire keeps its public payload with its encoding |
| Setup-fixed private RHS | `RhsId`, lowered as `RhsRef` | Supplied only to `preprocess`; never exposed to `online_eval` |
| Public left operand of `MulPublic` | Ordinary `WireId` marked `PublicPayload` | Its public plaintext is paired with its encoding and may be used online |

This separation is security-relevant. In particular, the online graph cannot
recover RHS plaintexts from a `RhsRef`: it contains only an ID and an optional
symbolic family index.

## Program representation

The Program implementation is split by responsibility:

| File | Responsibility |
| --- | --- |
| `src/program/mod.rs` | immutable Program, IDs, declarations, LUT tables, layout, validation, canonical identity, derived metadata |
| `src/program/builder.rs` | checked mutable construction and rollback before freezing a Program |
| `src/program/lowering.rs` | common scalar walker and backend lowering contract |
| `src/program/graph.rs` | graph-family projection, parallel execution, reduction, and `RhsRef` graph schema |

### Immutable identity and opaque IDs

`ExponentLutProgramBuilder` is the only normal way to allocate `WireId`,
`RhsId`, and `LutId`. `finish` validates the body and produces an immutable
`ExponentLutProgram` with a canonical `ExponentLutProgramId`. The identity is
derived from the validated structure, rather than from an allocation address
or a caller-selected name, so it can safely bind checkpoints and refresh
declarations to the exact computation.

The Program owns all authoritative structural metadata:

- scalar versus family shape;
- public versus private payload knowledge;
- explicit LUT input and output representation;
- family ranges and lexical scopes;
- declared outputs and operation order.

RNS-BGG does not maintain a competing wire-type registry. It queries
`wire_knowledge` and `wire_is_coefficient_output`, whose results are derived
from the validated Program and memoized.

### Operations

The body supports:

| Operation | Meaning and validation |
| --- | --- |
| `Add` | Adds two values at the same logical level. Their payload knowledge must match. |
| `MulPublic` | Multiplies a **public exponent LHS** by a public or private RHS. Output knowledge follows the RHS. Operand order is part of the contract. |
| `MulPrivate` | Applies setup-fixed private RHS material to a private input. One `RhsId` must be consumed at one logical level throughout the Program. |
| `Lut` | Applies one resolved `LutTable` to a private input. The table explicitly declares exponent or coefficient output representation. |
| `Parallel` | Lowers a body over a checked symbolic family range without host-side family expansion. |
| `Sum` | Reduces a wire family through a balanced addition tree. |

`register_binary_lut` is only a builder convenience. It flattens the two input
domains into one table using `lhs + lhs_width * rhs`. After construction there
is no special “binary LUT” runtime representation and no `rhs_width` concept
in the evaluator.

### Families and lexical indices

`InputBinding<Scalar, Family>` is the shared scalar/family representation for
declarations and runtime bindings. `FamilyRange` records checked start, count,
and capacity expressions. `FamilyScope` keeps every lexical range together
with its selected index and computes a checked row-major flattened index.

Nested consumers substitute their loop indices into the original family
reference. They do not create new RHS samples or use a naked inner index as a
globally unique member identity. `LoweringSite` carries the current output wire
and complete lexical scope to a backend without putting backend state in the
Program.

### One common walker

Every execution mode uses the same `lower_program` walker and the same trait:

```rust
pub trait ExponentLUTProgramLowering {
    type Wire: Clone;
    type Rhs: Clone;
    type Error: std::error::Error + Send + Sync + 'static;

    fn add(&mut self, lhs: &Self::Wire, rhs: &Self::Wire)
        -> Result<Self::Wire, Self::Error>;
    fn mul_public(&mut self, lhs: &Self::Wire, rhs: &Self::Wire)
        -> Result<Self::Wire, Self::Error>;
    fn mul_private(&mut self, input: &Self::Wire, rhs: &Self::Rhs)
        -> Result<Self::Wire, Self::Error>;
    fn lut(&mut self, input: &Self::Wire, table: &LutTable)
        -> Result<Self::Wire, Self::Error>;
}
```

The walker, not the backend, resolves a `LutId` to `&LutTable`. A LUT callback
therefore handles exactly one gate and does not receive the whole Program.
Prepared helpers and private material live inside the concrete lowerer and are
accessed through `self`. There is no separate `ProgramLoweringTypes`,
`HelperSet`, material trait, or duplicate interpreter.

The trait is implemented for:

- public-key preparation;
- encrypted online evaluation;
- workload/level derivation;
- deterministic and average-case noise tracing.

That sharing prevents the noise model or preprocess graph from silently
interpreting a different Program than online execution.

## Public stage API

`src/api.rs` introduces a scheme-independent orchestration boundary. The four
public entry points are:

```text
preprocess -> encrypt -> online_eval
                 \
                  noise_simulate (same Program, separate symbolic lowering)
```

### `preprocess`

Preprocessing receives the complete immutable Program, setup-fixed RHS
plaintext bindings, a payload secret, and a domain-separated hash key. It:

1. binds ordinary input-key declarations and RHS plaintexts;
2. lowers the Program through the public-key/preparation implementation;
3. produces compact input keys, RHS ciphertexts, LUT switching material,
   public LUT data, and any refresh preparation material;
4. validates and executes the graph once;
5. publishes two typed checkpoints over the same production: one for the
   encryption consumer and one for the online consumer.

It does **not** need ordinary LHS plaintext values. Public LHS plaintexts enter
through `encrypt`, like other ordinary inputs.

### `encrypt`

Encryption receives only `program.encryption_inputs()`, not the Program body,
RHS declarations, or Program identity. It binds ordinary input plaintexts,
payload and mask secrets, and the preprocess checkpoint. It constructs the
input encodings and writes a typed encryption checkpoint.

This restricted projection makes it structurally impossible for input
encryption to become a second Program interpreter.

### `online_eval`

Online evaluation receives the Program and the two artifact checkpoints. It
imports compact prepared artifacts lazily, lowers the Program, optionally
appends scheme-owned refresh work, then executes the resulting graph.

It receives no RHS plaintext map. A private RHS is available only as a compact
preprocess artifact selected by `RhsRef`. The only plaintext consumed by
`MulPublic` is the public LHS payload already paired with that public encoding.

### `noise_simulate`

Noise simulation runs the same Program walker over symbolic values and returns
the backend's report. It has no runtime execution shortcut and no separate
hand-maintained operation list.

### Artifact boundaries

`StageGraph` contains a validated graph, runtime inputs, artifact catalog, and
runtime-owned outputs. `StageArtifacts` wraps a typed checkpoint descriptor.
Large values remain in the `SessionStore`; checkpoints carry identities,
manifests, schemas, and production references rather than copying matrix
payloads into application structs.

Production identities bind artifacts to the Program, parameters, stage,
session, and domain. Consumer-specific checkpoint identities prevent an
encryption descriptor from being reused as an online descriptor even when
both came from the same preprocess execution.

## RNS-BGG backend

The legacy `mxx-bgg` adapter was removed from Exponent-LUT. The crate's final
manifest has no `mxx-bgg` dependency, and `src/lib.rs` exposes only the
RNS-BGG path. The retained implementation is organized as follows:

| File | Responsibility |
| --- | --- |
| `src/rns_bgg/mod.rs` | parameters, errors, shared block types and arithmetic, refresh invocation |
| `src/rns_bgg/public_key.rs` | public-key carrier, preprocess material, `PublicKeyLowering`, preprocess adapter |
| `src/rns_bgg/encoding.rs` | encodings, distinct payload/mask secrets, encryption, online lowering, verification |
| `src/rns_bgg/noise.rs` | workload levels, noise contracts and tracing, parameter manifests and search |
| `src/rns_bgg/gpu.rs` | GPU-only support and sequential tests |

There is deliberately no scheme/engine facade and no catch-all backend
lowering file. Parameters own parameter-derived operations; stage builders own
stage state; the concrete lowerers own their prepared material.

### Parameters and modulus levels

`RnsBggParameters` stores:

- a validated vector of `DCRTPolyParams` levels;
- exact adjacent drop ratios;
- the fresh encoding-error sigma and its hard cutoff;
- the number of logical operations allowed per physical drop.

Every level has the same ring dimension and gadget base. Adjacent moduli must
be equal or strictly nested CRT bases. For a real reduction, the implementation
requires `q_before > q_after` and exact divisibility; equal moduli are an
explicit no-op with ratio one. Partial input bases and unchecked drop vectors
are rejected.

Logical levels are derived once by `RnsBggDerivedWorkload`. Wire levels and
RHS-consumer levels are not stored redundantly in carriers or Program nodes.

### Carriers and knowledge

The public-key carrier is a Program wire plus a 2-by-2 matrix:

```rust
RnsBggPublicKey { wire, matrix }
```

An encoding contains the public key, a two-polynomial ciphertext row, and an
optional public payload:

```rust
RnsBggEncoding { public_key, row, public_payload }
```

`public_payload` is `Some` exactly for a Program wire marked
`PublicPayload`. Private encodings reject a plaintext port; public encodings
reject a missing one. Keeping the payload beside the encoding eliminates
parallel plaintext registries and makes `MulPublic`'s authorization explicit.

`RnsBggPayloadSecret` and `RnsBggMaskSecret` remain distinct because they have
different cryptographic roles. Neither carries redundant Program, context, or
parameter metadata.

### `MulPublic`

`MulPublic(lhs, rhs)` is intentionally non-commutative at the API level:

- `lhs` must have `PublicPayload` knowledge;
- `rhs` may be public or private;
- the result's knowledge is the RHS knowledge;
- the operation requires exponent operands, not coefficient-LUT outputs.

After selecting and validating the RHS-owned target level, public carriers are
aligned to that modulus. The implementation decomposes the **RHS public-key
matrix**, not the public LHS matrix. At a high level, the encoding row follows

```text
c_l * G^-1(A_r) + c_r * mu_l
```

and the corresponding public key follows

```text
A_l * G^-1(A_r).
```

This preserves the expected encoding relation: the first term is the relevant
secret multiplied by the computed public key, while the message term is
scaled by the public LHS plaintext. A private-payload LHS is rejected rather
than routed through an alternate alignment rule.

### `MulPrivate` and next-level RHS material

For an operation evaluated at logical level `l`, preprocessing encrypts its RHS
compact matrix directly at modulus `q_(l+1)`, using the ordinary private
encoding shape based on `G_(l+1)`. Online evaluation then:

1. imports the compact matrix at `q_(l+1)`;
2. decomposes it against `G_(l+1)` immediately before use;
3. centered-extends only the bounded decomposition components to the
   evaluation ring;
4. applies the action and reduces the result to its derived output level.

The dense compact RHS ciphertext is never mod-reduced or extended online.
Preprocessing persists no recomputable decomposition. This keeps artifacts
compact, preserves the intended gadget equation, and bounds the active
working set.

One `RhsId` must have one consumer level, including all members of an RHS
family. A caller that needs the same plaintext at two levels must declare two
RHS IDs; the implementation does not silently clone or deduplicate them.

### LUT evaluation

Each Program LUT is one `LutTable`, even when it came from the binary builder
convenience. The table owns its input layout, values, output width, and output
representation.

For each table branch, the RNS-BGG lowerer constructs the selected
automorphism, imports compact switching material, performs the required
decompositions at the consuming levels, routes the input key/encoding, and
forms a branch result. Branches are combined with `balanced_sum_family`, so
the graph has logarithmic reduction depth rather than a long serial sum.

Names describe their roles (`switching_decomposition`, `routed`,
`switched_public_key`, and `switched_key_decomposition`) rather than using
opaque sigma-prefixed temporary names.

Exponent-output LUTs advance the logical level. A coefficient-output LUT keeps
the operation's special coefficient representation until final output
normalization, where it is reduced to the expected next modulus.

### Compact artifacts and namespacing

The stage boundary stores compact matrices and carrier values. Signed
decompositions use the DSL `Preimage` type but are reconstructed by the
consumer and are not serialized as application artifacts.

Artifact names include the identities necessary to prevent accidental reuse:

- Program and parameter/session identity at the stage boundary;
- wire or RHS ID and family scope;
- evaluation and decomposition levels;
- LUT ID, layout, automorphism branch, and matrix port;
- refresh declaration commitment, CRT slot, PRF label, and role.

This replaces the old public/private artifact classification and the old
`MulPublic` operand-position metadata. Public/private authority now comes from
the Program, while `MulPublic` has one fixed public-left contract.

## Noise analysis and parameter selection

### Symbolic values

`RnsBggNoiseValue` combines:

- a deterministic error and payload L1 envelope;
- payload L2-squared information for convolution gains;
- optional exact rational moments;
- stable source provenance for shared/independent correlation decisions;
- an explicit ledger of average-case heuristics.

The noise evaluator implements the same Program lowering trait. It validates
the same knowledge and level rules as the graph lowerers, records a dependency
graph, and tracks physical event multiplicities for scalar and family nodes.

### Worst-case and average-case modes

Worst-case mode checks deterministic bounds. Average-case mode propagates
means, radii, variances, and provenance, then derives a tail multiplier from a
failure-event budget. Acceptance is never based on treating an unavailable
moment model as zero variance.

The event budget counts physical coordinates and family multiplicities from
the actual trace. Refresh adds its route and transcript events. Tail
corrections are recorded in the report rather than silently changing the
requested failure policy.

For binary-message decoding, the message scale is `q/2` and the correctness
threshold is `q/4`. Tests cover values immediately below and above that
threshold.

### Parameter manifests

`RnsBggPlanRequest` specifies ring dimension, sigma, CRT bit size, depth
bounds, terminal value bound, and operations per physical drop.
`RnsBggParameterManifest` records:

- the request and Program identity;
- commitments to public noise contracts and selected parameters;
- the exact CRT primes and gadget bases;
- the accepted complete trace and its commitment;
- the selected average-case policy, if any;
- an estimated persistent artifact size.

This makes parameter acceptance reproducible and binds execution to the noise
evidence that justified it.

## Scheme-independent noise refresh

`src/noise_refresh/` is an application layer and has no dependency on BGG or
RNS-BGG. Backend-specific work is dispatched through the `NoiseRefresh` trait.
The directory contains:

| File | Responsibility |
| --- | --- |
| `refresh.rs` | immutable refresh declaration, CRT slots, digit plan, labels, backend trait |
| `prf.rs` | sparse-LWR PRF Program, canonical labels, setup and encryption bindings |
| `pbc/` | packed-Boolean selector layout, schedule, artifacts, and validation |
| `setup.rs` | refresh graph validation and simulation input metadata |
| `simulation.rs` | route/transcript gains, worst/average reports, storage estimates |

### Refresh declaration

A `RefreshDeclaration` binds the operation to:

- one source `ExponentLutProgramId` and declared exponent output;
- the checked output graph schema;
- CRT slots and their exact `q -> p -> q_t` divisibility chain;
- a canonical digit plan and PRF label index;
- the target RNS-BGG level;
- the selected noise-moment policy.

Its deterministic commitment is used in every refresh artifact namespace. It
contains no runtime graph handles, secret carriers, or backend matrices.

### Sparse-LWR PRF and PBC selectors

The sparse-LWR application builds a Program whose structural labels identify
the exact prepared selection families. PBC layout IDs, key-instance IDs,
canonical active-cell order, element schemas, and family shape are validated
when artifacts are imported. Private one-hot selector values are not placed in
public metadata.

Each refresh slot receives its own child PRF context. The slot schedule uses
the actual modulus sequence required by the construction, including repeated
`q` levels and the slot's `p` level. Child preparation is fully namespaced by
the refresh commitment, slot, Program IDs, levels, labels, and ports; it cannot
fall back to outer Program artifacts.

The child shares the intended reduced payload and mask secrets with the outer
refresh construction rather than sampling unrelated substitutes.

### Noise gains added for refresh

Refresh simulation accounts for details that were previously easy to omit:

- decomposition width at the actual slot modulus `p`;
- convolution gain from the decoder secret;
- label-specific weighted aggregate gains;
- mask-smudging and fresh-error terms;
- the complete physical route and transcript event count;
- exact source, `p`, `q_t`, and target-product modulus checks.

The refreshed output report is attached back to the main Exponent-LUT trace,
so correctness is checked at the true destination modulus rather than against
an unrelated nominal level.

## Application adapters

### NAND circuits

`NandCircuit` validates a topologically ordered Boolean NAND graph and provides
an independent cleartext evaluator. The compiler packs a three-register state
as `R0 + 2*R1 + 4*R2`. Each round uses one flattened LUT over an eight-valued
state and one original Boolean RHS input; an evaluated intermediate is never
repackaged as setup-fixed RHS material.

`CompiledNandCircuit::plaintext_inputs` produces the exact ordinary-input and
RHS maps required by the common API. The public adapter therefore remains
scheme-independent; RNS-BGG appears only when selecting concrete stage
builders.

The integration target builds a complete binary NAND tree of configured depth:
it has `2^d` leaves and one output. It rejects configurations whose expanded
LUT-round count exceeds the checked budget.

### Sparse-LWR PRF

The PRF adapter exposes immutable profile/layout metadata, canonical output
identities and labels, structural selector families, and prepared artifact
descriptors. The end-to-end test computes the sequential `D*T` exponent and
readout independently of the encrypted graph.

### Noise-refresh application

The refresh integration constructs and validates preprocessing, encryption,
and online graphs, independently computes the clear PRF state/readout, and
verifies the refreshed output with caller-owned secrets. Correctness-oracle
calculations are kept out of the graphs passed to the estimator.

## Integration tests and configuration

There are exactly three GPU-gated integration targets:

| Target | Test | Purpose |
| --- | --- | --- |
| `tests/nand_circuit.rs` | `nand_circuit_end_to_end` | dynamically sized binary NAND tree |
| `tests/lwr_prf.rs` | `lwr_prf_end_to_end` | sparse-LWR Program and independent readout |
| `tests/noise_refresh.rs` | `noise_refresh_end_to_end` | full refresh and verification path |

Every test is marked `#[sequential(dcrt_runtime)]` because the DCRT GPU
runtime owns process-global resources. Each runs the production order:

```text
parameter search -> noise acceptance -> security estimate
-> graph estimate -> preprocess -> encrypt -> online -> independent verification
```

Defaults and overrides are deliberately application-namespaced:

- `MXX_EXPONENT_LUT_NAND_*`
- `MXX_EXPONENT_LUT_LWR_PRF_*`
- `MXX_EXPONENT_LUT_NOISE_REFRESH_*`

The NAND depth is controlled by `MXX_EXPONENT_LUT_NAND_DEPTH`. Candidate
searches default to 40 CRT bits and derive gadget base bits as
`ceil(CRT_BITS / 2)`, subject to backend limits. Inclusive logarithmic
ring-dimension bounds and maximum CRT depth are configurable.

Average-case noise is the default. Setting the corresponding
`*_NOISE_MODEL=worst` selects deterministic worst-case acceptance;
`*_FAILURE_EXPONENT` configures the average-case event budget. Every accepted
candidate must also reach the positive `*_SECURITY_BITS` target through
`lattice-estimator-cli`. There is no successful “skip estimator” mode.

Shared environment parsing, candidate enumeration, security invocation,
Program audit, and balanced-family helpers live in public `src/utils.rs`, not
in a test-only `tests/common` module.

## Lean sources

The branch contains Lean models and Rust graph-export/certification sources for
Exponent-LUT, readout, modulus conversion, decomposition, routing, and refresh.
They document a substantial earlier verification effort.

However, the current `src/lib.rs` does not declare `mod lean`. These sources
are intentionally disconnected from the compiled crate while the Rust/RNS-BGG
implementation is stabilized. They should be treated as deferred work, not as
evidence that the final Rust path is currently Lean-certified. Integrating or
removing them is a separate change.

## Removed legacy surfaces

Although `main` sees only the final added crate, the branch history passed
through broader APIs. The final state intentionally does **not** contain:

- an Exponent-LUT BGG backend or `mxx-bgg` dependency;
- `RnsBggScheme`, another engine/context facade, or scheme selection;
- compatibility or `advanced` re-export namespaces;
- a monolithic `program.rs`, separate `layout.rs`, or duplicate scheme walker;
- public/private artifact-role enums;
- a `MulPublic` operand-position field;
- persisted recomputable RHS/LUT/refresh decompositions;
- RHS plaintext access during online evaluation;
- `ProgramLoweringTypes`, `HelperSet`, or material-provider traits;
- a backend-specific NAND implementation;
- skipped-stage integration switches or alternate legacy execution paths;
- format version bumps. Existing versioned runtime formats remain version 1,
  and new application descriptors do not invent a version field.

## Changes outside the Exponent-LUT crate

This is the most important section for merge planning. Relative to `main`, the
branch is not an isolated application-crate addition. It includes a repository
architecture migration and large foundational implementations. Another PR
touching any of these areas may conflict textually, semantically, or both.

### 1. Root package converted to a workspace — very high conflict risk

`Cargo.toml` changes from one root package named `mxx` to a resolver-3
workspace. It adds these members:

```text
ir-core, dsl, runtime, bench-estimator, bgg, exponent-lut,
primitives, gadgets, func-enc, we, io, fhe
```

Shared package metadata and dependencies move into `[workspace.package]` and
`[workspace.dependencies]`. The root features, dependencies, build script,
bench declarations, and dev-dependencies disappear or move to owning crates.
`Cargo.lock` is correspondingly rewritten.

Why this conflicts:

- any PR editing root dependencies/features or adding a crate must be rebased
  onto the workspace layout;
- old commands targeting package `mxx` no longer identify the same unit;
- dependency ownership and feature propagation now occur per crate;
- lockfile conflict resolution must preserve actual crate manifests, not just
  choose one side of the generated file.

Files most likely to conflict are `Cargo.toml`, `Cargo.lock`, `.cargo/config.toml`,
the moved root `build.rs`, and every newly introduced crate manifest.

### 2. Monolithic `src/`, `tests/`, `cuda/`, and `benches/` split — very high conflict risk

The previous root implementation is redistributed among crates. The diff
contains roughly 88,000 deleted lines under old `src/`, 15,000 deleted lines
under old `tests/`, and 8,000 deleted lines under old `cuda/`, together with
new or moved implementations under `crates/`.

This is more than a path-only rename. Module paths, crate visibility,
dependencies, features, public exports, tests, and build ownership change at
the same time. A PR based on old paths may apply as a delete/add conflict or,
worse, apply cleanly to code no longer compiled.

When porting such a PR, first identify the new owner from
`docs/architecture.md`; do not recreate an old root facade solely to make the
patch apply.

### 3. `mxx-ir-core` added — very high semantic conflict risk

`crates/ir-core/` adds the common typed IR:

- expressions and symbolic integer evaluation;
- graph, node, wire-type, and constraint definitions;
- graph validation;
- artifact catalogs and production identities;
- protocol declarations, bundles, and specifications;
- inventory/liveness descriptions;
- Lean export structures and fixtures.

Exponent-LUT relies on stable `IntExpr`, `WireType`, graph validation, artifact
schemas, and `ProductionId` behavior. PRs that change expression
canonicalization, family scopes, graph validation, artifact identity, or wire
schemas can alter Program validation and checkpoint compatibility even without
touching `crates/exponent-lut/`.

Likely conflict paths include `crates/ir-core/src/{expr,graph,validate,artifact,inventory}.rs`
and `crates/ir-core/src/protocol/`.

### 4. `mxx-dsl` added — very high semantic conflict risk

`crates/dsl/` adds graph-construction types used directly throughout
Exponent-LUT:

- `DslContext`, `Ring`, `Mat`, `Preimage`, `Bytes`, and `Int`;
- typed `GraphValue`/`GraphValueSchema` flattening;
- scalar and family graph operators;
- symbolic `parallel`, family packing/projection, captures, and subgraphs;
- arithmetic, automorphism, decomposition, sampling, hashing, and artifact
  input/output constructors.

The Exponent-LUT common walker assumes one DSL parallel body for a family and
uses `Preimage` as the semantic type of signed decomposition results. Changes
to family capture semantics, graph-value port ordering, decomposition nodes,
artifact constructors, or ring conversion therefore require coordinated
updates in RNS-BGG and noise-refresh.

### 5. `mxx-runtime` added — very high conflict risk

`crates/runtime/` is the largest external dependency in this branch. It adds:

- validated graph execution;
- memory and file-backed artifact stores;
- manifests, checkpoints, lazy import, and session aliases;
- CPU polynomial execution;
- GPU DCRT execution, compilation, scheduling, admission, inventory, liveness,
  transfer, and prepared-storage management;
- compact matrix/preimage codecs and raw-RNS staging;
- execution observers, resource accounting, and result leases.

Exponent-LUT's stage API assumes that a preprocess execution can publish two
typed checkpoints over one production, that artifacts remain lazy in the
store, and that online results retain runtime leases. It also relies on exact
artifact schema/production validation.

GPU-side changes are especially coupled: family waves, live-value release,
compact artifact import, native matrix ownership, prepared scratch admission,
and asynchronous host transfers determine whether large LUT/selector batches
fit without changing their semantics.

Any runtime PR touching checkpoint formats, `SessionStore`, execution result
ownership, artifact codecs, family scheduling, or GPU resource admission needs
an explicit integration review. Do not resolve those conflicts by bypassing
the common runtime in Exponent-LUT.

### 6. `mxx-primitives` and CUDA moved and expanded — very high conflict risk

The former root primitives and CUDA tree move into `crates/primitives/` and are
substantially extended. Relevant additions include:

- DCRT parameter and polynomial operations;
- exact modulus conversion/reduction and centered extension;
- signed matrix decomposition and compact preimage support;
- automorphisms, NTT, sampling, serialization, and trapdoor operations;
- small-RHS kernels and batch matrix arithmetic;
- GPU runtime/admission primitives, typed storage slots, event tracking, and
  asynchronous transfer support.

RNS-BGG assumes exact nested modulus bases, a stable signed decomposition
convention, and correct preservation of preimage semantic tags through
artifact serialization. A superficially compatible change to “small
decomposition,” digit ordering, centered representatives, or modulus
conversion can invalidate its equations and noise analysis.

CUDA and GPU-runtime PRs will also encounter path moves from `cuda/` to
`crates/primitives/cuda/` plus large edits to matrix headers and kernels. Such
conflicts should be resolved by preserving the current primitive contract and
then revalidating RNS-BGG, not by maintaining duplicate old/new kernels.

### 7. `mxx-bench-estimator` added — high conflict risk

`crates/bench-estimator/` adds dataflow estimation, GPU primitive calibration,
bounded GPU admission, batch measurement, and harness support. Exponent-LUT's
GPU integrations estimate the exact preprocess, encryption, and online graphs
before executing them. Artifact boundaries, family staging, raw-RNS transfer,
and compact codec costs are included separately from primitive GPU work.

PRs changing estimator node coverage, liveness, batch admission, artifact
transfer accounting, or output interpretation can change whether a parameter
candidate or benchmark is accepted without changing cryptographic code.

### 8. `mxx-gadgets` added and old circuit/gadget code moved — high conflict risk

`crates/gadgets/` becomes the owner of BGG-independent circuits and reusable
gadgets. It includes Boolean circuit construction/lowering, polynomial
circuits, nested-RNS arithmetic, modulus switching, NTT, convolution,
FHE/PRG gadgets, decoder and input-injection support, plus GPU tests.

This overlaps heavily with PRs based on old `src/circuit`, arithmetic,
decoder, or noise-refresh paths. Exponent-LUT's own NAND adapter is an
application compiler, not a replacement for these reusable gadgets. Shared
functionality should move to the owning gadget crate rather than introducing
an application dependency in the opposite direction.

### 9. Standalone `mxx-bgg` added, but Exponent-LUT no longer depends on it — high conflict risk

`crates/bgg/` contains the general BGG+ implementation: public keys,
encodings, circuits, Boolean lookup, slot operations, tall encodings,
commitment/opening code, and its Lean model.

This crate is still part of the workspace for other consumers, but the final
Exponent-LUT manifest deliberately removed its `mxx-bgg` dependency. A merge
must not restore the deleted Exponent-LUT BGG adapter or place application
specific NAND/PRF code into `mxx-bgg`. General BGG changes and RNS-BGG
Exponent-LUT changes should remain separate.

### 10. `mxx-fhe`, `mxx-we`, `mxx-func-enc`, and `mxx-io` added — medium to high conflict risk

The branch also introduces:

- `mxx-fhe`: BGV, Ring-GSW, parameter utilities, and GPU round trips;
- `mxx-we`: witness-encryption/DiamondWE graph construction and proofs;
- `mxx-func-enc`: a small functional-encryption crate boundary;
- `mxx-io`: the repository's I/O crate boundary.

These are not part of Exponent-LUT's retained dependency chain, but they share
the new primitives, DSL, IR, runtime, workspace dependencies, and GPU
infrastructure. Their presence greatly increases lockfile and foundational API
conflict surface. Changes to them should not be pulled into an Exponent-LUT
conflict resolution unless a shared contract actually requires it.

### 11. Repository documentation, CI, and agent/runbook changes — medium conflict risk

The branch rewrites or adds:

- `README.md` and `docs/architecture.md`;
- many design/implementation plans under `docs/`;
- `.github/workflows/ci.yml`;
- `.gitignore` and `.cargo/config.toml`;
- `AGENTS.md`, `BUILDER.md`, `GPU.md`, and `REVIEWER.md`;
- Runpod/Codex skills and operational scripts.

These changes encode the new workspace commands, crate ownership, GPU test
rules, review procedure, and remote benchmark workflow. They commonly conflict
with process/documentation PRs even when production Rust code does not.

Only normative current architecture and runbook statements should survive a
merge. Historical plan documents should not override current code or restore
obsolete compatibility paths.

## Recommended merge strategy

Because `main...HEAD` combines an application with foundational repository
changes, the safest integration order is:

1. establish the workspace split and final crate ownership;
2. integrate `ir-core`, `dsl`, primitives, and runtime contracts;
3. integrate GPU admission/artifact codecs and the estimator against those
   exact contracts;
4. add the final `mxx-exponent-lut` crate without restoring its deleted legacy
   BGG facade;
5. port only genuinely shared fixes from competing PRs into their new owning
   crates;
6. regenerate `Cargo.lock` from the resolved manifests;
7. run targeted unit checks, then compile and execute the three GPU integration
   targets in their sequential lane.

For a conflicting PR based on old root paths, treat the old code's intent as
the patch to port. Do not mechanically retain both the old root implementation
and the new crate implementation.

## Current validation status

At the time this document was written, the implementation had the following
validation evidence from the branch work:

- 225 non-ignored Exponent-LUT unit tests passed;
- all three release GPU integration targets compiled;
- the final design/code review completed without remaining findings.

Compilation of GPU targets is not the same as executing them on a device.
Device execution still requires a GPU-visible environment and the sequential
`dcrt_runtime` lane. The Lean tree is also not compiled through `src/lib.rs`,
as described above.

## File map for first-time readers

A practical reading order is:

1. `crates/exponent-lut/src/program/mod.rs`
2. `crates/exponent-lut/src/program/builder.rs`
3. `crates/exponent-lut/src/program/lowering.rs`
4. `crates/exponent-lut/src/api.rs`
5. `crates/exponent-lut/src/rns_bgg/mod.rs`
6. `crates/exponent-lut/src/rns_bgg/public_key.rs`
7. `crates/exponent-lut/src/rns_bgg/encoding.rs`
8. `crates/exponent-lut/src/rns_bgg/noise.rs`
9. `crates/exponent-lut/src/noise_refresh/refresh.rs`
10. `crates/exponent-lut/src/noise_refresh/prf.rs`
11. `crates/exponent-lut/src/noise_refresh/simulation.rs`
12. `crates/exponent-lut/tests/README.md`

For repository-wide ownership and cross-crate dependencies, read
`docs/architecture.md` before resolving conflicts.
