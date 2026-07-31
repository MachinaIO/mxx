# Workspace architecture

This repository is a virtual Cargo workspace. It intentionally has no root package or compatibility facade: consumers depend on the crate that owns the required abstraction.

## Dependency layers

Arrows below point from a consumer to one of its dependencies:

```text
mxx-runtime          -> mxx-ir-core, mxx-primitives
mxx-ir-symbolic    -> mxx-ir-core
mxx-noise-simulator -> mxx-ir-symbolic, mxx-primitives
mxx-bench-estimator  -> mxx-ir-core, mxx-runtime
mxx-gadgets          -> mxx-ir-core, mxx-primitives
mxx-bgg              -> mxx-ir-core, mxx-gadgets
application crates   -> graph, execution, primitive, gadget, and BGG layers
```

Dependencies follow these rules:

- `mxx-ir-core` has no dependency on another workspace crate.
- `mxx-runtime` depends on `mxx-ir-core` and `mxx-primitives`.
- `mxx-ir-symbolic` depends only on `mxx-ir-core`.
- `mxx-noise-simulator` depends on `mxx-ir-symbolic` and `mxx-primitives`.
- `mxx-bench-estimator` depends on `mxx-ir-core` and reuses the runtime liveness schedule.
- `mxx-primitives` has no dependency on another workspace crate.
- `mxx-gadgets` depends on `mxx-ir-core` and `mxx-primitives`.
  It owns BGG-independent circuits and circuit gadgets. Its generic circuit
  lowerer owns recursive traversal and parameter binding;
  scheme crates supply only operation-specific Graph IR formulas. Every
  lowering callback receives the stable path of enclosing direct or summed
  sub-circuit calls, so per-gate artifact identities remain distinct across
  repeated child instances.
- `mxx-bgg` depends on `mxx-ir-core` and `mxx-gadgets`. It owns BGG+ wire
  bundles and operations whose formulas require those bundles, including
  lookup, input injection, slot transfer, commitment, masked decoding, and
  noise refresh. Decoder and noise-refresh execution exists only as canonical
  Graph IR; there is no parallel direct-execution API.
- `mxx-func-enc`, `mxx-we`, and `mxx-io` may depend on the graph,
  execution, primitive, gadget, and BGG compiler layers.
- Application crates must not depend on one another.

Code shared by two application crates belongs in the lowest layer that can
express it naturally. Graph scheduling and resource measurement belong in
`mxx-bench-estimator`; graph-derived magnitude analysis belongs in
`mxx-noise-simulator`; application-only parameter policy remains in its
application crate.

## Directory ownership

### `mxx-primitives`

`crates/primitives/` owns operations that are broadly useful across lattice-cryptography schemes:

| Path | Responsibility |
| --- | --- |
| `crates/primitives/src/element/` | Ring element abstractions. |
| `crates/primitives/src/poly/` | Polynomial traits, DCRT parameters, CPU polynomials, and GPU polynomial wrappers. |
| `crates/primitives/src/matrix/` | Generic in-memory matrices, DCRT matrices, and GPU matrices. |
| `crates/primitives/src/sampler/` | Uniform, hash, trapdoor, and GPU samplers plus generic sampling bounds. |
| `crates/primitives/src/rlwe_enc.rs` | Low-level RLWE encryption helpers. |
| `crates/primitives/src/env.rs` | Primitive-operation configuration. |
| `crates/primitives/cuda/` | Native CUDA headers and kernels. |
| `crates/primitives/build.rs` | OpenFHE and CUDA compilation/linking. |
| `crates/primitives/benches/` | Primitive CPU/GPU matrix and preimage benchmarks. |

GPU implementations of primitive operations stay in this crate even when a higher-level crate is their main caller.

The GPU implementation of `CrtRecompose` performs exact integer reconstruction
without a host staging fallback. Its initial supported domain is a single GPU,
at most 64 active RNS limbs, and plaintext moduli representable as `u64`.
Input levels are inverse-transformed independently before one batched
coefficient-recomposition kernel. Batching those inverse transforms and
reducing the kernel's per-thread fixed workspace are performance follow-ups.

### Graph infrastructure

| Crate | Path | Responsibility |
| --- | --- | --- |
| `mxx-ir-core` | `crates/ir-core/` | Executable typed graph structure, exact compile expressions, concrete type and shape validation, canonical identities, and runtime artifact manifests. This is the core Graph IR. |
| `mxx-ir-symbolic` | `crates/ir-symbolic/` | Optional symbolic atom and term identities, source provenance, elaboration, rewrite machinery, and cross-graph symbolic manifests. It retains only the structural `Large`/`Bounded` classification and performs no numerical bound calculation. It does not define BGG-, Diamond-, or AKY-specific invariants. |
| `mxx-noise-simulator` | `crates/noise-simulator/` | Target-driven numerical analysis of elaborated symbolic graphs. It owns all magnitude arithmetic, declared virtual-atom metadata, stable statistical dependencies, selection branch joins, threshold-decode reports, and the historical `PolyNorm`/`PolyMatrixNorm` rules. |
| `mxx-runtime` | `crates/runtime/` | CPU/GPU graph execution over existing primitive APIs, transcript recording/replay, shared liveness, indexed artifact-family persistence, and manifest production. |
| `mxx-bench-estimator` | `crates/bench-estimator/` | Per-node measurement composition, binding-sensitive subgraph reuse, critical paths, parallel waves, and persistent/workspace peak modeling. |
| `mxx-bgg` | `crates/bgg/` | BGG+ public-key, encoding, polynomial-encoding, and naive-vector wire bundles; Graph IR samplers; digit reconstruction; recursive `PolyCircuit` graph lowering; BGG-specific lookup, input-injection, slot-transfer, and commitment compilers; and thin BGG adapters for cohesive decoder/noise-refresh gadgets. BGG arithmetic is expressed only as generic Graph IR nodes and executed by `mxx-runtime` backends. |

The supported BGG public-lookup compiler is the LWE construction. GGH15 lookup
and WEE25 commitment-backed lookup evaluators are unsupported, and their direct
implementations have been deleted. Standalone WEE25 commitment remains a
BGG-owned feature.

Circuits that combine supported LWE lookup with slot transfer use a small
delegate-based advanced lowering: lookup gates go to the LWE compiler and slot
transfer/reduction gates go to the representation-specific slot compiler. No
cross-product enum or compatibility branch is provided for excluded lookup
schemes.

The production dependency graph is acyclic: `mxx-runtime` is scheme-independent,
`mxx-gadgets` does not depend on `mxx-bgg`, and `mxx-bgg` does not depend on
`mxx-runtime`. BGG unit tests may use `mxx-runtime` as a dev-dependency to
execute the graphs they construct; runtime types do not appear in BGG
production APIs.

BGG artifact names encode only deterministic structural coordinates. Lookup
table commitments remain in BGG descriptors and are checked when artifacts are
bound to a compiler; commitments are deliberately not copied into artifact
names. Manifests and descriptors are trusted metadata, so the design favors one
simple validation path over a second canonical-name protocol.

Trapdoor types and sampling nodes carry `gadget_base` and `digit_count`
explicitly. The runtime validates those values against the concrete backend
parameters before calling the unchanged trapdoor or preimage sampler.

Graph IR version 3 gives persisted and imported artifacts an explicit type and
confidentiality. Artifact families are represented by one `IndexedFamily`
wire; their cardinality is metadata rather than a number of graph ports.

`GadgetDecompose` and decomposed hash nodes normally derive their digit count
from the modulus and base. DCRT small decomposition is different: its width is
defined per CRT tower and cannot be recovered from the aggregate modulus.
Those nodes therefore accept an optional compile-time `digit_count`; DCRT
small-gadget graph builders must set it explicitly.

#### Runtime sessions

`mxx-runtime::execute_in_session` fixes the `ProductionId` before the first
random draw. A session descriptor binds that identity to the Graph IR spec
hash, IR version, graph name, and a canonical digest of all runtime inputs.
Reopening one identity with different inputs or another graph is an error.

Session stores atomically record each logical draw batch before dependent
execution continues. Existing identical records are replayed, missing records
are sampled and appended, and conflicts are errors. Artifact payloads are
written before their typed completion handles; the final manifest is written
last. Repeated payloads, transcript records, and manifests are idempotent only
when byte- and descriptor-identical. One session permits only one active
writer. Private artifact handles expose type, confidentiality, and logical
coordinates but never payload bytes or a filesystem path.

#### Runtime scheduling

`mxx-runtime::ExecutionConfig::max_parallel_instances` is a nonzero hard bound
on the number of `ParallelLoop` instances executed in one wave. Each wave
retains preimage batching within that bound, completes its child liveness
schedule, and releases child intermediates before the next wave begins. The
default bound is 64; applications with tighter RAM or VRAM budgets must pass an
explicit configuration through `execute_with_config` or
`execute_in_session_with_config`.

The production polynomial backend creates one runtime placement per device id
carried by its GPU parameters, deriving an equivalent single-device parameter
set for each placement; the default GPU helper obtains those ids from
`detected_gpu_device_ids`. `ParallelLoop` iterations are assigned round-robin
across those placements within every bounded wave.
Matrices never split CRT limbs across devices. Broadcast matrices and
trapdoors are staged once and loaded once per device before the first loop
wave; their per-device runtime values are reference-shared across iterations.
Zip inputs are materialized directly on the device that owns their iteration.
Preimage requests are batched independently by wave and placement, so one
backend batch never mixes device contexts. Returning from a nested loop restores
the caller's placement.

Artifact-compatible `ParallelLoop` outputs are encoded and staged after each
wave instead of remaining as backend matrices. `FilesystemArtifactStore`
keeps those payloads off-heap and off-device, atomically publishes transcript
batches and completion markers, and uses an OS advisory lock for one live
session writer. A persisted staged family is copied member-by-member into its
final typed coordinates. Payload and completion metadata both retain the
declared layout. The final manifest is published before scratch members are
removed, and the session writer lock is released only after that cleanup. The
returned value then becomes a lazy final family handle.

Aggregate public-family hashes are verified once per immutable
production/name/hash tuple and cached for later member loads, avoiding
quadratic verification when a family is consumed sequentially. Ephemeral
streamed output families carry explicit `StagedFamilyLease`s in the execution
result; callers release their scratch payloads with
`ExecutionResult::cleanup_staged`. Failed executions clean all scratch payloads
before returning. If storage itself prevents that cleanup, the returned
`ExecutionError::StagedCleanup` owns retryable leases and exposes the same
cleanup operation; ownership is never discarded on a partial failure. The
in-memory store remains available for tests and small graphs but does not
provide a bounded-RAM guarantee.

#### Symbolic overlays

`mxx-ir-symbolic` can optionally apply a `SymbolicOverlay` while elaborating
an unchanged executable Graph IR. An overlay never becomes a graph node and
does not affect the Graph IR spec hash or runtime behavior.

- A **fold** replaces selected terms with a derived fold atom whose
  `DefExpr::Fold` retains the exact replaced expression. It is an identity by
  construction and is therefore a symbolic fact.
- An **unfold** replaces a wire description with an assumed term list that may
  contain shared virtual atoms. It is an axiom. The local assumption hash and
  all transitively imported assumption digests are retained in elaboration
  results and symbolic manifests.

This fact/axiom distinction is the overlay trust boundary. An empty
`assumption_digests` set is the only indication that a result is
assumption-free; merely importing and re-exporting a result cannot erase its
provenance.

Fold validation:

| Check | Result |
| --- | --- |
| Expected and actual canonical term lists differ | Error with both descriptions |
| Groups overlap, omit a position, or use an invalid position | Error |
| A signal group lacks a common non-scalar suffix or has a large prefix | Error |
| A noise group contains a large atom | Error |
| A whole-signal group (`suffix_len = 0`) contains standalone noise | Error |
| A selector targets a non-matrix wire or overlaps another selector | Error |
| A folded prefix carries preimage references | Warning |
| A selector matches no concrete wire | Warning |

Unfold validation:

| Check | Result |
| --- | --- |
| A factor chain does not compose to the selected wire type | Error |
| An assumed preimage does not satisfy `uniform × preimage = target` by type | Error |
| Assumed preimage targets form a cycle | Error |
| Bounded/large character differs from the current description | Error |
| A non-source description is replaced without `replace_derived` | Error |
| A non-source description is replaced with `replace_derived` | Warning |
| A preimage description is discarded | Warning |
| A virtual declaration is invalid or references an undeclared virtual | Error |
| A virtual atom, assumed term list, or selector is unused | Warning |

Symbolic manifests use content-derived atom and term-list identifiers, retain
the original production namespace of re-exported records, and export the
complete closure of definitions, dependencies, uniforms, and preimage targets.
Multiple projections of one production merge only when their interpretation
digests and every overlapping record agree.

Symbolic manifest format version 5 exports full concrete artifact wire types,
self-contained source parameters,
assumed bounded metadata, and selection-domain roles. Locally declared
dependency labels and selection domains are qualified by their production
identity during export, and re-export preserves that origin.

#### Noise simulation

`mxx-noise-simulator::simulate` evaluates only matrix graph outputs and the
inputs of instantiated `ThresholdDecode` nodes. Atom results are memoized.
Reports distinguish signal from noise and state that their values are
high-probability coefficient envelopes using the existing CLT eligibility
rules, not worst-case bounds.

- Gaussian samples use the existing `6.5 * sigma` envelope.
- Preimage samples derive sigma from the explicit trapdoor sigma, gadget base,
  digit count, public-matrix row count, and target block-row count, then apply
  the same `6.5` envelope.
- Balanced gadget digits retain
  `6.5 * sqrt((base^2 + 2) / 12)`.
- Selection joins use the maximum branch norm. Dependencies are unioned,
  `is_const_poly` is combined with logical AND, equal `zero_rows` metadata is
  retained, and the joined result is not CLT-ready.
- Exact rational `RealExpr` values convert directly to `BigDecimal`; square
  roots use `BigDecimal::sqrt` without an intermediate `f64`.
- A bounded external input requires imported manifest metadata or an unfold
  assumption. The simulator does not guess a bound.
- Unknown modulus-switch shapes are reported as unsupported instead of being
  assigned a new universal formula.

Opaque tensor nodes are unsupported in the initial symbolic elaborator.
Concat, reshape, and folds reject any operation that would hide a signal/noise
mixture inside one atom. The range-versus-modulus comparison for
`UniformSample` remains in `mxx-ir-symbolic` because it determines the
structural `Large`/`Bounded` classification rather than a numerical noise
bound.

### `mxx-gadgets`

`crates/gadgets/` owns reusable cryptographic components assembled from primitives:

| Path | Responsibility |
| --- | --- |
| `crates/gadgets/src/circuit/` | BGG-independent circuit structure, serialization, analysis, and canonical Graph IR traversal/lowering interfaces. |
| `crates/gadgets/src/circuit_gadgets/` | Arithmetic, convolution, FHE/Ring-GSW, PRG, NTT, and other cohesive circuit gadgets. |

BGG-specific lookup, input injection, slot transfer, commitment, masked
decoding, and noise refresh live in `mxx-bgg`. `mxx-gadgets` provides only the
BGG-independent circuit structures and helpers used to construct those Graph
IR programs. There is no direct BGG execution API in this crate.

### Application crates

| Crate | Paths | Responsibility |
| --- | --- | --- |
| `mxx-func-enc` | `crates/func-enc/src/` | Functional-encryption trait. The AKY24 implementation is disabled pending a separate specification of its raw-mask semantics. |
| `mxx-we` | `crates/we/src/` | Witness-encryption trait. Diamond WE source is present but excluded from the crate root pending a separate application cutover. |
| `mxx-io` | `crates/io/src/` | iO trait. AKY24 iO and Diamond iO sources are present but excluded from the crate root pending separate application cutovers. |

Diamond WE, Diamond iO, and AKY24 iO are disabled at their crate roots and are
not compiled as production modules. Re-enabling any of them requires a separate
application-level cutover to the current BGG Graph IR and runtime APIs. AKY24
functional encryption is also disabled until its raw-mask semantics are
specified separately.

## Features and native build ownership

Crates with GPU-specific behavior define a `gpu` feature as needed. Upper
crates only forward that feature to their dependencies. `mxx-primitives` is
the sole owner of:

- OpenFHE linkage;
- `cc` build dependencies;
- CUDA source compilation.

Primitive CPU matrices are always in memory. Durable and streamed artifacts
are owned by `mxx-runtime`; there is no feature-selected primitive matrix
backing implementation.

Use workspace-wide commands when checking a cross-cutting change:

```sh
cargo check --workspace
cargo check --workspace --features gpu
```

## Tests

Unit tests remain beside the implementation they test. Integration tests are grouped by owner:

- `crates/bgg/src/` unit-test modules for BGG lookup, injection,
  slot-transfer, commitment, and BGG adapter regressions;
- `crates/gadgets/src/` unit-test modules for BGG-independent circuit,
  decoder, and noise-refresh regressions;
- `crates/we/tests/` for Diamond WE;
- `crates/io/tests/` for AKY24 iO and Diamond iO.

Integration tests are not part of routine validation and must only be run when a task explicitly requests them. Compiling their targets with `cargo check --workspace --tests` is permitted for boundary validation.
