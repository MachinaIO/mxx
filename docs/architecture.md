# mxx architecture

This document is the design entry point for the `mxx` workspace. It is meant to be read top-down:
the first sections give the mental model and the crate boundaries, and later sections go into each
layer in the order data flows through it. Every type and path named here exists in the current
source tree; paths are relative to the repository root. When the code and this document disagree,
the code is authoritative and this document should be corrected.

Contents:

1. [What mxx is and the mental model](#1-what-mxx-is-and-the-mental-model)
2. [Workspace crate map and dependency direction](#2-workspace-crate-map-and-dependency-direction)
3. [`mxx-ir-core`: the executable graph IR](#3-mxx-ir-core-the-executable-graph-ir)
4. [`mxx-dsl`: building graphs](#4-mxx-dsl-building-graphs)
5. [`mxx-backends`: values, primitives, and CPU execution](#5-mxx-backends-values-primitives-and-cpu-execution)
6. [`mxx-backends`: the GPU runtime](#6-mxx-backends-the-gpu-runtime)
7. [Application crates](#7-application-crates)
8. [Testing, validation, and where to look next](#8-testing-validation-and-where-to-look-next)

## 1. What mxx is and the mental model

`mxx` is a Rust and CUDA workspace for lattice-cryptography research. It provides polynomial and
matrix arithmetic over RNS (CRT) rings `Z_q[X]/(X^N + 1)`, bounded samplers (uniform, Gaussian,
hash-derived, lattice trapdoors and preimages), and constructions built from them: BGG+ encodings,
circuit gadgets, TFHE NAND bootstrapping and leveled BGV, and Diamond witness encryption.

Cryptographic algorithms are not written as eager function calls. They are written once as a
typed dataflow graph and then interpreted by several consumers. The life of a computation is:

```text
 Rust construction code (mxx-dsl: DslContext, Ring, Mat, Int, Family, parallel/iterate/select)
        |  builds immutable node handles
        v
 Graph (mxx-ir-core)      Graph::freeze keeps reachable nodes, one scope per body
        |  validate(graph, ParamEnv, CRT-basis resolver) [+ artifact manifests]
        v
 ValidatedGraph           concrete wire types, rings, execution order, liveness
        |
        +--> CPU executor:  mxx_backends::execute(..., &mut CpuDcrtBackend, ...)
        |
        +--> GPU runtime:   GpuRuntime::plan(graph, inputs) -> GpuExecutionPlan
        |                   GpuRuntime::execute(&mut plan, inputs)
        |
        +--> Lean exporter: mxx_ir_core::lean (execution relations, linked correctness claims)
        v
 Outputs, artifacts, sessions (ArtifactStore / SessionStore, ProductionId, Manifest)
```

Key ideas that recur throughout the code:

- **Construction is not execution.** DSL expressions such as `&a + &b` create graph nodes. No
  matrix arithmetic or sampling happens until a backend executes the validated graph.
  `crates/fhe/src/lib.rs` states this explicitly for FHE: its methods build graphs, and key
  generation, sampling, arithmetic, and decryption run in `mxx-backends`.
- **Compile parameters versus runtime values.** Shapes, loop counts, moduli, and sampler
  parameters are compile expressions (`IntExpr`, `RealExpr`) resolved by a `ParamEnv` during
  validation. Runtime `Int`/`Bool` values can select family members or candidates but never
  change a shape or a loop count.
- **Structure is kept, not unrolled.** Subgraph bodies and loop bodies are stored once in the
  frozen graph and validated once. Executors instantiate them per call or per loop index;
  runtime identities carry an instantiation path (`InstantiationFrame` in
  `crates/ir-core/src/types.rs`).
- **Sampler placement is semantics.** A sampler outside a loop is shared by every instance; a
  sampler inside a loop body produces a fresh value per executed instance. Sampler nodes carry
  their authoritative integer coefficient cutoffs.
- **Artifacts link stages.** A graph can export outputs as persisted artifacts, identified by a
  `ProductionId` (graph specification hash plus execution nonce). Another graph imports them by
  that identity. Protocol declarations in `mxx_ir_core::protocol` link stages, ideal
  specifications, and requirements for correctness checking.
- **One GPU path.** All GPU computation goes through `GpuRuntime`, which lowers a validated graph
  to explicit CUDA Graph regions. There is no separate eager GPU matrix, polynomial, or sampler
  API.

## 2. Workspace crate map and dependency direction

The repository is a virtual Cargo workspace (`Cargo.toml`) with no root facade crate; `Cargo.toml`
is the authoritative member list. Consumers depend directly on the crate that owns an
abstraction.

| Crate (path) | Package | Responsibility |
| --- | --- | --- |
| `crates/ir-core` | `mxx-ir-core` | Executable typed graph IR, compile expressions, rings, validation, canonical hashing, artifact manifests, protocol declarations, Lean export. |
| `crates/dsl` | `mxx-dsl` | Typed, declarative Rust DSL that builds `mxx-ir-core` graphs. |
| `crates/backends` | `mxx-backends` | Polynomial/matrix primitives (OpenFHE via cxx), samplers, runtime values, CPU executor, GPU runtime and native CUDA, artifacts, sessions, transcripts. |
| `crates/gadgets` | `mxx-gadgets` | Reusable, BGG-independent circuits and circuit gadgets (nested RNS arithmetic, NTT, FHE gadgets, noise refresh, input injector). |
| `crates/bgg` | `mxx-bgg` | BGG+ public keys, encodings, circuit lowering, lookups, slot transfer, Tall encodings, WEE25 commitments. |
| `crates/fhe` | `mxx-fhe` | TFHE (integer LWE with NAND bootstrapping) and leveled BGV graph builders. |
| `crates/we` | `mxx-we` | Witness-encryption interfaces and Diamond WE with Lean-checked parameter search. Excluded from the workspace (`exclude` in `Cargo.toml`); its manifest is standalone. |
| `crates/func-enc` | `mxx-func-enc` | Functional-encryption interface trait only (`FuncEnc`). |
| `crates/io` | `mxx-io` | Indistinguishability-obfuscation interface trait only (`Obfuscation`). |

### Crate boundaries and dependency direction

Normal (non-dev) dependencies, taken from each `crates/*/Cargo.toml`:

```text
mxx-ir-core   -> (no workspace crates)
mxx-dsl       -> mxx-ir-core
mxx-backends  -> mxx-ir-core                      (dev: mxx-dsl)
mxx-gadgets   -> mxx-dsl, mxx-ir-core, mxx-backends   (dev: mxx-bgg)
mxx-bgg       -> mxx-ir-core, mxx-dsl, mxx-gadgets, mxx-backends
mxx-fhe       -> mxx-dsl, mxx-ir-core, mxx-backends
mxx-we        -> mxx-backends, mxx-gadgets, mxx-bgg, mxx-dsl, mxx-ir-core
mxx-func-enc  -> (no dependencies)
mxx-io        -> (no dependencies)
```

Rules that follow from this layout:

- `mxx-ir-core` is the bottom layer. It knows nothing about the DSL or any backend; the CRT basis
  resolver it needs is passed in as a function pointer (`ResolveCrtBasis`).
- `mxx-dsl` and `mxx-backends` are siblings over `mxx-ir-core`. The backend executes core graphs
  and does not depend on the DSL (it uses the DSL only in tests).
- Application crates (`mxx-fhe`, `mxx-we`, `mxx-func-enc`, `mxx-io`) never depend on one another.
  Shared application code moves down to the lowest natural layer.
- Reusable gadgets live in `crates/gadgets/`; circuit gadgets live in
  `crates/gadgets/src/circuit_gadgets/`. `mxx-bgg` is the BGG+-specific layer above gadgets,
  and `mxx-we` builds on it.
- The `gpu` feature is owned by `mxx-backends` (`crates/backends/Cargo.toml`). `mxx-gadgets`,
  `mxx-fhe`, and `mxx-we` forward their `gpu` feature to it (`mxx-we` also forwards to
  `mxx-gadgets`). The `gpu` features of `mxx-func-enc` and `mxx-io` are empty.
- Native CUDA sources, GPU wrappers, and the GPU runtime are owned by `mxx-backends` under
  `crates/backends/cuda/` and `crates/backends/src/`; higher crates use its public API. The one
  exception is a subgraph kernel (section "Subgraph kernels"): a higher crate may implement the
  native kernel of one of its named subgraphs, compiled by its own build script against
  `crates/backends/cuda/include/SubgraphKernel.cuh` only. `mxx-backends` publishes that
  directory as `DEP_MXX_BACKENDS_CUDA_INCLUDE` (its `links = "mxx_backends"` key).

Diamond iO and AKY24 iO, and the AKY24 functional-encryption implementation, were removed from
this branch during the DSL migration; `README.md` links the `main`-branch implementations.

## 3. `mxx-ir-core`: the executable graph IR

`crates/ir-core/src/lib.rs` describes the crate as owning "executable graph structure, compile
expressions, concrete type validation, canonical identities, and runtime artifact metadata". It
has no dependency on any other workspace crate.

| Module | Owns |
| --- | --- |
| `graph.rs` | Node/value handles, construction scopes, subgraph sealing and captures, `Graph::freeze`, frozen scopes, serialization. |
| `node.rs` | `NodeKind` and its payload types (`ConstantMatrix`, `ParallelLoop`, `SequentialLoop`, `LoopInputMode`, `HashVariant`, ...). |
| `types.rs` | `NodeId`, `Port`, `WireRef`, `WireId`, `MatrixType`, `WireType`, `ConcreteWireType`, `CoefficientBoundDomain`. |
| `ring.rs` | Symbolic rings (`RingRef`/`RingExpr`), concrete rings (`ConcreteRing`), CRT basis resolution. |
| `expr.rs` | `IntExpr`, `RealExpr`, `Rational`, `ParamEnv`, `ExprError`, `IndexExpr`. |
| `validate.rs`, `checks.rs`, `constraints.rs` | Structural and concrete validation, `ValidatedGraph`, liveness, parameter constraints. |
| `encoding.rs` | Canonical JSON, SHA-256 hashing, `spec_hash`, `IR_VERSION`. |
| `artifact.rs` | `SpecHash`, `ProductionId`, `Manifest`, `ManifestArtifact`, `ArtifactType`, `ArtifactAvailability`. |
| `protocol/` | Protocol declarations linking stages (`declaration.rs`), closed protocol bundles (`bundle.rs`), pure specifications (`spec.rs`). |
| `lean/` | Lean export of execution relations (`mod.rs`) and linked correctness claims (`claim.rs`, `protocol.rs`). |
| `inventory.rs` | Structural snapshot (`GraphInventory`) for checkers, without evaluation. |

### 3.1 Handles, scopes, and freezing

Graph construction uses immutable, reference-counted handles (`crates/ir-core/src/graph.rs`):

- `NodeHandle` wraps a node: its `NodeKind`, argument `ValueHandle`s, output `WireType`s, source
  location, construction scope, and optionally a structural child (a subgraph or loop body).
  Equality is by identity, so cloning a handle shares the node rather than duplicating it.
- `ValueHandle` is a `(node, port)` pair. Constructors are `NodeHandle::new`,
  `NodeHandle::subgraph_call`, `NodeHandle::parallel_loop`, and `NodeHandle::sequential_loop`.
- Construction scopes (`ConstructionScopeId`, `with_new_construction_scope`,
  `current_construction_scope`) are a thread-local lexical stack. A body may read values from
  ancestor scopes but never from a completed sibling or child scope.
- `SubgraphHandle::seal(..., CapturePolicy)` turns a body into a sealed definition. With
  `CapturePolicy::Lexical`, every outer value the body reads becomes a `__capture_N` input
  placeholder (`CapturedValue { outer, placeholder, mode }`). The capture mode is a
  `LoopInputMode`: `Broadcast` by default; a `FamilyGetDynamic` read indexed by the parallel
  binder (optionally plus a nonnegative constant) becomes `Zip` or `ZipOffset { offset }`, so the
  loop receives one member per instance instead of the whole family.

`Graph::freeze(name, parameters, outputs, retained_roots, effect_roots, real_constants)` produces
the immutable `Graph` and a `FreezeMap`:

- Only nodes reachable from outputs, retained roots, and effect roots are kept; a node shared by
  several handles is frozen once. `NodeId`s are postorder indices within a scope.
- Scopes are keyed by `FrozenGraphScopeId`: `Root`, `Subgraph { canonical_name }`,
  `ParallelBody { parent, owner }`, and `SequentialBody { parent, owner }`. A named subgraph has
  exactly one scope regardless of how many call sites it has, and each loop body is one scope
  owned by its loop node; bodies are never unrolled.
- Freezing rejects cycles, foreign-scope edges, invalid ports, duplicate input names, and two
  different subgraph definitions with the same name (`FreezeError`).
- `FreezeMap::resolve_unique` maps a construction handle to its frozen `ScopedWireRef`, rejecting
  handles reachable along more than one structural path.
- Outputs are `GraphOutput { value, availability: Option<ArtifactAvailability> }` during
  construction and `OutputRoot { value: WireRef, availability }` after freezing.

A `Graph` serializes to JSON with a ring table (`{"ring_table": [...], "graph": ...}`); rings
are interned and referenced as `{"$ring": i}`. Source locations, construction scopes, and
benchmark roles are not serialized. Deserialization checks that node ids are contiguous, edges
point backward, and ports exist; this is a structural consistency check, not an authentication
of provenance.

### 3.2 Node kinds

`NodeKind` (`crates/ir-core/src/node.rs`) is the complete executable vocabulary:

| Category | Variants |
| --- | --- |
| Inputs and constants | `Input { name, wire_type, artifact }`, `ConstantInt`, `EvaluateInt(IntExpr)`, `ConstantReal`, `ConstantBool`, `ConstantMatrix { matrix_type, value }` |
| Trapdoor structure | `GadgetTrapdoor { matrix_type, base }`, `TrapdoorPublic` |
| Scalar arithmetic | `IntBinary(Add/Subtract/Multiply/Divide/Remainder)`, `IntCompare(Equal/Less/LessEqual)`, `BitExtract`, `IntToReal`, `BoolToInt`, `RealBinary`, `RealSqrt`, `IntMatrixVectorProduct { transpose }` |
| Matrix arithmetic | `MatrixBinary(Add/Subtract/Multiply)`, `MatrixMulAccumulate { coefficients, has_bias }`, `MatrixMulSmallRhs`, `MatrixNegate`, `MatrixScale`, `RingAutomorphism { index }`, `MultiplyMonomial` |
| Ring and modulus conversion | `ModulusSwitch`, `ModulusReduce`, `CenteredRebase` (each with a destination `RingRef`), `CenteredRoundDivide { divisor }`, `RnsModUp { destination, digit_size, normalize }`, `RnsModDown { destination, plaintext_modulus }`, `BlockModSwitch { destination, plaintext_modulus }` |
| Shape | `Transpose`, `Slice { rows, columns }`, `Tensor`, `Concat { axis: Rows/Columns/Diagonal }` |
| Samplers | `UniformResidueSample`, `UniformIntervalSample`, `GaussianSample { sigma, max_coefficient_bound }`, `HashSample { variant: Plain/Decomposed/SmallDecomposed, tag_prefix, tag_components, base, digit_count }`, `HashIntFamily { count, modulus, tag_prefix, tag_components }`, `TrapdoorSample`, `PreimageSample { max_coefficient_bound }` |
| Decomposition and coefficients | `GadgetDecompose { base, small, digit_count }`, `ExtractCoefficient`, `LiftIntegerToConstantPolynomial`, `PackPolynomialCoefficients`, `PolynomialFromValues { evaluation }`, `PolynomialValues { evaluation }` |
| Decoding and CRT | `ThresholdDecode { plaintext_modulus, length, output_bool }`, `CrtRecompose` |
| Control | `SubgraphCall`, `ParallelLoop`, `SequentialLoop`, `Select { count }` |
| Families | `FamilyPack { count }`, `FamilyGetStatic { index }`, `FamilyGetDynamic` |

`ConstantMatrix` covers `Zero`, `Identity`, `UnitRow`, `UnitColumn`, `Gadget { base, small }`,
`PowerOfBase`, `Rotation`, and `Polynomial`. Hash-tag components are typed
(`HashTagComponent::{Bytes, Integer, Decimal, U64Le, Operand}`) so different framings cannot
collide.

`HashIntFamily` returns a `Family<Int>` of `count` integers uniform on `[0, modulus)`, where
`modulus` is a power of two above one. It reuses the `HashSample` transcript: integer `i` is
coefficient `i` of entry `(0, 0)`, truncated to `log2(modulus)` bits, so no candidate is
rejected. Both hash samplers share key and tag validation (`validate_hash_key_and_tag` in
`validate.rs`: a 32-byte key and integer tag operands). Like the other samplers, `HashIntFamily`
is rejected in sampler-free protocol specifications and has no Lean sampler relation.

`MultiplyMonomial` multiplies every matrix entry by `X^k` in the negacyclic ring, where `k` is a
runtime integer (argument 1) of any sign taken modulo `2n`. `IntMatrixVectorProduct { transpose }`
is the integer product of a row-major matrix family (argument 0) with a vector family (argument
1); the vector length fixes the inner dimension, so the matrix count must be a nonzero multiple of
it. It computes `M v` (`out[i] = sum_j M[i, j] v[j]`), or `v^T M` with `transpose`. Neither node
has a Lean relation; the Lean emitter reports both as unsupported.

The ring-conversion nodes are fused CRT operations with explicit destination rings. They are
not a generic, implicit modulus-changing mechanism: nested-RNS level switching, for example,
remains a circuit gadget in `mxx-gadgets`.

Control nodes carry their own metadata:

- `ParallelLoop { count, minimum_count, index_slot, bindings, input_modes }`: independent
  instances over `0..count`; each argument has a `LoopInputMode` (`Broadcast`, `Zip`,
  `ZipOffset { offset }`). Outputs are indexed families.
- `SequentialLoop { count, index_slot, bindings, carried_count }`: the first `carried_count`
  arguments are the carried state; the rest are captures.
- `SubgraphCall { definition, bindings, canonical_input_exclusive_uppers }`: calls a named body,
  optionally with per-argument canonical-coefficient upper bounds for matrix arguments.
- `Select { count }`: eager selection of one candidate by a runtime selector. Both candidates are
  part of the graph; selection is not lazy branching.

### 3.3 Wire types, rings, and CRT bases

`WireType` (`crates/ir-core/src/types.rs`) has `ConstantInt`, `ConstantReal`, `ConstantBool`,
`Int`, `Real`, `Bool`, `Bytes { length }`, `TypedBlob { type_name, schema_hash }`,
`Matrix(MatrixType)`, `Trapdoor { matrix, sigma, gadget_base, digit_count,
preimage_max_coefficient_bound }`, `SmallMatrix { matrix, max_coefficient_bound, bound_domain }`,
`Preimage { .. }` (same fields, distinct semantics), and `IndexedFamily { element, count }`.
`ConcreteWireType` mirrors it with resolved sizes. `MatrixType { ring, rows, columns }` refers to
a ring, not a single modulus. `CoefficientBoundDomain` says whether a bounded coefficient is a
single `Global` integer or a signed residue per CRT limb (`PerCrtLimb`).

Rings are ordered CRT bases (`crates/ir-core/src/ring.rs`):

- `RingRef` wraps a `RingExpr`: `Generated { crt_bits, crt_depth, ring_dimension }`,
  `Explicit { crt_moduli, ring_dimension }`, `Slice { source, start, end }`,
  `Select { source, indices }`, or `Concat { left, right }`. Basis order is significant.
- Validation resolves each ring to a `ConcreteRing` (ordered `u64` primes plus dimension).
  `ConcreteRing` requires a power-of-two dimension, a nonempty basis, and distinct primes
  `2 < q < 2^60` with `q = 1 mod 2N`.
- The actual prime generation or checking is delegated to a `ResolveCrtBasis` callback,
  `fn(ring_dimension, crt_depth, crt_bits, explicit_moduli) -> Result<Vec<u64>, String>`.
  Production callers pass `mxx_backends::openfhe_guard::gen_modulus_and_warmup`. For an explicit
  basis the resolver must return it unchanged and in the same order.

### 3.4 Compile expressions and parameters

`IntExpr` (`crates/ir-core/src/expr.rs`) has `Const`, `Var`, `LoopIndex`, `Add`, `Sub`, `Mul`,
`Div` (exact: a nonzero remainder is an error), `FloorDiv`, `Rem` (floor remainder),
`RoundDiv` (nearest, ties toward positive infinity, positive denominator), `Log2Ceil`,
`Select { selector, branches }`, and ring properties `RingModulus`, `RingCrtDepth`,
`RingCrtModulus { ring, index }`. Serialization always goes through a canonical polynomial normal
form, so equivalent expressions encode identically.

`RealExpr` has `Rational`, `Var`, `FromInt`, `Add`, `Sub`, `Mul`, `Div`, and `Sqrt`, evaluated as
exact rationals where possible (`evaluate_rational`, `evaluate_f64`, `close`). `Rational` holds a
normalized `BigInt` numerator and denominator.

`ParamEnv { integers, reals, loop_indices }` binds named integer parameters (`BigInt`), real
parameters (`Rational`), and loop-index slots (managed by executors). Compile parameters are
declared on the graph (`CompileParameter`). `derive_param_constraints`
(`crates/ir-core/src/constraints.rs`) collects the parameter-only conditions (`ParamConstraint`)
that concrete validation also enforces.

Runtime integer division is different from `IntExpr` division: see section 4.3.

### 3.5 Validation

`validate(graph, bindings, resolve_basis)` and `validate_with_manifests(graph, bindings,
manifests, resolve_basis)` (`crates/ir-core/src/validate.rs`) produce a `ValidatedGraph`:

```text
ValidatedGraph { source: Graph, bindings: ParamEnv,
                 scopes: BTreeMap<FrozenGraphScopeId, ValidatedScope>,
                 warnings, resolved_rings }
ValidatedScope { execution_order, liveness: LivenessSchedule { last_use, retained },
                 wire_types: BTreeMap<WireRef, ConcreteWireType>, artifact_inputs }
```

The steps are: structural validation (`validate_structure`: topological order, declared compile
variables, legal loop-index use, dynamic family access in loop-dependent reads, subgraph bound
arity), manifest checks, parameter bindings and constraints, then per-scope concrete type
checking of every node, ring resolution, and checks that call and loop boundaries agree with
their child scopes. Each loop body is checked once as a template at loop index zero rather than
per iteration. A named subgraph is one scope for all of its calls, so it is checked once under each
distinct call binding; the stored `ValidatedScope` is the first call's, and a ring is recorded in
`resolved_rings` only when every call resolves it to the same value. `execution_order` is the frozen postorder, and
`LivenessSchedule::last_use` lets executors release intermediates after their last reader.

Validation proves structural and type correctness under concrete parameters. It does not prove
cryptographic norm bounds; those are application-owned.

### 3.6 Hashing, artifacts, and manifests

- `encoding::spec_hash(graph, bindings)` hashes canonical JSON of `{ir_version, graph, integer and
  real bindings}` with SHA-256 (`IR_VERSION` in `crates/ir-core/src/encoding.rs`). It commits to
  every binding, so artifacts produced under different dimensions or moduli cannot be swapped. It
  depends on the serialized structure (node kinds, postorder ids, canonical expressions, rings,
  graph name), not on allocation addresses, source locations, or construction order. The graph
  caches one `(ParamEnv, SpecHash)` pair in a `OnceLock`; other bindings are recomputed.
- `artifact::ProductionId { spec_hash, execution_nonce }` identifies one execution of one graph
  instantiation. A `Manifest { ir_version, production_id, artifacts }` lists each exported
  `ManifestArtifact` (artifact type, optional family count, availability, layout).
- `ArtifactAvailability::Transferred` means the consumer receives the payload from an external
  producer; `Cached` means the consumer could regenerate it deterministically from public context
  and uses stored bytes as a cache. Availability is about transport, not secrecy.
- `export_validated_manifest` builds the manifest for outputs that declare an availability.

### 3.7 Protocol declarations and pure specifications

`mxx_ir_core::protocol` (`crates/ir-core/src/protocol/`) links several executable graphs:

- `ProtocolDecl { params, bundle: ClosedProtocolBundle }` with `ProtocolStage { id, graph,
  bindings: Vec<ArtifactBinding> }`. Validation checks each artifact binding's name, type, and
  availability against the producer, parameter agreement across graphs, and reachability.
- `ClosedProtocolBundle` holds the workflow, an `IdealSpec`, requirement `PurePredicateSpec`s, a
  comparator, endpoint bindings, operational decoder targets, input contracts
  (`InputValueContract`), and input bindings.
- `IdealSpec::new(Graph)` and `PurePredicateSpec::new(Graph)` (`protocol/spec.rs`) reject every
  sampler kind in every scope; a predicate must have exactly one Boolean output. The graph is
  private and read through `graph()`, so these checks cannot be bypassed.
- `OutputRef { stage, output }` names an executable result. `OperationalDecoderKind` is
  `ThresholdDecode { plaintext_modulus }` or `BooleanInterval`; validation checks the decoder's
  executable node chain against the residual it names.

There is no separate correctness crate and no generic symbolic noise simulator; correctness
evidence comes from application-owned bounds and generated Lean claims.

### 3.8 Lean export

- `lean::export(&ValidatedGraph, &ExportOptions) -> LeanArtifact` (`crates/ir-core/src/lean/mod.rs`)
  emits one Lean relation per frozen scope, referencing backend-owned primitive relations
  (`PrimitiveNames`) and concrete CRT layouts (`BackendLayout`). Loops are not unrolled;
  families remain functions on `Fin N`.
- `lean::claim::assemble_claim` renders an application-independent linked claim
  (`LinkedClaim`, `ClaimBackend`, `ClaimSemantics`, `Endpoint`). The Boolean-interval renderer
  requires a scalar-polynomial residual.
- `lean::protocol::export_claim` exports every stage, requirement, and ideal graph of a
  `ProtocolDecl` and writes the final `Claim.lean`. It infers no noise bounds; applications
  supply decoder semantics and proofs.
- The handwritten Lean package `crates/ir-core/lean/` (`MxxIR`) supplies shared definitions such
  as `IterRuns` for sequential loops; `crates/ir-core/lean/README.md` explains the fixture tests
  that generate `test_data/lean_ir_fixtures/`.

## 4. `mxx-dsl`: building graphs

`mxx-dsl` (`crates/dsl/src/`) is a Rust embedded DSL. Running the Rust construction code builds
core nodes immediately; there is no separate parser or symbolic reinterpretation layer. Ordinary
Rust (`if`, `for`, functions, tuples, vectors) organizes construction, but it runs once at
construction time and cannot branch on a graph `Bool`.

### 4.1 A first graph

```rust
use mxx_dsl::{DslContext, Ring};
use mxx_ir_core::ParamEnv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ring dimension 8 with a generated basis of two 30-bit CRT primes.
    let ring = Ring::new(30, 2, 8);
    let input = ring.input("input", (2, 2));
    let doubled = &input + &input;
    let built = DslContext::new("double").output("result", doubled)?.build()?;
    // The CRT-basis resolver is supplied by the backend; mxx-dsl itself does not depend on it.
    let _validated =
        built.validate(&ParamEnv::default(), mxx_backends::openfhe_guard::gen_modulus_and_warmup)?;
    Ok(())
}
```

The three stages are construction (Rust builds handles; `build()` freezes reachable outputs and
runs structural validation), validation (`validate` resolves parameters, rings, and concrete
types), and execution or analysis (a backend or the Lean exporter consumes the
`ValidatedGraph`).

### 4.2 Context, values, and rings

- `DslContext` (`crates/dsl/src/lib.rs`): `new(name)`, `int_parameter(name)`,
  `real_parameter(name)`, `input::<V>(name, schema)`, `evaluate_int(expr)`,
  `int_family_input(name, count)`, `hash_int_family(key, tag, count, modulus)`,
  `output(name, value)`, `transferred_output`, `cached_output`,
  `transferred_trapdoor_output`, `transferred_trapdoor_family_output`, and `build()`. Output
  methods consume and return the context. Composite outputs are flattened as `x.0`, `x.1`, ...;
  names must be unique after flattening. `hash_int_family` builds a `HashIntFamily` node and
  prefixes the tag with the domain `mxx/hash-int-family/v1\0`, so its stream never coincides
  with a `hash_matrix` stream under the same key and tag.
- `BuiltGraph { graph }` has `validate(bindings, resolve_basis)` and
  `validate_with_manifests(bindings, manifests, resolve_basis)`.
- `Ring` wraps a `RingRef`: `Ring::new(crt_bits, crt_depth, ring_dimension)` (generated basis),
  `Ring::from_crt_moduli(moduli, ring_dimension)` (explicit ordered basis), and `from_ref`. Basis
  operations `slice_crt`, `prefix`, `select_crt`, and `concat_crt` build related rings;
  `modulus()`, `crt_depth()`, and `crt_modulus(i)` return compile expressions. Applications
  derive a DSL ring from backend parameters, for example `mxx_gadgets::ring_from_params`.
- Ring methods declare inputs (`input`, `bool_input`, `bytes_input`, `small_matrix_input`,
  `preimage_input`, family variants, and `*_with_domain` variants taking a
  `CoefficientBoundDomain`), artifact inputs (`artifact_input`, `family_artifact_input`,
  `trapdoor_artifact_input`, bounded and bytes variants), constants (`zero`, `identity`,
  `gadget`, `constant`, `polynomial`, `pack_polynomial_coefficients`, `from_coefficients`,
  `from_evaluations`), samplers (`uniform_residue`, `uniform_interval`, `gaussian`,
  `hash_matrix`, `hash_decomposed`, `hash_small_decomposed`, `sample_trapdoor`), and the
  public deterministic `gadget_trapdoor`.
- Value handles: `Mat`, `SmallMatrix`, `Preimage`, `Trapdoor`, `Int`, `Bool`, `Bytes`, and
  `Family<T>`. Cloning a handle shares the value; it never re-samples. Schemas (`MatType`,
  `SmallMatrixType`, `PreimageType`, `TrapdoorType`, `IntType`, `BoolType`, `BytesType`,
  `FamilyType`) and the `GraphValue`/`GraphValueSchema` traits (`crates/dsl/src/value.rs`) let
  tuples, vectors, and domain records flatten into wires.

### 4.3 Operations

- `Mat` supports `+`, `-`, `*`, unary `-` for owned and borrowed operands; a `1 x 1` matrix
  multiplies as a scalar, and `mat * scalar` scales by a constant polynomial. Other operations
  include `Mat::multi_row_gemm_accumulate`, `mul_small_rhs`, `ring_automorphism`,
  `multiply_monomial` (by `X^k` for a runtime `Int` `k`), `transpose`,
  `slice`, `tensor`, `decompose`/`small_decompose` (returning `Preimage`), `extract_coefficient`,
  `canonical_coefficient_bits`, `threshold_decode_ints`/`threshold_decode_bools`, `concat` (with
  the `concat_rows!`, `concat_cols!`, `concat_diag!` macros), `crt_recompose`, `coefficients`,
  and `evaluations`.
- Ring conversions take a destination `&Ring`: `modulus_switch` (scale and round),
  `reduce_modulus` (reduce into a divisor ring without scaling), `centered_rebase`,
  `block_mod_switch`, `rns_mod_up(dest, digit_size, normalize)`, and
  `rns_mod_down(dest, plaintext_modulus)`; `centered_round_divide(divisor)` rounds a centered
  quotient.
- `Trapdoor::sample_preimage(target, shape)` returns a `Preimage` whose cutoff comes from the
  trapdoor schema. Gaussian and preimage samplers always take explicit integer cutoffs.
- `Int` supports `+ - * / %` and `equal`, `less`, `less_equal`, `bit`, `lift_to_constant_polynomial`,
  and `expression()` (recovering a compile expression when possible). Runtime division computes
  `q = floor(a / |b|)` and `r = a - |b| q`, so `0 <= r < |b|`; division by zero is a runtime
  error. `Bool` supports `& | ^ !` and `to_int`. Rust `==`, `<`, `&&`, `||` are not graph
  operators.
- Hash tags (`HashTag`, `tag!`, `HashTagPart`) preserve component order and type framing, which
  are part of sampled-value identity.

### 4.4 Families and control flow

`crates/dsl/src/family.rs` and `crates/dsl/src/control.rs`:

- `Family<T>` is an ordered collection with one element schema. `family.at(i)` lowers to
  `FamilyGetStatic` when `i` is compile-known and loop-independent, and to `FamilyGetDynamic`
  otherwise. `Family::pack`, `count`, and `field` (structural projection of existing fields) are
  also available. Nested families (families of families) are not supported, but an `iterate`
  state, `select` candidate, or subgraph argument may be a whole family.
- `Family<Int>::matrix_vector_product(&v)` (`M v`) and `vector_matrix_product(&v)` (`v^T M`)
  treat the family as a row-major matrix whose inner dimension is the length of `v`, and build
  one `IntMatrixVectorProduct` node.
- `parallel(count, |i| body)` builds a `ParallelLoop` whose instances are independent and
  returns a `Family<T>` in index order. The closure runs once, inside a new construction scope,
  with `i` bound to a loop-index slot. The body is sealed with lexical captures: outer values
  become explicit arguments, `inputs.at(i)` becomes a `Zip` member input (`ZipOffset` for
  `i + c`), and other outer values are `Broadcast`. An indirect read such as
  `table.at(indices.at(i))` keeps the whole table as a `Broadcast` input plus an in-body dynamic
  lookup.
- `iterate(count, initial, |i, state| body)` builds a `SequentialLoop` that carries a state of
  invariant schema; a zero count returns the initial state.
- `select(selector, candidates)` builds one eager `Select` per leaf. A `Bool` selects candidate
  0 for false and 1 for true; an `Int` selects a zero-based candidate.
- `Subgraph::define(name, schema, body)` and `subgraph.call(input)` (`crates/dsl/src/subgraph.rs`)
  define a reusable named body; `call_with_canonical_input_exclusive_uppers` attaches canonical
  upper bounds to matrix arguments.

Sampler placement follows the scopes: a sampler outside a loop is shared by all instances; a
sampler inside a body samples once per executed instance, and a zero-count loop samples nothing.

### 4.5 Errors

`DslError` covers compile-time index recovery (`CompileTimeIndex`), freezing (`Freeze`),
duplicate outputs, schema errors, subgraph bound errors, `FamilyCountMismatch`, structural
validation, and specification errors. `ValidationBuildError::Core` wraps concrete validation
errors. Numerical failures (division by zero, a dynamic index out of range, artifact mismatches,
sampler failures) are runtime errors reported by the executor.

## 5. `mxx-backends`: values, primitives, and CPU execution

`mxx-backends` (`crates/backends/src/lib.rs`) is the only crate with concrete arithmetic. It has
two layers:

- **Primitive layer:** `element`, `poly`, `matrix`, `sampler`, `modulus`, `openfhe_guard`,
  `utils`, `env`. Polynomials, matrices, samplers, and codecs over OpenFHE.
- **Execution layer:** `backend` (runtime values and the CPU/GPU backends), `executor` (CPU
  executor), `artifact`, `session`, `transcript`, `authority`, `host_control`, `lean`, and, under
  the `gpu` feature, `gpu_runtime` (section 6).

### 5.1 Runtime value model

`RuntimeValue` (`crates/backends/src/backend.rs`) is what executors accept as inputs and return as
outputs:

| Variant | Meaning |
| --- | --- |
| `Int(BigInt)`, `Real(f64)`, `Bool(bool)`, `Bytes(Arc<[u8]>)` | Host scalars and byte strings. |
| `TypedBlob { type_name, schema_hash, bytes }` | Opaque typed payload. |
| `Matrix(PolyMatrix)` | A matrix with its concrete wire type; also used for `SmallMatrix` and `Preimage` wires. |
| `Trapdoor(TrapdoorValue)` | Public matrix plus secret `DCRTTrapdoor`. |
| `Resident(Arc<GpuResidentValue>)` (`gpu` only) | A non-matrix value resident on the device (integer/bool families, reals, bytes). |
| `IndexedFamily { element_type, values }` | An in-memory family. |
| `LazyArtifact`, `LazyArtifactFamily` | An artifact reference not yet loaded from the store. |
| `StagedArtifact`, `StagedArtifactFamily` | Family members streamed to the store during execution. |

`PolyMatrix` (the runtime struct in `backend.rs`, distinct from the `matrix::PolyMatrix` trait)
pairs a `ConcreteWireType` with shared storage: `CpuFull(DCRTPolyMatrix)`,
`CpuCompact(CpuSmallMatrix<DCRTPolyMatrix>)` for bounded matrices, `Gpu(GpuResidentValue)` under
the `gpu` feature, or `Encoded(bytes)`. Constructors (`cpu_full`, `cpu_compact`, `encoded`, `gpu`)
check that the storage agrees with the wire type: shape, ring dimension, ordered CRT basis, and
bounded-matrix bound and domain. Clones are shallow.

Helpers: `RuntimeValue::matrix`, `small_matrix`, `preimage`, `integer_values`, `indexed_family`,
`matches_wire_type`, and under `gpu` `RuntimeValue::gpu_matrix` (binds an evaluation-domain
`GpuDCRTPolyMatrix`, for example one created by `GpuDCRTPolyMatrix::from_cpu_matrix`, as a
resident input) and `RuntimeValue::gpu_signed_family`.

`TrapdoorValue::new` requires a secret and checks that the public matrix type and the secret's
ordered ring match the trapdoor wire type.

Lazy and staged values are materialized by `ExecutionResult::materialize_output(name, backend,
store)`; staged family members are removed with `ExecutionResult::cleanup_staged(store)`.

### 5.2 Polynomial and matrix primitives

- `poly::PolyParams` and `poly::Poly` (`crates/backends/src/poly/mod.rs`) are the parameter and
  polynomial traits. `DCRTPolyParams` (`crates/backends/src/poly/dcrt/params.rs`) holds the ring
  dimension, CRT depth and bit width, the exact ordered basis `moduli`, gadget base bits, and
  dropped moduli. `try_new` validates capability limits (power-of-two dimension, CRT width at
  most 60 bits, `base_bits <= ceil(crt_bits / 2)`) and generates or checks the basis through
  `openfhe_guard::gen_modulus_and_warmup`.
- `DCRTPoly` (`crates/backends/src/poly/dcrt/poly.rs`) wraps an OpenFHE `DCRTPoly`. The OpenFHE
  Rust bindings come from the `openfhe` crate; repository-owned C++ adapters in
  `crates/backends/native/ExactBasis.{h,cc}` are bridged with `cxx` in
  `crates/backends/src/poly/dcrt/native.rs` (exact-basis sampling, NTT tables, RNS conversions,
  matrix entry copies). `crates/backends/native/openfhe/README.md` explains the vendored
  declaration header. `crates/backends/build.rs` compiles the bridge and links OpenFHE and
  OpenMP.
- `matrix::PolyMatrix` (`crates/backends/src/matrix/mod.rs`) is the primitive matrix trait:
  arithmetic, batch operations, automorphisms, slicing, concatenation, tensor products,
  decomposition, modulus conversions, and the compact byte codec (`to_compact_bytes`,
  `from_compact_bytes`, `validate_compact_bytes`). `DCRTPolyMatrix = BaseMatrix<DCRTPoly>`
  (`crates/backends/src/matrix/dcrt_poly.rs`, `crates/backends/src/matrix/base/memory.rs`)
  implements it with Rayon-parallel entry loops. `CpuSmallMatrix<M>` stores a bounded matrix with
  its `max_coefficient_bound` and `CoefficientBoundDomain`.
- `modulus::modulus_raise` is the exact centered lift into a larger ring.

### 5.3 Samplers

`crates/backends/src/sampler/`:

- `DCRTPolyUniformSampler` (`uniform.rs`) draws uniform residues, bits, ternary values, and
  Gaussians. With a cutoff, Gaussian sampling resamples individual coefficients whose centered
  magnitude exceeds the cutoff; it never clips.
- `DCRTPolyHashSampler<H>` (`hash.rs`) derives matrices from `H(key || tag)` with per-entry and
  per-column framing, so column windows are consistent with whole-matrix sampling.
  `sample_hash_integers` (same file) samples a `HashIntFamily` from the entry `(0, 0)`
  coefficient streams of that transcript (`key || tag || row || column || coefficient ||
  attempt || block`).
- `DCRTPolyTrapdoorSampler` (`trapdoor/sampler.rs`) samples gadget trapdoors (`DCRTTrapdoor`) and
  preimages. CPU preimage sampling rejects and redraws a whole candidate that exceeds its cutoff,
  so the preimage equation always holds.
- `bounds.rs` defines the authoritative cutoffs: `hard_cutoff_from_sigma_bound` is
  `floor(6.5 * sigma_bound)`, and `default_preimage_cutoff` gives the minimum preimage cutoff for
  concrete parameters (`TrapdoorPreimageCutoffPolicy` rejects explicit cutoffs below it).

Correctness uses these enforced integer cutoffs and deterministic worst-case bounds. Lattice-
security estimation separately models the ordinary untruncated distributions.

The executor encodes `HashSample` and `HashIntFamily` tags (`hash_key_and_tag` in
`crates/backends/src/executor/cpu.rs`) as the fixed prefix followed by typed, length-framed
components, so tags such as `(1, 23)` and `(12, 3)` differ. Changing the tag encoding changes
every hash-derived value: rebuild serialized graphs and hash-derived artifacts together.

### 5.4 CPU backend and executor

`CpuDcrtBackend` (`crates/backends/src/backend/poly.rs`) is a concrete struct, not a trait
implementation. It is constructed from the `DCRTPolyParams` of every ring the graph will use
(`CpuDcrtBackend::new(params)`); rings are looked up by ordered CRT basis and dimension, and a
missing ring is a `MissingParameters` error. It implements every node's primitive (sampling,
arithmetic, conversions, codecs).

Entry points (`crates/backends/src/executor.rs`):

```rust
pub fn execute<S: SessionStore>(validated: &ValidatedGraph, backend: &mut CpuDcrtBackend,
    inputs: BTreeMap<String, RuntimeValue>, artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>, config: ExecutionConfig)
    -> Result<ExecutionResult, ExecutionError>;
```

- `execute_in_session(..., execution_nonce, config)` opens or resumes a durable session for the
  `ProductionId(spec_hash, nonce)` and binds it to a digest of the inputs.
- `execute_prepared` uses a session when the graph exports artifacts and plain `execute`
  otherwise.
- `execute_with_trace` additionally returns every intermediate value (`ExecutionTrace`).
- `SamplingMode` (`crates/backends/src/transcript.rs`) is `Fresh`, `Record(&mut
  TranscriptRecorder)`, or `Replay(&TranscriptReplayer)`; transcripts key sampled values by
  `DrawSite` (instantiation path, node, port).

`ExecutionConfig { max_parallel_instances (default 64), preimage_progress,
release_fence_interval }` controls execution. The executor (`crates/backends/src/executor/cpu.rs`):

1. walks each scope's validated `execution_order`, releasing values at their liveness
   `last_use` unless retained (trace mode retains everything);
2. executes subgraph calls and sequential loops by instantiating the body scope with extended
   instantiation paths and loop-index bindings;
3. executes a `ParallelLoop` in waves of at most `max_parallel_instances` instances.
   `Broadcast` inputs are materialized once, `Zip` inputs supply one member per instance, and
   the instances of one wave run node by node in lockstep so arithmetic and preimage sampling are
   issued as batched backend requests. Artifact-typed family outputs are streamed per wave to the
   store as staged artifacts; scalar families accumulate in memory;
4. applies dispatch-only fusions on validated root plans (row sums, tensor row sums,
   concatenation/slice aliases), cached per thread in `executor/plan_cache.rs`. Fusions never
   change the graph, its identity, or its outputs, and are disabled in trace mode.

Parallelism lives in the primitives: matrix operations (including `MultiplyMonomial`), integer
matrix-vector products, samplers, and bound checks use Rayon.
The executor itself orders nodes deterministically.

`ExecutionError` reports missing inputs or wires, value-kind mismatches, runtime integer errors
(`DivisionByZero`, `SelectIndexOutOfRange`), backend, artifact, transcript, and manifest errors.

### 5.5 Artifacts, sessions, transcripts, and authorities

- `ArtifactStore` (`crates/backends/src/artifact.rs`) loads manifests and payloads
  (`ArtifactKey { production, name, index }`, `ArtifactPayload`), reports payload sizes without
  loading them, stores payloads, and manages staged family chunks. `MemoryArtifactStore` and
  `FileArtifactStore` (alias `FilesystemArtifactStore`) implement it and `SessionStore`.
- `SessionStore` (`crates/backends/src/session.rs`) adds durable sessions: `SessionDescriptor
  { production_id, graph_name, ir_version, input_digest }`, `SessionStatus::{Running, Finalized}`,
  transcript recording, artifact commits, and finalization (payloads, then commit, then the
  manifest last). Stable aliases (`SessionAliasDescriptor`) resolve to a durable nonce.
- `ExecutionAuthority<S>` (`crates/backends/src/authority.rs`) is the prepare-then-run boundary
  used by applications: `prepare(validated, &inputs)` then `run(&mut prepared, inputs, store,
  nonce)`. `CpuExecution` implements it with `execute_prepared`; under `gpu`, `GpuRuntime`
  implements it with `Prepared = GpuExecutionPlan` and a borrowed result type.
- `host_control.rs` holds the shared dispatch of structural nodes and host primitives, used both
  by the executor and by setup-time measurement so the two cannot drift.

### 5.6 Caller and storage contracts

These contracts apply to both CPU and GPU execution:

- Execution accepts trusted, complete inputs prepared for the exact validated graph and bindings.
  Every declared non-artifact input must be supplied before execution starts, including session
  execution. Matrix dimensions, ring dimension, ordered CRT basis, representation, and bounded-
  matrix metadata must match the concrete wire type; for host-staged matrices, both the embedded
  metadata and the payload must describe the same value. These apply recursively to family
  elements.
- A supplied trapdoor must have the validated matrix type, sigma, gadget base, digit count, and
  preimage cutoff, and its public matrix and secret must come from the same construction.
  Execution does not copy resident matrices to the host or scan their contents to re-validate
  them; producers enforce these properties.
- GPU execution checks only what addressing needs: the exact input set, the physical layout of
  each resident input, and the frozen range of each host integer while encoding it. A resident
  integer input carries its producer-proven range. A host integer input uses its entry in
  `GpuRuntimeOptions::integer_input_ranges`, or else the full signed range of the fewest 64-bit
  words (at least one) that hold its planning values.
- A session nonce binds an immutable, complete input map. A failed execution can leave a durable
  session descriptor; changing inputs requires a new nonce (and a new alias). Retrying with the
  original nonce resumes the original inputs.
- Artifact stores must return intact payloads produced by the matching backend codec, schema, and
  parameters. The compact matrix decoder is not an untrusted-data parser: malformed payloads can
  panic. Manifests do not authenticate payload integrity; applications reading untrusted storage
  must establish integrity first. Serialized graphs must come from the graph serializer and pass
  validation before execution.

### 5.7 Backend Lean layout

`crates/backends/src/lean/` exports concrete CRT gadget layouts (`export_dcrt_layouts`,
`render_backend_context`) taken from the actual `DCRTPolyParams`. The handwritten Lean packages
in `crates/backends/lean/` provide `MxxPrimitives` (bounds, CRT decomposition, radix, negacyclic
arithmetic, preimage and sampling facts) and `MxxRuntime` (successful-sampling relations and
matrix operations used by generated scope relations).

## 6. `mxx-backends`: the GPU runtime

The GPU runtime is compiled only with the `gpu` feature. Its public module is
`mxx_backends::gpu_runtime` (source file `crates/backends/src/gpu_runtime_direct.rs`); the planning
metadata in `gpu_execution_plan.rs`, `gpu_schedule.rs`, and `gpu_warmup`
(`gpu_runtime_metrics.rs`) is always compiled. The lowering and I/O modules
(`gpu_physical_lowering.rs`, `gpu_physical_control.rs`, `gpu_io_worker.rs`, `gpu_runtime_io.rs`,
`gpu_runtime_import.rs`, `gpu_runtime_digest.rs`) are crate-private.

Two principles define the design:

- **GPU computation runs only through `GpuRuntime`.** The old eager GPU matrix, polynomial, and
  sampler API has been removed: no GPU type implements `Poly` or `matrix::PolyMatrix`, and there
  is no GPU sampler type. `GpuDCRTPolyMatrix` (`crates/backends/src/matrix/gpu_dcrt_poly.rs`) is
  only a device allocation that plans bind; `GpuDCRTPolyMatrix::from_cpu_matrix` uploads a CPU
  matrix as evaluation-domain residues, and downloads go through `GpuRuntime::download_*`.
- **Matrix encodings belong to the plan.** Whether a matrix is in coefficient or evaluation form
  is a property of the plan's `PhysicalValue`, not of its native allocation. NTT and inverse NTT
  are planned operations.

### 6.1 Lifecycle: plan once, execute many times

```rust
let backend = mxx_backends::backend::poly_gpu::gpu_backend(gpu_params); // or gpu_backend_on(params, devices)
let mut runtime = GpuRuntime::new(backend)?;          // reads GpuRuntimeOptions::from_env()
let mut plan = runtime.plan(graph, &inputs)?;         // a BuiltGraph or a ValidatedGraph
let result = runtime.execute(&mut plan, inputs)?;     // or execute_with_artifacts(.., store, nonce)
let next_inputs = BTreeMap::from([("ct".into(), result["ct"].clone())]);
let matrix = runtime.download_matrix(&result["result"])?;
```

- `GpuRuntime::plan(graph, &inputs)` takes a `ValidatedGraph` or a DSL `BuiltGraph` (validated
  with default parameters through `IntoValidatedGraph`), lowers the graph, allocates it, compiles it to CUDA
  Graph regions, measures candidates, and returns a `GpuExecutionPlan`. `plan_with_store` also
  queries the store for artifact payload sizes (never payload bytes) so that integer and
  typed-blob imports get fixed, pointer-stable destinations.
- Inputs and outputs are keyed by their DSL names. A composite DSL value (a ciphertext, a key)
  flattens into the graph leaves `name.0`, `name.1`, ...; the runtime accepts and returns it as one
  `RuntimeValue::Composite` (`expand_composite_values` and `group_composite_values` in
  `crates/backends/src/backend.rs`). Two inputs that plan on one resident allocation (the same
  example value for both operands) are planned on a device copy of the repeat, since input
  rebinding redirects views by allocation.
- `GpuRuntime::execute(&mut plan, inputs)` draws fresh sampling randomness and needs no artifact
  store; a plan with artifact inputs or outputs runs through
  `execute_with_artifacts(&mut plan, inputs, store, nonce)`. Both rebind new inputs and replay the frozen
  program. It does not plan, measure, or re-validate. A plan executes only on the backend
  instance that planned it (`GpuRuntimeError::StalePlan` otherwise). A resident input produced by
  another plan is moved into this plan's storage slots when it is rebound
  (`with_planned_storage` in `gpu_physical_lowering.rs`), so outputs of one plan can feed
  another plan regardless of which plan produced them. `PhysicalFrame::rebind_inputs` collects
  the replaced allocations in a hash map keyed by owner pointer; only inputs whose storage
  actually changed contribute (binding the same input again leaves every view untouched), and
  only plan values aliasing a replaced allocation are re-derived (`GpuResidentValue::rebound` in
  `crates/backends/src/backend.rs`). A member or view of a family
  (`GpuResidentValue::with_physical_view`) binds only the storages its parts use, so it does not
  carry or rebind every allocation of the family. Input ready events are waited once per event,
  and a `GpuNativeEvent` caches completion after a successful host wait. Binding a compiled
  graph (`mxx_gpu_graph_bind` in `crates/backends/cuda/src/Runtime.cu`) updates only nodes whose
  patched argument bytes changed; nodes start bound to their build-time arguments, and an update
  of a conditional body, which reconciles the whole executable, reapplies every top-level node.
- The result, `GpuExecutionResult`, owns its outputs: `result[name]` (or `into_outputs`) gives
  values that can be kept, downloaded, or bound as inputs of any plan, including the same plan.
  Each output has one caller handle (`PhysicalFrame::output_handles`), reused while its plan
  owner is unchanged; the next execute writes an output whose handle still has another reference
  into fresh storage, together with every plan value viewing its old allocation, before inputs
  rebind. An output the caller released is written in place with no allocation. Planning uploads
  each compiled executable, so the first production launch pays no device-side graph setup. `result.output(name)` returns a `GpuOutputRef` for the typed downloads
  (`download_matrix_output`, `download_matrix_member_output`, `download_integer_family_output`,
  `download_bool_output`, `download_real_output`, `download_bytes_output`).
- `GpuExecutionPlan::report()` returns the `GpuWarmupReport` selected during planning; reading it
  performs no measurement. `compiled_region_count` and `compiled_launch_count` expose compiled
  structure.
- Applications normally use the `ExecutionAuthority` implementation for `GpuRuntime`
  (`crates/backends/src/authority.rs`), which maps `prepare` to planning and `run` to
  `execute`.

`GpuRuntimeOptions` (`crates/backends/src/runtime_env.rs`) is read once by `GpuRuntime::new` and
can be adjusted with `options_mut`:

| Field | Environment variable | Default |
| --- | --- | --- |
| `max_parallel_instances` | `MXX_GPU_MAX_PARALLEL_INSTANCES` | 64 |
| `measurement_warmups` | `MXX_GPU_MEASUREMENT_WARMUPS` | 1 |
| `measurement_iterations` | `MXX_GPU_MEASUREMENT_ITERATIONS` | 2 |
| `release_fence_interval` | `MXX_GPU_RELEASE_FENCE_INTERVAL` | unset |
| `integer_input_ranges` | (set in code) | empty |

`crates/backends/src/env.rs` also defines `MXX_CUDA_STREAM_POOL_SIZE` (compute streams per
context and device, default 32), `MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS` (GPU preimage retry
bound per column tile, default 64; zero or malformed values are errors), and
`MXX_GPU_LOGICAL_DEVICES` (`gpu_logical_devices`: one physical CUDA device id per logical device,
for example `0,0`; unset or empty means one logical device per detected GPU, and an invalid value
panics; see section 6.4), `MXX_GPU_MEMORY_FRACTION` (`gpu_memory_fraction`: the fraction of each
device's memory that one plan's persistent allocations may use, default 0.8, values outside
`(0, 1]` are errors), and `MXX_GPU_SMALL_RHS_CHUNK_COLUMNS` (`gpu_small_rhs_chunk_columns`: the
right-operand columns the fused small-RHS multiplication transforms per chunk, default 16).
A copy between two physical GPUs without peer access is staged through a pinned host buffer
(a device-to-host and a host-to-device graph node). The staged copies between one GPU pair in
one graph take turns on two buffers, so pinned memory is bounded by the GPU pairs and the largest
copy rather than the number of copies; `MXX_GPU_HOST_STAGED_COPIES=1` stages every
copy between distinct logical devices, which lets `MXX_GPU_LOGICAL_DEVICES=0,0` test that path
on one GPU. The CUDA
library reads `MXX_GPU_NTT_RADIX` once per process in
`crates/backends/cuda/src/matrix/MatrixNTT.cu`: the butterfly radix of the register-blocked
NTT, a power of two from 2 to 32 (default 4). Each thread holds that many coefficients and runs
`log2(radix)` stages in registers between shared-memory exchanges, so a larger radix needs fewer
barriers but gives each thread a longer serial chain; small batches such as TFHE blind rotation
are latency-bound and run fastest at 2 or 4. An invalid value makes every NTT launch fail.

### Subgraph kernels

A named subgraph (`mxx_dsl::Subgraph`, an IR `SubgraphCall`) may have a native GPU kernel that
executes the whole call. The IR is unchanged: validation, liveness, the CPU executor, and Lean
export read the subgraph body as usual. Only GPU planning consults
`GpuRuntimeOptions::subgraph_kernels` (`crates/backends/src/gpu_subgraph_kernel.rs`):

- A call whose definition name equals a registered `GpuSubgraphKernel::name` lowers to one
  `SubgraphKernel` operation instead of its body. Its arguments (explicit inputs, then captures)
  and results must have the registered `GpuKernelOperandKind`s; otherwise planning fails.
- Matrix operands are passed in the evaluation domain, integer operands with their signed
  encoding, and matrix families as a per-member limb table refreshed before every launch (as for
  a dynamic member read). The runtime allocates the results, a scratch buffer of
  `scratch_bytes`, and a status word.
- While the CUDA graph is built, the runtime calls `entry` once with an `MxxSubgraphLaunch`
  (`crates/backends/cuda/include/SubgraphKernel.cuh`). The entry adds its kernels with
  `mxx_gpu_launch_kernel` or `mxx_gpu_launch_cooperative_kernel`, declaring every resident
  address as a patch of its operand's binding so replays rebind it.
- An empty list runs every subgraph from its body.

`mxx-fhe` registers the blind rotation (`tfhe.blind_rotation`,
`TfheParams::gpu_blind_rotation_kernel`, `crates/fhe/cuda/tfhe_blind_rotation.cu`): the whole
CMUX loop runs as one cooperative launch. `crates/fhe/tests/gpu_tfhe.rs` checks that its
ciphertexts equal those of the subgraph body bit for bit.

Errors: planning returns `GpuPlanError` (`InvalidInput`, `Resource`, `Measurement`,
`GraphCompile`, `InvalidCompiledSchedule`). Execution returns `GpuRuntimeError`: `StalePlan`,
`Execution`, `LaunchUncertain`, `DeviceStatus`, `Artifact`, `Session`. `StalePlan` means only
that the plan came from another backend instance; an input that does not match the planned
layout fails as `Execution("input rebinding failed: ...")`. A data-dependent failure
reported by a device status word after its launch joined (for example integer division by zero,
an invalid selected index, or exhausted preimage retries) is `DeviceStatus`: outputs are
suppressed and the plan remains reusable. An uncertain launch drains the device and poisons the
plan; later executions fail with `LaunchUncertain`.

### 6.2 Planning: measured W/C candidates

Planning (`GpuRuntime::plan_with_payload_sizes` in `gpu_runtime_direct.rs`) chooses two numbers:

- **W**, the wave width: the number of parallel-loop instances one replay of a wave template
  handles. One W is shared by every wave loop site of the plan; each site uses
  `min(W, count)` lanes and a tail wave for the remainder (`GpuLoopChoice { key:
  GpuLoopSiteKey, loop_count, wave_instances, tail_instances }` in `gpu_execution_plan.rs`). The
  largest candidate is the largest finite loop count found by a probe lowering, capped at
  `max_parallel_instances`.
- **C**, the column tile width: the number of matrix columns processed per job, derived from the
  widest matrix (or matrix-family) output. With several devices, each node freezes one width per
  logical device (section 6.4); the trial grid itself does not depend on the device count.

Both use a geometric grid (`geometric_candidates`): W in `1, 2, 4, ...` plus the maximum itself,
and C in `ceil(columns / t)` for `t = 1, 2, 4, ...` plus `t = columns`, so both extremes are always
tried. For every `(W, C)` pair the planner lowers the graph (`single_root_physical_plan`,
`plan_physical_graph`), actually allocates the physical frame, checks its persistent allocations
against the device budget, compiles the CUDA Graph regions, and runs
`measurement_warmups + measurement_iterations` trials. A candidate whose allocation, compilation,
or trial fails is rejected and the search continues with the next candidate; no VRAM requirement
is predicted. Graph-owned scratch (section 6.3) is the one exception, because CUDA keeps the
reservation of a Graph upload or launch that runs out of memory and the process can then no
longer allocate: `DirectGraph::compile` (`crates/backends/src/gpu_runtime_direct.rs`), after
building all regions and before `executable.upload(...)` of any of them, calls
`admit_graph_scratch`, which compares the scheduled scratch peak (the bytes of allocations whose
memory cannot yet be reused, plus 1/32 for CUDA's rounding) with the memory currently free and
fails compilation with `GpuPlanError::Resource` if it does not fit, so the candidate is rejected
and the search continues. Uploading every region while planning means the first production launch
does not pay the device-side Graph setup. After each candidate,
`gpu_release_cached_memory` synchronizes the device, trims the Graph and default pools, and makes
one whole-device `cudaMalloc` request that cannot succeed: the driver caches the device memory of
destroyed Graph executables (several KiB per kernel node) and releases it only to `cudaMalloc`,
not to pool or Graph allocations. This device-wide synchronization happens only while planning;
`execute` waits on streams and events. Each
region's measured time is weighted by how often production replays it (waves times active parent
occurrences). The feasible candidate with the smallest measured time is re-lowered and frozen;
if no candidate is feasible, planning fails with the collected rejection reasons. Trials measure
time only; device status words are data-dependent and are checked by `execute`.

The frozen, value-only record of these choices is `FrozenGpuPlan { contract: GpuPlanContract,
layouts, loops: Vec<GpuLoopChoice>, nodes: Vec<GpuNodeChoice> }`. It never owns device buffers,
pointers, or command objects. `GpuNodeChoice::columns_per_job` holds one column width per logical
device (0 marks an inactive device), and the report's `columns_per_job` likewise has one entry per
device.

### 6.3 Physical lowering

Lowering (`gpu_physical_lowering.rs` for the frame and matrix operations,
`gpu_physical_control.rs` for control flow, scalars, and families) turns the validated graph into
a `PhysicalFrame`: a `CompiledGpuProgram` plus the resident owners, export slots, import
templates, control resets, and wave descriptors needed to run it.

- **Physical values.** `PhysicalValue { ty, encodings, parts, integer_ranges }` describes a value
  as one or more `PhysicalPart { leaf, storage: StorageRef, device, view: PhysicalView }`, where
  `device` is a logical device id (section 6.4).
  `StorageRef` is `Input(i)`, `Output(i)`, or `Scratch(i)`: an allocation slot local to the plan
  or to a resident value. `PhysicalView` is an origin/extent/stride view into that allocation.
  `PhysicalEncoding` covers `FullCoeff` and `FullEval` matrices, compact bounded coefficients
  (`CompactCoeff`, `CompactCoeffPerCrtLimb`), signed integers (`Signed`), `BoolI64`, `RealF64`,
  `Bytes`, `TypedBlobLengthPrefixed`, and the public-only gadget trapdoor `PublicGadgetEval`.
- **Integer family storage.** An integer family is stored as `SignedWords(words)` (one sign word
  and `words` magnitude words per member) or as `CanonicalU64` (one word per member, range in
  `[0, 2^64)`). Hash integer families with modulus at most `2^64` are `CanonicalU64`. A returned
  (graph output) integer family is `CanonicalU64` exactly when its proven range lies in
  `[0, 2^64)` and SignedWords otherwise (`plan_physical_graph`), so its layout depends on its
  range, not its producer, and outputs of different graphs rebind into one another's plans.
  Polynomial import (`PolynomialFromValues`) reads SignedWords, so it widens a canonical family
  with an integer `Copy` first.
- **Compiled operations.** `CompiledGpuOp { implementation, arguments, outputs, device, grid,
  block, shared_bytes, predecessors, body }` is one native launch with explicit dependency edges.
  `body` holds the nested operations of a CUDA conditional node (`GpuNativePrimitive::BranchIf`
  or `LoopWhile`). `GpuNativePrimitive` enumerates the native operations (NTTs, arithmetic,
  RNS conversions, samplers, preimage stages, copies, exports, control, ...). A
  `GpuImplementation` is a primitive with its argument schema, and `GpuImplementationRegistry`
  is keyed by the full implementation, so one primitive may have several schemas (for example
  `GpuImplementation::matrix_copy_views(count)`, one six-argument group per copied window).
- **Explicit CUDA Graph regions.** `DirectGraph::compile` splits the operation list into
  `GraphRegion`s at wave-body boundaries, import points, and external-I/O loop bodies, and builds
  each region with `GpuNativeGraphBuilder` (`crates/backends/src/poly/dcrt/gpu.rs`). One region is
  one CUDA Graph even when its operations run on several devices (section 6.4). Unrelated
  operations keep independent paths because each operation declares its predecessors. At bind
  time a region rejects any owner whose physical descriptor differs from the planned one.
- **Waves.** A `ParallelLoop`, at the root or nested inside another loop body, lowers to a
  reusable W-lane wave template (`PhysicalWave` in `gpu_physical_control.rs`) that is replayed
  once per wave with fresh lane bindings; nested wave templates lie inside their parent's body
  and are replayed for every active parent occurrence. Every template of one plan uses the same
  per-site W choice from section 6.2. Each member returned by a wave has its own owner.
- **Vectorized scalar loops.** A parallel loop whose body contains only scalar `Int`/`Bool`
  arithmetic, comparisons, selection, and family reads (`is_vectorized_scalar_loop`) is not
  replayed in waves. It lowers once, with one elementwise operation per body node over all lanes.
- **Integer control and status words.** Runtime `Int`/`Bool` operations, `Select`, and
  sequential loops execute on the device. Integer operations report errors (division by zero,
  overflow, invalid index, inexact division, invalid ring property; `MxxGpuControlStatus` in
  `crates/backends/cuda/include/Control.cuh`) into resident status words. Errors are first-wins, each
  replay resets a status word once, and status words are checked only after the launch joins,
  instead of one host reset and readback per operation. Each device has its own status word
  (`PhysicalLoweringContext::integer_status` is a per-device map).
- **Sequential-loop carries.** A sequential loop may carry matrices, scalars, and `Int` families
  (`is_integer_carry` in `gpu_physical_control.rs`). An `Int` carry, scalar or family, is sized
  by a range closed over every iteration (`carry_range`), not by its initial value. The integer
  range analysis gives a Euclidean remainder by a positive divisor the exact range
  `[0, divisor)` (capped by a nonnegative dividend's upper bound), so a carry reduced modulo a
  constant each iteration keeps a finite range.
- **Hash samplers.** `HashSample` and `HashIntFamily` share the `HashSample` native primitive:
  `hash_tag_resource` registers the tag and `push_hash_sample` emits the sample
  (`gpu_physical_lowering.rs`), and `lower_hash_int_family` (`gpu_physical_control.rs`) lowers
  the integer family. An integer family uses a `GpuHashSamplePlan` without CRT moduli, and
  `GpuHashSamplePlan::emit_raw_hash_integers` (`crates/backends/src/poly/dcrt/gpu_hash.rs`)
  launches `raw_hash_integer_kernel` through `gpu_raw_hash_integers_emit`
  (`crates/backends/cuda/src/matrix/MatrixHash.cu`); its `sign_words` argument (0 or 1) writes
  either a `CanonicalU64` or a SignedWords family. The output matches the CPU transcript.
- **Monomial products.** `MultiplyMonomial` (`lower_multiply_monomial` in
  `gpu_physical_control.rs`) takes its operand as `FullEval` and adds no NTT of its own:
  `raw_monomial_multiply_kernel` (`crates/backends/cuda/src/matrix/MatrixRawRemaining.cu`)
  reduces the resident exponent modulo `2n` and multiplies each evaluation slot by the matching
  twiddle or its negation.
- **Integer matrix-vector products.** `lower_int_matrix_vector_product`
  (`gpu_physical_control.rs`) emits integer operation 19 (`GpuIntegerOperation::MatrixVectorProduct`,
  `crates/backends/cuda/src/Control.cu`). `M v` uses one warp per row with shuffle reduction;
  `v^T M` clears the output, accumulates 64-row chunks per column with atomics, then converts the
  sums to sign-magnitude. Operands must be one-word integer families, the body must not be
  vectorized, and the operand ranges and the proven output range must fit in `int64`, so the
  kernel accumulates exactly without widening.
- **Slices and concatenation.** A matrix `Slice` that is not a root graph output is a view: the
  window retyped to the output shape over the source allocation, with no copy. A `Concat` piece
  whose writers are elementwise or product operations (add, subtract, multiply, tensor, scale,
  monomial, automorphism), and which nothing else reads, is written straight into its window:
  every view of the piece's scratch allocation moves into the output at the window offset when
  the strides agree (`concat_piece_writers` in `gpu_physical_control.rs`). The remaining pieces
  are copied into their windows by one `MatrixCopyView` operation.
- **Returned matrices.** A root output that is a whole plan-owned scratch matrix is returned in
  place: its storage is relabeled as return storage, which scratch sharing never reuses, instead
  of being copied into a separate return allocation.
- **Preimage retry loop.** Each preimage column tile is a device `LoopWhile` body that derives a
  per-attempt seed (`crates/backends/cuda/src/matrix/MatrixPreimageSeed.cu`), samples a
  candidate, and checks its cutoff, repeating until acceptance or until the frozen
  `preimage_max_attempts` bound (`MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS`) is reached. Exhaustion is
  a `DeviceStatus` error after the join.
- **Conversions only when encodings differ.** `full_eval_value` and `full_coeff_value` return an
  operand unchanged if it already has the requested encoding; otherwise the value gets exactly one
  forward or inverse NTT, cached in `converted` and shared by all later consumers. When the
  source has no later reader, the conversion writes in place. Coefficient-domain operations
  (automorphisms, modulus and RNS conversions, rounding, CRT recomposition, gadget
  decomposition, samplers, packs, imports, and exports) consume and produce `FullCoeff` directly.
- **Graph-owned scratch.** Lowering creates scratch values as deferred owners
  (`GpuDeferredScratch` in `crates/backends/src/gpu_graph_memory.rs`) that carry only a layout.
  `DirectGraph::compile` calls `plan_graph_scratch`, which makes a scratch value Graph-owned when
  it is written before it is read, lives on one device, and stays within one region (or spans
  regions whose endpoints are not replayed). Each region's operations are emitted in order: a
  CUDA memory-allocation node precedes the first writer and a free node follows the last reader,
  so CUDA reuses freed memory inside the Graph, as a caching allocator would. The builder keeps no
  implicit chain: `mxx_gpu_graph_builder_add_memory_alloc` and `_free`
  (`crates/backends/cuda/src/Runtime.cu`) take explicit `after` tokens, and `GraphScratchPlan`
  (`enter_operation`, `push_memory_node`) supplies them. Outside parallel loops the memory nodes
  form one chain in operation order, each following the previous one, which serializes them and
  keeps the peak low; the column-parallel jobs of one operation write column views of one output
  allocation with no memory nodes between them, so the chain does not serialize them. Lowering
  records the top-level operation range of every lane of each parallel loop in
  `PhysicalFrame::parallel_lanes` (`lower_parallel_loop` in
  `crates/backends/src/gpu_physical_control.rs`; lanes lowered inside a device body are not
  recorded, because the body is one top-level operation). Inside the outermost parallel loop every
  memory node follows only the chain at the loop's start and no memory is reused: a free node joins
  all readers of its allocation, and making later work depend on such joins made CUDA run the W
  lanes one after another (a 64-lane keygen region went from about 4.4 ms to 62 ms). When the loop
  ends, the chain joins every memory node created inside it, so memory freed by the W lanes is
  reused by the next W-wave replay and by later operations, and the loop's scratch peak grows with
  W. `peak_bytes` counts each allocation from its first use until its memory can be reused: after
  its last use, or after the end of the outermost parallel loop containing its last use. Memory
  nodes are not allowed in conditional bodies, so scratch used inside a loop body is allocated
  before the loop and freed after it. An allocation freed by a later region is recorded on the
  `DirectGraph`, freed on the launch stream after that region's launch (or when the `DirectGraph`
  is dropped). Every other scratch value, together with inputs, outputs, wave-bound members, and
  imports, is materialized as a persistent allocation.

### 6.4 Multiple GPUs

A plan distributes selected work over every device of its backend. Everything the plan and the
backend call a device (`PhysicalPart::device`, `CompiledGpuOp::device`, backend device ids) is a
**logical** device id; the **home device** is the first logical device.

- **Logical devices.** A process-wide native table maps logical to physical CUDA devices
  (`gpu_configure_logical_devices`, `mxx_physical_device`, `mxx_set_device`, `mxx_get_device` in
  `crates/backends/cuda/src/Runtime.cu`, declared in `crates/backends/cuda/include/Runtime.cuh`).
  It is the identity unless `MXX_GPU_LOGICAL_DEVICES` is set; `ensure_logical_devices`
  (`crates/backends/src/poly/dcrt/gpu.rs`) installs it once, before any device query, and
  `detected_gpu_device_ids()` returns logical ids. `MXX_GPU_LOGICAL_DEVICES=0,0` gives two logical
  devices on GPU 0, so multi-device plans run and are tested on one GPU. Native code selects
  devices only through `mxx_set_device`, never raw `cudaSetDevice`. Logical devices sharing a
  physical GPU each report that GPU's full memory budget, which is fine for tests but not for
  capacity planning.
- **Peer access.** `gpu_context_create` calls the native `enable_peer_access`, which enables CUDA
  peer access and default-mempool access between distinct physical GPUs where the hardware allows
  it.
- **One Graph across devices.** `DirectGraph::compile` emits each operation through the launch
  stream of its own device (`GpuNativeGraphBuilder::replace_launch_stream`; the native
  `mxx_gpu_graph_builder_for_stream` falls back to the thread's active builder for another
  device's stream, and conditional gate kernels select the stream's device). The graph launches on
  the home device's stream, and `GraphRegion::peer_streams` holds one stream per other device of
  the plan. Before each launch, every per-launch resource (reals, bytes inputs, seeds, indexed
  tables, preimage attempt/status words, prepared workspaces) is prepared on its own device's
  stream after the previous launch and joined into the launch stream with events
  (`GpuNativeLaunchStream::record_event`, `GpuNativeEvent::enqueue_wait`); after the launch,
  resources are protected per device.
- **Copies only.** Cross-device data moves only through copy nodes; no kernel reads remote memory.
  The `Copy` primitive (`compiled_span_part` in `crates/backends/src/backend/poly_gpu/fleet.rs`)
  copies one contiguous span, runs on one of its two devices, and may cross to the other
  (`cudaMemcpyDefault` when the devices differ). CUDA cannot update pitched (2D/3D) memcpy nodes
  in an instantiated graph, so strided windows are first packed on their own device
  (`MatrixCopyView`) and then moved as contiguous spans. `replicate_to_device`
  (`crates/backends/src/gpu_physical_lowering.rs`) copies any value into same-layout storage on
  another device with one contiguous copy per storage, covering only the bytes its parts view.
- **Column ownership.** `GpuLayout::owner_intervals` assign contiguous column blocks to devices
  (`balanced_owner_intervals`), and `GpuLayout::schedule` turns them into per-device column jobs
  using the per-device widths of `GpuNodeChoice::columns_per_job`.

What runs where:

- **Matrix products** (`MatrixBinaryOp::Multiply` in `lower_matrix_node`) split output columns
  evenly over all devices. A remote job computes on copies of the whole left operand and of its
  right column window (windows are packed first by `matrix_columns_on_device`), cached per value,
  column range, and device within the node; its output shard is copied home and packed into the
  output's columns. Elementwise addition and subtraction stay on the home device, where copies
  would cost more than they save.
- **Preimage sampling** (`lower_preimage_sample_node`) assigns column tiles to devices in
  contiguous blocks from the layout schedule. A remote tile runs its whole retry loop (the
  `LoopWhile` body) on its device, with copies of the public matrix, the trapdoor leaves `r`, `e`,
  and `re`, the inverse trapdoor, and its target window. Each device writes a compact output block
  that `copy_compact_block_home` copies home row by row, because the compact payload is row-major.
- **Parallel loops** (`lower_parallel_loop` in `crates/backends/src/gpu_physical_control.rs`)
  spread the lanes of an outermost wave loop over the devices: lane `l` runs on device `l mod G`.
  Broadcast inputs are copied once per loop to each remote device before the wave template; `Zip`
  members bind the home placeholder per wave and are copied into the lane inside the template;
  outer device-loop indices are copied into the lane; a remote lane's results are copied home, so
  family members always live on the home device. Nested wave loops inside a lane and anything
  inside a device body (conditional, retry, or sequential-loop body) stay on the current device;
  products and preimages inside a remote lane run on that lane's device.
- **Everything else** runs on the home device.

Tests that assert work lands on every logical device when run with `MXX_GPU_LOGICAL_DEVICES=0,0`
(or `0,0,0`): `matrix_products_shard_columns_over_devices`
(`crates/backends/tests/gpu_direct_node_semantics.rs`),
`test_gpu_direct_parallel_lanes_spread_over_devices` (`gpu_physical_control.rs`, using
`plan_with_fixed_geometry_for_test`), and `direct_preimage_sample_tiled_columns_match_public_relation`
(`gpu_physical_lowering.rs`). The whole GPU suite passes in identity, `0,0`, and `0,0,0` modes.

### 6.5 Artifact I/O

- **On-demand import.** An artifact input is not read at planning time. The plan records an
  `ImportTemplate` with the operation before which the payload is needed; at execution the region
  boundary joins the preceding work and `load_import_template` (`gpu_runtime_import.rs`) uploads
  the payload into a plan-owned destination. Selected family members
  (`family.at(dynamic_index)`) are imported individually by selector, without loading the whole
  family.
- **Preallocated export slots.** Exported outputs write into export slots reserved at planning
  time (`reserve_gpu_export_slots` in `gpu_execution_plan.rs`, `PlannedExportSlot` in
  `gpu_runtime_io.rs`). An observer thread watches for ready slots during the launch and forwards
  references to the I/O worker; commits happen only after the whole launch succeeds.
- **I/O worker.** `with_scoped_producer_io_worker` (`gpu_io_worker.rs`) runs one scoped worker
  thread that owns the `SessionStore` and serializes imports, export slots, commits, transcoding,
  and session finalization (`IoCommand`). Before reusing input or scratch storage, the runtime
  joins both the previous GPU launch and all artifact readers.
- Producer executions open a session keyed by `ProductionId(spec_hash, nonce)` and a digest of
  the canonical inputs (`gpu_runtime_digest.rs`), mirroring CPU `execute_in_session`.

### 6.6 Backend, fleet, and related contexts

- `GpuDcrtBackend` (`crates/backends/src/backend/poly_gpu/fleet.rs`) is the set of registered CUDA
  contexts: one placement per logical device, each holding one `GpuDCRTPolyParams` per ring,
  selected by ordered CRT basis and ring dimension. Each fleet context is single-device and holds
  every CRT limb of its matrices. The backend carries a unique execution identity used to bind
  plans to their backend.
- `gpu_backend(params)` uses every logical device from `detected_gpu_device_ids()`
  (`crates/backends/src/poly/dcrt/gpu.rs`); `gpu_backend_on(params, device_ids)` restricts the
  fleet (`crates/backends/src/backend/poly_gpu.rs`). A plan uses every device of its backend.
- **Related contexts.** On each device, the first registered ring is the anchor and the others
  are created as related contexts (`GpuDCRTPolyParams::params_for_device(device, related)`, native
  `gpu_context_create` in `crates/backends/cuda/src/Runtime.cu`). Related rings with different
  bases share one execution owner: identity, stream pool, release streams, and Graph builder, so
  operations mixing rings (for example RNS mod up/down) stay on one ordered execution.

### 6.7 Native CUDA layer

Native code lives in `crates/backends/cuda/`. Headers in `include/` declare only cross-file and
Rust-facing functions; bodies live in `src/`.

| File | Role |
| --- | --- |
| `src/Runtime.cu`, `include/Runtime.cuh` | C ABI for contexts, execution owners, streams, memory, export slots, the explicit Graph builder (including conditional `if`/`while` nodes), binding, launch, and events. |
| `src/Control.cu`, `include/Control.cuh` | Device integer control operations (`integer_operation_kernel`, the integer matrix-vector product kernels of operation 19) and status codes. |
| `src/Primitive.cu`, `include/Primitive.cuh` | Scalar-polynomial primitives (polynomial values, coefficient extraction, packing, threshold decoding). |
| `src/Real.cu`, `include/Real.cuh` | Device `f64` operations with their own status word. |
| `src/ChaCha.cu`, `include/ChaCha.cuh` | Device ChaCha RNG (included by the matrix unity build). |
| `src/matrix/Matrix.cu` | Unity build of the matrix layer: includes `../ChaCha.cu`, `MatrixUtils.cu`, `MatrixNTT.cu`, `MatrixData.cu`, `MatrixDecompose.cu`, `MatrixSampling.cu`, `MatrixTrapdoor.cu`, `MatrixSerde.cu`, `MatrixCrt.cu`, `MatrixRawRns.cu`, `MatrixSmallRhs.cu`, `MatrixPreimageRaw.cu`, `MatrixPolynomialValues.cu`, `MatrixRawRemaining.cu`, `MatrixIndexed.cu`, `MatrixPreimageSeed.cu`, and `MatrixHash.cu`. |
| `include/matrix/*.cuh` | Matrix structure (`Matrix.cuh`), CRT, data, NTT, serialization, compact small-RHS, and utility declarations. |

`crates/backends/build.rs` compiles exactly `Runtime.cu`, `Primitive.cu`, `Control.cu`,
`Real.cu`, and `matrix/Matrix.cu` into the `gpupoly` library when the `gpu` feature is enabled
(`CUDA_ARCH`, default `89`; `CUDA_HOME`; `CUDA_LIB_DIR`; `NVCC`), and embeds a hash of the CUDA
sources as `MXX_NATIVE_KERNEL_BUILD_REVISION`. The matrix sources included by `Matrix.cu` are not
compiled on their own. GPU-specific Rust code lives in files whose names contain `gpu`, as
required by `GPU.md`.

Raw matrix kernels:

- **Limb batching.** One launch covers up to 8 CRT limbs (`RawLimbSet`, `RawNttBatch`,
  `RawCopyBatch`, `RawDecomposeBatch`); `blockIdx.z` selects the limb (or the window-limb or
  source-output limb pair), and graph patches target the nested address fields of these argument
  structs by byte offset. Addition and subtraction, multiplication, tensor products, NTTs, copies
  (all windows of one `MatrixCopyView`), and decompositions are batched this way.
- **Fused NTT** (`src/matrix/MatrixNTT.cu`). `raw_ntt_fused_local_kernel` runs up to ten
  butterfly stages of a 1024-coefficient tile in shared memory, with the stage twiddles staged in
  shared memory, and applies the twist or scaling when the tile is the whole transform.
  `raw_ntt_fused_top_kernel` runs the stages above one tile with warp XOR shuffles
  (`Width = n / 1024` lanes per group). A forward transform runs the top kernel and then the local
  kernel in place; an inverse runs them in the opposite order. Ring dimensions up to 32768 are
  supported. Shoup multiplication uses a 32-bit path for moduli below `2^31`.
- **Matrix product.** The product kernel splits the inner dimension over 4 thread rows and sums
  products in 128 bits, with one modular reduction per batch that cannot overflow.
- **Decomposition** (`src/matrix/MatrixDecompose.cu`) peels every balanced digit of a coefficient
  in one pass.
- **Compact pack** (`gpu_raw_compact_pack_emit` in `src/matrix/MatrixCrt.cu`) uses a single-limb
  kernel when the bound is below half of every source modulus, because the centered residue of
  one limb is then the value (under the trusted-input contract of section 5.6); otherwise it runs
  the full CRT recombination kernel.
- **Integer division** in `integer_operation_kernel` (`src/Control.cu`) uses native 128-bit
  register division when the magnitudes have at most two words.

### 6.8 Current limitations

These restrictions are explicit errors in the current code:

- **Partial multi-device distribution.** Only matrix products, preimage column tiles, and the
  lanes of outermost parallel loops are distributed (section 6.4); every other value lives on the
  home device. Multi-physical-GPU execution uses the same code path but has been validated in this
  repository only with logical devices on one physical GPU. The legacy native multi-partition
  context (CRT limbs partitioned by `dnum`, `GpuDCRTPolyParams::new_with_gpu`) still exists for
  native tests, but plans never use it; binding such a matrix fails ("GPU physical matrix needs an
  explicit multi-device shard plan").
- **Nested device loops.** A parallel loop inside a device body (a sequential-loop, retry, or
  branch body) runs all of its occurrences in one template. A sequential loop inside a device body
  restarts its index on the device, and it cannot read artifacts per iteration ("GPU
  artifact-reading sequential loop cannot run inside a device body"). Sibling conditional nodes
  inside one conditional body are ordered one after another, because concurrently executing
  nested conditionals did not complete on the device.
- **Parallel-loop bodies** cannot import scoped artifacts ("GPU device body artifact import needs
  an on-demand region boundary"); nested loop counts and types cannot depend on the enclosing
  index; matrix-valued loop outputs must be homogeneous matrix families ("GPU loop can return only
  matrix members", "GPU parallel loop needs a homogeneous matrix family"); a vectorized scalar
  loop cannot be nested in another vectorized body.
- **External I/O and waves** cannot be combined: external-I/O loops and selected artifact imports
  cannot be nested in a wave schedule (`DirectGraph::compile`).
- **Artifacts.** Integer and typed-blob artifact inputs need `plan_with_store` for payload sizes;
  host `Int` inputs without an `integer_input_ranges` entry get a signed word range; matrix
  exports need a `FullEval` source; raw transcoding does not support every wire type ("raw
  artifact transcode does not support this wire type yet" in `fleet.rs`).
- **Trapdoors and preimages** require the exact regular gadget layout, sigma, and shapes.
- **Integer matrix-vector products** need one-word operand families and operand and output ranges
  within `int64`, and cannot run in a vectorized body ("GPU integer matrix-vector product may
  exceed int64").
- **NTTs** support ring dimensions up to 32768.

The GPU planner is under active development (support for more loop structures is being added),
so check the current error messages in `gpu_physical_lowering.rs`, `gpu_physical_control.rs`,
and `gpu_runtime_direct.rs` before relying on this list.

## 7. Application crates

All application-level crates build graphs through `mxx-dsl` and execute them through
`mxx-backends`; none performs cryptographic arithmetic eagerly.

### 7.1 `mxx-gadgets`: circuits and reusable gadgets

`crates/gadgets/src/` is BGG-independent:

- `circuit/`: the circuit model and its lowering. `PolyCircuit<P>` (`circuit/poly_circuit/`) with
  `PolyGate`/`PolyGateKind` (`circuit/gate.rs`), sub-circuit calls, and serialization
  (`circuit/serde.rs`); Boolean circuits (`circuit/boolean.rs`: `BooleanCircuitShape`,
  `BooleanCircuitData`, `BooleanGateKind` with constant false/true, copy, not, and, xor);
  DSL-level Boolean circuit families and their validity and satisfaction predicates
  (`circuit/boolean_dsl.rs`); public lookup programs (`circuit/public_lut.rs`). The lowering
  framework (`circuit/lowering.rs`) defines `GateInstance` (call path, local gate, operation
  occurrence) and the traits a concrete encoding scheme implements
  (`CircuitLoweringTypes`, `ArithmeticCircuitLowering`, `SlotOperationLowering`,
  `PublicLookupLowering`, `StructuredCircuitLowering`), driven by `lower_circuit`.
- `circuit_gadgets/`: gadgets written as circuits or DSL graphs.
  - `arith/`: modular arithmetic contexts (`ModularArithmeticContext`, `CrtWindow`), lane-packed
    nested RNS arithmetic (`arith/nested_rns/`: `NestedRnsPolyContext`, `NestedRnsPoly`), and
    carry/Montgomery arithmetic (`arith/carry_montgomery/`).
  - `conv_mul/`: negacyclic convolution without NTT.
  - `ntt/`: radix-2 NTT and inverse NTT over `NestedRnsPoly`.
  - `mod_switch/nested_rns.rs`: nested-RNS modulus switching and its error bounds.
  - `fhe/`: Ring-GSW gadgets (`ring_gsw.rs`, `ring_gsw_nested_rns.rs`); `ckks.rs` is present but
    not compiled (commented out in `fhe/mod.rs`).
  - `fhe_prg/goldreich.rs`: a Goldreich PRG evaluated over Ring-GSW bits.
  - `secret_ip.rs`: secret inner products.
- `decoder/`: mask decryption and PRG helpers.
- `input_injector.rs`: the Diamond input injector (`DiamondInputConfig`, `DiamondInputInjector`).
- `noise_refresh/`: BGG-independent noise-refresh circuits (decrypt, merge, PRG) and material.
- `ring_from_params` (`crates/gadgets/src/lib.rs`) converts `DCRTPolyParams` to a DSL `Ring` with
  the same ordered basis. The `test-support` feature exposes `test_utils` to dependent crates'
  tests.

### 7.2 `mxx-bgg`: BGG+ encodings

`crates/bgg/src/` implements BGG+ on top of the gadget lowering traits:

- `public_key.rs` (`BggPublicKeyCompiler`, `BggPublicKeySampler`) and `encoding.rs`
  (`BggEncodingCompiler`, `BggEncodingSampler`, `BggSamplerLayout`) define the wires, schemas,
  and samplers.
- `circuit.rs`: `PolyCircuitCompiler` lowers a `PolyCircuit` to public-key or encoding graphs
  (`compile_public_keys`, `compile_encodings`, naive and Tall variants, each with a
  `*_with_lowerings` form). `compile_encodings` takes a decomposition provider called with each
  `GateInstance`; it returns the preprocessing-produced `Preimage` for each multiplication, so the
  online encoding graph never builds public-key matrices or gadget decompositions. The producer
  must bind each cached decomposition to the right gate; shapes and bounds alone do not establish
  that binding.
- `boolean.rs`: BGG+ evaluation of dynamic Boolean circuit families
  (`evaluate_boolean_public_key_layers`, `evaluate_boolean_encoding_layers`).
- `lwe_lookup.rs`: LWE-based public lookup tables with preprocessing artifacts.
- `naive_vec.rs`, `slot_operation.rs`: per-slot vectors, slot transfer, and rotation.
- `tall_encoding.rs`, `tall_rotation_encoding.rs`: Tall encodings with one row per slot and
  their linear-transform preprocessing.
- `wee25_commitment.rs`, `wee25_opening.rs`, `wee25_public_parameters.rs`: WEE25 commitments and
  public parameters. The commitment-backed lookup evaluator is intentionally absent.

### 7.3 `mxx-fhe`: TFHE and BGV

`crates/fhe/src/` builds FHE graphs: TFHE with NAND bootstrapping and leveled BGV (BGV has no
bootstrapping). Graph handles are not
authenticated cryptographic objects, and key/ciphertext compatibility beyond ring and shape is the
caller's responsibility.

- `FheCommonParams { ring: DCRTPolyParams, secret_range, error_sigma, error_cutoff }`
  (`params.rs`). Level zero keeps the first CRT prime and higher levels keep longer prefixes.
  `FheScheme` (`lib.rs`) is the shared matrix-plaintext `keygen`/`encrypt`/`decrypt`/`add`/`mul`
  interface, implemented by BGV; TFHE has its own integer LWE API.
- **TFHE** (`tfhe.rs`): `TfheParams::new(common, lwe_dimension, lwe_modulus, lwe_error_sigma,
  lwe_error_cutoff)` pairs integer LWE over a power-of-two modulus `q` with the CRT ring `R_Q`;
  secrets are binary and both Gaussian cutoffs must be at least 16 sigma.
  `LweCiphertext { a, b, .. }` has phase `b - <a, s>` modulo `q` and encodes a bit as
  `+floor(q/8)` (true) or `-floor(q/8)` (false). `RingCiphertext` holds ring-LWE `(1,1)` values and ring-GSW `(1,2L)` key entries.
  `keygen(hash_key)` returns `TfheKeys { lwe_secret, ring_secret, bootstrapping_key,
  key_switch_key }`: a `BootstrappingKey` of GSW encryptions of each LWE secret coordinate and a
  flat `KeySwitchKey` (base `2^b` with `d` digits, both set in `TfheParams::new`; digit `j`
  encrypts the coefficient times `2^(log2 q - b d + b j)`, and `key_switch` rounds each
  coefficient to its leading `b d` bits before decomposing). `encrypt`, `decrypt`, and
  `can_decrypt` work on single bits. Bootstrapping is four public stages, `pre_blind_rotation`, `blind_rotation`
  (external products), `sample_extract` (with rounded `Q -> q` modulus switching), and
  `key_switch`; `bootstrap` chains them and `nand` applies them to `nand_input` (`floor(q/8) -
  ct1 - ct2`) with the `nand_accumulator` sign LUT. LWE `a` vectors come from
  `DslContext::hash_int_family` keyed by a fresh caller-supplied 32-byte key per ciphertext (and
  per `keygen` for the KSK); secrets and errors are independent samples. KSK errors are sampled
  as whole Gaussian polynomials outside the key loop and read by coefficient
  (`lwe_error_sources`, `lwe_error_at`). The KSK `b` dot products `<a_k, s>` are one
  `matrix_vector_product` of the row-major `a` family with the LWE secret, and `key_switch`
  computes both of its sums as `vector_matrix_product`s of the flat key arrays with the digits of
  the extracted coefficients. `blind_rotation` carries the accumulator as one `(a; b)` column, so
  each step multiplies it by `X^k` (`Mat::multiply_monomial`), subtracts, and takes the external
  product on both components in single operations; `pre_blind_rotation` also uses
  `multiply_monomial` for the initial rotation.
- **BGV** (`bgv.rs`): `BgvParams::new(common, plaintext_modulus, hybrid: Option<BgvHybridParams>)`
  and `BgvCiphertext { components, correction_factor, noise_bound }`. Messages are SIMD slots
  (`Family<Int>`, 1 to N values modulo a prime `t = 1 mod 2N`); encoding lifts the inverse NTT
  modulo `t` into `R_Q`. Supported operations: addition, `mul` (with relinearization) or
  `mul_unrelinearized` plus `relinearize`, `mod_switch_to` (CRT modulus reduction),
  `rotate_rows`, and `swap_rows`. Key switching is hybrid RNS key switching (ePrint 2021/204,
  Appendix B.2.3) over `Q_level * P`, using the fused `RnsModUp`/`RnsModDown` graph nodes. With
  `None`, the hybrid parameters default to about three digits and 60-bit auxiliary primes.
  `runtime_parameters()` lists every ring the runtime must register, in tower order, and
  `key_switch_parameters(level)` gives the ring for importing evaluation keys. Security must be
  assessed at `Q * P`.
- Noise bounds are public declarations propagated by the evaluator; `can_decrypt` checks a
  conservative sufficient condition and does not inspect secret values. A matrix artifact alone
  does not carry correction factors or bounds; carry them alongside when connecting stages.
- `utils.rs` provides parameter helpers and GPU helpers used by the GPU tests; default error
  cutoffs come from `mxx_backends::sampler::bounds::hard_cutoff_from_sigma_bound`.

Execution follows the general pattern: build with `DslContext`, validate with the backend CRT
resolver, register the exact ordered ciphertext bases (`runtime_parameters()` for BGV and TFHE) with the
backend, and call `execute` (CPU) or `GpuRuntime` (GPU) with a `MemoryArtifactStore`.
`README.md` summarizes the FHE API.

### 7.4 `mxx-we`: Diamond witness encryption

`mxx-we` is currently excluded from the workspace: its protocol family is bound to one ring, so
it is not parameter-independent across rings. Until that is redesigned, the root `Cargo.toml`
lists it under `exclude`, so no workspace command (`--workspace` included) builds it, and its
standalone manifest is built only by naming it: `cargo test --manifest-path crates/we/Cargo.toml`.
The description below is kept for that work.

`crates/we/src/` defines implementation-independent WE interfaces and the Diamond construction:

- `lib.rs`: `WitnessEncryptionProtocolDecl` (a validated `ProtocolDecl` with its interface),
  `WitnessEncryptionProtocol`, and `WitnessEncryptionRuntime`.
- `diamond/graph.rs`: `DiamondWeProtocolFamily` (the parameter-independent protocol declaration,
  fixed only by its BGG domain tag, so changing a runtime `ParamEnv` does not change the protocol
  hash) and `DiamondWeCompiler`, which builds the encryption and decryption graphs
  (`build_encryption`, `build_decryption`). Both stages declare `instance_width`,
  `witness_width`, `depth`, and `max_layer_width` as compile parameters, while gate opcodes and
  predecessor indices are runtime families (`BooleanCircuitData` from `mxx_gadgets::circuit`).
  Layers run in one carried-state `SequentialLoop`, gates in structural `ParallelLoop`s with
  dynamic predecessor reads. The encryption stage exports input-injector transitions, BGG+ public
  keys, and witness projection preimages as family artifacts; the decryption stage imports them.
  Witness bits are decryption-only inputs.
- `diamond/runtime.rs`: `DiamondWeRuntime<E: ExecutionAuthority<S>, S: SessionStore>` runs
  encryption and decryption through any execution authority, including `GpuRuntime` under the
  `gpu` feature (`DiamondBooleanOutput` abstracts the Boolean result).
- `diamond/parameter_search.rs`: `DiamondParameterSearch` fixes the circuit shape and searches ring
  dimension and CRT depth. Correctness uses deterministic worst-case bounds with sampler cutoffs
  `floor(6.5 * sigma)` (`diamond/config.rs`); lattice security is estimated with untruncated
  distributions. A candidate is accepted only after Lean checks a freshly generated theorem for
  the same frozen workflow, backend layout, and parameters; the selected result keeps its
  verified certificate. Numerical rejection and checker failures are distinct errors, and the
  search is a heuristic, not a minimality claim.
- `lean.rs`, `lean/`: WE decoder semantics and backend bindings for
  `mxx_ir_core::lean::protocol::export_claim`, Lean checking (`lean/check.rs`), and numeric
  certificates (`lean/numeric.rs`). Handwritten proofs live in `crates/we/lean/`; the audit entry
  point is `crates/we/lean/Certificate.lean` (`DiamondCertificate.correctness`). To edit proofs
  against a generated candidate, select it with `scripts/select_we_lean_candidate.py` and follow
  `crates/we/lean/README.md`.

### 7.5 `mxx-func-enc` and `mxx-io`

`crates/func-enc/src/lib.rs` defines only the `FuncEnc` trait and `crates/io/src/lib.rs` only the
`Obfuscation` trait. Their former implementations were removed during the DSL migration.

## 8. Testing, validation, and where to look next

### 8.1 Test layout

- **Unit tests** live next to the code they test (`#[cfg(test)]` modules) and run with
  `cargo test -r --workspace --lib`. GPU unit tests live in files whose names contain `gpu` and
  compile only with `--features gpu`; `crates/fhe/src/tests_gpu.rs` is one example. Tests that
  need a real CUDA device and are not safe to run by default carry `#[ignore = "requires a CUDA
  GPU"]` (in `crates/backends/src/gpu_physical_lowering.rs` and `gpu_physical_control.rs`).
  Tests that need Lean or long runs are also `#[ignore]` (for example in `crates/we/src/lean/`
  and `crates/fhe/src/bgv.rs`).
- **Integration tests** are all GPU tests:

  | Test target | Gate |
  | --- | --- |
  | `crates/backends/tests/gpu_control_resident.rs` | `#![cfg(feature = "gpu")]` |
  | `crates/backends/tests/gpu_direct_node_semantics.rs` | `#![cfg(feature = "gpu")]`; compares direct GPU nodes with the CPU backend |
  | `crates/fhe/tests/gpu_bgv.rs` | `required-features = ["gpu"]` in `crates/fhe/Cargo.toml` |
  | `crates/fhe/tests/gpu_tfhe.rs` | `required-features = ["gpu"]`; the TFHE-rs `TFHE_LIB_PARAMETERS` Boolean profile (LWE `n = 630`, `q = 2^32`, sigma `2^17`; ring `N = 1024`, double-CRT `Q = 65537 * 79873`, sigma 156, about `2^-25` of `Q`; gadget base `2^9`, two digits per limb; KSK base `2^2` with 8 digits). `test_gpu_tfhe_round_trip` mirrors the BGV round trip: key generation, encryption, one bootstrapped NAND program per gate, and decryption, each planned once with keys kept resident; checks the NAND truth table and four chained gates that feed each output back as the next left input (profile in `utils::tfhe_params`, overridable through `FHE_TEST_TFHE_*`) |
  | `crates/gadgets/tests/test_gpu_tall_bgg_nested_rns_modq_arith.rs` | `#![cfg(feature = "gpu")]`; long modes are `#[ignore]` |
  | `crates/we/tests/test_gpu_diamond_we.rs` | `#![cfg(feature = "gpu")]` and `#[ignore]`; Lean-checked parameter search plus a GPU round trip |

  `AGENTS.md` requires an explicit request before running integration tests.
- **Benchmarks** are `harness = false` targets in `crates/backends/benches/`
  (`bench_matrix_mul_cpu`, `bench_matrix_mul_gpu`, `bench_preimage_cpu`, `bench_preimage_gpu`).
- Test parameters are overridable through environment variables with small defaults, for
  example `MXX_PRIMITIVE_TEST_*` (`crates/backends/src/env.rs`), `FHE_TEST_*` (`crates/fhe`),
  `MXX_TALL_NESTED_RNS_*`, and `MXX_DIAMOND_WE_GPU_*`. Each test writes to its own directory under
  `test_data/`.
- `scripts/run_tests.sh` is the reference validation script: Python unit tests under
  `scripts/lib/tests`, `cargo +nightly fmt --all`, `cargo test -r --workspace --lib --features gpu
  --no-run`, `cargo test -r --workspace --lib`, and a conditional repeated GPU run
  (`scripts/lib/repo_validation.py`).
- Lean fixtures are generated by ordinary unit tests (`lean::fixtures` in `mxx-ir-core` and
  `mxx-backends`) into `test_data/`; see `crates/ir-core/lean/README.md`.

### 8.2 Where to look next

| Document | Use it for |
| --- | --- |
| `AGENTS.md` | Repository rules, scope, and which guide applies to a task. |
| `BUILDER.md` | Implementation, debugging, design style, testing, and benchmark rules. |
| `REVIEWER.md` | Review criteria and result format. |
| `GPU.md` | GPU dataflow, synchronization, memory complexity, and GPU validation requirements. |
| `README.md` | Project overview, FHE API summary, and requirements (OpenFHE, OpenMP, CUDA). |
| `docs/correctness/` | Correctness specifications, for example `docs/correctness/operational-protocol-inventory.md`. |
| `docs/plans/` | Local design plans and progress records; ignored by git and not part of the repository. |
| `crates/ir-core/lean/README.md`, `crates/we/lean/README.md` | Lean packages and fixture workflows. |
| `references/` | Read-only specifications and papers. |
