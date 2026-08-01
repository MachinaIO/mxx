# Workspace architecture

This repository is a virtual Cargo workspace. It has no root facade crate.
Consumers depend directly on the crate that owns the abstraction they use.

Detailed guides are available for the [typed DSL](dsl.md),
[core IR](ir-core.md), [symbolic IR](ir-symbolic.md),
[noise simulator](noise-simulator.md), and
[runtime](runtime.md).

## Dependency layers

Arrows point from a consumer to a dependency:

```text
mxx-runtime          -> mxx-ir-core, mxx-primitives
mxx-ir-symbolic      -> mxx-ir-core
mxx-noise-simulator  -> mxx-ir-core, mxx-ir-symbolic, mxx-primitives
mxx-bench-estimator  -> mxx-ir-core, mxx-runtime
mxx-dsl              -> mxx-ir-core, mxx-ir-symbolic
mxx-gadgets          -> mxx-dsl, mxx-ir-core, mxx-primitives
mxx-bgg              -> mxx-dsl, mxx-gadgets, mxx-ir-core
mxx-func-enc/we/io   -> lower layers, never one another
```

Application crates currently expose interfaces only. Their protocol graphs
will be implemented separately with the declarative DSL.

## Crate responsibilities

### `mxx-primitives`

`crates/primitives/` owns polynomial and matrix representations, samplers,
OpenFHE integration, sampling bounds, and all native CUDA code. GPU versions
of primitive operations stay here even when a higher layer is their primary
consumer.

### `mxx-ir-core`

`crates/ir-core/` is the only executable graph representation. Nodes are
immutable and shared through `NodeHandle` and `ValueHandle`. Reusing a cloned
handle reuses the same node; constructing the same operation twice creates two
distinct nodes.

A graph is frozen by traversing named output roots. Freezing assigns stable
scope-local node identifiers, seals subgraphs and parallel-loop bodies, and
produces canonical serialization. Validation evaluates compile expressions,
checks concrete types and shapes, and derives deterministic execution and
liveness schedules. Runtime order is not graph semantics.

Subgraph calls and parallel loops are structural nodes. Their bodies are stored
once and are not expanded per call or iteration. Parallel-loop execution is
bounded by the runtime configuration so callers can respect RAM and VRAM
budgets.

### `mxx-dsl`

`crates/dsl/` is the typed construction API. Ordinary expressions create core
nodes immediately:

```rust
let error = ring.gaussian((1, columns), sigma);
let encoding = secret * public_key - plaintext * secret_gadget + error;
```

`DslContext` only declares parameters and output roots. It does not own a
mutable graph builder. `Family<Mat>` represents indexed values and its
`parallel_map` and `parallel_zip` methods create one structural parallel loop,
not one node expansion per element.

`VirtualMat` and `SymbolicExpr` are the only non-executable expression wrappers.
They build a small typed symbolic expression for `Mat::assume` and never form a
second executable DAG:

```rust
let secret = VirtualMat::bounded("s", secret_type, secret_metadata);
let error = VirtualMat::bounded("e", error_type, error_metadata);
let encoding = encoding.assume(secret * public_key + error)?;
```

`assume` attaches an axiom-like reinterpretation and creates no executable
node. Cloning a virtual value preserves identity; constructing another value,
even with the same name, creates another identity.

### `mxx-ir-symbolic`

`crates/ir-symbolic/` elaborates core nodes into a typed, hash-consed symbolic
expression arena. It tracks exact source and instantiation identities, imports
and exports symbolic manifests, applies `assume` annotations, and performs
targeted preimage and aligned-block rewrites. Products of unrelated sums remain
compact instead of being expanded into a global sum of products.

For a preimage relation

```text
c = s B + e
B K = S' P + E
```

the elaborator directly derives

```text
c K = (s S') P + s E + e K.
```

The resulting expression retains every source identity. Memoization is an
internal rewrite or analysis cache and has no serialized identity.

### `mxx-noise-simulator`

`crates/noise-simulator/` owns all numerical magnitude calculations. Symbolic
IR stores source parameters and declared metadata but does not calculate noise
bounds. The simulator preserves the established `PolyNorm` and
`PolyMatrixNorm` rules, including the 6.5-sigma envelopes, balanced gadget
digit bound, dependency-aware CLT eligibility, and conservative addition when
dependency disjointness is unknown.

Signal and noise stay separate. The simulator traverses additive and
multiplicative alternatives lazily, so it does not materialize a global sum of
products. `Select` is evaluated under one domain-wide branch assignment, and
`Tensor` uses its bilinear norm rule. Generic modulus conversion is not part of
the current core or symbolic IR; nested-RNS switching remains in the circuit
gadget layer.

### `mxx-runtime`

`crates/runtime/` executes validated schedules on CPU or GPU primitive
backends. It owns runtime inputs, reproducible sampling transcripts, in-memory
artifacts, sessions, and bounded parallel-loop waves. Applications own durable
artifact persistence outside the runtime crate.

Parallel-loop waves intentionally remain bounded by
`ExecutionConfig::max_parallel_instances`. This bound is a resource policy and
must not be replaced by unbounded host parallelism.

### `mxx-gadgets`

`crates/gadgets/` owns BGG-independent `PolyCircuit` structure and reusable
circuit gadgets. `circuit_gadgets` contains arithmetic, convolution, FHE,
Ring-GSW, PRG, NTT, decoder templates, and noise-refresh circuit templates.

### `mxx-bgg`

`crates/bgg/` owns BGG+ public keys, encodings, samplers, indexed encoding
families, circuit evaluation, masked decoding, naive and cryptographic slot
transfer, preprocessing artifacts, and noise refresh. All active BGG graph
code uses `mxx-dsl`.

The LWE-based public LUT evaluator, including its preprocessing artifacts and
scalar, polynomial-slot, and naive per-slot circuit lowerings, is implemented
in `mxx-bgg` with the declarative DSL. Public lookup implementations provide
only `PublicLookupLowering`; slot-transfer implementations provide only
`SlotOperationLowering`. `GraphCircuitLowering` is the complete traversal
interface and inherits those traits together with ordinary arithmetic
lowering. `PolyCircuitCompiler::compile_*_with_lowerings` accepts lookup and
slot providers separately when a circuit uses either or both gate families.
The WEE25 commitment-backed lookup evaluator is deliberately outside the
current implementation. Decoder and noise-refresh circuit templates remain in
`mxx-gadgets` because they are fundamentally `PolyCircuit` components.

## Parallelism

Independent host computations use Rayon where output ordering can be restored
by indexed collection. Graph construction, topological scheduling, transcript
mutation, and backend calls are kept sequential when ordering or mutable device
state is semantic. GPU-facing loops whose concurrency is deliberately bounded
for VRAM are controlled by runtime waves instead of Rayon.

## Features

`gpu` enables CUDA-backed primitive operations. `mxx-primitives` alone owns
CUDA compilation and OpenFHE linkage. Upper crates only forward the feature.

## Validation

Use the narrowest owning-crate unit tests during development. For cross-cutting
changes, use:

```sh
cargo check --workspace
cargo +nightly fmt --all
```

Integration tests are run only when explicitly requested.
