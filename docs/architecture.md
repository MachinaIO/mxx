# Workspace architecture

This repository is a virtual Cargo workspace. It has no root facade crate.
Consumers depend directly on the crate that owns the abstraction they use.

Detailed guides are available for the [typed DSL](dsl.md),
[core IR](ir-core.md), [symbolic IR](ir-symbolic.md),
[noise simulator](noise-simulator.md), and
[runtime](runtime.md). The active Diamond witness-encryption application is
described in [diamond-we.md](diamond-we.md).

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

`mxx-we` contains the active Diamond witness-encryption protocol graphs.
`mxx-io` contains the active AKY24 and Diamond indistinguishability-obfuscation
graphs. AKY24 functional encryption remains disabled; the private prFE
machinery in `mxx-io` is internal to the AKY24 iO cascade rather than a public
functional-encryption API.

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
The Diamond input-injection gadget builds the initial `p` vector and transition
preprocessing shared by Diamond WE and Diamond iO. It returns the final
trapdoors for application-specific projections but does not construct BGG+
encoding preimages.

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

### `mxx-we`

`crates/we/` owns Diamond witness encryption. Its preprocessing, encryption,
and decryption computations are declarative DSL graphs over `mxx-bgg`; runtime
execution is provided for CPU and GPU backends without a separate protocol
implementation. Public preprocessing values are passed through runtime
sessions and typed artifact manifests.

The crate also owns Diamond-specific symbolic noise simulation, automatic
ring-dimension and modulus search, and graph cost estimation. See
`docs/diamond-we.md` for the protocol graph and validation details.

### `mxx-io`

`crates/io/` owns the Diamond indistinguishability-obfuscation application.
Preprocessing and evaluation are separate declarative DSL graphs linked by
typed public artifact manifests. Private trapdoors are used only as internal
preprocessing witnesses and are never exported as graph outputs or artifacts.

The AKY24 implementation source is retained under `crates/io/src/aky24/`, but
its crate module and integration target are temporarily disabled pending full
end-to-end validation of the private-prFE cascade. The Diamond implementation
reuses the shared input injector, BGG+ public lookup, Ring-GSW, decoder, and
noise-refresh gadgets. Its graph definitions drive runtime execution and linked
noise simulation, and it provides automatic ring-dimension and modulus search.
The heavyweight GPU round-trip test is an explicit ignored integration target.

Diamond's native Ring-GSW public key always uses a strictly positive Gaussian
error. When native ciphertexts enter the declarative graph, their flattened top
row is Large signal and their bottom row is Large signal plus a bounded `eR`
term. Its physical nested-RNS bound is the conservative coefficient envelope
`6.5 * sigma * public_key_width * ring_dimension + max(p_j) - 1`. The second
term covers residue changes when `(c mod q_i) mod p_j` crosses a `q_i`
boundary. Dependency metadata remains unknown, so the simulator cannot apply
an unjustified CLT reduction. Parameter search checks lattice security for both
the ordinary protocol error and this native Ring-GSW error distribution, and
accepts a candidate only when the linked graph remains within every decode
threshold.

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
