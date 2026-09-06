# Declarative graph DSL

Use `mxx-dsl` to describe a computation once, validate its types and parameters, and execute it
with `mxx-runtime`. It is a Rust embedded DSL: you write ordinary Rust expressions whose values
build a graph. There is no separate source language to parse.

This program doubles each matrix in an input family and exposes the resulting family as an output.
The closure describes one loop body; the graph executes that body for four indices.

```rust
use mxx_dsl::{DslContext, Ring, parallel};
use mxx_ir_core::ParamEnv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ring = Ring::new(17, 8);
    let inputs = ring.input_family("inputs", 4, (2, 2));
    let outputs = parallel(4, |i| {
        let input = inputs.at(i);
        Ok(input.clone() + input)
    })?;
    let built = DslContext::new("double").output("outputs", outputs)?.build()?;
    built.validate(&ParamEnv::default())?;
    Ok(())
}
```

## Start with values, then choose how to combine them

A DSL value describes a computation. For example, `a + b` builds an addition node; it does not
add matrices in the Rust process constructing the graph. `Mat`, `Int`, `Bool`, and `Bytes` are
handles to such values. `SmallMatrix`, `Preimage`, and `Trapdoor` carry the additional type
information required by their cryptographic operations.

Most programs need four operations:

| What the program needs | What to write |
| --- | --- |
| Compute a value | Ordinary arithmetic such as `&a + &b`, or a typed method |
| Compute the same operation for each index | `parallel(count, |i| ...)`, returning a `Family<T>` |
| Carry a result into the next step | `iterate(count, initial, |i, state| ...)` |
| Make a result available to the caller | `DslContext::output(name, value)` |

Rust bindings, tuples, vectors, and records organize the code around those operations. They do not
need separate DSL declarations. The reference later in this document lists every supported API;
the following sections explain when to use them.

### Use a family when the computation needs an index

`Family<T>` is an ordered collection whose elements all have the same schema. Use it for inputs,
tables, or results that the graph must read by index. A family count can be a compile expression
resolved from parameters before execution. The opening example uses `Family<Mat>` because each
loop instance reads one matrix with `inputs.at(i)`.

`family.at(index)` returns an element. Its index can be a constant, compile expression, loop index,
or runtime `Int`. Known static indices are validated before execution; dynamic indices are checked
at runtime. Negative or out-of-range indices are errors. A loop can read a prefix of a longer
family; domain operations that require equal lengths check that requirement separately.

An element can also be a tuple or domain record. `Family<CircuitEncoding>`, for example, lets
`encodings.at(i)` return the vector, public key, and plaintext belonging to the same encoding.
`Family<Trapdoor>` similarly keeps a public matrix with its private trapdoor. These groupings
preserve correspondence; they do not prove an algebraic relation between the fields.

### Use Rust containers when the code only needs to group values

A tuple such as `(matrix, flag)` passes two existing values together. A record gives related fields
names. A Rust `Vec<T>` serves the same purpose when the number of fields is determined while
constructing the graph. Its `GraphValue` implementation lets it pass through `output`, `iterate`,
and `Subgraph` without callers unpacking it by hand. It is a convenience for composing Rust code,
not another collection operation that a DSL user must learn.

For example, this fragment assumes matrices `left` and `right` with matching schemas and a graph
integer `index`:

```rust
use mxx_dsl::Family;

let fields = vec![left, right];
let first = fields[0].clone();       // Rust chooses an existing handle now.
let table = Family::pack(fields)?;
let chosen = table.at(index);       // The graph chooses an element during execution.
```

The distinction matters because the two containers express different requirements:

| | Rust `Vec<T>` | DSL `Family<T>` |
| --- | --- | --- |
| Purpose | Pass several fields together | Read or compute elements by index in the graph |
| Length | A Rust `usize` fixed during construction | A compile expression resolved before execution |
| Element schemas | May differ; each position has its own schema | Must all agree |
| Indexing | Rust `usize`; returns an existing handle | `.at(...)`; accepts a graph integer |
| Graph representation | The fields' individual wires | An indexed family for each element field |

Thus a `Vec<Mat>` can group matrices of different shapes; packing those matrices into a
`Family<Mat>` fails. A family cannot replace every vector without changing the meaning or
restricting the values a Rust function can accept. Conversely, use `Family` when the algorithm
needs symbolic indexing or one reusable loop body. Use tuples or named records for a small,
fixed group with distinct roles.

### Keep element schemas consistent

A schema describes what is known before execution: the ring, matrix shape, coefficient bounds,
and the fields of a composite value. `GraphValue` and `GraphValueSchema` make tuples, vectors,
domain records, and families usable by the same construction functions. Ordinary callers use
these values directly; domain-library authors implement the traits for new records.

`Family::pack` requires a nonempty collection of same-schema values. A zero-count `parallel` can
produce a typed empty family. Rings, shapes, bounds, vector lengths, and static record metadata
must agree across elements. Optional record fields must be present in every element or absent in
every element. Ragged arrays and iteration-dependent output schemas are unsupported.

The current storage format supports one family dimension. Consequently, a `Family` cannot contain
another family, even inside a record. A fixed-length Rust vector or tuple can group several fields
within an element. A complete family can still be the state of `iterate`, a candidate in `select`,
or an argument to a subgraph: none of these uses adds another family dimension.

`family.field(|record| record.field)` selects an existing field from every element. It also
supports tuple and vector fields, including reordering and duplication. This is useful when an
operation needs all public keys from a family of encodings. It preserves the producer and sampling
identities and adds no computation loop. Use `parallel` when the projection needs arithmetic;
arithmetic inside `field` is rejected.

## Compute with ordinary expressions

Use arithmetic to describe how output values depend on input values. These operators build graph
nodes; Rust bindings and borrowing still behave normally.

Matrices support owned or borrowed `+`, `-`, `*`, and unary `-`. Integer/compile-expression scalar
multiplication uses a scalar constant polynomial and the existing multiplication primitive, such
as `&matrix * 2` or `2 * &matrix`. Scalar addition has no implicit broadcasting rule.
Primitive integers on the left of runtime `Int` arithmetic or matrix multiplication use `i32`,
so unsuffixed literals infer naturally. For other primitive types, put the scalar on the right
of multiplication or explicitly convert it to a graph integer for integer arithmetic.

Runtime `Int` supports arithmetic operators and borrowed operands. Division rounds down using the
absolute divisor; remainder is nonnegative modulo the absolute divisor. Division by zero fails at
runtime. `equal`, `less`, and `less_equal` accept integer literals and borrowed integers;
Rust comparison operators cannot return a symbolic `Bool`.

Booleans support `&`, `|`, `^`, and `!`, including borrowed operands and Boolean literals. These
construct Boolean computations without short-circuiting. Rust `&&` and `||` cannot be overloaded
for graph values. Domain operations that require compiler parameters or preprocessing inputs retain
explicit method arguments.

### Use compile expressions for shapes and counts

Matrix dimensions and loop counts must be known before execution so the graph can be validated
and storage planned. `IntExpr` describes those quantities; `RealExpr` describes exact real-valued
parameters such as a Gaussian sigma. Runtime `Int` describes a value computed during execution.
This is why an input integer can choose a family element but cannot set the graph's loop count.

Compile expressions support natural owned and borrowed arithmetic. The following fragment assumes
`start`, `columns`, `rows`, and `variance` are compile expressions:

```rust
let end = &start + &columns;
let entries = &rows * &columns;
let sigma = (&variance + 1) / 2;
```

`IntExpr` supports `+`, `-`, `*`, `/`, `%`, unary `-`, and `floor_div`. Its `/` retains
exact compile-time division: a nonzero remainder is an error. `%` retains floor remainder.
Cancellation and multiplication by zero preserve errors from partial operations, including during
canonicalization and serialization; for example, `(1 / 0) * 0` remains an error.
`RealExpr` supports `+`, `-`, `*`, `/`, and unary `-`; integer/rational conversions remain exact.
No implicit floating-point conversion is introduced.

## Repeat a computation

Use `parallel` when each result can be computed independently from shared inputs. Use `iterate`
when a step needs the previous step's result, as in a recurrence or a sequence of circuit layers.
Both describe one reusable body, even when the count is supplied later through parameters.

This fragment assumes compatible matrix families `left`/`right`, a family-valued `initial` state,
integer-family `circuit_sources`, and compile expressions `count`, `depth`, and `width`:

```rust
let products = parallel(count.clone(), |i| {
    Ok(left.at(&i) * right.at(i))
})?;

let result = iterate(depth, initial, |layer, previous| {
    parallel(width.clone(), |slot| {
        let flat = &layer * width.clone() + slot;
        Ok(previous.at(circuit_sources.at(flat)))
    })
})?;
```

`parallel(count, body)` constructs one reusable body, executes independent instances, and returns
`Family<T>` in index order. `iterate(count, initial, body)` supplies the previous state to each
instance and returns the final state. A zero-count iteration returns the initial state. The complete
state schema must remain unchanged. Both bodies return `Result<_, DslError>`; there are no separate
map, values, bundle, zip, gather, or broadcast interfaces.

Closures run once during graph construction. They do not execute Rust side effects once per runtime
iteration. Runtime indices are ordinary `Int` values, with compile-time binder provenance retained
internally. Counts and shapes still use compile expressions resolved by `ParamEnv`; runtime input
values cannot determine graph structure. `Int::expression()` performs a checked conversion only for
operations whose metadata must be known before execution, such as static matrix slice bounds.

Outer values are read-only lexical dependencies, regardless of whether they are scalars, families,
records, or artifacts. The DSL derives explicit graph inputs from their uses. Nested closures may
read ancestor values; a value leaked from a completed child or sibling scope is rejected.

## Choose a value with `select`

Use `select` when the choice depends on a graph `Int` or `Bool`. A Rust `if` can choose which graph
to construct from a Rust condition; it cannot choose based on a value that will only exist during
execution.

`select(selector, candidates)` selects a same-schema value, including a record or an entire family.
An integer selects a zero-based candidate; a Boolean selects candidate zero for false or one for
true. Every leaf of a record uses the identical selector. Empty candidates, schema disagreement,
and out-of-range selectors are errors.

This is value selection, not a lazy control-flow statement. Do not rely on it to guard an invalid
array access or to suppress a sampler in an unselected candidate. Runtime artifact materialization
can remain lazy without providing general lazy branch semantics.

## Decide where randomness is shared

Sampler placement states whether an algorithm reuses one sample or requires a fresh sample for
each loop instance. Place a sampler outside the body for a shared value, and inside the body for
independent samples. Cloning a value handle keeps the same sample.

This fragment assumes a `ring`, compatible `inputs`, compile expressions `count` and `bound`,
and a real compile expression `sigma`:

```rust
let shared = ring.gaussian((1, 1), sigma.clone(), bound.clone());
let outputs = parallel(count, |i| {
    let independent = ring.gaussian((1, 1), sigma, bound);
    Ok(inputs.at(i) + shared + independent)
})?;
```

The outer Gaussian is one shared sample. The body-local Gaussian is sampled separately for each
executed instance. Repeated reads of an already-produced value retain that same value. A zero-count
loop does not execute its local samplers. Structural rewrites must preserve sampler placement and
sharing; loop fusion and changes to sample scheduling require separate validation. Changed graph
structure may require regenerated transcripts, artifact identities, and proof artifacts.

Every Gaussian and preimage sampler requires an integer coefficient cutoff. Parameterized cutoffs
reference declared integer parameters resolved by `ParamEnv`. The DSL does not infer a cutoff from
an unresolved real sigma. `SmallMatrix` and `Preimage` retain their different semantics and bounds.

Loop-index hash tags preserve the index encoding used by existing sampling programs. Domain builders
must preserve the distinction between a logical table index and a separate row identifier used in
hash tags when changing indexing syntax.

## Build a graph with explicit inputs and outputs

Inputs name the values supplied at execution. Outputs name the results that execution must make
available. Grouping values in a Rust container does not automatically expose or persist them;
choosing outputs is a separate step.

`DslContext::input` accepts a complete value schema, including a family of records. Existing typed
ring input builders remain convenient for primitive inputs. `output`, `public_output`, and
`private_output` accept complete values. A single leaf retains its supplied name; composite leaves
use `name.0`, `name.1`, and so on in schema order. Field projections allow applications to retain
explicit domain artifact names.

Public and secret persistence remains explicit. For trapdoor families, use `public_matrices` for
the public projection and `private_trapdoor_family_output` for the secret projection; import the
pair with `trapdoor_family_artifact_input`. Composite values do not erase confidentiality boundaries.

### Reuse a computation with `Subgraph`

Use a subgraph when several call sites share the same computation and input schema. Define its
body once, then call it with compatible arguments. This gives a library operation an explicit
interface within the graph.

`Subgraph::define` stores one reusable body and accepts a closure returning `Result<_, DslError>`,
just like `parallel` and `iterate`. Formal inputs use explicit schemas;
outer lexical reads are lifted to hidden arguments using the same ancestry rule as iteration.
Artifact tables passed as formal arguments remain reusable across calls with the same schema.
Subgraph reuse does not require exposing zip/broadcast details to callers.

### Validate before execution

`build()` freezes the graph and checks structure. `validate(&ParamEnv)` resolves parameters and
checks concrete types, shapes, bounds, and execution planning. Artifact consumers can additionally
use `validate_with_manifests`. See [docs/runtime.md](runtime.md) for execution.

### Declare protocol checks on outputs

A computation graph describes what to execute. A protocol declaration adds the output relationships
that must be checked for that protocol. Keeping those references on the declaration lets ordinary
arithmetic and loop code work with values alone.

Correctness declarations use core-owned `IdealSpec::new` and `PurePredicateSpec::new`. Their graphs
must be sampler-free. Protocol endpoints identify executable results with `OutputRef` (stage and
output name). Each operational decoder target identifies an endpoint and a residual `OutputRef`;
the decoder is resolved from the endpoint's executable output. Validation checks the referenced
outputs, their types, and the supported decoder's executable connections and formula. No semantic
labels, rule attachments, or annotation propagation are part of graph values. Packing a record
does not itself establish an algebraic relation.

Domain records implement `GraphValue` using their schema, flattened values, and reconstruction from
those values. Computation and scope conversion carry no separate protocol metadata. Protocol-specific
proofs remain responsible for algebraic relations beyond the executable decoder checks.
See [docs/correctness/operational-protocol-inventory.md](correctness/operational-protocol-inventory.md)
for checking semantics.

## API reference

This reference covers the public API defined by
[crates/dsl/src/lib.rs](../crates/dsl/src/lib.rs),
[crates/dsl/src/integer.rs](../crates/dsl/src/integer.rs),
[crates/dsl/src/operators.rs](../crates/dsl/src/operators.rs),
[crates/dsl/src/family.rs](../crates/dsl/src/family.rs),
[crates/dsl/src/control.rs](../crates/dsl/src/control.rs),
[crates/dsl/src/subgraph.rs](../crates/dsl/src/subgraph.rs), and
[crates/dsl/src/value.rs](../crates/dsl/src/value.rs), including exported macros.
It specifies a Rust embedded DSL, not a separate parser or textual grammar. Ordinary Rust binding,
borrowing, tuple/vector construction, field access, closures, and `?` compose these operations.
A Rust `if`, `for`, or `match` executes during construction and cannot branch on a symbolic `Bool`.
Rust types and trait bounds determine which expressions are accepted.

In the tables, `E` abbreviates an argument accepting `impl Into<IntExpr>`, `R` abbreviates
`impl Into<RealExpr>`, `S` abbreviates `impl IntoShape`, and `N` abbreviates
`impl Into<String>`. These are documentation abbreviations, not exported types. Arguments appear
in declaration order; `self`/`&self` receivers are omitted. Methods returning a value directly still
require graph validation: constructing a node does not prove its concrete parameter constraints.

### Imports and supported value types

The crate root exports `DslContext`, `BuiltGraph`, `Ring`, `Shape`, `IntoShape`, `Mat`, `SmallMatrix`,
`Preimage`, `Trapdoor`, `Int`, `Bool`, `Bytes`, `Family`, `Subgraph`, `GraphValue`, `GraphValueSchema`,
`MatType`, `SmallMatrixType`, `PreimageType`, `TrapdoorType`, `IntType`, `BoolType`, `BytesType`,
`FamilyType`, `HashTag`, `HashTagPart`, `DslError`, `ValidationBuildError`, the three control functions,
and the four macros listed below. Core-owned reexports are `Rational`, `Confidentiality`,
`IdealSpec`, `PurePredicateSpec`, and `ConcatAxis`.

`IntExpr`, `RealExpr`, `ParamEnv`, `MatrixType`, `ValueHandle`, `Graph`, `ProductionId`,
`ConstantMatrix`, and `IndexRange` come from `mxx_ir_core` or its `artifact`/`node` modules;
they are not reexported by `mxx_dsl`. Core graph construction, runtime execution, artifact manifests,
and protocol declarations retain their own APIs; they are not additional DSL control syntax.

| Type or syntax | Complete public data or contract |
| --- | --- |
| `Shape { rows, columns }` | Both fields are `IntExpr`. `IntoShape::into_shape(self)` consumes a shape. It is implemented for `Shape` and `(A, B)` where each member converts into `IntExpr`; `(2, 3)` is the usual shape syntax. |
| `MatType(matrix)` / `MatType::new(matrix)` | Wraps `MatrixType { modulus, ring_dimension, rows, columns }`, all `IntExpr`. |
| `IntType`, `BoolType` | Unit schemas for ring-independent scalar values. |
| `BytesType { length }` | Byte length is `IntExpr`. |
| `SmallMatrixType { matrix, max_coefficient_bound }` | Matrix type plus an integer coefficient bound; a bounded matrix without a gadget/preimage relation. |
| `PreimageType { matrix, max_coefficient_bound }` | Matrix type plus integer coefficient bound, retaining the distinct preimage wire kind. |
| `TrapdoorType { matrix, sigma, gadget_base, digit_count, preimage_max_coefficient_bound }` | Matrix type, `RealExpr` sigma, and three `IntExpr` metadata fields; flattened order is public matrix then secret trapdoor. |
| `FamilyType { element, count }` | Element schema and `IntExpr` count; one indexed wire per flattened element field. |
| `(A,)` through `(A, ..., L)` | Graph-value/schema tuples of arity 1 through 12, with fields in tuple order. The unit tuple has no supplied `GraphValue` implementation. |
| `Vec<T>` / `Vec<T::Schema>` | Rust grouping supported by `GraphValue`: each position contributes its own schema and wires, in vector order. Length is fixed during construction; element schemas may differ. See [Rust containers](#use-rust-containers-when-the-code-only-needs-to-group-values) for why this differs from `Family<T>`. |
| `Confidentiality::{Public, Private}` | Alias of core `ArtifactConfidentiality`; controls declared artifact visibility. |
| `ConcatAxis::{Rows, Columns, Diagonal}` | Row concatenation, column concatenation, or block diagonal construction. |

All graph values and schemas implement `Clone`. Cloning values preserves producer identity rather
than sampling or calculating again. Empty vectors can be represented by the trait implementation,
but inputs, outputs, loops, packing, and selection reject empty flattened schemas where they require
a graph value. A custom record may contain optional data only through its own fixed schema;
`Option<T>` has no built-in `GraphValue` implementation.

### Context, outputs, and validation

| API | Result and semantics |
| --- | --- |
| `DslContext::new(name: N)` | New context with no parameters or outputs. |
| `context.int_parameter(name: N)` | Consumes and returns the context, declaring an integer compile parameter. |
| `context.real_parameter(name: N)` | Consumes and returns the context, declaring a real compile parameter. |
| `context.input::<V>(name: N, schema: V::Schema)` | `Result<V, DslError>`; named runtime input leaves, no artifact provenance. Includes records and families. |
| `context.evaluate_int(expression: E)` | `Int`; materializes a compile expression as a runtime integer wire, including expressions combining a loop binder with parameters. |
| `context.int_family_input(name: N, fixed_count: E)` | `Family<Int>`; convenience builder for a runtime integer family. |
| `context.output(name: N, value: V)` | `Result<DslContext, DslError>`; consumes the context and graph value; no explicit confidentiality declaration. |
| `context.public_output(name: N, value: V)` | Same naming/flattening, sets every output leaf to `Some(Public)`. |
| `context.private_output(name: N, value: V)` | Same naming/flattening, sets every output leaf to `Some(Private)`. |
| `context.private_trapdoor_output(name: N, trapdoor: Trapdoor)` | Outputs only the secret trapdoor wire, explicitly private. |
| `context.private_trapdoor_family_output(name: N, trapdoors: Family<Trapdoor>)` | Outputs only the aligned secret trapdoor family wire, explicitly private. |
| `context.build()` | `Result<BuiltGraph, DslError>`; freezes reachable named outputs and validates graph structure. |
| `built.graph` | Public core `Graph` field, for runtime, serialization, and protocol construction. |
| `built.validate(bindings: &ParamEnv)` | `Result<ValidatedGraph, ValidationBuildError>`; validates concrete parameters and plans execution. |
| `built.validate_with_manifests(bindings: &ParamEnv, manifests: &BTreeMap<ProductionId, Manifest>)` | Same validation with authoritative artifact manifests. |

Output methods consume the context and return it for chaining. Names must be unique after flattening;
for example, composite output `x` can conflict with existing `x.0`. A scalar or single-leaf family
keeps `x`, even if supplied through a one-element tuple/vector. Composite outputs use `x.0`, `x.1`,
... without recursively preserving Rust field names. An explicit public/private operation applies
to every flattened leaf: `private_output` on a `Trapdoor` includes both public and secret leaves,
whereas `private_trapdoor_output` projects the secret leaf only. Visibility is an output declaration; artifact validation checks that an imported declaration
agrees with its producer manifest. It does not implement information-flow analysis or access control. Only declared outputs and their dependencies are retained by the DSL builder.

### Ring inputs and artifact inputs

`Ring::new(modulus: E, ring_dimension: E)` stores symbolic ring metadata.
`ring.matrix_type(shape: S)` returns its `MatrixType`. Concrete matrices require modulus greater
than one, positive ring dimension, and positive row/column counts. Further backend restrictions are
checked by the relevant validator/runtime. Integer, Boolean, and byte inputs are ring-independent.

| API on `ring` | Return type |
| --- | --- |
| `input(name: N, shape: S)` | `Mat` |
| `small_matrix_input(name: N, shape: S, max_coefficient_bound: E)` | `SmallMatrix` |
| `preimage_input(name: N, shape: S, max_coefficient_bound: E)` | `Preimage` |
| `bool_input(name: N)` | `Bool` |
| `bytes_input(name: N, length: E)` | `Bytes` |
| `input_family(name: N, count: E, shape: S)` | `Family<Mat>` |
| `small_matrix_input_family(name: N, count: E, shape: S, max_coefficient_bound: E)` | `Family<SmallMatrix>` |
| `preimage_input_family(name: N, count: E, shape: S, max_coefficient_bound: E)` | `Family<Preimage>` |
| `artifact_input(production_id: ProductionId, artifact_name: N, shape: S, confidentiality: Confidentiality)` | `Mat` |
| `small_matrix_artifact_input(production_id: ProductionId, artifact_name: N, shape: S, max_coefficient_bound: E, confidentiality: Confidentiality)` | `SmallMatrix` |
| `preimage_artifact_input(production_id: ProductionId, artifact_name: N, shape: S, max_coefficient_bound: E, confidentiality: Confidentiality)` | `Preimage` |
| `bytes_artifact_input(production_id: ProductionId, artifact_name: N, length: E, confidentiality: Confidentiality)` | `Bytes` |
| `family_artifact_input(production_id: ProductionId, artifact_name: N, count: E, shape: S, confidentiality: Confidentiality)` | `Family<Mat>` |
| `small_matrix_family_artifact_input(production_id: ProductionId, artifact_name: N, count: E, shape: S, max_coefficient_bound: E, confidentiality: Confidentiality)` | `Family<SmallMatrix>` |
| `preimage_family_artifact_input(production_id: ProductionId, artifact_name: N, count: E, shape: S, max_coefficient_bound: E, confidentiality: Confidentiality)` | `Family<Preimage>` |
| `trapdoor_artifact_input(production_id: ProductionId, public_artifact_name: N, trapdoor_artifact_name: N, rows: E, sigma: R, gadget_base: E, digit_count: E, preimage_max_coefficient_bound: E)` | `Trapdoor` |
| `trapdoor_family_artifact_input(production_id: ProductionId, public_artifact_name: N, trapdoor_artifact_name: N, count: E, rows: E, sigma: R, gadget_base: E, digit_count: E, preimage_max_coefficient_bound: E)` | `Family<Trapdoor>` |

Artifact builders declare dependencies; they do not load files during graph construction. Metadata
must agree with the producer and supplied manifest. Trapdoor imports use a public matrix artifact
and a private secret artifact from the same production. Their public matrix has
`rows × (rows * (digit_count + 2))` entries. For other complete schemas, including `Family<Bool>`,
`Family<Bytes>`, or a custom record, use `context.input` rather than inventing a typed ring method.

### Constants, samplers, and hash domains

| API on `ring` | Result and meaning |
| --- | --- |
| `zero(shape: S)` | `Mat`, all zero. |
| `identity(size: E)` | Square identity `Mat`. |
| `gadget(rows: E, base: E, digit_count: E)` | Gadget `Mat` of shape `rows × (rows * digit_count)`, ordinary decomposition mode. |
| `constant(shape: S, value: ConstantMatrix)` | `Mat` using the exact core constant descriptor below. |
| `polynomial(coefficients: impl IntoIterator<Item = IntExpr>)` | Scalar `Mat`; coefficients in ascending polynomial degree, length at most ring dimension. |
| `pack_polynomial_coefficients(bits: Family<Bool>, coefficient_bits: usize)` | Scalar polynomial `Mat`; inverse layout of coefficient bit serialization. Bit count must equal ring dimension times positive coefficient width, with `2^coefficient_bits >= modulus`. |
| `uniform_residue(shape: S)` | `Mat`, independent uniform residue coefficients. |
| `uniform_interval(shape: S, minimum: E, maximum: E)` | `Mat`; only inclusive intervals `[-1, 1]` and `[0, 1]` are supported. |
| `gaussian(shape: S, sigma: R, max_coefficient_bound: E)` | `Mat`; Gaussian sampling with explicit nonnegative sigma and integer cutoff. |
| `hash_matrix(key: Bytes, tag: impl Into<HashTag>, shape: S)` | Hash-derived `Mat`. |
| `hash_decomposed(key: Bytes, tag: impl Into<HashTag>, shape: S, base: E, digit_count: E)` | Hash-derived `SmallMatrix`, declared shape unchanged, ordinary digit mode. |
| `hash_small_decomposed(key: Bytes, tag: impl Into<HashTag>, shape: S, base: E, digit_count: E)` | Hash-derived `SmallMatrix`, declared shape unchanged, small digit mode. |
| `sample_trapdoor(rows: E, sigma: R, gadget_base: E, digit_count: E, preimage_max_coefficient_bound: E)` | `Trapdoor`; one producer with public matrix and secret outputs, public shape `rows × (rows * (digit_count + 2))`. |

Hash keys must contain exactly 32 bytes. Bounded hash/decomposition operations require base greater
than one and positive digit count. For bounded hashes the requested row count must be divisible by
the digit count. Their coefficient bounds are `round(base / 2)` for ordinary
mode and `base - 1` for small mode. Hash results carry generic bounded-matrix types, not a relation
to a separately declared source matrix. Gaussian cutoffs and bounded input coefficient bounds must be
nonnegative. Trapdoor sampling requires strictly positive sigma, absolute gadget base greater than one,
positive digit count, and a nonnegative preimage cutoff. Preimage dimensions must satisfy the
public-matrix/target product contract. These checks do not by themselves prove a cryptographic norm
bound.

The full `ConstantMatrix` argument vocabulary is:

| Variant | Parameters and interpretation |
| --- | --- |
| `Zero` | Zero entries. |
| `Identity` | Identity entries. |
| `UnitRow { index }` | Unit-row constant; nonnegative `IntExpr` index below the matrix column count. |
| `UnitColumn { index }` | Unit-column constant; nonnegative `IntExpr` index below the matrix row count. |
| `Gadget { base, small }` | `IntExpr` base with absolute value greater than one and explicit Boolean mode. |
| `PowerOfBase { base, exponent }` | Nonzero integer base and nonnegative integer exponent. |
| `Rotation { exponent }` | Polynomial rotation constant; exponent in `[0, ring_dimension)`. |
| `Polynomial { coefficients }` | `Vec<IntExpr>` in ascending degree, no longer than ring dimension. |

These descriptors are core primitive constants, not new control forms. Their matrix realization
is the one used by the runtime's trusted constant constructors.

`HashTag::new()` and `HashTag::default()` create empty prefix/component sequences.
`HashTag::from(Vec<u8>)` and `HashTag::from(&[u8])` set a fixed raw prefix.
`tag.push(part)` appends a typed component; `tag.push_decimal(index: impl Into<Int>)` appends a
compile-known integer's decimal representation or returns `CompileTimeIndex`.
`HashTagPart::append_to(self, tag: &mut HashTag)` is the extension trait implemented by:

| Component type | Encoding contract |
| --- | --- |
| `&str`, `String` | UTF-8 bytes component. |
| `IntExpr` | Compile-resolved unsigned 64-bit little-endian component; value must fit `u64`. |
| `Int` | A directly usable loop binder uses the same unsigned little-endian encoding; other integer nodes become explicit integer operands. |

`tag![part, ...]` expands to an empty `HashTag` plus ordered `push` calls; it permits zero parts and a
trailing comma. A byte slice is a prefix via `HashTag::from`, not a `HashTagPart` implementation.
Primitive Rust integer literals must be converted into `IntExpr` or `Int` before `push`/`tag!`.
Component order and type framing are part of sampled-value identity. Distinct frames and prefixes
must not be silently substituted when reusing persisted artifacts.

### Matrix and bounded-value operations

Except where stated, matrix operands must have the same ring. Shape checks happen during
validation. `Mat` operators consume owned operands or clone borrowed handles; method calls consume
`self` unless they are inspectors or explicitly borrow it.

| API | Result, shape, and constraints |
| --- | --- |
| `mat.matrix_type()` | `&MatrixType`. |
| `mat + rhs`, `mat - rhs`, `-mat` | `Mat`; addition/subtraction require equal shapes. |
| `mat * rhs` with `rhs: Mat` | Matrix product; a scalar `1 × 1` matrix on either side scales the other matrix. Otherwise inner dimensions must agree. |
| `mat * scalar` | `Mat`, scalar converts into `IntExpr` and becomes a constant polynomial. |
| `scalar * mat` | Supported directly for left-hand `i32`; use a scalar matrix for other left-hand forms. |
| `Mat::multi_row_gemm_accumulate(products: Vec<(C, Mat, Mat)>, bias: Option<Mat>)` where `C: Into<IntExpr>` | Fused sum of coefficient-weighted products plus optional bias. Products must be nonempty and have compatible result shapes; empty products panic at construction. |
| `mat.mul_small_rhs(rhs: SmallMatrix)` | `Mat`, computes `mat * rhs` with a bounded right operand; requires `mat.columns == rhs.rows`, with no scalar broadcasting. |
| `mat.ring_automorphism(index: E)` | Same-shape `Mat`, entrywise negacyclic substitution `X -> X^index`; ring dimension must be a power of two, and index must be odd with `1 <= index < 2 * ring_dimension`. |
| `mat.modulus_switch(modulus: E)` | Same-shape `Mat`; rounds coefficients scaled by destination/source modulus. Source and destination must be odd; destination greater than one must divide source. |
| `mat.reduce_modulus(modulus: E)` | Same-shape `Mat`; coefficient reduction into a divisor ring, without scaling. |
| `mat.transpose()` / `mat.t()` | `Mat`, exchanges rows and columns. `t` is a supported shorthand. |
| `mat.slice(rows: Option<IndexRange>, columns: Option<IndexRange>)` | `Mat`; `None` retains that axis; `IndexRange { start, end }` is a compile-expression half-open range, checked against the input extent. |
| `mat.tensor(rhs: Mat)` | Kronecker product `Mat`, multiplying each corresponding shape extent. |
| `mat.decompose(base: E, digit_count: E)` | `Preimage`, row count multiplied by digit count, ordinary gadget decomposition. |
| `mat.small_decompose(base: E, digit_count: E)` | `Preimage`, same shape rule, small gadget decomposition. |
| `mat.extract_coefficient(position: E)` | `Int`; scalar polynomial only, coefficient position in range. |
| `mat.extract_coefficient_with_canonical_input_exclusive_upper(position: E, canonical_input_exclusive_upper: Option<BigUint>)` | Same extraction; optional authoritative compile-time contract that the canonical integer is less than `U`. This is metadata, not a runtime clamp. |
| `mat.canonical_coefficient_bits(ring_dimension: usize, coefficient_bits: usize)` | `Result<Family<Bool>, DslError>`; serializes requested coefficients, coefficient-major and little-endian within each coefficient. Requires scalar input, in-range positions, and a nonempty packed bit list. |
| `mat.threshold_decode_ints(plaintext_modulus: E, length: usize)` | `Vec<Int>` from one threshold decoder, one output port per decoded entry. |
| `mat.threshold_decode_bools(plaintext_modulus: E, length: usize)` | `Vec<Bool>` from the Boolean threshold mode. |
| `Mat::concat(axis: ConcatAxis, values: Vec<Mat>)` | Concatenates same-ring matrices. Rows require equal columns; Columns require equal rows; Diagonal sums both extents with zero off-diagonal blocks. Empty input panics. |
| `Mat::crt_recompose(values: Vec<Mat>, plaintext_moduli: Vec<IntExpr>, reconstruction_coefficients: Vec<IntExpr>, modulus: IntExpr)` | Core CRT recomposition into the destination modulus. Inputs must be nonempty one-row matrices with equal column count/ring dimension; both metadata lists match input count. Empty inputs panic. |

Threshold decoding requires a scalar polynomial input, a positive output length no greater than its
ring dimension, and `plaintext_modulus > 1`. Its integer/Boolean result is the runtime primitive's
threshold result, not a Rust cast. CRT plaintext moduli satisfy `1 < plaintext_modulus <= input modulus` for each
input and output modulus is greater than one; each reconstruction coefficient must satisfy `0 <= coefficient < destination modulus`.

`concat_rows![a, b, ...]`, `concat_cols![a, b, ...]`, and `concat_diag![a, b, ...]` expand to
`Mat::concat` with the corresponding axis. They require at least one expression and permit a
trailing comma. All owned/borrowed combinations of two `Mat` operands are supported for `+`, `-`,
and `*`, and both owned/borrowed negation are supported. There is no implicit scalar addition,
no matrix division operator, and no generic arithmetic implementation on `SmallMatrix` or `Preimage`.

| API on bounded values and trapdoors | Result and contract |
| --- | --- |
| `small.matrix_type()`, `preimage.matrix_type()` | `&MatrixType`. |
| `small.max_coefficient_bound()`, `preimage.max_coefficient_bound()` | `&IntExpr`. |
| `preimage.mul_small_rhs(lhs: Mat)` | `Mat`; computes `lhs * preimage`, preserving the preimage as the bounded right operand despite the receiver position; requires `lhs.columns == preimage.rows`, with no scalar broadcasting. |
| `trapdoor.public_matrix()` | `Mat`; clones the existing public output handle. |
| `trapdoor.preimage_max_coefficient_bound()` | `&IntExpr`. |
| `trapdoor.sample_preimage(target: Mat, shape: S)` | `Preimage`; samples using the public matrix, secret trapdoor, and target. Requested output must multiply with the public matrix to the target shape; cutoff comes from the trapdoor schema. |

### Runtime scalar operations and conversions

| API | Semantics |
| --- | --- |
| `Int::constant(value: impl Into<BigInt>)` | Exact integer constant wire. |
| `Int::evaluate(expression: E)` | Materializes a compile expression; unlike `DslContext::evaluate_int`, it may initially carry the constant-integer wire kind. |
| `int.add(rhs)`, `int.sub(rhs)`, `int.mul(rhs)`, `int.div(rhs)`, `int.rem(rhs)` | Each RHS accepts `impl Into<Int>`; returns `Int`. These implement `+`, `-`, `*`, `/`, `%`. |
| `-int` | `Int`, implemented as zero minus the operand. |
| `int.equal(rhs)`, `int.less(rhs)`, `int.less_equal(rhs)` | `Bool`; RHS accepts `impl Into<Int>`. |
| `int.bit(position: impl Into<Int>)` | `Result<Bool, DslError>`; position must be recoverable as compile-known metadata and nonnegative when validated. |
| `int.lift_to_constant_polynomial(matrix_type: MatrixType)` | `Mat`; embeds the integer into a scalar polynomial over the supplied ring. Non-scalar declared shape panics at construction. |
| `int.expression()` | `Result<IntExpr, DslError>`; recovers only constants, compile expressions, and supported arithmetic with valid lexical provenance. Division/remainder recovery requires a positive constant divisor. |
| `Bool::constant(value: bool)` | Boolean constant wire. |
| `boolean.to_int()` | `Int`, false maps to zero and true to one. |
| `a & b`, `a \| b`, `a ^ b`, `!a` | Boolean conjunction, disjunction, exclusive-or, and negation; graph computations with no short-circuiting. |

Runtime integer division computes `q = floor(a / abs(b))` and `r = a - abs(b)*q`, so
`0 <= r < abs(b)`; the divisor sign does not change either result (for example `7 / -3 = 2`). Division by zero is a runtime
error. Compile-expression division is different, as specified below.

`Int::from` / `.into()` accept every Rust signed/unsigned integer width including `isize` and
`usize`, owned `BigInt`/`BigUint`, borrowed `&BigInt`/`&BigUint`, owned/borrowed `IntExpr`, `&Int`,
and owned/borrowed `Bool`. Primitive integer references are not provided as conversions.
The same RHS conversions apply to owned and borrowed `Int` arithmetic. Left-hand arithmetic with
`Int` or `&Int` is implemented for `i32`, `BigInt`, `BigUint`, their big-integer references,
`IntExpr`, and `&IntExpr`. Other primitive widths can be explicitly converted to `Int` first.
`Bool::from` accepts `bool`, `&bool`, and `&Bool`; Boolean operators accept owned/borrowed graph
Booleans and literal Boolean RHS values. A left-hand `bool` supports a `Bool` or `&Bool` RHS.

The DSL does not define arithmetic on `Bytes` or convert raw byte vectors into `Bytes`; use a byte
input. Standard Rust `==`, `<`, `<=`, `&&`, `||`, and indexing `family[index]` are not symbolic DSL
operators. Use the methods and control functions in this reference.

### Families, control, and subgraphs

| API | Result and complete argument contract |
| --- | --- |
| `Family::<T>::pack(elements: Vec<T>)` | `Result<Family<T>, DslError>`; nonempty, identical complete schemas, normalizes constant scalar leaves. |
| `family.count()` | `&IntExpr`. |
| `family.at(index: impl Into<Int>)` | `T`, complete element including its static metadata. |
| `family.field(project: impl FnOnce(T) -> U)` | `Result<Family<U>, DslError>`; structural projection of placeholder fields only, preserving selected wire identities. |
| `Family<Mat>::element_type()` | `&MatrixType`. |
| `Family<Preimage>::element_type()` | `&MatrixType`. |
| `Family<Preimage>::max_coefficient_bound()` | `&IntExpr`. |
| `Family<Trapdoor>::public_matrices()` | `Family<Mat>`; existing public field projection. |
| `parallel(count: E, body: impl FnOnce(Int) -> Result<T, DslError>)` | `Result<Family<T>, DslError>`; nonnegative count, ordered independent instances. |
| `iterate(count: E, initial: S, body: impl FnOnce(Int, S) -> Result<S, DslError>)` | `Result<S, DslError>`; here `S: GraphValue` is a Rust state type, not the table's shape abbreviation. Nonnegative count and invariant complete state schema. |
| `select(selector: impl Into<Int>, candidates: Vec<T>)` | `Result<T, DslError>`; nonempty same-schema candidates, shared selector for every leaf. |
| `Subgraph::<I, O>::define(name: N, input_schema: I::Schema, body: impl FnOnce(I) -> Result<O, DslError>)` | `Result<Subgraph<I, O>, DslError>`; captures lexical dependencies, normalizes output scalar leaves, and seals one named body. |
| `subgraph.call(input: I)` | `Result<O, DslError>`; exact complete input schema match. |
| `subgraph.call_with_canonical_input_exclusive_uppers(input: I, canonical_input_exclusive_uppers: Vec<Option<BigUint>>)` | Same call with one optional canonical integer bound per flattened explicit argument. Vector length must match, `Some(U)` must have `U > 0`, and bounds are only accepted for matrix arguments. Automatically captured arguments get no bound. |

Constant integer/Boolean leaves are promoted to ordinary scalar wire kinds when packing, selecting,
sealing subgraphs, or constructing loop outputs. This permits constant and runtime candidates with
the same DSL schema without changing numeric values. An `iterate` body returning a changed schema
is rejected before sealing. `Subgraph` implements `Clone` by sharing its definition/captures.

### Compile expressions accepted by metadata arguments

These are core-owned types imported from `mxx_ir_core`; the DSL consumes them as metadata and
through `Int::evaluate`. Parameters are declared on the context and bound with
`ParamEnv { integers, reals, loop_indices }`: integer bindings use `BigInt`, real bindings use
`Rational`. Loop-index bindings are managed by graph execution. Domain authors normally obtain a
loop expression through `i.expression()` instead of constructing binder slots themselves.

| `IntExpr` form | Meaning and errors |
| --- | --- |
| `Const(BigInt)`, `IntExpr::constant(value)`, integer conversions | Exact integer constant. |
| `Var(String)` | Named integer parameter; missing binding errors. |
| `LoopIndex(u32)` | Lexically bound loop index; requires the matching binder. |
| `Add(a,b)`, `Sub(a,b)`, `Mul(a,b)` | Boxed operands; same syntax as `+`, `-`, `*`. |
| `Div(a,b)` / `/` | Exact integer division; zero divisor and nonzero remainder error. |
| `FloorDiv(a,b)` / `a.floor_div(b)` | Quotient rounded toward negative infinity; zero divisor errors. |
| `Rem(a,b)` / `%` | Floor remainder with divisor's sign; zero divisor errors. |
| `RoundDiv(a,b)` | Nearest-integer division, ties toward positive infinity; denominator must be positive. |
| `Log2Ceil(value)` | Ceiling of binary logarithm; argument must be at least one. |
| `Select { selector, branches }` | Compile-time zero-based branch lookup; only selected branch evaluated; invalid index errors. |

Recursive integer operands use `Box<IntExpr>`; `Select` has `selector: Box<IntExpr>` and
`branches: Vec<IntExpr>`. Binary operators accept `IntExpr` or `&IntExpr` on the left and any
`Into<IntExpr>` value on the right. A primitive integer on the left must be converted explicitly,
for example `IntExpr::from(2) * &columns`.

`IntExpr` supports unary negation, owned/borrowed expression arithmetic, conversions from every
primitive integer width, owned/borrowed `BigInt`/`BigUint`, and `&IntExpr`. `evaluate(&ParamEnv)`
returns `Result<BigInt, ExprError>`. `canonicalize()` returns the canonical expression;
`contains_variable(&str)` inspects named variables. Partial-operation errors survive canonicalization,
cancellation, zero multiplication, and serialization.

| `RealExpr` form | Meaning |
| --- | --- |
| `Rational(Rational)` | Exact rational constant. |
| `Var(String)` | Named real parameter. |
| `FromInt(IntExpr)` | Exact embedding of compile integer metadata. |
| `Add(a,b)`, `Sub(a,b)`, `Mul(a,b)`, `Div(a,b)` | Boxed real arithmetic; operators `+`, `-`, `*`, `/` and unary negation are available. Zero divisor errors on evaluation. |
| `Sqrt(value)` | Square root; negative argument errors. |

Recursive real operands use `Box<RealExpr>`; `FromInt` contains an unboxed `IntExpr`.
Binary operators accept `RealExpr` or `&RealExpr` on the left and any `Into<RealExpr>` value on the
right. Convert a primitive left operand explicitly, for example `RealExpr::from(2) * &variance`.

`RealExpr` accepts owned/borrowed `RealExpr`, `IntExpr`, `Rational`, big integers, and primitive
integer values through the applicable `From` implementations; borrowed primitive integers are not
implicit conversions. `RealExpr::from_integer` creates an exact constant.
`RealExpr::from_f64_exact` is an explicit fallible conversion of a finite floating-point number into
its exact binary rational, not a decimal approximation. `contains_variable` inspects parameters,
`evaluate_f64` evaluates numerically, `evaluate_rational` evaluates rational-only expressions and
rejects square roots, and `close` substitutes parameter bindings while preserving symbolic real
operations. Each evaluation/closing method accepts `&ParamEnv` and returns `Result`.

`Rational::new(numerator: BigInt, denominator: BigInt)` normalizes sign/common factors and rejects
zero denominator. `Rational::from_integer(BigInt)` is exact;
`Rational::from_f64_exact(f64)` rejects non-finite values and preserves the finite binary value
exactly. `numerator()` and `denominator()` return `&BigInt`. These core numeric types are not
runtime graph scalar types: there is no DSL `Real` wire wrapper.

### Domain records, inspection, and error boundaries

The sealer preserves member correspondence: `a.at(i)` can become a Zip input, and supported constant
offsets become ZipOffset inputs. Indirect reads such as `table.at(indices.at(i))` retain the shared
table and dynamic lookup. These are internal representations, not caller-selected execution modes.
Outer producers are not moved into consumers or re-sampled during capture conversion.

A `GraphValue: Clone` supplies `type Schema: GraphValueSchema<Value = Self>`,
`flatten(&self) -> Vec<ValueHandle>`, `schema(&self) -> Self::Schema`, and
`from_values(schema: &Self::Schema, values: &[ValueHandle]) -> Result<Self, DslError>`.
Flattening and reconstruction must agree in field count/order and preserve static record metadata.
Implementations must return `Schema` when the supplied values cannot reconstruct their declared
schema; graph validation remains authoritative for concrete wire compatibility.

`GraphValueSchema: Clone + PartialEq` supplies `type Value: GraphValue<Schema = Self>`,
`wire_types(&self) -> Vec<WireType>`, and
`placeholders_from(&self, next: &mut usize) -> Self::Value`. The provided
`placeholders()` starts numbering at zero. `placeholders_from` is a hidden implementation hook:
record schemas must delegate using the same counter so placeholder names remain unique.
Complete schema equality must include every static property needed to interpret a value; flattening
alone does not establish that two records have the same schema.

The following public inspection/extension methods expose core IR handles, not extra computation
syntax: `value_handle()` on `Mat`, `SmallMatrix`, `Preimage`, `Trapdoor`, `Int`, `Bool`, `Bytes`,
and `Family<T>`; `Family<Trapdoor>::secret_value_handle()`; `GraphValue::flatten` and
`GraphValue::from_values`; `GraphValueSchema::placeholders`/`placeholders_from`/`wire_types`;
and the `BuiltGraph::graph` field. Handles are borrowed `&ValueHandle`; `Family::value_handle` panics for
composite elements because it requires exactly one flattened family wire. Trapdoor `value_handle`
and the secret-family handle identify the secret component. These escape hatches serve reusable
domain builders, runtime integration, and inspection; their presence does not make arbitrary core
`NodeKind` values part of the ordinary DSL vocabulary.

`IdealSpec::new(graph: Graph)` and `PurePredicateSpec::new(graph: Graph)` return `Result` with core
`SpecificationError`. Both reject all samplers, including hash sampling, throughout the graph;
a predicate additionally requires exactly one Boolean or constant-Boolean output. Both wrappers
expose their core `graph` field. Neither proves the protocol's algebraic correctness by construction.

| `DslError` variant | Boundary |
| --- | --- |
| `CompileTimeIndex` | Runtime/escaped integer cannot be used as compile-known metadata. |
| `Freeze(FreezeError)` | Invalid graph/scoped dependency during sealing or freezing. |
| `DuplicateOutput(String)` | Repeated flattened output name. |
| `Schema` | Empty/incompatible schema, invalid field projection, or invalid reconstruction. |
| `CanonicalInputUpperCount` | Subgraph bound-vector length mismatch. |
| `CanonicalInputUpperZero` | Zero exclusive canonical bound. |
| `CanonicalInputUpperNonMatrix` | Canonical bound applied to a non-matrix argument. |
| `FamilyCountMismatch` | Available domain-builder error for unequal parallel family counts. |
| `StructuralValidation(ValidationError)` | Core structural validation failed. |
| `Specification(SpecificationError)` | Core pure-specification construction failed. |

`ValidationBuildError::Core(ValidationError)` wraps concrete validation failures. Numerical execution
can additionally fail on runtime inputs (for example division by zero or a dynamic index outside a
family), artifact mismatches, or sampler/backend errors; these are runtime errors rather than new
DSL syntax. Methods that return values directly may defer checks until build/validation/execution;
explicit construction-time panics are noted in the tables. This specification does not substitute
for backend capability checks in `docs/runtime.md` or protocol correctness checks in
`docs/correctness/operational-protocol-inventory.md`.
