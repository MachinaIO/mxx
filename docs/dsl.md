# Declarative graph DSL

`mxx-dsl` builds the canonical executable `mxx-ir-core` graph. Its control vocabulary consists of
ordinary values, homogeneous indexed families, indexing, independent iteration, carried-state
iteration, and selection. Arithmetic creates immutable shared nodes immediately; numeric execution
happens after validation in `mxx-runtime`.

```rust
use mxx_dsl::{DslContext, Ring, parallel};
use mxx_ir_core::ParamEnv;

let ring = Ring::new(17, 8);
let inputs = ring.input_family("inputs", 4, (2, 2));
let outputs = parallel(4, |i| {
    let input = inputs.at(i);
    Ok(input.clone() + input)
})?;
let built = DslContext::new("double").output("outputs", outputs)?.build()?;
let validated = built.validate(&ParamEnv::default())?;
```

## Values and families

`Mat`, `SmallMatrix`, `Preimage`, `Trapdoor`, `Int`, `Bool`, and `Bytes` are graph values. Tuples,
fixed-length vectors, domain records implementing `GraphValue`, and families are graph values too.
`GraphValueSchema` describes the complete static schema, including domain metadata. Library authors
implement these traits for domain records; ordinary callers use those records directly.

`Family<T>` is an ordered collection with a common element schema and compile-time count. An
element can contain several graph wires. For example, `Family<Trapdoor>` keeps each public matrix
aligned with its private trapdoor, and `Family<CircuitEncoding>` keeps a Boolean circuit encoding's
vector, public key, and plaintext together. Storage remains a collection of aligned leaf families;
there is no separate record IR or second expression language.

`Family::pack` requires a nonempty collection of same-schema values. A zero-count `parallel` builds
a typed empty family. Rings, matrix shapes, coefficient bounds, vector lengths, and static record
metadata must agree across elements. Optional fields have one fixed presence/absence throughout a
family. Ragged arrays and iteration-dependent output schemas are not supported.
Nested `Family` elements are also unsupported, including records containing another family:
the current artifact storage format has one family dimension. Use fixed-length Rust vectors or
tuples for composite elements. A complete family can still be an `iterate` state, a selected value,
or a subgraph argument; these operations do not add a family dimension.

`family.at(index)` reads a complete element. The same API accepts constants, compile expressions,
loop indices, and runtime integers. Known static indices are validated before execution; dynamic
indices are checked at runtime. Negative or out-of-range indices are errors. Indexing a longer
family over a shorter iteration domain intentionally reads a prefix; domain operations requiring
equal lengths validate that separate contract rather than silently truncating inputs.

`family.field(|record| record.field)` projects existing fields without computing values or adding a
loop. Tuple and vector field projections work the same way. It preserves the original producer and
sampling identities. Arithmetic inside a field projection is rejected; use `parallel` for computation.

## Arithmetic notation

Compile expressions support natural owned and borrowed arithmetic:

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

Runtime `Int` supports arithmetic operators and borrowed operands. Division rounds down using the
absolute divisor; remainder is nonnegative modulo the absolute divisor. Division by zero fails at
runtime. `equal`, `less`, and `less_equal` accept integer literals and borrowed integers;
Rust comparison operators cannot return a symbolic `Bool`.

Matrices support owned or borrowed `+`, `-`, `*`, and unary `-`. Integer/compile-expression scalar
multiplication uses a scalar constant polynomial and the existing multiplication primitive, such
as `&matrix * 2` or `2 * &matrix`. Scalar addition has no implicit broadcasting rule.
Primitive integers on the left of runtime `Int` arithmetic or matrix multiplication use `i32`,
so unsuffixed literals infer naturally. For other primitive types, put the scalar on the right
of multiplication or explicitly convert it to a graph integer for integer arithmetic.

Booleans support `&`, `|`, `^`, and `!`, including borrowed operands and Boolean literals. These
construct Boolean computations without short-circuiting. Rust `&&` and `||` cannot be overloaded
for graph values. Domain operations that require compiler parameters or preprocessing inputs retain
explicit method arguments.

## Independent and carried-state iteration

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

The sealer preserves member correspondence: `a.at(i)` can become a Zip input, and supported constant
offsets become ZipOffset inputs. Indirect reads such as `table.at(indices.at(i))` retain the shared
table and dynamic lookup. These are internal representations, not caller-selected execution modes.
Outer producers are not moved into consumers or re-sampled during capture conversion.

## Selection

`select(selector, candidates)` selects a same-schema value, including a record or an entire family.
An integer selects a zero-based candidate; a Boolean selects candidate zero for false or one for
true. Every leaf of a record uses the identical selector. Empty candidates, schema disagreement,
and out-of-range selectors are errors.

This is value selection, not a lazy control-flow statement. Do not rely on it to guard an invalid
array access or to suppress a sampler in an unselected candidate. Runtime artifact materialization
can remain lazy without providing general lazy branch semantics.

## Sampling and sharing

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

## Inputs, outputs, and reusable definitions

`DslContext::input` accepts a complete value schema, including a family of records. Existing typed
ring input builders remain convenient for primitive inputs. `output`, `public_output`, and
`private_output` accept complete values. A single leaf retains its supplied name; composite leaves
use `name.0`, `name.1`, and so on in schema order. Field projections allow applications to retain
explicit domain artifact names.

Public and secret persistence remains explicit. For trapdoor families, use `public_matrices` for
the public projection and `private_trapdoor_family_output` for the secret projection; import the
pair with `trapdoor_family_artifact_input`. Composite values do not erase confidentiality boundaries.

`Subgraph::define` and `try_define` store one reusable body. Formal inputs use explicit schemas;
outer lexical reads are lifted to hidden arguments using the same ancestry rule as iteration.
Artifact tables passed as formal arguments remain reusable across calls with the same schema.
Subgraph reuse does not require exposing zip/broadcast details to callers.

`build()` freezes the graph and checks structure. `validate(&ParamEnv)` resolves parameters and
checks concrete types, shapes, bounds, and execution planning. Artifact consumers can additionally
use `validate_with_manifests`. See `docs/runtime.md` for execution.

Correctness declarations use core-owned `IdealSpec::new` and `PurePredicateSpec::new`. Their graphs
must be sampler-free. Domain builders retain semantic anchors and derivation attachments through
scope conversion and freezing; packing a record does not itself establish an algebraic relation.
See `docs/correctness/operational-protocol-inventory.md` for checking semantics.
