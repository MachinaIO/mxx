# Declarative graph DSL

`mxx-dsl` is the typed construction API for executable graphs. Arithmetic on
`Mat` creates immutable `mxx-ir-core` nodes immediately; there is no separate
executable expression tree and no mutable graph builder.

```rust
let ring = Ring::new(q, n);
let input = ring.input("input", (1, 1));
let error = ring.gaussian((1, 1), sigma);
let output = input * ring.identity(1) + error;
let built = DslContext::new("example").public_output("output", output)?.build()?;
```

`Subgraph::define` creates one reusable graph body. Its runtime values must be
explicit arguments; implicit runtime captures are rejected. Matrices, integer
values, bytes, indexed matrix families, trapdoors, preimages, tuples, and typed
structs implementing `GraphValue` may cross a subgraph boundary.

`Family::parallel_map`, the `parallel_zip` variants, and `Parallel::range`
create structural `ParallelLoop` nodes. Their closures describe one body and
are not executed once per element during construction. A primary family is a
`Zip` input, explicitly captured scalar values are `Broadcast` inputs, and
offset family relationships use `ZipOffset`.
Symbolic instantiation records the offset as part of the source iteration
identity: static member `i` of a `ZipOffset { offset }` input refers to source
member `i + offset`, while dynamic identities retain the offset alongside the
index wire.

See [ir-symbolic.md](ir-symbolic.md) for `assume` and symbolic expressions and
[runtime.md](runtime.md) for bounded execution waves.

An operation involving a `VirtualMat` produces a typed symbolic expression
instead of an executable node. The expression uses the same ordinary `+`, `-`,
unary `-`, and `*` syntax, preserves ordered multiplication and nested sums,
and is accepted only by `Mat::assume`. No factor-list macro or aggregate-label
operation is part of the public API.

Concrete values referenced only by an assumption are retained as existing core
values during graph freezing, so callers do not need to publish them as extra
outputs. The assumption itself still creates no executable core node.
