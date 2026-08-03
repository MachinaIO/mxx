# Declarative graph DSL

`mxx-dsl` is the typed construction API for the one executable `mxx-ir-core` DAG. Arithmetic on
`Mat` creates immutable shared nodes immediately.

```rust
let ring = Ring::new(q, n);
let input = ring.input("input", (1, 1));
let error = ring.gaussian((1, 1), sigma, max_coefficient_bound);
let output = input * ring.identity(1) + error;
let built = DslContext::new("example").public_output("output", output)?.build()?;
```

Every Gaussian and preimage sampler carries a required integer coefficient cutoff. Parameterized
cutoffs reference declared integer parameters and are resolved by `ParamEnv`; the DSL never
converts an unresolved `RealExpr` sigma itself.

`Subgraph::define` stores one reusable body. Runtime values crossing its boundary are explicit.
`Family::parallel_map`, `parallel_zip`, and `Parallel::range` create structural
`ParallelLoop` nodes rather than expanding one body per member.

Correctness declarations use `IdealSpec::new` and `PurePredicateSpec::new`. These wrappers reject
sampling nodes and retain a deterministic graph consumed by `mxx-correctness::ProtocolDecl`.
There is no virtual matrix, assumption, symbolic overlay, or second expression DAG.

See `docs/runtime.md` for execution and `docs/lean.md` for correctness verification.
