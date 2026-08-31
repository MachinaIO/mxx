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
`Family` is one rank-N, row-major abstraction. `Parallel::grid` and the one-dimensional convenience
combinators create structural `ParallelGrid` nodes rather than expanding one body per member.

Sealing remaps body-local handles and captured values to the exact sealed scope before freezing.
Ordinary DSL users receive only `BuiltGraph`; construction state is not a second expression
language.

Fixed public circuit descriptions use `DslContext::int_family_input`. Deterministic
`FamilyReindex` uses typed `IndexMap` expressions, while runtime-dependent indexing uses
`FamilyGather` or `FamilySelectAxis`. Selector identity is preserved by explicit aliases, not
inferred from equal ranges.

`Parallel::range(count).map_values` can also return `Trapdoor`. Because one trapdoor consists of a
public matrix wire and a private trapdoor wire, the result is a `TrapdoorFamily` that keeps those
two indexed families aligned. `public_matrices` exposes the public half, while `get`,
`parallel_map_values`, and `parallel_zip_mat_values` feed matching trapdoors into preimage
preprocessing without expanding a parameterized count. Persist the public half with
`public_family_output` and the private half with `private_trapdoor_family_output`; import the pair
with `trapdoor_family_artifact_input`.

`Preimage` is distinct from `Mat`. `Mat::apply_preimage` and `Mat::mul_decomposed` lower to
`ApplyPreimage`, the only multiplication that consumes a relation. `Preimage::materialize_exact`
is accepted by the simulator only when the registered relation target has zero error.
`Decomposition` exposes relation-preserving consumption and guarded scalar entries; there is no
unrestricted conversion to `Mat`.

See `docs/runtime.md` for execution and `docs/noise-simulator-spec.md` for simulation semantics.
