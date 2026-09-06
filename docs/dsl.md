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

Protocol builders may use crate-internal traced variants of these loop combinators to retain the
handles of operations they just created for correctness certificates. Sealing remaps body-local
handles and captured values to the exact sealed scope before freezing. Ordinary DSL users still
receive only `BuiltGraph`; construction traces and the temporary freeze map are not runtime graph
state or a second expression language.

Fixed public circuit descriptions use `DslContext::int_family_input`. `parallel_gather` broadcasts
one read-only source family and dynamically gathers one member for every zipped index. Heterogeneous
zip bundles keep composite values such as a BGG encoding's vector, public key, and plaintext in one
parallel iteration while still lowering every component to ordinary core wires.

`Parallel::range(count).map_values` can also return `Trapdoor`. Because one trapdoor consists of a
public matrix wire and a private trapdoor wire, the result is a `TrapdoorFamily` that keeps those
two indexed families aligned. `public_matrices` exposes the public half, while `get`,
`parallel_map_values`, and `parallel_zip_mat_values` feed matching trapdoors into preimage
preprocessing without expanding a parameterized count. Persist the public half with
`public_family_output` and the private half with `private_trapdoor_family_output`; import the pair
with `trapdoor_family_artifact_input`.

Correctness declarations use core-owned `IdealSpec::new` and `PurePredicateSpec::new`. These wrappers
reject sampling nodes and retain a deterministic graph consumed by
`mxx_ir_core::protocol::ProtocolDecl`; construction remains in the DSL.
There is no virtual matrix, assumption, symbolic overlay, or second expression DAG.

See `docs/runtime.md` for execution and `docs/correctness/operational-protocol-inventory.md` for checking semantics.
