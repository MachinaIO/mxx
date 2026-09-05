# Core graph IR

`mxx-ir-core` owns the canonical executable DAG. Nodes and values are immutable
handles during construction. `Graph::freeze` keeps reachable nodes, assigns
stable scoped identities, preserves explicit sharing, and produces the flat
serializable graph.

Subgraph calls and parallel loops remain structural nodes. Validation checks a
subgraph body once under its parameter environment instead of expanding every
call or loop iteration. Concrete runtime identities add call and loop-index
frames to the scoped wire identity.

Validation resolves compile parameters and artifact manifests, checks wire
types and structural constraints, and builds the execution and liveness plans
consumed by `mxx-runtime`. The graph specification hash depends on semantics,
not allocation addresses, construction scheduling, source locations, or
unreachable expressions.

Correctness declarations and deterministic hard-bound proofs are separate from
execution. The core stores executable semantics, artifact metadata, and the
authoritative integer cutoffs on sampler nodes.

`protocol` owns the declarations connecting multiple executable graphs: stages,
artifact links, external-input contracts, ideal specifications, requirements,
and decoder endpoints. `IdealSpec::new(Graph)` and `PurePredicateSpec::new(Graph)`
check sampler-free graphs without depending on the DSL. The DSL still owns
construction and returns a `BuiltGraph`; pass its `graph` field to these constructors.

`lean::protocol::export_claim` accepts a `ProtocolDecl`, concrete parameter and
manifest bindings, a `ClaimBackend`, and `ClaimSemantics`. It exports the graphs
and derives their connections and endpoints from the declaration, then invokes
`lean::claim` to render the final statement. Neither layer infers noise bounds;
applications retain their bound calculations and Lean proofs. Structural families
remain symbolic rather than being expanded into one graph or claim per element.

Tensor product, concatenation, selection, transpose, slice, reshape,
constant-coefficient extraction, and CRT recomposition are ordinary executable
core nodes. Generic modulus-down and modulus-up operations are not exposed
by the current core IR. Nested-RNS level switching remains a circuit gadget and
is not represented as a generic core conversion.
