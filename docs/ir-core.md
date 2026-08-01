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

Symbolic reinterpretation and magnitude analysis are separate layers. The core
stores only executable semantics and artifact metadata.

Tensor product, concatenation, selection, transpose, slice, reshape,
constant-coefficient extraction, and CRT recomposition are ordinary executable
core nodes. A symbolic layer may mirror these nodes, but it cannot change their
runtime dataflow. Generic modulus-down and modulus-up operations are not exposed
by the current core IR. Nested-RNS level switching remains a circuit gadget and
is not represented as a generic core conversion.
