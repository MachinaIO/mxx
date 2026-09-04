# Graph runtime

`mxx-runtime` executes a validated `mxx-ir-core` plan. It does not accept a
mutable builder or reinterpret symbolic annotations as executable operations.

The liveness plan releases intermediates after their last use unless trace mode
requests retention. Subgraph bodies share one validated plan while each call
has its own concrete instantiation path. Parallel loops execute in bounded
waves according to `max_parallel_instances`; each iteration receives its
concrete loop index and produces an ordered indexed-family result.

CPU-independent work uses Rayon where iterations are independent. GPU work may
intentionally use a smaller wave size to respect VRAM limits. DAG construction,
dependency-ordered traversal, and deterministic reductions remain ordered.

Artifacts are supplied and returned through the runtime artifact interfaces.
Final applications decide how artifact payloads are persisted.
