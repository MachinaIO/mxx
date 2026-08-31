# Graph runtime

`mxx-runtime` executes a validated `mxx-ir-core` plan. It does not accept a
mutable builder or reinterpret symbolic annotations as executable operations.

The liveness plan releases intermediates after their last use unless trace mode
requests retention. Subgraph bodies share one validated plan while each call
has its own concrete instantiation path. Parallel loops execute in bounded
waves according to `max_parallel_instances`; in `Family` output mode each
iteration receives its concrete loop index and produces an ordered
indexed-family result.

CPU-independent work uses Rayon where iterations are independent. GPU work may
intentionally use a smaller wave size to respect VRAM limits. DAG construction,
dependency-ordered traversal, and deterministic reductions remain ordered.

Parallel loops have two explicit output contracts. `Family` preserves every
iteration as an indexed-family value for consumers that need the index. The
`CollectColumns` contract is for a matrix body with exactly one column: the
runtime allocates one final matrix and writes body results into consecutive
columns in ascending loop-index order. It executes only a bounded wave at a
time, releases that wave after the write, and never materializes an indexed
family for the collected result.

The collected destination belongs to the parent placement. Each completed
child is normalized to that placement before the wave is written, so placement
changes do not alter ordering or matrix values. The generic matrix sink uses
one ordinary block copy per column; a dedicated multi-device GPU column sink
is intentionally deferred. Consequently, GPU assembly retains the final
matrix plus the live wave and any backend transfer workspace, while the
estimator charges the wave's transient storage and the final output through
ordinary liveness accounting.

Artifacts are supplied and returned through the runtime artifact interfaces.
Final applications decide how artifact payloads are persisted.
