# Tall Operational Selection Checker Status

## Purpose

This document describes the operational checker used to estimate noise for the Tall BGG+
nested-RNS graph. It is written for readers who are new to the checker and describes only the
current design.

The checker is implemented primarily in `lean/Mxx/Certificate/OperationalBounds.lean`. Binary
Graph IR decoding is implemented in `lean/Mxx/Ir/BinaryFormat.lean`.

The executable Rust and CUDA graph does not contain the operational expression representation
described below. The representation is private to Lean analysis and does not change protocol
execution.

## Tall Workload

The small diagnostic configuration currently used for Tall parameter analysis has:

- ring dimension 8;
- CRT depth 1;
- CRT modulus width 10 bits;
- gadget base width 5 bits;
- scale 1024;
- one nested-RNS multiplication;
- parameter-simulation parallelism 1;
- required security level 0 bits.

The generated protocol contains 31 lookup tables, 30,720 lookup preimages, and 48 slot-operation
preimages. The checker must therefore represent large families compactly while retaining the
identity information required by decomposition and preimage relations.

## Operational Facts and Expressions

An ordinary matrix operation is evaluated to an `OperationalMatrixFact`. Such a fact stores the
matrix type and parameters, a flat sum of ordered products, the hard-bound expression, metadata,
canonical-range information, public identity, and relation snapshots.

An unresolved dynamic choice is stored in a request-local `OperationalExprArena`. Each node has a
monotonically allocated `OperationalExprId` and one checked matrix type. The arena supports:

- concrete matrix facts;
- addition and subtraction;
- matrix multiplication together with its derivation rule and right-hand wire;
- tensor products;
- concatenation;
- structural transforms;
- dynamic selection.

`OperationalExprId` is only an array index for local sharing and memoization. It is not serialized,
does not become part of matrix or relation identity, and is not evidence that two expressions are
symbolically equal.

Concrete arithmetic continues to use the flat-polynomial semantics. Expressions are kept
unresolved only when a dynamic selection prevents immediate evaluation without expanding a large
or independent branch domain.

### Analysis-local polynomial interning

Polynomial normalization uses request-local `OperationalFactorId` and `OperationalMonomialId`
values. Canonical factors are interned first, and an ordered product is then interned from its factor
IDs, product modes, and output type. Coefficients are accumulated through a monomial-ID index rather
than by repeatedly scanning all previously normalized products.

The compact fingerprints used by the interners are only bucket selectors. A bucket hit is accepted
only after comparing the complete canonical factor or monomial key, so hash collisions do not alter
symbolic equality. The IDs remain private to one operational evaluation and are neither serialized
nor used as protocol identities. Matrix facts are normalized into the shared request-local interning
arena before entering expression or scope state, which lets later normalization reuse the same IDs.

## Dynamic Selection

A selection is identified by the executable index value and stores its alternatives in one of two
forms.

### Exact alternatives

An exact selection stores an array of expression IDs. It preserves each branch's value identity,
public identity, polynomial, and relations. The arena checks that all alternatives have the same
matrix type. Empty selections and invalid expression references are rejected.

If every exact alternative is the same expression ID, the selection is reduced to that ID without
allocating a selection node. This reduction requires complete expression identity; equal bounds or
equal schemas are not sufficient.

### Schema envelopes

A schema envelope stores:

- the logical alternative count;
- one representative expression;
- a complete uniform operational schema;
- whether the alternatives are relation-free;
- the common public-matrix boundary template, when one exists;
- the common relation boundary template, when one exists.

The representative does not claim that the alternative matrix values are equal. For example, the
30,720 lookup preimages are distinct matrices with distinct identities even when their operational
schemas have the same shape and bound behavior.

Envelope construction checks the summary against the representative before accepting it. A stale
summary whose bound, relation status, or boundary template no longer describes the representative
is rejected. A transformation may retain an envelope only when an explicit transfer rule can
derive a complete output summary from the previously established all-alternative schema. It may
not infer uniformity from the transformed representative alone.

Packed matrix families perform their full all-element schema check when they are first packed.
Subsequent registered deterministic maps transfer that validated summary through the transformed
representative instead of rebuilding every branch schema. Rebinding, loop-index instantiation,
protocol-family selection, and recurrence-bound shifting use this path. An absent source summary,
an unsupported operation, or an incomplete transfer discards the summary and fails closed at any
later operation that requires an envelope; it does not manufacture uniformity from one branch.

## Arithmetic over Selections

Operations on concrete operands are evaluated immediately with the existing operational formulas.
This includes relation rewriting and bounded-noise compression.

For one selected operand, an operation is applied within that selection. Exact alternatives are
processed branch by branch. An envelope is processed through its representative only when the
operation has a valid summary-transfer rule; otherwise evaluation fails closed.

Two selections with the same identity and domain are evaluated branch-wise. Two independent
selection identities remain nested expression nodes. The checker does not positionally zip them and
does not allocate their Cartesian product.

This is important for Tall evaluation. A value chosen by one loop can be multiplied by a preimage
chosen by another loop, and the result can then be added to a term controlled by only one of those
loops. The expression arena preserves both choices without flattening all branch combinations into
one array.

## Relation Rewriting

Decomposition and preimage rewrites remain noncommutative and identity-sensitive. A rewrite is
accepted only when the concrete multiplication boundary satisfies the existing checks for:

- compatible matrix types and moduli;
- exact factor order;
- matching relation producer;
- matching public-matrix identity;
- matching selection identity and branch domain;
- an exact target snapshot or a complete envelope schema that justifies representative evaluation.

A shared boundary template may justify checking an envelope representative, but it is not copied
blindly to the result. Operations that consume relations or change factor boundaries recompute the
output summary. An operation without a reviewed summary-transfer rule rejects an envelope instead
of dropping relation or identity information.

The operational derivation attachments used by BGG grouping are also preserved. The checker
recognizes the BGG encoding family pairing, BGG public-key signal grouping, and protocol Boolean
signal grouping attachments by their owner and rule names.

Relation rewriting processes only terms produced by a successful rewrite. Each input term is
rewritten to completion independently, and unchanged terms are not rescanned merely because
another term changed. The existing 64-step fail-closed limit remains attached to each rewrite chain,
and the completed terms are normalized together once at the boundary. This changed-term work queue
preserves the existing noncommutative matching rules while avoiding repeated full-polynomial passes.

## Loop Handling

Parallel loops are evaluated from one abstract loop body. Packed inputs can retain a compact
selection expression rather than expanding every lane before the body is analyzed. Loop-index,
namespace, protocol-family, and dynamic-selection instantiation map origins, public identities,
relations, and bound expressions consistently through the expression tree.

Expression transformations use a request-local cache keyed by an explicit transformation namespace
and source expression ID. A repeated transformation in the same namespace reuses its prior output,
including the no-op case in which the mapped root is already available. Namespaces include the
relevant node, lane, selection, or coordinate information at their call sites, so results from
different instantiation environments are not conflated. A per-traversal array additionally prevents
revisiting shared children while a new transformation result is being constructed.

Sequential loops summarize their carried values through the existing recurrence analysis.
Relation-bearing carried values remain fail-closed where the checker cannot prove a valid recurrence
schema.

## Noise-Bound Evaluation

For a mutually exclusive dynamic selection, the checker evaluates the complete bound of each exact
alternative and takes their maximum. It does not maximize partial terms independently and combine
pieces that belong to different alternatives.

A validated schema envelope evaluates its representative schema once. Nested independent
selections apply a maximum at each selection node without allocating all branch combinations.

Expression-bound evaluation memoizes results in an array indexed by `OperationalExprId`. The
current fixtures verify that evaluating the same root a second time produces a memo hit. General
structural interning is not part of the design.

Decoder checking traverses concrete facts and operational expressions, obtains the maximum complete
noise bound, and checks the strict inequality

```text
2 * plaintext_modulus * noise_bound < ciphertext_modulus
```

The multiplication form is used directly so integer division cannot change threshold behavior.

The Rust runner groups requests that have the same environment, layout, residual stage, and residual
output. Lean computes the structural residual bound once for that group, after which each compatible
request applies only its own numeric decoder threshold. This separates graph-derived bound work from
cheap threshold checks, including normal and diagnostic requests for the same candidate.

The generated runner emits start and finish records for graph evaluation and decoder-bound
evaluation, including elapsed nanoseconds. Rust forwards Lean stdout and stderr line by line while
retaining the exact byte streams for report parsing and errors. Long evaluations therefore expose
their current phase without waiting for the child process to exit. The final JSON diagnostics also
include expression and memo counts, envelope logical and stored counts, relation rewrites, transform
cache hits and misses, and maximum stored polynomial size.

## Fail-Closed Boundaries

The checker rejects a graph when it cannot preserve the information required for a sound bound.
Current explicit rejection boundaries include:

- an invalid expression ID or expression type mismatch;
- an empty or out-of-range selection;
- an unsupported operation on a schema envelope;
- a missing, ambiguous, malformed, or unavailable matrix relation;
- a public-identity mismatch;
- selection-dependent coefficient extraction or threshold decoding where complete branch conditions
  cannot be established;
- a nested executable `Select` whose alternatives already contain unresolved selections and cannot
  be represented by the supported expression rules;
- structural operations whose selection domains cannot be aligned safely;
- relation-bearing sequential-loop state without a validated recurrence schema.

These errors are analysis failures. They must not be replaced with a guessed bound or by discarding
identity and relation metadata.

## Focused Fixtures

`lean/Mxx/Certificate/OperationalBounds.lean` contains focused fixtures for:

- relation-bearing dynamic extraction followed by relation-consuming multiplication;
- agreement with an explicitly unrolled relation rewrite;
- a 30,720-alternative schema-envelope bound;
- rejection of stale envelope summaries;
- branch-wise decoder maximums;
- request-local expression memoization statistics;
- independent selections without a Cartesian arena allocation;
- exact equal-alternative reduction;
- static selection range errors;
- packed-family and loop identity preservation;
- decomposition, preimage, subgraph, and recurrence relation transport.

The independent-selection fixture verifies an expression arena with two two-way choices without
allocating four Cartesian alternatives. Its endpoint bound is computed from complete branch sums.

## Current Verification Status

This commit is an auditable development checkpoint, not a completed Tall checker. `lake build Mxx`
passes, including the native operational fixtures. The current Tall diagnostic command is:

```sh
RUST_LOG=info \
MXX_TALL_NESTED_RNS_MIN_CRT_DEPTH=1 \
MXX_TALL_NESTED_RNS_MAX_CRT_DEPTH=1 \
MXX_TALL_NESTED_RNS_PARAMETER_SIMULATION_PARALLELISM=1 \
cargo test -p mxx-gadgets \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  --features gpu \
  test_tall_bgg_nested_rns_parameter_simulation \
  -- --ignored --exact --nocapture
```

The latest run used ring dimension 8, CRT depth 1, 10-bit CRT moduli, 5-bit gadget base, scale
1024, and one nested-RNS multiplication. It generated 31 lookup tables, 30,720 lookup preimages,
48 slot-operation preimages, and a producer scope with 92,758 nodes.

The main expression-growth problem is substantially reduced. Before primitive-selection
summarization, the observed producer evaluation took approximately eight minutes and left roughly
337,000 expression nodes. At this checkpoint it reaches the encoding stage with 102,717 expression
nodes; the complete diagnostic run reaches its current failure in 142.93 seconds, including Rust
graph construction, source emission, Lean module preparation, and operational evaluation. These are
development observations rather than controlled benchmark results, but they establish that the
original producer blow-up is no longer the immediate blocker.

The previously failing family choice in encoding parallel body 371 now succeeds. That node selects
one LUT chunk family and then dynamically extracts a row. It is represented pointwise as one
`familyUniform` template containing an exact choice expression, so the family length is not
materialized.

The current first failure is:

```text
OperationalError.inScope
  (parallelBody (root (workflowStage "encoding")) 418)
  (unsupportedOperationalExpr 0)
```

Within that body, node 17 is a matrix multiplication whose operands contain unresolved selection
expressions. Evaluating one two-way exact selection as a single representative finds that one branch
has a different identity-erased polynomial schema. The checker correctly fails closed instead of
discarding the distinction. The unresolved design issue is not another missing primitive formula;
it is that operation lifting and representative construction are separate, partially overlapping
mechanisms. A nested choice can reach a caller that requests one representative even when no valid
uniform representative exists.

The following completion claims therefore remain pending:

- successful traversal of the complete Tall preprocessing, public-key, encoding, and residual
  graphs;
- completion within the 30-minute checker target;
- a successful parameter-search verdict;
- agreement between the resulting simulated threshold and the GPU runtime residual;
- end-to-end Tall runtime correctness and performance.

These checks are required before the operational selection work can be declared complete.

## Complexity Added After the Main Performance Fix

The following mechanisms were introduced to retain the performance improvement while repairing
later semantic failures. They are listed separately because some are essential representation
choices, while others are symptoms of duplicated control flow.

| Mechanism | Why it was introduced | Optimization or correctness property it protects | Current complexity cost |
| --- | --- | --- | --- |
| `OperationalMatrixExpr` arena | Keep unresolved choices as shared expression nodes | Avoids expanding 30,720 LUT alternatives and avoids Cartesian products of independent selections | Every matrix operation now has concrete and expression paths |
| Exact selections and schema envelopes | Preserve distinct branch identities when needed, but store one representative for proven-uniform families | Exact relations remain available; uniform families become O(1) downstream | Two representations require conversion, validation, and fallback rules |
| Cached `containsSelection` | Detect nested choices without recursively rescanning a growing arena | Makes the shape test O(1) per expression node | Correctness now depends on every arena constructor maintaining the cache |
| Operation-specific push-through functions | Apply add, multiply, tensor, concat, transform, scale, and BGG grouping within a choice when safe | Consumes preimage/decomposition relations before joining and avoids premature maxima | Similar alignment and fallback logic is repeated across operations |
| `SelectedMatrixSummary` and transfer registry | Prove that a compact representative still describes every logical branch after an operation | Allows deterministic maps over huge families without rescanning all branches | Summary fields can become stale; every operation must classify each field as recomputed or invalidated |
| Family binders, loop coordinates, and selected identity wrappers | Preserve the distinction between a family lane, a chunk selector, and an outer loop instance | Prevents a preimage relation from matching the wrong lane or branch | Substitution is spread over value, matrix, public, hash, relation, polynomial, and summary structures |
| Namespaced expression-map cache | Reuse binder substitution and structural maps over shared DAGs | Avoids repeated whole-expression traversal for the same instantiation | Cache namespaces are manually assembled at call sites |
| Factor and monomial interning | Replace repeated product rendering and linear product comparison | Makes polynomial normalization close to linear in the number of terms | Adds fingerprints, collision buckets, canonical-key arrays, and request-local ID threading |
| Changed-term relation work queue | Rewrite only terms changed by a successful relation rule | Avoids repeatedly scanning a full polynomial after every local rewrite | Adds another normalization boundary and relation-processing state machine |
| Representative and bound memo arrays | Evaluate shared expression subtrees once | Keeps repeated endpoint and threshold evaluation linear in arena size | A representative is not defined for every valid nonuniform choice, which is the current failure class |
| Pointwise `IndexedFamily` selection | Select a LUT chunk without materializing all rows | Complexity is proportional to chunk alternatives, not chunk length times alternatives | It requires alpha-renaming lane binders and separately preserving the chunk-selection identity |

The arena, compact choice representation, arrays, memoization, and interning are directly required by
the measured performance problem. The large bug surface comes mainly from operation-specific choice
lifting, summary lifecycle management, and provenance substitution being implemented as separate
features rather than as one abstraction.

## Why the Current Design Produces Late Bugs

The checker currently has three partially overlapping representations of the same semantic value:

1. a concrete flat `OperationalMatrixFact`;
2. an exact or compact `OperationalMatrixExpr` choice;
3. a `SelectedMatrixSummary` attached beside a representative fact.

Each operation decides independently whether to:

- evaluate concrete facts immediately;
- distribute over exact alternatives;
- transform an envelope representative and transfer its summary;
- retain a delayed expression node;
- request a single conservative representative;
- or reject.

These decisions are individually reasonable, but they are distributed through separate add,
multiply, tensor, concat, transform, scale, grouping, family, loop, and endpoint implementations.
Consequently, a new nested shape can be accepted by one layer and rejected much later by another.
The body-418 failure is the clearest example: pointwise family selection correctly preserves two
branches, matrix multiplication correctly keeps the choice unresolved, but a later representative
request assumes those branches have one uniform schema.

The same duplication affects provenance. A family operation must update matrix origins, value
origins, public identities, deterministic-hash identities, relation targets, complete relations,
polynomial factors, selection identities, and summary boundary templates. Missing one map produces
a late relation mismatch rather than an error near the operation that lost the information.

## Recommended Unified Replacement

A simpler replacement should retain the flat-polynomial evaluator and all measured compactness, but
make choice lifting one generic mechanism.

### One typed abstract value

Use one analysis value with the following conceptual shape:

```text
MatrixValue =
  | Concrete(OperationalMatrixFact)
  | Choice(domain, alternatives)

alternatives =
  | Exact(Array<MatrixValue>)
  | Uniform(logical_count, representative, validated_schema)
```

`FamilyValue` should contain a lane binder and one `MatrixValue` template. Selecting a family then
only alpha-renames each branch template to the output lane and creates one ordinary `Choice`.
There should not be a separate fact-level selected-family representation.

### One generic choice-lifting algorithm

Every primitive operation should provide only its concrete transfer function and a declaration of
whether it is relation-sensitive. A single generic lifting algorithm should then enforce:

1. all-concrete inputs: call the existing concrete transfer;
2. choices with the same domain: zip alternatives branch-wise;
3. independent domains: retain a nested choice without Cartesian expansion;
4. a uniform alternative: map the representative only through a registered summary transfer;
5. a relation-sensitive operation: keep exact alternatives until the concrete relation rewrite has
   run, then attempt one uniform join;
6. no valid uniform join: keep the exact choice, never call a general representative function.

Add, multiply, tensor, concat, transform, scale, and grouping should use this same algorithm rather
than each implementing their own selection cases.

### Make summaries derived caches

`SelectedMatrixSummary` should be a cache derived from an alternative set, not an independently
mutable semantic object. A transformed summary should be produced only by one central operation
registry. Any unregistered operation invalidates the cache and retains the exact choice. This makes
stale-summary acceptance impossible by construction and removes scattered summary repair code.

### Eliminate the general representative escape hatch

Replace `evaluateOperationalExprRepresentative` with two explicit APIs:

```text
tryUniformRepresentative(choice) -> Option<(fact, validated_schema)>
evaluateCompleteBound(value) -> bound
```

The first must return `none` for nonuniform exact choices. The second can recurse through choices and
take complete branch maxima without inventing a representative. A primitive that needs polynomial
structure must use generic choice lifting, not the bound-only API. This separation would have made
the body-418 path impossible.

### Centralize provenance mapping

Define one structural provenance map that traverses every identity-bearing field. Binder
substitution, loop instantiation, protocol-family selection, and dynamic selection should be four
instances of that map. Summary boundary metadata should be derived after the map rather than mapped
independently. This removes the current family of near-duplicate substitution functions.

### Preserve the performance-critical pieces

The replacement should retain:

- request-local append-only expression IDs;
- array-indexed memoization;
- exact selections without Cartesian expansion;
- validated uniform envelopes for large LUT families;
- factor and monomial interning with full equality after fingerprint lookup;
- changed-term relation rewriting;
- one abstract parallel-loop body rather than lane materialization.

The goal is not to return to eager SOP expansion. It is to make all compact selection behavior pass
through one typed lifting engine, so the optimization does not require a new hand-written case for
every later protocol shape.

## Suggested Handoff Order

1. Freeze the current body-418 failure as a small fixture that reproduces a nested nonuniform choice
   under relation-sensitive multiplication.
2. Introduce generic choice lifting beside the current paths and compare it against existing focused
   fixtures; do not add another protocol-specific rewrite.
3. Route multiplication first, because it consumes decomposition and preimage relations.
4. Route add/subtract and structural transforms next.
5. Replace general representative calls with `tryUniformRepresentative` and complete-bound folding.
6. Remove the superseded operation-specific selection branches in the same change; do not retain a
   permanent dual path.
7. Rerun the Tall diagnostic after each migrated operation and record arena size, evaluation time,
   relation rewrite count, and first unsupported scope.

## Audit Scope

The relevant review surface is:

- `lean/Mxx/Certificate/OperationalBounds.lean` for operational facts, the expression arena,
  selection evaluation, relation rewriting, loops, endpoint bounds, and fixtures;
- `lean/Mxx/Ir/BinaryFormat.lean` for array-based Graph IR decoding;
- generated correctness modules whose source hashes depend on these Lean files;
- `crates/gadgets/tests/test_gpu_tall_bgg_nested_rns_modq_arith.rs` for the eventual Tall
  parameter-simulation and GPU runtime validation.

The Rust/CUDA runtime protocol and proof-side symbolic derivation are outside the expression arena's
implementation boundary.
