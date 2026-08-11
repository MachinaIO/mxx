# Tall Operational Selection Checker Migration Status

## Purpose

This document explains the approved architecture for compact dynamic selections in the Tall BGG+
operational noise checker and records the current implementation status. It is intended for readers
who are new to the checker.

The implementation is primarily in `lean/Mxx/Certificate/OperationalBounds.lean`. Binary Graph IR
decoding is in `lean/Mxx/Ir/BinaryFormat.lean`. The selection representation described here is
request-local Lean analysis state. It is not serialized into Graph IR and does not change the Rust
or CUDA execution graph.

The migration is in progress. The target architecture below is approved, but it must not be read as
a claim that the implementation or Tall end-to-end validation is complete.

## Why Compact Selections Are Required

The small Tall diagnostic configuration uses ring dimension 8, CRT depth 1, 10-bit CRT moduli,
5-bit gadget base, scale 1024, and one nested-RNS multiplication. Its generated protocol contains
31 lookup tables, 30,720 lookup preimages, and 48 slot-operation preimages.

Materializing every logical branch at every downstream operation is therefore too expensive.
Flattening independent selections into Cartesian alternatives is even worse. At the same time, the
checker must preserve the branch identities required by decomposition and preimage relations. A
single scalar maximum is not a sufficient replacement for symbolic structure while relations remain
to be consumed.

The approved design represents a selection either exactly or by a validated shared envelope, and
uses one generic lifting engine for all matrix primitives.

## Approved Analysis Value

Every matrix value is an ID in one request-local append-only arena:

```text
MatrixValue := OperationalExprId

OperationalExprNode :=
  | Concrete  OperationalMatrixFact
  | Primitive PrimitiveOperation (Array OperationalExprId)
  | Choice    SelectionDomainId ChoiceStorage

ChoiceStorage :=
  | Exact  (Array OperationalExprId)
  | Shared OperationalExprId ValidatedSchemaId
```

Each arena node has one checked matrix type and an analysis-only `containsChoice` bit. An
`OperationalExprId` is only an array index for sharing and memoization. It is not serialized, is not
part of matrix or relation identity, and is not evidence of symbolic equality.

Ordinary operations whose inputs are concrete continue to use the existing flat-polynomial
operational formulas. `Primitive` nodes delay an existing concrete transfer when unresolved choices
prevent immediate evaluation. They do not introduce protocol-specific bound formulas.

## Interned Selection Domains

One `Choice` represents exactly one mutually exclusive domain. Independent domains remain nested.
The request-local `SelectionDomainId` interner uses the full key:

```text
(selection kind, selection identity, canonical branch count)
```

Fingerprints select candidate buckets only; full keys are compared before an interned ID is reused.
After interning, domain comparison is constant time. Loop-lane and protocol-selection domains are
different kinds and are never positionally zipped.

The domain owns the only canonical branch count. `Exact` construction checks once that the stored
array length equals that count. `Shared` construction checks that its representative and schema
describe the same domain. Downstream code must not maintain or revalidate a second count.

## Exact and Shared Choices

### Exact

`Exact` stores every distinct branch expression. It preserves branch-local value identities,
public identities, polynomials, and relations. It is used when branches are genuinely nonuniform or
when branch-local relations still need to be consumed.

Operations over an Exact domain may visit its stored branches once. If every branch is the same
expression ID, construction reduces the choice to that expression. Equal bounds or equal schemas
are not enough for this identity reduction.

After a producing operation and its relation rewrites finish, the checker performs one canonical
all-branch join. A successful join immediately converts the value to `Shared` and discards the Exact
array. A failed join retains `Exact`; it does not create a misleading representative.

### Shared

`Shared` stores one representative expression and one validated schema. The representative need not
be concrete and may contain a nested independent choice. It describes the structure common across
the outer domain, but it does not assert that all logical matrices have the same value or identity.

The `ValidatedSchema` owns the conservative join over the complete outer domain. It records all
bounds and metadata later consumers require, including branch-maximum total and noise bounds,
signal structure, constant-polynomial and zero-row metadata, canonical range, public and relation
boundaries, and selection provenance.

The meaning of a Shared value is always the pair `(representative, schema)`. No consumer may treat
the representative alone as the complete value. In particular, complete-bound evaluation:

1. evaluates any nested choices inside the representative once; and
2. applies the outer-domain envelope stored in the schema.

This preserves the outer maximum without traversing the outer logical branch count. There is no
separate mutable summary node and no fallback that reconstructs Exact branches from Shared.

A direct uniform family template must construct Shared in time proportional to the template and
schema sizes, independently of the logical branch count. It must not allocate a count-sized array or
instantiate each lane.

## Generic Primitive Lifting

All matrix operations use one storage-aware lifting engine. Primitives contribute their existing
concrete transfer behavior and a transfer-class declaration; they do not implement separate
selection strategies.

The lifting rules are:

1. All concrete operands invoke the existing concrete transfer.
2. Same-domain Exact operands are zipped branch-wise and joined once after the operation.
3. Same-domain Shared operands transform their representatives and schemas without logical branch
   traversal when the transfer supports it.
4. Same-domain Exact and Shared operands visit only the stored Exact branches. Operand order is
   preserved.
5. An ordinary primitive over one Shared argument creates one delayed Primitive over the
   representative and transfers the schema.
6. A relation-free operation over independent domains remains a nested delayed Primitive only when
   its registered transfer class supports composition.
7. Relation-consuming multiplication zips the domain named by the relation requirement. Other
   independent domains remain nested in coefficients.
8. Incompatible relation domains fail closed with an error that names both domains and the scope.

N-ary operations follow the same rules. The engine chooses the first immediate domain in argument
order, aligns operands belonging to that domain, and leaves other domains nested. It never creates
or visits a Cartesian product.

For relation-consuming Shared operands, matching domain IDs alone are insufficient. The checker
must also prove that:

- the public operand and preimage-relation boundaries correspond;
- relation producer and target identities match;
- both schemas use the same branch parameterization; and
- the representative rewrite is valid for every logical branch.

Failure of any condition rejects the operation.

## Closed Transfer Registry

Every `PrimitiveOperation` transfer class has exactly one row in a closed registry:

```text
CompositionalTransfer :=
  | Supported existingReviewedTransfer
  | RequiresConcreteStructure
```

A constructor with semantically different variants, such as ordinary and relation-consuming matrix
multiplication, contributes one row per transfer class. A build-time inventory must reject missing
or duplicate rows.

`Supported` reuses only an existing reviewed concrete/compositional transfer. The registry must not
derive a new bound formula from an operation name or a broad classification such as monotonicity.

`RequiresConcreteStructure` has one deterministic lifecycle:

1. while unresolved choices remain, construction stores a delayed Primitive;
2. when matching-domain lifting makes the required structure concrete, it invokes the existing
   concrete transfer; and
3. if an endpoint bound is requested while independent unresolved domains remain, evaluation fails
   closed with a distinct error.

It never substitutes a scalar child-bound approximation.

## Relation and Provenance Handling

`relationRequirement` classifies each value as having no relation, one uniform validated relation
schema, one branch-local relation domain, or an unknown relation. Unknown requirements fail closed.
The result is memoized per expression ID and parameter environment.

Decomposition and preimage rewrites remain noncommutative and identity-sensitive. Existing checks
for matrix types, moduli, factor order, producer identity, public-matrix identity, relation target,
selection domain, and branch parameterization remain mandatory.

One central provenance transformation maps every identity-bearing field for loop instantiation,
binder rebinding, protocol-family selection, and dynamic selection. The existing namespaced
transformation cache remains request-local and must not conflate different instantiation
environments.

Relation rewriting retains the changed-term work queue and its 64-step per-chain fail-closed limit.
Only terms changed by a successful rewrite are processed again, and completed terms are normalized
together at the boundary.

## Bound Evaluation and Memoization

`evaluateCompleteBound` takes the maximum of complete Exact branch bounds. It never maximizes
partial terms from different branches and then combines those unrelated maxima.

Request-local arrays indexed by `OperationalExprId` memoize relation requirements, schemas, total
bounds, and noise bounds. Schema derivation is lazy and occurs only when a join, Shared construction,
or representative query requires it. The first query may visit each reachable expression once;
repeated queries are constant time.

Decoder validation uses the strict inequality:

```text
2 * plaintext_modulus * noise_bound < ciphertext_modulus
```

The multiplication form avoids integer-division boundary changes.

## Fail-Closed Contract

Unsupported semantics are analysis errors, not invitations to guess a bound. The checker rejects:

- invalid expression references, matrix-type mismatches, and invalid choice cardinalities;
- missing or ambiguous relations and public-identity mismatches;
- unsupported schema transfers;
- relation-consuming operations whose correlation conditions cannot be proved;
- incompatible or unknown selection domains;
- concrete-structure-dependent primitives that still have unresolved independent domains at an
  endpoint; and
- selection-dependent endpoint operations whose complete branch conditions cannot be established.

No failure path may discard identity or relation metadata, invent branch uniformity from one
representative, synthesize Exact alternatives from Shared, or introduce a handwritten
protocol-specific noise formula.

## Performance Contract

The migration must preserve the following asymptotic behavior:

| Operation | Required cost |
| --- | --- |
| Uniform template to Shared | Independent of logical branch count |
| Packed Exact join | Linear once in stored branches |
| Same-domain Exact lifting | Linear in stored branches |
| Shared with Shared | Independent of logical branch count |
| Exact with Shared | Linear only in stored Exact branches |
| Single-Shared primitive lift | Linear in arity and schema transfer size |
| N-ary lift with at least one matching Exact operand | Linear in branch count times arity |
| Shared outer domain with nested Exact inner domain | Linear only in stored inner branches |
| Interned domain comparison and `containsChoice` | Constant time |
| Repeated memoized query | Constant time after first evaluation |
| Independent-domain Cartesian arrays or visits | Never performed |

Shared processing must not use a logical-count range, inspect unavailable logical alternatives, or
recover a count-sized Exact representation. Performance instrumentation must keep
`cartesianPairVisits` at zero.

## Migration Status

The current worktree is between the previous operation-specific implementation and the approved
architecture. The following table is deliberately conservative: a feature is not marked complete
until its implementation and focused validation are both present.

| Area | Current status |
| --- | --- |
| Request-local expression arena and concrete/expression values | Present from the earlier implementation |
| Interned selection-domain identity | Implemented with collision-safe full-key comparison; focused cardinality fixtures remain pending |
| `Exact` and `Shared` storage names | Partially implemented; Shared still carries the older `SelectedMatrixSummary` representation |
| One canonical domain-owned count | Implemented; `SelectionDomainId` is the sole owner |
| One lossless n-ary `PrimitiveOperation` node | Not yet implemented; operation-specific expression constructors remain |
| Schema-owned outer envelope through `ValidatedSchemaId` | Not yet implemented; the older summary object remains |
| Construction-time Exact-to-Shared join | Not yet complete |
| Generic lifting rules for all matrix primitives | Not yet implemented; operation-specific branches remain |
| Closed transfer-class registry and completeness inventory | Not yet implemented |
| Deterministic `RequiresConcreteStructure` lifecycle | Not yet implemented |
| Memoized structural `relationRequirement` | Not yet implemented |
| Lazy schema and complete-bound memo arrays | Partially present only in older bound/representative forms |
| Deletion of the general representative API and old selection paths | Not yet performed |
| Body-418 success fixture | Pending |
| Focused correctness and complexity fixtures | Pending migration to the approved representation |
| `lake build Mxx` after the complete migration | Not yet validated |
| Tall diagnostic after migration | Not yet run |

The last auditable checkpoint before this migration substantially reduced the original expression
growth: the producer phase decreased from approximately eight minutes and 337,000 expression nodes
to an encoding-stage checkpoint with 102,717 nodes, with the first later failure reached in about
143 seconds. These observations are development measurements, not controlled benchmarks. The
approved migration must keep the equivalent-phase expression count at or below 113,000 and must
perform zero logical traversal of the 30,720-branch uniform family.

At that checkpoint the first failure was:

```text
OperationalError.inScope
  (parallelBody (root (workflowStage "encoding")) 418)
  (unsupportedOperationalExpr 0)
```

The failure is the migration's primary frozen acceptance case. It arises when relation-sensitive
matrix multiplication receives a valid nonuniform nested choice but a later path requests a single
uniform representative. Under the approved design, generic branch-wise relation consumption keeps
the Exact distinction, joins only after the rewrite, and computes the final bound as the maximum of
complete branch bounds.

## Remaining Acceptance Work

Before the migration can be declared complete, it must provide evidence for all of the following:

1. body 418 succeeds with its hand-checked branch-wise relation bound;
2. every primitive transfer class has exactly one registry row;
3. positive and negative Shared relation-correlation fixtures behave as specified;
4. nested Shared/Exact representatives preserve the schema-owned outer maximum without traversing
   the outer logical domain;
5. `RequiresConcreteStructure` succeeds after matching-domain resolution and fails distinctly at an
   unresolved endpoint;
6. uniform counts 2, 1,024, and 30,720 have zero logical branch visits;
7. Exact counts 8, 32, and 65 exhibit linear stored-branch work;
8. independent selections never allocate or visit Cartesian alternatives;
9. repeated relation, schema, and bound queries hit their request-local memos;
10. obsolete operation-specific selection branches, old family representations, summary-repair
    paths, representative escape hatches, and test oracles have been deleted;
11. `lake build Mxx` and all focused operational fixtures pass without new axioms, `sorry`, `admit`,
    or `native_decide`; and
12. the Tall CRT-depth-1 diagnostic completes under its 30-minute wall limit and reports the required
    structural and cache metrics.

Successful parameter search, agreement between the simulated threshold and GPU residual, and Tall
end-to-end runtime correctness and performance remain later end-to-end requirements. None is
currently claimed by this document.

## Audit Surface

The primary review surface is:

- `lean/Mxx/Certificate/OperationalBounds.lean` for operational facts, arena values, lifting,
  relations, loops, bounds, and fixtures;
- `lean/Mxx/Ir/BinaryFormat.lean` for Graph IR decoding;
- generated correctness modules whose source hashes depend on these Lean sources; and
- `crates/gadgets/tests/test_gpu_tall_bgg_nested_rns_modq_arith.rs` for the eventual Tall parameter
  simulation and GPU runtime validation.

The Rust/CUDA protocol executor and proof-side symbolic derivation remain outside the operational
selection arena's implementation boundary.
