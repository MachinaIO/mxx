# Indexed Family Semantics for Operational Noise

## 1. Status and authority

This document is the implementation specification for runtime indices, indexed families,
family-valued preimage relations, finite selection, and fixed-count sequential loops in the Rust
operational-noise checker.

For those subjects, this document supersedes the Switch-centric rules in
`docs/correctness/egg-free-operational-noise-normal-form-spec.ja.md`, including its selector
terminology, Switch scope-minimization rules, family section, `Select`/`FamilyGet` transfer rows,
Stage 4, and the corresponding Switch acceptance gates. The older document remains authoritative
for matrix `PolynomialNF`, exact factor ordering, central scalars, numeric bounds, final Large
rejection, and the strict acceptance inequality.

This is a replacement design, not a compatibility layer. After the cutover, the checker must not
retain two semantic authorities for selection or family relations. Historical experiments remain
only in `docs/correctness/exact-signal-large-debugging-history.md`.

## 2. Goals

The implementation must:

1. Treat graph template parameters as concrete for one noise-simulation job.
2. Represent runtime integer indices with a small, typed, job-local SSA identity.
3. Distinguish index equality from family-element value equality.
4. Treat a parallel-loop family as a value over an exact finite integer domain.
5. Register preimage relations as universally quantified family relations.
6. Apply such a relation at any validated runtime index without equating the preprocessing binder
   with the runtime input.
7. Replace dynamic `Select` semantics with indexed lookup semantics.
8. Fully unroll fixed, small sequential loops during lowering.
9. Preserve family sharing and never enumerate a uniform family's logical lanes during checking.
10. Never form a Cartesian product of independent runtime indices.
11. Remain fail-closed when identity, range, type, layout, trapdoor, or relation evidence is
    incomplete.

The implementation must not use protocol names, Tall node numbers, fixture constants, debug text,
numeric bounds, or insertion-local `TermId`/`ScalarId` values as semantic identities.

## 3. Assumptions and trust boundary

### 3.1 Parameter specialization

The DSL may continue to contain `IntExpr::Var` template parameters. Before indexed-family lowering,
one complete `ParamEnv` must resolve every parameter that affects:

- loop counts;
- family domains;
- matrix dimensions and ring types;
- index-map fixed parameters;
- sampler and bound contracts.

An unresolved parameter is a typed error. Every parameter candidate runs a separate lowering and
normalization job, or uses a cache key containing the complete resolved `ParamEnv`. No indexed
identity may be reused across different parameter environments.

### 3.2 Small sequential counts

Sequential-loop counts are assumed to be small after specialization. The checker therefore uses
literal unrolling, not exponentiation, recurrence solving, widening, or a sequential-loop normal
form. Resource failure remains fail-closed, but no semantic shortcut may be introduced to avoid the
unroll.

### 3.3 Trusted index-map range

The first implementation permits a DSL index map to declare its output range without proving that
the Rust implementation always returns a value in that range. This declaration is part of the
trusted DSL contract. Consequently, checker soundness is conditional on that declaration being
correct.

The contract must be named `TrustedIndexRange` in code and diagnostics. It must not be presented as
an inferred or verified fact. Later validation or certificates may reduce this trust boundary, but
the first implementation must not silently sample the map or infer a tighter range.

## 4. Core distinctions

The implementation must keep the following concepts separate.

| Concept | Meaning | Used for equality? |
| --- | --- | --- |
| `FamilyDomain` | Exact integer keys owned by a family | Yes, for compatible pointwise operations |
| `TrustedIndexRange` | Conservative/trusted possible values of one runtime index | No; range checks only |
| `IndexValueId` | Complete pure SSA computation of one runtime index | Yes, for correlated lookup |
| `FamilyValueId` | Semantic identity of the complete family value | Yes |
| `FamilyElementId` | One family value at one runtime index | Yes |
| matrix/scalar value identity | Semantic contents of an element | Yes |
| local loop-binder owner | Which loop syntax introduced an alpha-renamable template variable | No |
| `ValueDefinitionId` | Which deterministic SSA definition produced a value | Yes |
| `SampleEventId` | Which concrete random or sampler invocation produced a value | Yes |
| occurrence diagnostics | Source location attached after semantic identity is known | No |

Equal domains permit pointwise computation. They do not prove that two families contain equal
values. Equal index ranges do not prove that two runtime indices choose the same element.

For example, with `domain(F) = domain(G) = [0, 8)`:

```text
F[x] and G[x] use the same runtime index but are normally different values.
F[x] and F[y] are normally different values even if range(x) = range(y).
F[x] and F[x] denote the same family element.
```

## 5. Exact domains and ranges

All domains use half-open intervals.

```rust
pub struct FamilyDomain {
    pub minimum: u64,
    pub maximum_exclusive: u64,
}

pub struct TrustedIndexRange {
    pub minimum: u64,
    pub maximum_exclusive: u64,
}
```

Both constructors must reject `minimum > maximum_exclusive`. A family domain must be nonempty
unless the Graph IR operation explicitly permits an empty family; the initial implementation does
not permit empty operational families.

Dynamic access is valid exactly when:

```text
family.minimum <= index.minimum
and index.maximum_exclusive <= family.maximum_exclusive
```

Range containment does not alter identity. One runtime index identity has one canonical trusted
range declaration; repeating the exact declaration is allowed and any different declaration is a
contract error. Range declarations are finalized before a family-access consumer runs, and adding a
runtime range declaration after finalization is rejected. This removes traversal-order dependence
without adding range facts to semantic identity.

## 6. Runtime index identity

### 6.1 Closed runtime values

`IndexValueId` is a copyable handle into one job-local, structurally hash-consed arena.

```rust
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct IndexValueId(u32);

pub enum IndexValueNode {
    Constant(u64),

    RuntimeScalar {
        value: ScalarIdentityHandle,
    },

    Apply {
        map: IndexMapInstanceId,
        inputs: Box<[IndexValueId]>,
    },

    FamilyScalarGet {
        family: FamilyValueId,
        index: IndexValueId,
    },

    ExtractCoefficient {
        matrix: MatrixIdentityHandle,
        position: u64,
    },
}
```

The ID identifies the complete expression, not merely its ultimate input. Thus `x / 8` and `x % 8`
have different IDs even though both depend on `x`.

`RuntimeScalar.value` must be a semantic scalar SSA identity. It must distinguish two independent
runtime inputs with identical names, sorts, and ranges. It must not use a display name or a raw
`ScalarId` as the key.

Removing loop-owner provenance does not remove value-producing occurrence identity. Deterministic
SSA leaves use a stable `ValueDefinitionId`. Random and sampler leaves use a stable `SampleEventId`
that distinguishes concrete invocations of the same reusable definition with the same parameters.
These IDs may be derived from a resolved graph occurrence, but they are semantic event identities,
not alpha-renamable loop-binder identities. They remain part of scalar, matrix, and family source
identity.

`TrustedIndexRange` is stored in facts beside the interned node and is excluded from
`IndexValueNode` equality.

### 6.2 Template-bound indices

A preprocessing or parallel-loop binder is not a runtime `IndexValueId`. It is a locally bound
variable in a family template.

The implementation uses one shallow `TemplateIndexArena` per family template. Its `TemplateIndexId`
handles are descriptor-local, so `Bound(0)` in two independent templates cannot share facts. The
arena is tied to the job's closed `IndexValueArena`; range facts are stored in a parallel table and
are not part of the node key.

```rust
pub struct TemplateIndexId(u32);

pub struct TemplateIndexArena {
    // one descriptor-local shallow node/facts arena
}

pub enum TemplateIndexNode {
    Bound(u32),
    Closed(IndexValueId),
    Apply {
        map: IndexMapInstanceId,
        inputs: Box<[TemplateIndexId]>,
    },
}

pub struct TemplateIndexFacts {
    pub range: TrustedIndexRange,
}
```

`Bound(0)` denotes the primary family index. If nested family templates are added later, larger
de-Bruijn indices denote outer binders. The initial cutover supports one primary bound index per
family value; a family body that would let an unresolved outer binder escape is rejected. Tall and
Diamond acceptance paths must be expressed using flat families plus fixed sequential steps, so they
do not require escaping nested binders.

Family-template equality is alpha-equivalence of this arena DAG. Loop node IDs, occurrence paths,
port numbers, and binder names are not part of the template identity.

At runtime access, instantiation substitutes `Bound(0) := Closed(index)`. A closed matrix/scalar DAG
must contain no unresolved `Bound` node.

### 6.2.1 Proof-only family-root binder

The implementation has one additional index node, `UniversalFamilyRoot(F)`, solely for checking
that a post-finalization family-root expression has the declared domain. It is not a second runtime
binder and is not runtime-evaluable by the executor or by ordinary lookup consumers. A scoped
`ProofRootIndex` capability may, after finalization, close symbolic Packed/Get/Template/Pointwise
values and specialize a family relation at that exact same runtime `X`; this is the only permitted
relation use. The node is created only after the ordinary family/value lowering has finalized its
`FamilyValueId` and domain; its trusted range is copied from that exact domain. A root whose family
or domain is not exactly the finalized descriptor is rejected.

All nested selectors and derived indices remain ordinary `IndexValueId` expressions derived from
the same closed runtime input. No root capability may be passed to the executor, an arbitrary
consumer, or ordinary `get_dynamic`; it may not escape its scoped close operation, enumerate a
domain, or establish equality across families or independently computed indices. This is the sole
root-binder exception in the checker.

Template bodies also need a typed open value representation. This is required for a `Select` or
family get whose selector or selected value depends on `Bound(0)`.

```rust
pub enum TemplateFamilyExpr {
    Closed(FamilyValueId),
    Packed {
        domain: FamilyDomain,
        elements: Box<[ResolvedTemplateValueIdentity]>,
        element_type: ResolvedValueType,
    },
}

pub enum ResolvedTemplateValueNode {
    Closed(ResolvedValueIdentity),
    Get {
        family: TemplateFamilyExpr,
        index: TemplateIndexId,
    },
    Operation {
        operation: TypedValueOperation,
        inputs: Box<[ResolvedTemplateValueIdentity]>,
    },
}
```

These descriptors are structurally hash-consed. `TemplateFamilyExpr::Packed` is the open form of a
finite family whose elements can depend on the local binder; it is not a materialized logical loop
domain. Instantiation recursively closes the descriptor in this order:

1. replace `Bound(0)` with the supplied closed index and its trusted range;
2. close map inputs and intern the resulting closed `IndexValueId`;
3. close explicit packed elements and intern the ordinary `Packed` family;
4. close `Get` to the ordinary `FamilyElementId`;
5. close typed operations through the existing scalar or matrix constructor.

Instantiation is linear in the open descriptor and the number of explicitly packed alternatives.
It never enumerates a template's logical family domain. A substituted index whose declared range is
not valid for its target family fails with the ordinary range-contract error.

### 6.3 Index maps

An index map is a pure, deterministic, total function from one or more `u64` values to one `u64`.

```rust
pub struct IndexMapDefinitionId(u32);

pub struct IndexMapInstanceKey {
    pub definition: IndexMapDefinitionId,
    pub resolved_parameters: Box<[u64]>,
}

pub struct IndexMapDefinition {
    pub stable_name: &'static str,
    pub arity: usize,
    pub evaluate: fn(&[u64], &[u64]) -> Result<u64, IndexMapError>,
}
```

The function pointer is not an identity. `IndexMapDefinitionId` comes from one checked registry;
`IndexMapInstanceId` is interned by definition ID plus resolved parameters. A map must not capture
mutable state, randomness, time, or ambient configuration. Returned errors and arity mismatches are
typed errors. An unwinding panic is an implementation bug and must not be used to report invalid
input.

Two independently registered maps remain different even if they happen to be extensionally equal.
This can cause a safe false negative, never a false positive. Built-in canonical maps such as
addition, multiplication, quotient, remainder, and affine flattening should share registered
definitions.

## 7. Family values

### 7.1 Representation

`FamilyValueId` is a copyable job-local handle into a structurally hash-consed family arena.

```rust
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FamilyValueId(u32);

pub enum FamilyValueNode {
    Source {
        source: FamilySourceIdentity,
        domain: FamilyDomain,
        element_type: ResolvedValueType,
    },

    Packed {
        domain: FamilyDomain,
        elements: Box<[ResolvedValueIdentity]>,
        element_type: ResolvedValueType,
    },

    Template {
        domain: FamilyDomain,
        body: ResolvedTemplateValueIdentity,
        element_type: ResolvedValueType,
    },

    Pointwise {
        domain: FamilyDomain,
        operation: PointwiseOperation,
        inputs: Box<[FamilyValueId]>,
        element_type: ResolvedValueType,
    },
}
```

`FamilySourceIdentity` is the semantic protocol/artifact source identity. It must remain stable
across the preprocessing/evaluation binding that denotes the same artifact family, and it must
distinguish independent random/sampler families. Deterministic sources retain their
`ValueDefinitionId`; sampler-produced sources retain their `SampleEventId`. Two concrete sampler
invocations of the same reusable definition with the same parameters remain different values. A
raw graph occurrence path is not itself the family identity, but may be used to derive the stable
definition or event identity before artifact binding is resolved.

`ResolvedValueIdentity` and `ResolvedTemplateValueIdentity` are typed structural identities. They
include semantic matrix/scalar sources, operations, and ordered children; they exclude numeric
bounds and range facts.

### 7.2 Packed family

`Packed([A, B, C])` is an explicitly stored family with domain `[0, 3)` and ordered element
identities `[A, B, C]`. Equal length alone does not imply equal family identity.

A constant get returns the corresponding element. A dynamic get remains one `FamilyGet` value for
exact identity. Its numeric bound is the maximum over only the stored elements reachable from the
validated range. Since `Packed` is explicit storage, this work is linear in actual stored elements,
not in an implicit logical domain.

### 7.3 Template family

A parallel loop produces:

```text
Template(domain = D, body = lambda i. body(i))
```

The body is lowered once with `TemplateIndexExpr::Bound(0)`. No logical lane is materialized.

### 7.4 Pointwise family

An operation on families is valid when all input domains and element types satisfy the operation's
normal type rules. Equal domains alpha-align only their local `Bound(0)` variables. They do not
merge the input `FamilyValueId`s or their element identities.

Canonical lookup distributes through a pointwise family:

```text
Get(Pointwise(op, [F, G]), x)
  = op(Get(F, x), Get(G, x))
```

This direction exposes ordinary matrix algebra and registered relations. The inverse collection
rule is not part of canonicalization.

### 7.5 Reindex

`Reindex(F, D, f)` means the derived family `lambda i in D. F[f(i)]`.

The initial implementation must not add a persistent `Reindex` variant. Lower it directly to a
`Template` whose body is a source-family get with the mapped `TemplateIndexExpr`. The map's trusted
range must be contained in `domain(F)`.

Thus reindexing preserves both the source `FamilyValueId` and the complete map identity without
adding a second semantic mechanism.

### 7.6 Family element identity

One dynamic element is identified by both family and index:

```rust
pub struct FamilyElementId {
    pub family: FamilyValueId,
    pub index: IndexValueId,
}
```

`F[x]` and `G[x]` remain different unless `F` and `G` have the same `FamilyValueId`. `F[x]` and
`F[y]` remain different unless the complete `IndexValueId`s are equal or an independent exact index
normalizer proves them equal.

The first implementation uses only structural `IndexValueId` equality. Identities such as `x + 0 =
x` may therefore be missed unless the built-in map constructor already canonicalizes them. Such
misses are safe false negatives.

### 7.7 Artifact family aliasing

A resolved workflow `ArtifactBinding` is an identity alias, not a copy and not a new source. Before
any family get or relation lookup, lowering follows the binding and reuses the producer's exact
`FamilyValueId` for the consumer input. The consumer must not intern a separate `Source` family for
that artifact.

An external or unresolved source with no binding receives its own source identity and cannot inherit
relations registered for a producer family merely because its type, domain, bounds, or display name
match. Missing or ambiguous bindings fail closed. Binding resolution applies recursively through
tuples and bundles, but never merges two values by shape or numeric facts alone.

## 8. Dynamic access and finite selection

### 8.1 Dynamic get

`FamilyGetDynamic(F, x)` performs these checks in order:

1. `x` is an integer `IndexValueId` with a `TrustedIndexRange`.
2. The range is contained in `domain(F)`.
3. The family element type matches the consumer's expected type.
4. Construct or reuse `FamilyElementId { family: F, index: x }`.

It must not enumerate a `Template` family and must not inspect unrelated family domains.

### 8.2 Static get

`FamilyGetStatic(F, k)` requires `k in domain(F)`. For `Packed`, it returns the stored element. For a
`Template`, it substitutes `Bound(0) := Constant(k)` once. Static get does not create a runtime
selector identity.

### 8.3 Value selection

The DSL operation `Select(selector, [A, B, ...])` lowers as:

```text
Get(Packed([A, B, ...]), selector)
```

This applies to Int, Bool, Real, Bytes, matrix, and other supported closed value sorts. Matrix NF
does not create a `SwitchData`/Switch-barrier identity for this operation.

If all reachable packed alternatives are bounded-only, the bound is their maximum. If exact/Large
content remains, the lookup remains an exact `FamilyElementId` until it cancels or a family relation
consumes it. Equal numeric bounds never identify alternatives.

### 8.4 Family selection

Selecting one of several equal-domain families lowers to a pointwise template:

```text
SelectFamily(s, [F0, F1, ...])
  = Template(D, lambda i. Get(Packed([Get(F0, i), Get(F1, i), ...]), s))
```

The body uses the open template-get representation from Section 6.2: each `Get(Fk, i)` and a selector
derived from `i` remain typed binder-open values until template instantiation. The selector is closed
with respect to the family's local `i` unless the DSL explicitly constructs an index map from `i`.
Different independent selectors remain nested opaque lookups. The checker never enumerates their
Cartesian product.

## 9. Family preimage relations

### 9.1 Registration model

A `PreimageSample` lowered inside a template family registers a universal relation, not one relation
whose key contains the preprocessing binder owner.

```rust
pub struct FamilyRelationRegistration {
    pub domain: FamilyDomain,
    pub public: ResolvedTemplateMatrixIdentity,
    pub preimage: FamilyValueId,
    pub target: ResolvedTemplateMatrixIdentity,
    pub matrix_type: ConcreteMatrixType,
    pub layout: Option<ResolvedLayoutIdentity>,
    pub trapdoor: TrapdoorSourceIdentity,
    pub source: SamplerSourceIdentity,
}
```

Its meaning is:

```text
for every i in domain:
    public(i) * preimage[i] = target(i)
```

The template identities use `Bound(0)`. The registration key includes every value-affecting source,
coordinate expression, type, layout, and trapdoor contract, but it excludes the temporary loop
owner, numeric bound, and debug name.

A family relation may be registered only when one uniform parallel-loop body produces the preimage
family. Conditional sampling, a body that returns unrelated samplers, an escaping outer binder, or
nonuniform relation contracts fail closed in the initial implementation.

Non-family `PreimageSample` continues to register the ordinary exact relation for one concrete
matrix value.

### 9.2 Runtime application

Given an exact product containing adjacent factors equivalent to:

```text
public(x) * Get(preimage_family, x)
```

the matcher:

1. Looks up registrations by `preimage_family` and static relation contract.
2. Requires `range(x)` to be contained in the registration domain.
3. Instantiates `public(Bound(0))` with `Bound(0) := x`.
4. Requires that instantiated public identity to equal the actual adjacent public factor.
5. Requires exact type, layout, trapdoor, and source contract equality.
6. Requires a unique target registration.
7. Instantiates `target(Bound(0))` with the same `x`.
8. Rewrites only in the registered direction.

The preprocessing binder and runtime `x` are intentionally different objects. No equality between
their roots is tested. The universal relation is specialized at `x`.

The following are therefore distinct:

```text
public(x) * K[x]   // may match
public(x) * K[y]   // does not match when x != y
public(x) * L[x]   // does not match a K-family relation
other(x)  * K[x]   // does not match when public identities differ
```

Range equality alone never enables a relation.

### 9.3 Ambiguity and termination

Duplicate registrations with identical complete contracts and target identities deduplicate.
Multiple distinct targets for the same complete instantiated left-hand side produce a typed
`AmbiguousFamilyRelation` error.

Family relation application uses the same one-directional relation dependency/cycle discipline as
ordinary matrix relations. The matcher must not enumerate all runtime indices to prove termination.

### 9.4 Tall logical flattening

Tall's checker-visible artifact must be one logical family over the table index:

```text
Low  : Family[0, table_length)
High : Family[0, table_length)
```

The runtime may retain chunked physical storage, but `chunk = x / C` and `offset = x % C` are a
storage access plan, not two independent semantic selectors. The DSL/runtime boundary must expose a
logical get `Low[x]`/`High[x]` to the checker, together with validated chunk-layout metadata used by
execution.

The checker must not reconstruct `x` from independently compared chunk and offset roots, and it must
not model the chunk choice as a nested Switch. This DSL adjustment is required for the initial
simple implementation.

## 10. Fixed-count sequential loops

### 10.1 Semantics

After parameter specialization, a sequential loop has an exact `u64` count and the semantics:

```text
state_0 = initial
state_(step + 1) = body(step, state_step, invariants)
result = state_count
```

All carried components are replaced simultaneously after each body instantiation.

### 10.2 Lowering algorithm

The lowerer implements the semantics literally:

```rust
let count = resolve_exact_u64(spec.count)?;
let mut state = lower_initial_values()?;

for step in 0..count {
    let next = lower_body_once(
        step,
        &state,
        &invariants,
    )?;
    require_same_carried_schema(&state, &next)?;
    state = next;
}

return state[requested_port].clone();
```

`lower_body_once` substitutes the iteration index with an ordinary exact scalar integer constant and
maps all state placeholders to the previous step's values. Only when that scalar is consumed as a
family index does the index consumer intern `IndexValueNode::Constant(step)`. It accepts every
already-supported carried value category, including matrix families; it does not restrict the loop
to scalar matrices.

The resulting scalar/matrix/family DAG contains only ordinary nodes. No
`TermSequentialRecurrence`, sequential-state exact factor, recurrence NF, widening, or separate
relation authority remains after the cutover.

### 10.3 Family state

Unrolling a sequential loop whose state is a family unrolls only the fixed sequential count. Each
state remains a `FamilyValueId`/template. The lowering must not enumerate the family domain.

For example, a sequential count of 8 over a family domain of 30,720 performs eight body
instantiations, not 245,760 lane evaluations.

Sequential iterations are dependency-ordered and are not parallelized with Rayon. Independent work
inside one body or unrelated roots may be parallelized later after measurement, using deterministic
merge order and bounded worker memory.

## 11. Identity soundness invariants

The implementation must maintain all of the following:

1. IDs are compared only inside the job-local arena that allocated them.
2. A runtime scalar leaf uniquely identifies its semantic SSA definition.
3. A bound variable is descriptor-local and cannot escape uninstantiated.
4. An index-map node includes the exact map instance and ordered input IDs.
5. Map implementations are pure, deterministic, and total.
6. Range facts, noise bounds, and provenance diagnostics are outside semantic identity.
7. A family element identity includes both family and complete index identity.
8. A family source identity distinguishes independent random/sampler values.
9. Separate invocations of the same sampler definition use distinct `SampleEventId`s.
10. A resolved artifact binding aliases the consumer to the producer's exact `FamilyValueId`.
11. Pointwise domain alignment alpha-renames only the local bound index; it never merges family value
   identities.
12. A family relation is applied only after substituting the same runtime index into every bound
    occurrence in its public and target templates.
13. Numeric ID order, thread completion order, and hash iteration order never affect accepted
    semantics or diagnostic order.

Under these invariants, equal complete `IndexValueId`s cannot create a branch-correlation false
positive. This follows structurally: equal constants evaluate equally; equal runtime leaves read the
same SSA value; and equal map nodes apply the same pure function to equal ordered inputs.

Sharing only an ultimate dependency is insufficient. For example, `x / C` and `x % C` share `x`
but have different complete IDs.

## 12. Fail-closed errors

The cutover must expose typed errors for at least:

- unresolved template parameter;
- invalid or empty family domain;
- missing or invalid trusted index range;
- dynamic range outside the family domain;
- unknown index-map definition, arity mismatch, or panic;
- unresolved bound index escaping a template;
- family element type mismatch;
- incompatible pointwise domains;
- unsupported nested/escaping family binder;
- missing semantic family source identity;
- nonuniform family preimage construction;
- relation public/target/type/layout/trapdoor mismatch;
- ambiguous family relation;
- sequential count that is nonexact, negative, or does not fit `u64`;
- sequential carried-schema mismatch;
- arithmetic overflow or allocation failure;
- final uncancelled exact/Large term.

These errors must not be collapsed into `UnsupportedMatrixProductExpansion`.

## 13. Determinism and complexity

Let:

- `I` be the number of unique runtime index nodes;
- `V` be the number of unique family value nodes;
- `P` be the number of explicitly packed elements actually stored;
- `N` be the fixed sequential count;
- `B` be the reachable body DAG size of one sequential iteration;
- `R` be the number of registered family relations reached by indexed lookup.

Required bounds are:

- index interning: `O(I log I)` with `BTreeMap`, memory `O(I)`;
- family interning: `O(V log V)`, memory `O(V + P)`;
- template dynamic get: independent of logical family size;
- packed dynamic bound: linear in reachable stored elements, never an implicit domain;
- pointwise lookup distribution: output-linear in the affected expression DAG;
- family relation lookup: indexed by preimage family/contract, `O(log R)` plus structural
  instantiation work;
- sequential lowering: `O(N * B)` before ordinary DAG sharing, with no family-lane factor.

No operation may enumerate combinations of two independent runtime indices. No new persistent cache
database is introduced; all arenas and memos are owned by one simulation job.

## 14. Implementation map

The implementation should modify the existing modules rather than create compatibility wrappers.

### `crates/correctness/src/operational_noise/identity.rs`

- Add the closed/template index descriptor types and trusted range contract.
- Retain semantic source, sampler, trapdoor, matrix, and scalar identities.
- Remove only local loop-owner provenance from canonical family-template equality.
- Retain `ValueDefinitionId` and `SampleEventId` in value-producing leaf identities.
- Keep raw occurrence information only in diagnostics and graph traversal keys after the semantic
  definition/event identity has been constructed.

### `crates/correctness/src/operational_noise/scalar.rs`

- Add or host the job-local `IndexValueArena` beside `ScalarStore`.
- Convert supported integer sources/operations to index nodes only at family-get consumers.
- Remove scalar `Switch` as an operational selection authority after all consumers migrate.
- Keep scalar arithmetic facts separate from index identity and trusted range facts.

### `crates/correctness/src/operational_noise/family.rs`

- Replace owner-keyed `LoopDomainKey`/`CoverageBinderDomain` semantics with exact `FamilyDomain` and
  descriptor-local template binding.
- Add the family value arena, packed/template/pointwise constructors, static/dynamic get, and
  template instantiation.
- Lower reindexing to a template; do not add a second reindex authority.
- Preserve semantic family source and element identities.

### `crates/correctness/src/operational_noise/lower.rs`

- Require complete parameter specialization before indexed lowering.
- Resolve every artifact-bound consumer input to the producer's exact `FamilyValueId` before family
  access or relation matching.
- Lower `ParallelLoop` to one template body.
- Lower `FamilyPack`, static/dynamic get, pointwise family operations, and `Select` according to this
  document.
- Register family preimage relations from uniform template bodies.
- Replace binder-owner alignment helpers with descriptor-local alpha-normalization.
- Fully unroll sequential loops and delete matrix-only sequential restrictions.
- Preserve memoization by semantic source plus closed runtime indices; never memoize by range alone.

### `crates/correctness/src/operational_noise/normal_form.rs`

- Represent an unresolved dynamic family element with `FamilyElementId` in exact identity.
- Distribute lookup through pointwise family operations in the canonical direction.
- Remove matrix Switch barrier/case-transform identity after migration.
- Keep all matrix polynomial, central scalar, bound, and final Large rules unchanged unless this
  document explicitly replaces them.

### `crates/correctness/src/operational_noise/normal_form_relation.rs`

- Extend the sole relation registry with `FamilyRelationRegistration` and an index keyed by
  preimage family plus static relation contract.
- Reuse ordinary target normalization and cycle handling.
- Do not create a second family relation registry with independent matching semantics.

### `crates/correctness/src/operational_noise/normal_form_family.rs`

- Retain family validation helpers that are still required.
- Delete `TermSequentialRecurrence` after lowering performs complete unrolling.

### DSL and Tall lookup

- Keep template `IntExpr::Var`; resolve it before the checker job.
- Express dynamic finite selection through packed family lookup.
- Expose Tall lookup artifacts as logical flat families to the checker. Physical chunking remains a
  runtime layout concern.
- Do not add Tall-specific branches to the checker.

## 15. Migration sequence

The authority switch must be staged but must not leave a long-lived dual production path.

1. Add `FamilyDomain`, trusted range, `IndexValueArena`, and identity tests without changing
   production selection.
2. Add the family value arena and migrate `FamilyPack`, parallel templates, pointwise operations,
   and static/dynamic get.
3. Add family preimage registration/application and migrate Tall's checker-visible artifacts to
   logical flat families.
4. Lower all finite `Select` forms through packed lookup; delete matrix/scalar Switch production
   paths and Switch NF rules immediately in the same bounded stage.
5. Replace sequential recurrence lowering with complete unrolling for all required carried sorts;
   delete `TermSequentialRecurrence` in the same stage.
6. Remove owner-keyed family alignment and dead binder/substitution code that no production path
   uses.
7. Run focused gates, then the explicitly authorized local Tall simulation. Do not start a pod until
   local Tall reaches finite bounded acceptance.

Every stage must keep the tree compiling and must remove dead compatibility names, including any
`legacy_` identifier, before handoff.

## 16. Mandatory tests

### 16.1 Index identity

- two runtime inputs with the same range remain distinct;
- two sampler invocations of the same definition and parameters remain distinct;
- the same runtime SSA input reused twice has one ID;
- same map instance plus same ordered inputs reuses one ID;
- different map, parameter, input order, or runtime leaf remains distinct;
- `x / C` and `x % C` remain distinct;
- range changes do not change identity and conflicting declarations fail closed;
- no cross-job numeric ID comparison is accepted;
- deep/shared index DAG construction is iterative and grows linearly.

### 16.2 Family identity and access

- equal packed elements in equal order reuse family identity;
- reordered or changed packed elements differ;
- equal-domain `F` and `G` can be combined pointwise but retain distinct identities;
- `F[x]` equals repeated `F[x]`, while `F[y]` and `G[x]` differ;
- static/dynamic range boundaries use half-open intervals;
- reindex preserves the source family and map in identity;
- template construction and access do not enumerate logical lanes;
- alpha-renamed local binders produce equal template fingerprints;
- an open template get whose index is derived from `Bound(0)` closes to the expected
  `FamilyElementId` without lane enumeration;
- a template map combining `Bound(0)` with a closed runtime index preserves both ordered inputs and
  its trusted result range;
- a binder-dependent packed `Select` closes only its explicit alternatives and creates no Switch
  barrier or logical family lanes;
- an escaping outer binder is rejected.

### 16.3 Selection

- scalar and matrix select lower to packed lookup;
- equal runtime selector IDs correlate repeated lookups;
- different selector IDs with equal ranges do not correlate;
- bounded-only alternatives take maximum, not sum;
- exact/Large alternatives retain family element identity;
- independent nested selectors do not form a Cartesian product;
- family selection remains pointwise and preserves every branch family identity.

### 16.4 Family preimage relations

- register `for all i: B(i) * K[i] = P(i)` from one uniform template;
- apply it at a runtime `x` whose root differs from the preprocessing binder;
- exporting producer family `K` and importing it through the resolved artifact binding reuses the
  exact producer `FamilyValueId` and permits the universal relation at runtime `x`;
- a wrong, missing, ambiguous, or external artifact binding does not inherit the producer relation;
- reject `B(x) * K[y]`, wrong family, wrong public, wrong target contract, wrong trapdoor, wrong
  layout, and out-of-domain range;
- index-independent public factors are supported when the template proves independence;
- duplicate identical registrations deduplicate;
- distinct targets fail with deterministic ambiguity;
- application does not enumerate the relation domain;
- relation application preserves prefix/suffix and ordinary central-scalar rules;
- a cyclic target fails through the existing relation-cycle mechanism;
- Tall logical flat lookup exposes the registered relation at `input_index`.

### 16.5 Sequential unrolling

- counts 0, 1, and N;
- iteration index becomes the exact step constant;
- all carried components update simultaneously;
- carried matrix, scalar, tuple, and matrix-family schemas required by Tall/Diamond;
- schema mismatch and nonexact count reject;
- family state unroll grows with sequential count and body size, not family domain;
- the final DAG contains no sequential recurrence node or state-placeholder factor;
- Diamond's fixed level scan and fixed bit scan lower successfully.

### 16.6 Production and deletion gates

- zero production references to matrix/scalar Switch semantic authority after Stage 4;
- zero `TermSequentialRecurrence` after Stage 5;
- zero owner-based family equality/alignment after Stage 6;
- zero `legacy_` identifiers in the migrated modules;
- no checker branch refers to Tall, Diamond, node numbers, or fixture values;
- full serial correctness library, `cargo check -p mxx-correctness`, nightly formatting, and
  `git diff --check` pass before local Tall.

## 17. Acceptance

The indexed-family cutover is accepted only when:

1. all mandatory focused tests pass;
2. the normal-form checker still rejects every final exact/Large residual;
3. local Tall simulation completes with all Large terms cancelled and a finite bound;
4. the strict threshold remains `2 * plaintext_modulus * noise < ciphertext_modulus`;
5. Diamond's existing graph is expressible after finite selects are written as family lookup and
   sequential loops are fixed-count unrolled;
6. measured work and memory do not scale with Tall's logical family size except for explicitly
   packed physical elements actually present in the Graph IR;
7. no pod is started before local semantic acceptance.

Rayon parallelization is deferred until the local Tall run identifies a genuinely slow,
independent, memory-safe loop. Sequential iterations and dependent relation rewrites are not such a
loop.
