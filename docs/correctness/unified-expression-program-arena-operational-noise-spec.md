# Unified Expression and Program Arenas for Operational Noise

## 1. Status and adoption boundary

This document specifies a possible simplification of the indexed-family operational-noise
checker. It is self-contained, but it is not the production authority while
`docs/correctness/indexed-family-operational-noise-spec.md` remains the active cutover
specification. Adopting this design requires a separately approved migration.

The current indexed-family cutover must first reach its own acceptance gates. This redesign must
not interrupt, partially replace, or add a second production authority to that work. It may be
developed behind non-production tests and switched into production only in one bounded migration
stage that deletes the replaced authorities.

The design has two job-local arenas:

- one `ExprArena` for all closed and binder-open value expressions; and
- one `ProgramArena` for typed, nonrecursive programs, including indexed families.

It deliberately removes separate semantic authorities for runtime indices, family values, family
elements, template values, and universal family-root indices. Validated views and proof
capabilities may wrap expression or program IDs, but those wrappers never create new semantic
identities.

## 2. Goals and non-goals

The implementation must:

1. identify semantically identical expressions by one job-local canonical DAG identity;
2. keep independent runtime values, sample events, artifacts, families, and indices distinct;
3. express `B(x) * K(x) = P(x)` by specializing one universal relation at one exact `x`;
4. analyze a generated family over its whole declared domain without enumerating its lanes;
5. avoid every Cartesian expansion of independent selectors;
6. fully unroll exact fixed-count sequential loops into ordinary DAG nodes;
7. preserve exact identities even when finite numeric bounds are available;
8. distinguish missing numeric evidence from a semantically known `Large` value;
9. remain stack-safe and bound peak memory by live DAG/NF work rather than historical work; and
10. fail closed whenever type, scope, range, source, layout, trapdoor, or relation evidence is
    incomplete.

The initial implementation does not attempt algebraic equivalence of arbitrary index programs,
recursive programs, dependent family domains, symbolic sequential recurrences, or proof of DSL
index-map range declarations.

No checker branch may depend on Tall or Diamond names, graph node numbers, fixture values, or
debug strings.

## 3. Checker job and trust boundary

One fully specialized parameter environment creates one checker job:

```rust
struct CheckerJob {
    expressions: ExprArena,
    programs: ProgramArena,
    facts: FactStore,
    relations: RelationRegistry,
    monomials: MonomialArena,
    normalization: NormalizationCache,
}
```

All template parameters, loop counts, matrix dimensions, sampler contracts, family domains, and
index-map fixed parameters are resolved before semantic lowering. An unresolved value is a typed
error. A different parameter candidate creates a different job.

Raw arena IDs are never persisted in artifacts or compared across jobs. Artifact import validates
the semantic artifact identity, version, confidentiality, type, layout, and domain, then resolves
the artifact to an ID in the receiving job.

Index-map output ranges remain explicitly trusted DSL contracts in the first implementation. The
code and diagnostics must call them `TrustedIndexRange`; they must not be described as inferred or
verified.

## 4. Unified expression DAG

Every scalar value, matrix value, trapdoor value, index computation, family body, and relation
target is represented in one immutable hash-consed DAG:

```rust
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ExprId {
    arena: ArenaToken,
    slot: u32,
}

struct ExprNode {
    operator: ValueOperator,
    inputs: Box<[ExprId]>,
}
```

`ArenaToken` identifies the owning job-local arena; `slot` identifies one node in that arena.
Both fields participate in ID equality. Validation rejects a foreign token before inspecting a
slot.

The interning key is the complete pair `(operator, ordered inputs)`. A hash is only an accelerator;
collisions are resolved by full key comparison. Nodes are immutable after interning. Numeric facts,
trusted ranges, source locations, diagnostic provenance, and normalization state are not part of
the key.

The arena uses deterministic maps, or deterministic sorting before externally observable output.
Numeric ID order, hash iteration order, and thread completion order must not affect accepted
semantics or diagnostic ordering.

All traversal, substitution, reachability, reference counting, and destruction of deep structures
uses iterative worklists. No operation relies on recursive Rust stack depth.

## 5. Operators and complete semantic identity

Zero-input leaves and ordinary operations use the same operator namespace:

```rust
enum ValueOperator {
    Argument { position: u32 },
    Constant(TypedConstant),
    Source(SemanticSourceIdentity),
    Sample { event: SampleEventId, descriptor: SampleDescriptor },
    OpaqueFamilyElement { source: SemanticFamilySourceIdentity },
    IndexMap { definition: IndexFunctionDefinitionId, parameters: Box<[u64]> },
    ExplicitElement { domain: FamilyDomain },
    ProgramCall { program: ValueProgramId },
    Scalar(ScalarOperation),
    Matrix(MatrixOperation),
    Trapdoor(TrapdoorOperation),
}
```

Every operator descriptor contains every value-affecting parameter. Examples include bit
positions, slice ranges, concat axes, matrix shapes, ring parameters, hash tags and dynamic tag
arguments, sampler parameters, gadget base, digit count, and decomposition kind.

`SemanticSourceIdentity` contains at least:

- the stable value definition;
- the concrete invocation identity;
- `SampleEventId` for independent randomness;
- output role;
- the complete sampler descriptor;
- resolved artifact identity when applicable;
- resolved type; and
- every value-affecting closed coordinate.

Two invocations of the same sampler definition with equal parameters have different
`SampleEventId`s. Temporary loop owners, binder names, occurrence display paths, and numeric bounds
are excluded after the stable definition or event identity has been constructed.

## 6. Types and validated construction

Every expression has exactly one resolved type in the job-local fact table:

```rust
enum ResolvedValueType {
    Bool,
    Int,
    Real,
    Bytes,
    Matrix(ResolvedMatrixType),
    Trapdoor,
}

struct ResolvedMatrixType {
    modulus: BigUint,
    ring_dimension: usize,
    rows: usize,
    columns: usize,
}
```

One shared validator checks operator arity, input types, output type, matrix dimensions, ring and
modulus compatibility, central-scalar rules, layout requirements, and trapdoor/public contracts
before interning. A matrix atom without a semantic source identity is forbidden.

Representative rules include:

- matrix addition requires identical complete matrix types;
- matrix product requires compatible inner dimensions and equal ring/modulus contracts;
- only validated `1 x 1` matrices use the central-scalar canonicalization rules;
- `LiftConstantPolynomial` has an explicit scalar-to-matrix output contract;
- `ExtractCoefficient` has an explicit matrix-to-integer contract and checked position; and
- `Slice`, `View`, `Concat`, and `Tensor` include complete shape/layout descriptors.

## 7. Scope, arguments, and program identity

`Argument(0)` is not meaningful without a program scope. Open references therefore use a validated
view:

```rust
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ScopedExprId {
    program: ValueProgramId,
    expression: ExprId,
}
```

`ScopedExprId` is not a second expression identity. It pairs an expression DAG node with the
program that binds its free arguments. The constructor is private to `ProgramArena` and verifies
that every free argument is in range and has the declared type.

Closed expressions have no free arguments and use a validated `ClosedExprId` view. Converting an
open expression to `ClosedExprId` without exact substitution is impossible through the API.

A program is immutable and nonrecursive:

```rust
struct ProgramInput {
    value_type: ResolvedValueType,
    trusted_index_range: Option<TrustedIndexRange>,
}

struct ProgramSignature {
    inputs: Box<[ProgramInput]>,
    output: ResolvedValueType,
}

struct ValueProgram {
    signature: ProgramSignature,
    root: ExprId,
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd, Hash)]
struct ValueProgramId {
    arena: ArenaToken,
    slot: u32,
}
```

The program interning key is `(signature, alpha-normalized root)`. Arguments are positional, so
`lambda x. x + 1` and `lambda y. y + 1` share a program. This is alpha-equivalence, not arbitrary
algebraic equivalence.

The program-call graph must be acyclic. Program construction may refer only to already finalized
callees. Finalization checks free arguments and computes an iterative structural fingerprint.
Across jobs, tests compare this fingerprint, never raw `ExprId` or `ValueProgramId` slots. Within
one arena, reinterning the same complete key returns the same ID regardless of unrelated insertion
order.

## 8. Runtime indices

There is no `IndexValueArena`. A runtime index is a validated view of an integer expression:

```rust
struct FamilyDomain {
    minimum: u64,
    maximum_exclusive: u64,
}

struct TrustedIndexRange {
    minimum: u64,
    maximum_exclusive: u64,
}

struct IndexValue {
    value: ClosedExprId,
    range: TrustedIndexRange,
}
```

Both intervals are half-open. Constructors reject `minimum > maximum_exclusive`; operational
families additionally reject an empty domain. Dynamic access requires complete range containment.

The expression identity, not the range, determines equality. The fact store allows repetition of
the exact same trusted range declaration and rejects every different declaration for the same
expression. Range declarations are finalized before any family access. Late declarations are
typed errors.

Index functions are ordinary `IndexMap` expression nodes. Each definition is pure, deterministic,
total, fixed-arity, and registered under a stable definition identity. An instance is identified by
definition plus resolved parameters and ordered input expression IDs. A panic is caught and
reported as a typed implementation error; it is never interpreted as an invalid runtime value.

Consequently `x / C` and `x % C` have different expression IDs even though they share `x` and have
related ranges. Equal ranges never correlate independent runtime values.

## 9. Families as programs

An indexed family is a validated one-argument program:

```rust
struct FamilyValueId(ValueProgramId);
```

The program input is `Int` with one exact, nonempty `FamilyDomain`; its output is the element type.
There is no family-specific semantic arena and no `Source`, `Packed`, `Template`, `Pointwise`, or
`Reindex` family enum.

### 9.1 Source family

A source family is:

```text
lambda i. OpaqueFamilyElement(source, i)
```

The source identity distinguishes artifacts, stable definitions, and sample events. The argument
expression makes different indices different exact values.

### 9.2 Explicit family

An explicitly stored family is:

```text
lambda i. ExplicitElement(i, [A0, A1, ..., A(P-1)])
```

Its domain width equals `P`, and ordered element identities are part of the program body. Dynamic
access remains the compact exact expression `ProgramCall(F, x)`; it does not beta-reduce into a new
node that clones the `P` inputs and does not expand into cases. The explicit descriptor is stored
once in the program DAG.

A derived range-maximum tree may answer numeric bound queries. Building the tree is `O(P)`;
full-range lookup is `O(1)` after construction and subrange lookup is `O(log P + output fragments)`.
This cache contains facts only and is not an equality authority.

### 9.3 Generated family

Parallel loops, pointwise operations, reindexing, gather, zip, zip-offset, and family selection are
ordinary generated programs:

```text
pointwise_add(F, G) = lambda i. F(i) + G(i)
reindex(F, h)       = lambda i. F(h(i))
select_family(s,F,G)= lambda i. ExplicitElement(s, [F(i), G(i)])
```

Their bodies are built once. Equal domains permit pointwise composition but never merge distinct
family program identities. An escaping or unresolved outer argument is rejected.

### 9.4 Family access

Static and dynamic access use `ProgramCall`:

```text
F(x) = ProgramCall { program: F }(x)
```

Construction checks the exact family program, complete index expression, trusted range containment,
and expected element type. There is no independent `FamilyElementId`. Repeated `F(x)` calls reuse
one expression node; `F(y)` and `G(x)` remain different.

Generated calls beta-reduce through iterative substitution and memoize by `(program, ordered
closed arguments)`. Source and explicit calls remain opaque exact `ProgramCall` nodes after their
contracts are validated.

## 10. Artifact aliases

A resolved artifact binding is an alias to the producer's exact family program in the current job.
The consumer must not construct a new source program. Resolution recurses through tuples and
bundles but never merges values by type, shape, domain, bounds, or display name.

An external or unresolved artifact obtains a distinct source identity and cannot inherit the
producer's relations. Missing, ambiguous, version-mismatched, type-mismatched, layout-mismatched,
or domain-mismatched bindings fail closed.

## 11. Proof-only family-root analysis

Family-root analysis uses the family's existing formal `Argument(0)`. It does not intern a
`UniversalFamilyRoot` expression or any other synthetic runtime index.

The formal argument must nevertheless be inaccessible to ordinary consumers. `ProgramArena`
therefore exposes one internal higher-ranked operation, conceptually:

```rust
fn with_family_root_proof<R>(
    &mut self,
    family: FamilyValueId,
    use_once: impl for<'proof> FnOnce(FamilyRootProof<'proof>) -> Result<R, ProofError>,
) -> Result<R, ProofError>;
```

`FamilyRootProof<'proof>` has private fields, is neither `Clone` nor `Copy`, and cannot be
constructed outside `operational_noise`. It records:

- the exact finalized family program;
- that program's exact finalized domain and element type;
- the scoped `Argument(0)` expression already owned by that program; and
- a nonsemantic generation token used only to reject stale or foreign capabilities.

The capability is issued only after the expression facts, program, domain, artifact bindings, and
relation registry are finalized. Callers cannot supply or widen the domain. The closure consumes
the capability exactly once.

### 11.1 Allowed proof operations

While the capability is borrowed, a proof-local context may:

1. inspect the family body at its scoped formal argument;
2. conservatively propagate typed facts over the entire declared argument range;
3. beta-reduce generated calls, close source/explicit calls symbolically, and normalize pointwise
   structure without enumerating the domain;
4. record reached family applications whose index is exactly the same scoped formal expression or
   a complete derived expression of it; and
5. specialize a universal relation by substituting that exact scoped expression into every
   occurrence in its public, preimage, target, and trapdoor plans.

Proof-local exact expressions, reached-call records, specializations, and normalization entries use
a local cache owned by the closure. The cache is destroyed before `with_family_root_proof` returns.
Substitution is an iterative proof-local environment overlay on the shared immutable DAG. It does
not intern a synthetic closed argument or proof-specialized node into the ordinary `ExprArena`.

### 11.2 Forbidden proof operations

The capability, its scoped formal argument, and every proof-local value may not:

- be returned from the closure or stored in the job's ordinary expression/program/fact caches;
- enter the executor, runtime values, serialization, artifacts, or public APIs;
- call ordinary dynamic/static family access or ordinary materialization APIs;
- be accepted by an arbitrary scalar, matrix, index, or layout consumer;
- enumerate or sample the family domain;
- establish equality with another family's formal argument;
- establish equality with an independently computed runtime index merely because ranges agree;
- specialize a relation at different expressions for public, preimage, target, or trapdoor; or
- populate the ordinary runtime specialization cache.

Proof-root normalization returns only an owned closed result that contains no proof-scoped ID, such
as a finite bound, exact zero, or a typed failure. A nonzero proof-local exact residual cannot be
exported as an ordinary expression; it is summarized in a bounded diagnostic and rejected.

These API and lifetime rules make the formal argument an unforgeable proof binder without adding a
second semantic index node.

## 12. Fixed-count sequential loops

After parameter specialization, a sequential-loop count must be an exact nonnegative integer that
fits `u64`. Lowering implements the literal semantics:

```text
state_0 = initial
state_(step + 1) = body(step, state_step, invariants)
result = state_count
```

All carried components update simultaneously after each body instantiation. Matrix, scalar,
trapdoor, tuple, and family-program values use their ordinary representations. A family state
remains one `FamilyValueId`; the sequential loop does not enumerate its domain.

There is no semantic policy ceiling and no recurrence, widening, exponentiation, or summary path.
The implementation must not reject a count merely because it exceeds a hard-coded threshold.
Checked arithmetic, `u64` conversion, arena exhaustion, allocation failure, or an explicitly
configured whole-job resource budget may abort with a typed resource error, but such an abort does
not change the loop semantics or authorize a shortcut.

Lowering costs `O(N * B)` before ordinary DAG sharing, where `N` is the exact sequential count and
`B` is the reachable body size. Diamond's fixed level and bit scans follow this rule.

## 13. Universal family relations

A family relation is a program-level rule:

```text
forall i in D: public(i) * preimage(i) = target(i)
```

An index-independent public value is represented by a closed expression in the public plan.

```rust
struct UniversalFamilyRelation {
    domain: FamilyDomain,
    public: ScopedValuePlan,
    preimage_family: FamilyValueId,
    target: ScopedValuePlan,
    trapdoor: ScopedValuePlan,
    source: SamplerSourceContract,
    matrix_type: ResolvedMatrixType,
    layout: Option<ResolvedLayoutIdentity>,
    factor_order: FactorOrderContract,
}
```

Plans use one alpha-normalized formal argument. Registration rejects unresolved free arguments,
conditional/nonuniform sampling, source mismatch, matrix mismatch, layout mismatch, invalid
trapdoor/public pairing, and an inconsistent domain.

## 14. Two-stage relation index

The registry is one authority for closed and universal relations. Universal lookup uses two
deterministic `BTreeMap` stages.

### 14.1 Stage A: universal dispatch

Registration constructs a canonical dispatch key from facts available at a reached preimage call:

```rust
struct UniversalDispatchKey {
    preimage_family: FamilyValueId,
    preimage_source: SamplerSourceContract,
    matrix_type: ResolvedMatrixType,
    trapdoor_source: TrapdoorSourceContract,
}
```

The registry stores:

```text
UniversalDispatchKey
  -> BTreeMap<StaticLhsKey, BTreeMap<CanonicalTargetPlan, Registration>>
```

`StaticLhsKey` contains the alpha-normalized public plan, preimage plan, trapdoor/public pairing,
domain, layout, factor order, and every remaining value-affecting static contract. It excludes the
target. Equal complete registrations deduplicate. Multiple target plans remain grouped under the
same static LHS until an actual instantiated LHS is reached.

A reached `ProgramCall(K, x)` constructs one exact `UniversalDispatchKey` and performs one
`BTreeMap` lookup. It never scans registrations for another preimage family, source, type, or
trapdoor source. Public layout and factor order are checked against the exact monomial in Stage B;
they are not guessed from the preimage call.

### 14.2 Stage B: canonical specialization

For every Stage-A match, the specializer:

1. validates `range(x) subseteq D`;
2. substitutes the identical complete `x` expression into public, preimage, target, and trapdoor;
3. validates types, layout, source, trapdoor/public pairing, and factor order again after
   substitution;
4. structurally canonicalizes the instantiated public and preimage expressions;
5. builds the complete canonical LHS key, including central-factor multiset and adjacent ordered
   factor word;
6. canonicalizes the RHS through the same exact normalization entry point; and
7. inserts the result into a specialization-local map:

```text
CanonicalInstantiatedLhsKey -> BTreeSet<CanonicalRhsKey>
```

The actual monomial queries this exact key. Zero RHS candidates means no rewrite. One distinct RHS
rewrites in the registered direction. More than one distinct RHS produces
`AmbiguousFamilyRelation`; the error lists a deterministic bounded prefix of sorted RHS keys.
Identical RHS keys deduplicate.

Different public factors or different runtime indices produce different instantiated LHS keys and
are not ambiguous. Range equality never changes this result.

### 14.3 Caches and complexity

Ordinary runtime specialization may memoize by:

```text
(UniversalDispatchKey, exact index expression, finalized registry generation)
```

The memo is job-local and invalidated when registration is still permitted. Registration is frozen
before normalization, so production normalization observes one generation.

Proof-root specialization uses only the one-shot proof-local cache from Section 11 and never enters
the ordinary memo.

Let `R` be the number of dispatch keys and `M` the number of registrations returned by the exact
dispatch key. Lookup is `O(log R + M)` plus the unavoidable structural substitution and
canonicalization work of those `M` matches. It is independent of the family domain. The second
stage uses `O(log M)` deterministic map/set operations per produced canonical key. No algorithm
enumerates all registered relations, all runtime indices, or combinations of independent indices.

## 15. Closed relations and exact matching

Closed relations such as `G * D(A) = A` use the same canonical LHS/RHS registry. Their complete key
includes exact `G`, exact `D(A)`, exact target `A`, gadget parameters, decomposition kind, matrix
type, layout, factor order, and any required range proof.

Ordered noncentral matrix factors must be adjacent and retain order. Only validated central `1 x 1`
factors form a sorted multiset. A structural wrapper containing a relation participant is not itself
that participant.

Relations are one-directional and use the existing dependency/cycle discipline. Only changed terms
return to the worklist; blind whole-NF equality iteration is forbidden.

## 16. Facts, bounds, and exact identity

Facts are stored separately from expression identity:

```rust
enum NumericContract<T> {
    Missing,
    Known(T),
}

enum CoefficientBound {
    ExactZero,
    Finite(BoundExpression),
    Large,
}

struct MatrixFacts {
    matrix_type: ResolvedMatrixType,
    coefficient_bound: NumericContract<CoefficientBound>,
    polynomial: NumericContract<PolynomialFacts>,
    metadata: MatrixMetadata,
}
```

`Missing` means no sound numeric transfer contract is available. `Known(Large)` means the checker
knows the value is an exact signal-scale/large factor that must remain available for exact
cancellation. They are never converted into one another.

An analyzed value keeps both identity and facts:

```rust
struct AnalyzedValue {
    semantic: ClosedExprId,
    exact_nf: Option<NormalFormId>,
    coefficient_bound: NumericContract<CoefficientBound>,
}
```

A finite bound never erases the semantic expression. A later exact query normalizes the retained
expression; it never reconstructs identity from a bound, provenance list, representative element,
or display source.

### 16.1 Typed transfer rules

Every production operation has one complete transfer function over resolved types and complete
matrix bounds. The operator registry is exhaustive: adding an operator requires its identity,
type-validation, exact-normalization, and numeric-transfer entries in the same change. The
implementation must not use coefficient-only multiplication where convolution, ring support, or
inner-dimension factors are required.

Required rules include:

- zero produces `Known(ExactZero)`;
- negate and transpose preserve a known coefficient bound;
- slice and validated views do not increase a known coefficient bound;
- concat takes the maximum of known input bounds after validating compatible types/layouts;
- addition uses a checked sum of finite bounds, preserves exact zero identities, and yields
  `Known(Large)` if a semantically large residual survives exact cancellation;
- product applies the established typed matrix-product formula, including ordered inner dimension,
  polynomial convolution/support, ring reduction, and central-scalar handling;
- tensor applies its established typed tensor bound rather than reusing coefficient multiplication;
- lift and coefficient extraction use their explicit scalar/matrix and canonical-residue contracts;
- pointwise/generated-family operations apply the same transfer functions to the body once; and
- any missing required input fact or unsupported operation produces `Missing`, never `Large`.

Exact relation normalization precedes numeric propagation. Thus `s * B * K` first becomes `s * P`
when the exact relation applies, and only then receives a numeric bound.

For explicit families, a dynamic access bound is the maximum over physically stored elements in
the validated index range. For generated/source families, root transfer is performed once over the
formal argument range and never over logical lanes.

## 17. Exact normal form

Only expressions requiring exact addition/product reasoning enter polynomial normal form:

```rust
struct PolynomialNF {
    exact_terms: TermMap,
    bounded_summary: BoundedSummary,
}

struct MonomialDescriptor {
    central_factors: Box<[ScopedExprId]>,
    ordered_factors: Box<[ScopedExprId]>,
}

struct MonomialId {
    arena: ArenaToken,
    slot: u32,
}
```

The monomial arena stores each factor list once. `TermMap` stores only `MonomialId -> signed
multiplicity`. Map keys and values must not duplicate recursively owned factor trees.

Canonicalization performs typed add flattening, zero removal, negate normalization, product identity
removal, central-factor sorting, ordered-factor preservation, double-transpose removal, validated
slice/view rules, exhaustive concat/slice inverses, generated-program beta reduction, and exact-term
coefficient cancellation.

Mutable flags such as `relation_live`, `relation_protected`, bounded origins, or reified provenance
must not become semantic authorities. Relation eligibility is queried from exact current factors and
the finalized registry.

The final root is accepted only if all exact/Large residuals have been consumed and a finite proven
bound satisfies the protocol's strict inequality, including `2 * p * noise < q` where applicable.
Equality fails.

## 18. Selection and Cartesian-growth prohibition

The checker has no Switch semantic node, Switch normal form, selector provenance equality, or
casewise relation fixed point. Finite selection is one exact explicit-family call:

```text
select(s, [A0, ..., An]) = ExplicitFamily([A0, ..., An])(s)
```

Nested selections remain nested opaque expressions. Two independent selectors are never expanded
to their Cartesian product. Equal complete selector expressions correlate repeated access; equal
ranges do not.

Tall lookup is one logical family over the table index. Chunk and offset remain runtime storage
details and are not independent checker selectors. Gather and reindex preserve the source program
and complete index-map expression.

## 19. Memory and lifetime rules

Identity is always a small ID. The implementation forbids deep owned identity trees, root-by-root
flattening, duplicate factor storage, recursive drop, and full-map clone fallbacks.

Allowed exact-map strategies are:

- streaming merge from borrowed inputs into a new map;
- persistent maps with path copying; or
- proven-last-use in-place reuse of an owned map.

The following patterns are forbidden for large shared maps:

```text
keys().cloned().collect()
Arc::make_mut(shared_large_map)
shared_large_map.clone() as fallback
all terms -> Vec -> process
```

Reachability computes remaining-use counts. A child NF is released after its last consumer. N-ary
add, product, concat, and tensor process children incrementally instead of retaining every child NF
simultaneously.

Parallelism is deferred until single-threaded semantic completion and memory shape are measured.
Later bounded batches may parallelize independent work, but peak memory must not multiply one huge
NF per worker. Sequential iterations and dependent relation rewrites remain ordered.

Diagnostics contain compact IDs, operator kinds, counts, mismatch classes, bounded samples, omitted
counts, and stage counters. They never format complete DAGs, explicit-family element lists, or
monomial keys.

## 20. Complexity requirements

Let:

- `V` be reached expression nodes;
- `Q` be reached programs;
- `P` be physically stored explicit elements;
- `N` be a fixed sequential count;
- `B` be one sequential body size;
- `R` be universal dispatch keys;
- `M` be matches for one exact dispatch key; and
- `L` be simultaneously live exact monomials.

Required bounds are:

```text
expression interning                  O(V log V), memory O(V)
program interning                     O(Q log Q), memory O(Q)
generated family root                 independent of domain length
explicit family construction          O(P), stored once
explicit range bound                  O(P) build, O(log P) query with cache
universal relation dispatch           O(log R + M) plus match specialization
fixed sequential lowering             O(N * B), no family-domain factor
exact NF live memory                   O(L) plus shared arena storage
```

Forbidden growth includes:

```text
O(generated family domain)
O(selector_A * selector_B)
O(number of roots * shared-prefix size)
O(total historical terms retained until the root)
O(runtime indices * all relation registrations)
```

## 21. Fail-closed errors

Typed errors cover at least:

- unresolved parameter or loop count;
- invalid or empty family domain;
- missing, conflicting, late, or out-of-domain trusted index range;
- foreign arena/program ID;
- unresolved or escaping free argument;
- cyclic program call;
- unknown index function, arity mismatch, evaluation error, or panic;
- incomplete operator/type/layout contract;
- missing semantic source or sample-event identity;
- artifact alias mismatch;
- nonuniform family preimage construction;
- public, target, source, matrix, layout, or trapdoor relation mismatch;
- deterministic family-relation ambiguity;
- sequential count that is negative, nonexact, or does not fit `u64`;
- sequential carried-schema mismatch;
- checked arithmetic, arena, allocation, or configured job-resource exhaustion;
- missing required numeric contract;
- unsupported typed bound transfer;
- final unknown bound; and
- final uncancelled exact or `Large` residual.

These errors must not collapse into a generic matrix-product-expansion error.

## 22. Migration plan

This redesign follows a measured, deletion-oriented migration after the active indexed-family
cutover is accepted.

1. Record current focused-gate results, Tall/Diamond behavior, wall time, peak RSS, arena sizes,
   specialization counts, and live-NF high-water marks.
2. Add `ExprArena`, `ProgramArena`, scope validation, and structural fingerprint tests behind a
   non-production module. Do not route production lowering through it.
3. Add family-as-program constructors, explicit/generated access, artifact aliases, and typed fact
   transfer tests. Demonstrate domain-independent work before production routing.
4. Add the two-stage relation registry and exact `B(x) * K(x) - P(x)` regressions, including
   proof-local specialization with the same formal argument.
5. Add the opaque `with_family_root_proof` API and prove by visibility/lifetime tests that proof
   values cannot enter ordinary lookup, materialization, executor, or caches.
6. Port fixed-count sequential lowering and Diamond's required carried schemas to literal unroll.
7. In one bounded authority-switch stage, route production lowering to the unified arenas and
   delete `IndexValueArena`, `FamilyValueArena`, `FamilyElementId`, template-value authorities,
   `UniversalFamilyRoot`, old specialization caches, and compatibility wrappers.
8. Delete old Switch/Select and recurrence authorities if any remain. Zero migrated-module
   `legacy_` identifiers are permitted.
9. Run focused gates and memory fixtures. Only after local semantic acceptance, run the explicitly
   authorized local Tall simulation. Integration or remote tests still require explicit approval.

Every intermediate production commit compiles. Non-production shadow code must not influence
acceptance, diagnostics, caches, or runtime behavior before the authority switch.

## 23. Mandatory tests

### 23.1 Expression, scope, and source identity

- repeated complete keys in one arena reuse one `ExprId`;
- different build orders yield equal structural fingerprints across jobs, not comparable raw IDs;
- equal ranges with different runtime leaves remain different;
- one runtime SSA value reused twice has one identity;
- equal sampler parameters with different sample events remain different;
- equal map definition/parameters/ordered inputs reuse identity;
- changed map, parameter, input order, or leaf changes identity;
- `x / C` and `x % C` remain different;
- conflicting ranges and foreign IDs fail;
- alpha-renamed programs share identity;
- different program scopes do not equate their `Argument(0)` runtime meanings; and
- depth-4096 shared DAG traversal and destruction are stack-safe.

### 23.2 Families and artifacts

- `F(x)` is reused, while `F(y)` and `G(x)` differ;
- source, explicit, generated, pointwise, reindex, gather, zip, and selection families preserve
  complete program identity;
- explicit access stays one compact call and stores alternatives once;
- generated domains `30,720` and `1,000,000` perform the same structural work;
- pointwise and reindex lookup do not enumerate domains;
- nested independent selectors do not create a Cartesian product;
- producer/consumer artifact aliasing reuses the exact program; and
- missing, external, wrong-version, ambiguous, type, layout, and domain bindings do not inherit
  producer relations.

### 23.3 Proof-root capability

- issuance before finalization fails;
- the capability is nonconstructible, noncloneable, and consumed once;
- ordinary family call, materialization, executor, serialization, and caches reject proof-scoped
  values;
- the formal argument cannot escape the closure;
- two family roots do not compare equal merely because both use `Argument(0)`;
- the exact domain is copied from the finalized family and cannot be widened;
- public, preimage, target, and trapdoor specialize at the identical scoped argument;
- derived index maps preserve that argument and ordered closed inputs; and
- proof-root work is independent of logical domain length.

### 23.4 Relations

- register and apply `forall i: B(i) * K(i) = P(i)` at runtime `x`;
- preprocessing binder identity may differ from runtime `x`;
- a production-shaped `B(X) * K(X) - P(X)` proof root has one exact `X` in public, preimage,
  target, and trapdoor and normalizes to zero;
- wrong index, family, public, target, source, trapdoor, type, layout, factor order, and range reject;
- index-independent public values work;
- identical registrations and RHS values deduplicate;
- one exact LHS with distinct canonical RHS values reports deterministic ambiguity;
- distinct exact LHS values coexist;
- dispatch with many unrelated families/types/contracts inspects only the exact Stage-A bucket;
- measured lookup follows `O(log R + M)` plus structural specialization;
- prefix/suffix, central-scalar, composite target, and cycle behavior remain correct; and
- closed `G * D(A) = A` relations reject `D(A')` and every contract mismatch.

### 23.5 Typed bounds and exact normalization

- `Missing` is observably distinct from `Known(Large)`;
- unsupported or incomplete transfer returns `Missing`, not `Large`;
- known finite bounds larger than the threshold report `ExceedsThreshold`, not missing/unknown;
- bounded secret times bounded public matrix uses the full typed product formula;
- exact `s * B * K` rewrites to `s * P` before bound propagation;
- equal bounds never identify or cancel different values;
- no API constructs exact identity from a bound or provenance representative;
- Tall-shaped slice/concat algebra preserves exact blocks and rejects partial/incompatible layouts;
- noncommutative factor order remains intact; and
- final exact/`Large` residuals fail closed.

### 23.6 Sequential loops

- exact counts `0`, `1`, and `N` use literal semantics;
- iteration indices are exact ordinary constants;
- all carried fields update simultaneously;
- required scalar, matrix, trapdoor, tuple, and family states lower;
- nonexact, negative, overflowed, and schema-mismatched loops reject;
- no fixed policy threshold changes semantics;
- configured resource exhaustion is typed and introduces no shortcut; and
- work grows with `N * B`, not family domain.

### 23.7 Memory and diagnostics

- expression and monomial identities are stored once;
- no shared exact map uses a full-clone fallback;
- a 20,000-node shared-prefix DAG has linear node/memory growth;
- peak NF memory follows live terms rather than total processed terms;
- explicit alternatives are not cloned per dynamic index;
- proof-local caches disappear after one root analysis;
- diagnostic size has a fixed cap; and
- increasing bounded worker count does not replicate one huge NF per worker.

## 24. Acceptance gates

The unified-arena migration is accepted only when:

1. all mandatory focused tests pass;
2. the production tree has exactly one expression identity authority and one relation registry;
3. no production `IndexValueArena`, `FamilyValueArena`, `FamilyElementId`, template-value semantic
   arena, `UniversalFamilyRoot`, Switch authority, or sequential recurrence authority remains;
4. no proof-root type or proof-scoped expression is publicly exported or referenced by runtime or
   executor code;
5. family relation lookup is deterministic and measured as `O(log R + M)` plus structural work;
6. generated-family work and memory do not scale with logical domain length;
7. independent selectors never form a Cartesian product;
8. the exact production-shaped `B(X) * K(X) - P(X)` regression passes with all four contracts at
   the same `X` and all mismatch regressions reject;
9. Diamond gather/reindex/preimage and fixed sequential scans lower without recurrence nodes;
10. generic Tall-shaped fixtures have no uncancelled exact/`Large` residual;
11. full serial correctness-library tests, `cargo check -p mxx-correctness`, nightly formatting,
    and `git diff --check` pass;
12. wall time and peak RSS are compared with the pre-migration baseline and meet an explicitly
    approved regression budget; and
13. after explicit authorization, local Tall completes with a finite bound satisfying the strict
    protocol threshold.

No pod or remote benchmark is started before local semantic acceptance. Parallelization remains a
later measured optimization, not part of the semantic migration.
