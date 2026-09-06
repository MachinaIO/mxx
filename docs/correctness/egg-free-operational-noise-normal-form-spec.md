# Egg-Independent Operational Noise Normal Form Specification

## 1. Status of this document

This document specifies how to implement exact signal cancellation and noise upper-bound calculation in the Operational Noise checker without depending on egg/e-graphs. Here, **exact signal** means exact algebraic terms composed of public keys, secrets, gadgets, preimages, and similar values, rather than encryption error. These terms must cancel completely through registered mathematical relations.

This is a design specification for a future implementation; it does not claim that the current implementation is complete. The final-leaf filter in the current egg implementation is a temporary safeguard during migration and must be removed along with egg.

Terms are introduced by explaining their intuitive meaning and role, followed by a formal definition.

## 2. Goals and non-goals

### 2.1 Goals

1. Correctly cancel all exact signals in honest Tall and Diamond WE protocol definitions.
2. Compute a finite coefficient upper bound for only the remaining noise, without underestimation.
3. Always obtain the same normal form from the same input.
4. Avoid unnecessary exhaustive search, Cartesian selector expansion, and e-graph saturation as matrix-product depth or loop counts grow.
5. Reuse existing lowering results, symbol tables, relation registries, and type/bound contracts without creating a new cache database.
6. Define rules concretely enough to prevent implementers from adding special cases based on protocol names, node numbers, or fixture values.

### 2.2 Non-goals

- Accepting protocols with maliciously constructed cycles.
- Inferring bounds from runtime observations or candidate values.
- A general-purpose CAS that automatically proves arbitrary noncommutative algebra.
- Empirical cancellation of expressions without registered relations.
- Enumerating all selector combinations.
- Retaining the old egg representation for backward compatibility.

## 3. Glossary for beginners

- **DAG**: A directed acyclic graph that allows the same subexpression to be shared from multiple locations. It processes the same computation only once instead of duplicating expression trees.
- **normal form (NF)**: A representation that always gives the same mathematical expression the same form. Comparison and cancellation can use structural comparison alone.
- **monomial**: A product of ordered factors without addition. Factor order is not exchanged because these are matrix products.
- **factor**: One value forming a monomial, such as a public matrix, secret, gadget, or scalar.
- **exact signal**: An algebraic term that must vanish exactly through a relation, rather than an approximation error.
- **bounded-only**: A noise term whose components all have finite coefficient upper bounds. Its internal expression can be summarized by one safe upper bound.
- **selector**: An integer value that determines which case a Switch or dynamic family access selects.
- **relation**: An exact equality, such as `B*K=P`, usable only when registered identities match.
- **producer DAG rank**: A well-founded rank on the producer DAG that generated a relation target. Registration verifies that the target is strictly smaller than the source, proving recursive termination.
- **fail-closed**: Rejecting with a typed error when evidence is insufficient, instead of guessing and continuing.
- **first-Large witness**: The first causal atom found in a deterministic traversal order when the final residual is Large. It is for diagnostics only and is not stored in persistent fields or caches.

## 4. Overall architecture

### 4.1 Intuitive flow

The checker reads the structure once and normalizes exact signals into a sum of terms, rather than repeatedly searching for alternative representations. Factor order is preserved within each term. Terms consisting only of noise are frequently combined into a single summarized noise term. Finally, the checker confirms that no exact signal remains and compares the summarized noise upper bound against the threshold.

### 4.2 Fixed pipeline

The processing order is fixed as follows. Later stages must not run ahead of earlier stages.

1. Resolve input contracts and owner-aware identities.
2. Resolve source bounds and integer domains.
3. Normalize the expression DAG bottom-up. Construct every product exclusively with the deterministic product constructor described below.
4. Aggregate bounded-only terms immediately after each operation.
5. Canonicalize exact monomials, combining same-sign terms and canceling opposite-sign terms.
6. Confirm that the final residual is bounded-only.
7. Check the strict threshold.

This sequence must not become e-graph saturation. However, normalizing relation targets and processing the Switches/relations newly exposed by the result are finite recursion included in the product constructor's definition. This does not mean executing every phase only once.

## 5. Existing structures to reuse and egg components to remove

### 5.1 What to reuse

- Owner-aware atom/source identities generated by lowering.
- The symbol table and resolved types, matrix metadata, and integer domains.
- The source, expected public, target, trapdoor, and ordered indices in `RelationRegistration`.
- Producer-output bound inheritance through protocol/artifact bindings.
- Existing `BoundClass` and matrix metadata semantics.
- Sampler descriptors and the sampler interner. The existing interner may be reused for Gaussian/UniformInterval, but no relations are registered for them.
- Stored family cases. Do not regenerate all logical elements of a family.

### 5.2 What to remove

- The operational-noise egg language, e-classes, rewrite runner, and saturation loop.
- Selection of raw relation left-hand sides by extraction cost.
- Structural preferences, selected-redex epochs, and re-extraction of e-class alternatives.
- Paths that reselect raw `B * K` from the same e-class after applying a relation.
- The final-leaf filter. This is a temporary measure that only prevents a raw left-hand side from returning from an egg equivalence class.
- The operational-checker source-hash check, if that hash protects only artifacts generated for the old Lean checker.
- The Lean operational checker itself and code dedicated to its generation and integration.

The removal inventory must identify at least the following by name.

- `crates/correctness/src/operational_noise/extract.rs` and `ProposalCost`.
- The egg runner, normalization epoch, selected phase, and structural preference.
- Relation searchers/appliers and replacement materialization.
- E-class analysis merges and relation provenance merges.
- Rewrite ownership budgets, reservation counters, and saturation iteration budgets.
- The operational checker's egg Cargo dependency.

Before removal, record a one-to-one mapping in the migration ledger from each responsibility to the function in the new pipeline that takes it over.

## 6. Canonical PolynomialNF

### 6.1 Intuitive meaning

**PolynomialNF** is a small normal form that stores a matrix expression as a sum of signed monomials. A monomial is an ordered sequence of factors; matrix-product factors must not be arbitrarily reordered. Fully bounded terms discard their internal structure and are combined into one bounded noise atom.

### 6.2 Formal definition

```text
PolynomialNF = {
  exact_terms: ordered map MonomialKey -> SignedMultiplicity,
  bounded_summary: ExactZero | Bounded(MatrixBound),
}

MonomialKey = ordered list of ExactFactor
SignedMultiplicity = nonzero signed integer
```

Compare `MonomialKey` values lexicographically by factor identity. Do not use insertion order or hash iteration order. Add coefficients with the same key and immediately remove any key whose coefficient becomes 0.

`bounded_summary` is a safe upper bound on the noise aggregated up to that point, not an equivalent representation of a mathematical expression. Cases that later multiply it by Large and require its internal structure are unsupported. Therefore, expressions that multiply summarized bounded-only terms by exact/Large factors must fail closed. Acceptance tests must establish that the Tall and Diamond WE protocols targeted for acceptance do not require this form.

### 6.3 ExactFactor identity

Intuitively, ExactFactor identity represents the same value at the same runtime coordinates under the same owner, rather than merely the same appearance.

Formally, atom identity includes at least the following.

```text
AtomIdentity = (
  source owner,
  source kind,
  output port,
  coordinate_binders,
  ordered runtime Atom.indices,
  public/target/layout identity,
  optional trapdoor identity
)
```

Do not infer runtime coordinates from binder counts or positions alone. Use the combination of `coordinate_binders` and ordered `Atom.indices`. Pass existing IDs that require canonical comparison once through a canonical resolver that preserves ownership.

A trapdoor may be a protocol input. Relation comparison requires the same input owner and the same ordered coordinates, not merely matching trapdoor presence.

## 7. Normalization rules

### 7.1 Zero

`ExactZero` means the entire matrix has been proved to be exactly 0.

- `0 + X = X`
- `-0 = 0`
- `0 * X = X * 0 = 0`
- integer scalar `0 * X = 0`
- An input with CRT reconstruction coefficient 0 contributes 0 even if that input is Large.

Perform zero annihilation before checking for Large.

### 7.2 Add and Negate

- `Add` adds its children's `exact_terms` by key.
- Add coefficient upper bounds when combining bounded summaries.
- `Negate` reverses exact multiplicity signs. It leaves bounded upper bounds unchanged.
- `X - X` and `X + (-X)` must cancel when they have the same canonical monomial key.
- Do not try multiple association orders because the structure appears ambiguous. Flatten Add once and insert it into an ordered map.

### 7.3 Deterministic product constructor and full expansion

Construct every product exclusively with the following single constructor. Do not omit or reorder any step.

1. First normalize every child to `PolynomialNF`.
2. Distribute over child Adds and flatten products into ordered factor lists.
3. Minimize the scope of currently visible Switches by moving common prefixes, suffixes, and additive terms outside them.
4. Scan the ordered factor list from the left and select the leftmost applicable checked relation.
5. Recursively normalize the relation target to `PolynomialNF`.
6. Reattach the original prefix and suffix to each target monomial, preserving order.
7. Minimize any Switches newly exposed by the target again.
8. Canonicalize central scalars, add coefficients of identical monomials, cancel opposite signs, and fold bounded-only terms.
9. If relations remain, return to step 4, but continue only if the rank multiset described below strictly decreases.

Do not try multiple forms and select the best candidate. This constructor's output is the unique canonical result.

Products distribute over child exact monomials while preserving order.

```text
(A + B) * (C + D)
  -> A*C + A*D + B*C + B*D
```

Do not identify `A*C` with `C*A`. Flatten only matrix-product association, so `(A*B)*C` and `A*(B*C)` produce the same ordered factor list `[A,B,C]`.

Even during full expansion, aggregate bounded-only monomials into `bounded_summary` immediately after each binary multiplication. Assume that the number of exact signal terms stays within a practical range for the targeted protocols. Do not reject by hard-coding an unjustified term-count budget. Allocation failures, integer overflow, and type mismatches are errors.

### 7.4 central scalar

A **central scalar** is a constant-polynomial scalar that can move to either side of the relevant ring/matrix product without changing its mathematical value.

- Move only factors proved central by their types and metadata to the monomial's designated position.
- A scalar key includes its value and owner-resolved matrix/ring type.
- Do not assume that nonconstant polynomials, rectangular matrices, secret matrices, or gadgets are central.
- Numerically combine scalar products only for integer arithmetic or quotient-ring arithmetic explicitly specified by the specification. If the modulus, representative convention, or type is unspecified, do not fold numerically; retain and sort scalar identities in deterministic order. Reapply zero annihilation only when 0 can be proved exactly.

### 7.5 bounded-only aggregation

A **bounded-only monomial** is a term whose factors all have finite upper bounds and that contains no exact/Large signal factor.

Immediately after constructing each Add, Negate, Multiply, Tensor, CRT, or Switch/Select, combine bounded-only terms into one `bounded_summary`. First compute the upper bound according to the operation rules. Do not retain the original bounded expression tree. This prevents noise term counts from continually growing during full expansion.

### 7.6 exact preimage relation `B * K = P`

Intuitively, this rule replaces the product of a registered public matrix `B` and a preimage `K` made from its trapdoor with the registered target `P`. Do not arbitrarily add sampler error to this equality.

Formally, replace an adjacent boundary `[..., B, K, ...]` in an ordered monomial only when the following full match key matches uniquely.

The relation's full match key is the entire following tuple.

```text
(source, ordered indices, public, target, matrix type, layout,
 trapdoor, selector identity and reachable-case mapping)
```

A registration is applicable only if every field of this key matches. Zero matches means the rule is inapplicable at this boundary, not an error. Deduplicate registrations representing exactly the same target. Two or more candidates with different targets must fail closed with `AmbiguousRelation`.

Replacement is one-way only: `B*K -> P`. Do not expand in reverse. Preserve prefix and suffix order.

```text
prefix * B * K * suffix
  -> prefix * P * suffix
```

If a monomial contains several independent boundaries, the product constructor always processes the leftmost applicable boundary. Do not replace `E_B * K`, `K * B`, or cases with a different public matrix or different coordinates.

Do not judge relation termination by the raw boundary count. Each applicable relation source has a natural-number rank on the producer DAG. Form a multiset of the ranks of all currently pending relations and compare it as a descending vector. Continue recursion only if the rank multiset after recursively normalizing the target is strictly smaller lexicographically than before application. Compute ranks from the producer DAG at registration; do not infer them from names, node numbers, or expression sizes. Same-rank cycles or increases produce a typed termination error.

### 7.7 Switch scope minimization

**Switch scope minimization** moves common computations independent of the selector outside the cases, leaving only the parts the Switch actually selects.

For the cases of the same selector, compute the longest common ordered prefix and ordered suffix once.

```text
Switch(s, A*X0*D, A*X1*D)
  -> A * Switch(s, X0, X1) * D
```

For addition as well, move identical canonical terms independent of the selector outside.

```text
Switch(s, A+B0, A+B1)
  -> A + Switch(s, B0, B1)
```

If `G` in `Switch(s, case0*G, case1*G)` is independent of the selector, it must be moved outside. Complete this before cancellation.

Do not perform Cartesian expansion across different selectors. Align Switches case-wise only when they share the same selector. Do not access cases unreachable from the selector domain. Fail closed if the correspondence between reachable cases is not unique.

### 7.8 family

A family is a collection of candidates selected at runtime. Use the cases saved once by lowering rather than enumerating all logical indices.

- Static access uses only the specified case.
- Dynamic access takes the maximum over stored cases reachable within the validated integer domain.
- An endpoint rule may be used if the shared affine template has been reviewed.
- Do not construct a Cartesian product of selectors.
- Handle nested families as explicit stored-case nesting, only to the depth required by Tall/Diamond acceptance fixtures. Do not implicitly flatten or expand all combinations.

### 7.9 PolynomialNF transfer for every operation

The following is the complete per-operation dispatch table. "Preserve" means applying the same structural operation to both exact monomial keys and the bounded summary. Operations absent from the table produce `UnsupportedOperation`; do not guess their behavior.

| Operation | Exact processing in `PolynomialNF` | Bounded processing and validation |
| --- | --- | --- |
| Zero | Empty exact map | `ExactZero` |
| Add/Sub/Negate | Signed addition by key | Add upper bounds; Negate preserves the upper bound |
| MatrixMultiply | Constructor from 7.3 | Zero-first, `K*R*A*B`, fold immediately afterward |
| Transpose | Validate types and normalize in the order `T(A+B)=T(A)+T(B)`, `T(-A)=-T(A)`, `T(A*B)=T(B)*T(A)`, `T(T(A))=A`. Attach transpose view identity only to atoms that cannot be decomposed further | Preserve the upper bound; exchange rows/columns |
| Slice | Preserve the owner-resolved slice specification in exact identity | Preserve the upper bound; validate the range and output shape |
| Concat | Distribute by corresponding position only over pointwise Adds with matching axis, arity, every input shape, and corresponding positions. Otherwise preserve the axis and ordered inputs as one canonical structural factor | Maximum over reachable inputs; validate axis/shape |
| Tensor | Distribute ordered tensor factor pairs without adding a matrix inner-sum factor | Zero/Large-first; polynomial factor `R` |
| exact integer scale | Scalar 0 yields zero. Move only proved central scalars under 7.4 | `abs(s)*B` |
| interval integer scale | When multiplying an exact signal, preserve an undetermined scalar identity and do not use it for cancellation | `max(abs(min),abs(max))*B` |
| CRT recompose | Multiply each input by its nonzero reconstruction coefficient as an exact central scalar, then Add | `sum abs(c_i)B_i`; a zero coefficient yields 0 without reading the input |
| LiftConstantPolynomial | Convert integer exact/domain identity to a constant-polynomial factor | `max(abs(min),abs(max))`, constant metadata=true |
| PackPolynomialCoefficients | An exact pack factor with ordered Bool bit identities and bit weights. Do not reinterpret it as a general polynomial product | Bit-bound formula from 8.6; validate shape/bit count |
| HashPlain | Preserve query identity and ordered arguments in exact atom identity | `Large` without an authoritative hard range |
| Select | Put selector identity and ordered cases into the same canonical form as Switch | Maximum over reachable cases; validate the domain |
| FamilyGetStatic | Reference one case's NF by owner and static index | That case's bound |
| FamilyGetDynamic | Selector identity and the NFs of stored reachable cases | Maximum over stored cases. Do not enumerate all indices |
| Identity/rotation/permutation view | Preserve the owner-resolved view specification in the exact factor. Fold only proved compositions | `Bounded(1)` or preserve the input bound |
| MatrixScale/other coefficient-preserving view | Preserve exact identity and ordered children | Operation-specific scalar rule, or preserve the bound |

Do not turn structural operations such as Transpose, Slice, Concat, Tensor, and Select into opaque leaves such as `Existing` for later processing by another extractor. The canonical constructors in the table directly build the final representation.

## 8. Source and bound rules

### 8.1 BoundClass

- `ExactZero`: Every coefficient is exactly 0.
- `Bounded(B)`: Every polynomial coefficient `c` satisfies `|c| <= B`.
- `Large`: A value for which the protocol explicitly declares that no small finite coefficient bound is used.
- Missing/unspecified bounds are contract errors, not `Large`.

### 8.2 ProtocolInput and GraphWire

Do not classify ProtocolInput or ordinary GraphWire as Large solely because of their kind. Resolve exactly one of the following authoritative sources.

1. The producer output bound from an exact upstream protocol/artifact binding.
2. Explicit `MatrixBounded(B)`.
3. The `MatrixExact` canonical contract `0 <= c < U`. Validate `0 < U <= Q` and produce `Bounded(U-1)`.
4. Explicit `Large`.

If none exists, return `MissingInputBoundContract`. Do not infer from names, shapes, or runtime candidates. `is_constant_polynomial` is multiplication metadata separate from the bound value.

### 8.3 Bool and Int

Bool/Int carry integer domains rather than matrix BoundClass values. Bool has domain `[0,1]`. Lifting `[min,max]` to a constant polynomial gives upper bound `max(abs(min), abs(max))`. A missing domain is an error.

`ExtractCoefficient` uses selector-only `[0,U-1]` when the input matrix has an authoritative `canonical_coefficient_exclusive_upper_bound = U`. Otherwise it falls back to the full modulus range. Do not use runtime observations. Preserve selector-only provenance after Divide/Remainder; only range checks, IntCompare, BitExtract, FamilyGetDynamic, and Select may consume it. Reject it in matrix scaling, dimensions, sampler cutoffs, and noise arithmetic.

### 8.4 Samplers and decomposition

- Gaussian + explicit nonnegative hard cutoff `C` -> `Bounded(C)`. Do not infer from sigma.
- UniformInterval `[min,max]` -> `Bounded(max(abs(min),abs(max)))`.
- UniformResidue -> `Large`.
- Preimage sampler -> `Bounded` from the explicit cutoff, independently of whether a relation is used.
- Decomposition digit, base > 1:
  - regular digit -> `Bounded(max(floor(base/2),1))`
  - small digit -> `Bounded(base-1)`
- Regular Gadget matrix -> `Large`.
- Small Gadget matrix -> `Bounded(base-1)`.

Do not share helpers between Gadget matrices and decomposition digits.

### 8.5 matrix constants

- Zero -> `ExactZero`
- Identity, UnitRow, UnitColumn, valid rotation/permutation -> `Bounded(1)`
- Explicit polynomial -> maximum absolute coefficient; `ExactZero` if all coefficients are 0
- PowerOfBase -> `Bounded(abs(base)^exponent)`
- Invalid base, shape, or index -> error. No fallback to Large

### 8.6 PackPolynomialCoefficients

From coefficient-major, little-endian Bool bits, compute

```text
c_j = sum_k 2^k bit[j,k]
B_j = sum_k 2^k max(bit[j,k])
B(output) = max_j min(B_j, q-1)
```

Validate a Bool family, 1x1 output, positive bit width, and bit count `ring_dimension * width`. Do not always classify the result as Large. Known zero bits may tighten the upper bound.

### 8.7 Operation bounds

- Add/Sub: Zero is the identity; add bounds for bounded operands; otherwise Large. Correlated Large terms cancel only when they have the same exact monomial identity.
- Matrix product: Give zero highest priority. A Large factor makes the result Large. `Bounded(A)*Bounded(B)` yields `Bounded(K*R*A*B)`. `K` is the number of potentially nonzero inner summands, or 1 for a 1x1 scalar. `R=1` if either operand is a constant polynomial; otherwise it is the ring dimension.
- Integer scalar scale: `S=max(abs(min),abs(max))`, `Bounded(B)->Bounded(SB)`. A nonzero scalar times Large is Large.
- Tensor: The same zero/Large priority and polynomial factor `R` apply. There is no matrix inner sum.
- Concat and Switch/Select: The maximum over reachable alternatives, not their sum.
- CRT: `sum_i abs(c_i) B_i`. `c_i=0` contributes zero even for a Large input.
- Transpose, slice, and coefficient-preserving views preserve the class.
- HashPlain is Large without an authoritative hard output range.
- SequentialState inherits the previous carried bound inside a recurrence. An unresolved state outside a recurrence is an error. SequentialRecurrence evaluates the initial/transition expressions.

### 8.8 first-Large witness

Compute the first-Large witness on demand for diagnostics only. Traverse lexicographically by `PolynomialNF` ordered monomial key, factor index, and Switch case index. Report only the authoritative source identity that first returned `Large` and the operation path from the root to that source. Do not use candidate search order, rayon completion order, or hash iteration order. Do not store the witness in analysis fields, e-classes, persistent caches, or protocol artifacts.

## 9. Fail-closed and deferred cases

Reject the following with typed errors.

- Missing bound contracts, integer domains, or producer bindings.
- Mismatched matrix types, layouts, owners, runtime coordinates, or trapdoors.
- Multiple non-unique relation registration candidates.
- Relations that require exchanging ordered factors to apply.
- Expressions that later multiply a bounded summary by an exact/Large factor.
- Expressions requiring Cartesian combinations of Switches with different selectors.
- Families that cannot be represented by stored cases.
- Unsupported operations, invalid shapes/bases/indices, or arithmetic overflow.
- Exact/Large monomials remaining in the final residual.

Full protection against malicious inputs for CyclicGraphDependency may be deferred. However, retain a DFS visiting state capable of detecting direct cycles caused by human error in honest protocols. Do not impose an unjustified owned-element budget.

If slot-transfer Tall encoding is covered by a separate specification and remains unimplemented, explicitly return `Unimplemented` for the corresponding gate. Do not generate that gate in Tall integration fixture configurations that can reconstruct using cyclic rotation alone.

## 10. Determinism, termination, and confluence

### 10.1 Determinism

- Use ordered containers or explicit sorting for map/set iteration.
- Specify comparison keys for atoms, monomials, and cases; do not use node insertion order.
- An applicable relation must be unique across the entire full match key in 6.2, or it is an error. Registrations may share the same source identity while differing in ordered coordinates, public/target/layout/trapdoor/selector provenance, or similar fields.
- The same input bundle and parameter environment must produce a byte-for-byte identical diagnostic order.

### 10.2 Termination

- Memoize each expression DAG node once, bottom-up.
- Add/Multiply flattening visits DAG edges a finite number of times.
- Relations are one-way, and every recursive step strictly decreases the producer-DAG rank multiset lexicographically.
- Switch scope minimization only moves common prefixes/suffixes outside, with no rule to move them back inside.
- Bounded aggregation only reduces terms and never re-expands them.

### 10.3 confluence

Do not rely on confluence of arbitrary rewrite orders. Construct the canonical result directly through a fixed phase order, ordered maps, and relation application from the left. Implementation tests must verify that different Add/Multiply association trees, input node insertion orders, and thread schedules produce the same `PolynomialNF`.

## 11. Complexity, memory, and parallelism

Let `N` be the number of reachable DAG nodes, `E` the number of edges, `T` the number of exact monomials actually generated, `F` the number of stored family cases, `L` the total `PolynomialNF` size of reachable stored cases, and `G` the total NF size actually generated by relation-target normalization.

- Source/bound resolution: `O(N + E)`.
- Add flattening and canonical insertion: `O(T log T)`.
- Multiply: Proportional to the number of output exact monomials. Full expansion can increase `T`, but bounded-only terms are aggregated into one term immediately after each operation.
- Relation lookup: An ordered-map lookup by full match key, `O(log R)`, for each factor boundary. Total time including target processing is proportional to the lookup count and generated size `G`, not the raw logical family size.
- Switch/family: `O(L)` in the total size `L` of stored case NFs actually read, not just the case count. Selector Cartesian products are prohibited.
- Family maximum: Traverse reachable stored cases once. Reuse existing DAG memo results rather than rebuilding the NF at each use.

Do not create a new persistent cache database. Reuse node memoization, interned atom identities, and stored family cases within one simulation job. Limit memo lifetimes to that job.

Bound calculations for independent nodes/cases may run in parallel with rayon. Merge each worker's local results by deterministic keys to obtain the same ordered result. Use existing configurable batch sizes rather than unconditionally parallelizing small loops and increasing overhead. Peak memory must scale with the worker count and active batch, not the total protocol loop count.

## 12. Staged migration

### Stage 0: Establish the evidence baseline

- Record the exact source/bound/relation chains of Tall, Diamond WE, and noiseless fixtures in a ledger.
- Save the current checker's first Large witness, time, peak RAM, and relation count.
- Verify the meaning of existing fields such as `input_max_plaintext_norm_ranges` against producer code and runtime uses.

### Stage 1: Egg-independent identity/bound resolver

- Extract existing owner-aware identity and bound resolvers into a pure API.
- Add differential tests between egg and the new API.

### Stage 2: PolynomialNF builder

- Implement zero, Add/Negate, ordered Multiply, central scalars, and bounded aggregation.
- Relations and Switches still fail closed.

### Stage 3: exact relation

- Implement the one-way `B*K=P` rule, prefixes/suffixes, multiple boundaries, trapdoor inputs, and runtime coordinates.

### Stage 4: Switch and family

- Implement scope minimization, same-selector case-wise processing, and stored family cases.

### Stage 5: Connect the checker

- Connect final-bound and threshold checks to the new pipeline.
- Restrict the differential mode that runs alongside the egg checker to tests only.

### Stage 6: Remove egg

- Remove the egg language, runner, rewrites, extractor, preferences, and final-leaf filter.
- Remove the Cargo dependency and dead diagnostics.
- Remove the old Lean checker and its dedicated source-hash check.

Make each stage a separate commit and obtain focused test results and reviewer acceptance before proceeding to the next stage.

## 13. acceptance gates

### 13.1 focused tests

- `0*Large=0`, nonzero `*Large=Large`.
- Identical cancellation under every Add/Negate association order.
- Ordered product flattening. Swapping factors does not yield cancellation.
- Bounded-only aggregation after every operation, with consistent upper bounds no smaller than the unaggregated calculation.
- `prefix*B*K*suffix -> prefix*P*suffix`.
- Preserve `E_B*K`, `K*B`, and cases with wrong public matrices, coordinates, or trapdoors.
- Multiple relation boundaries in one monomial.
- Nested relations with a decreasing producer-DAG rank multiset succeed; same-rank cycles and rank increases are rejected.
- Zero full-match-key matches are inapplicable; same-target duplicates are deduplicated; multiple different targets are rejected as ambiguous.
- Relations whose trapdoor is a protocol input.
- Common Switch prefixes/suffixes, common Add terms, and `Switch(cases*G)->Switch(cases)*G`.
- Same-selector success and different-selector fail-closed behavior.
- Family static/dynamic boundaries, `U<=count` success, and `U>count` rejection.
- Every source-contract path and missing contracts.
- MatrixExact `U-1`, with and without constant metadata.
- Regular Gadget as Large and decomposition digits as bounded.
- Pack with general/known-zero bits.
- CRT zero coefficients.
- Transpose distribution over Add/Negate/products, reversal of product factors, and double Transpose elimination.
- Concat distributes only over aligned pointwise Adds and preserves a canonical structural factor when axis, arity, shape, or corresponding positions differ.
- Focused tests for exact identity, shape, and bounds for every operation-table row.
- First-Large witnesses independent of insertion order, hash seed, and rayon thread count.

### 13.2 differential tests

- The same final bound on existing fixtures that the egg version completes correctly.
- On fixtures where the egg version reselects the raw left-hand side, the new version generates the canonical target only once.
- Identical NF under different node insertion orders, Add/Multiply associations, and rayon thread counts.
- Record the specification rule and witness when the new version accepts a case that the egg version considers unsupported.

### 13.3 noiseless runtime gate

Use small nested-RNS parameters and set all additive noise to 0. At runtime, compare the Tall output encoding exactly against the expected plaintext product computed independently of the checker, and require a residual of exactly 0. Checker success alone is not a substitute.

### 13.4 Tall gate

- Use the existing Tall integration test and control how far benchmark estimation proceeds through environment variables. Do not add a new integration-test target.
- Preserve the agreed parameters, with nested-RNS scale `32`.
- Require noise simulation to succeed through completion.
- The subsequent benchmark estimate may terminate because it runs out of VRAM.
- Emit progress logs at a bounded cadence containing phase, processed/total nodes, exact term count, bounded aggregation count, relations remaining/applied, Switch cases processed, and elapsed time.
- Do not branch on node numbers or Tall-specific names except in temporary diagnostics.

### 13.5 Diamond WE gate

Diamond WE/iO also uses the Rust checker. The existing Diamond WE integration path must leave no exact signal and complete finite-bound and strict-threshold checks. While the iO crate is explicitly disabled, record only the compile gate without changing its disabled state.

### 13.6 noisy Runpod gate

Run noiseless/noisy integration requiring GPUs on Runpod. The fixed conditions for noisy Tall are `MXX_TALL_NESTED_RNS_SECURITY_BITS=0`, `MXX_TALL_NESTED_RNS_MIN_LOG_RING_DIMENSION=3`, `MXX_TALL_NESTED_RNS_MAX_LOG_RING_DIMENSION=3`, and `MXX_TALL_NESTED_RNS_ERROR_SIGMA>0`; up to four RTX 5090 GPUs may be used. Save the execution commit, all environment variables, GPU count, start/end times, peak VRAM/RAM, and complete logs. Do not run Tall and Diamond simultaneously.

Completed noise simulation and satisfaction of the strict threshold are prerequisites for hardware testing; they must not themselves count as final acceptance. Fix the parameters selected by that simulation and run three consecutive actual Tall runtime roundtrips with `error_sigma > 0`, using the same commit and parameters. Accept only if decoded/runtime output matches the expected value in all three runs.

If a hardware roundtrip fails, first escalate `crt_depth` as permitted by the specification, pass simulation and the strict threshold again, and then repeat hardware testing. If it still fails, or runtime noise exceeds the checker bound, perform an underestimation audit before relaxing parameters further: check for missing terms, incorrect constant-polynomial factors, inner-sum counts, ring-dimension factors, and CRT coefficients. Benchmark OOM is allowed only after noise simulation completes.

### 13.7 Final threshold

Only for a finite residual bound `B`, check the following strict inequality.

```text
2 * plaintext_modulus * B < ciphertext_modulus
```

Reject equality. Reject Large, contract errors, and uncanceled exact terms before calculating the threshold.

## 14. Complexity and simplicity audit

Audit every introduction of three new concepts or data structures. The audit table must include the following.

1. The general problem being solved.
2. Why existing structures cannot be reused.
3. Correctness benefits.
4. Changes in time and memory complexity.
5. Old code that can be removed.
6. The concrete need in Tall/Diamond.
7. Evidence for rejecting simpler alternatives.

Reject changes in the following cases.

- They depend on protocol names, node numbers, or fixture values.
- A new cache/database duplicates existing memoization or symbol tables.
- The same relation/bound decision is implemented in two places.
- They increase traversal counts or peak memory without improving correctness.
- The reason for failing closed cannot be expressed through typed errors.
- They work only by trying multiple candidates in sequence, with no definable canonical rule.

Append past optimizations and their results to `docs/correctness/exact-signal-large-debugging-history.md`, and check whether the same idea has already been tried before changing the specification. After the new specification's implementation passes the acceptance gates, leave no migration-only egg compatibility code or temporary diagnostics behind.
