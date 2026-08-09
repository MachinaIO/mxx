# Operational Affine and Noise Reuse Audit

## Purpose

This document records which parts of the operational affine/noise checker are reused from
earlier implementations and which parts are intentionally new. It is also a guardrail against
reintroducing the retired recursive symbolic analyzer.

The approved representation is a flat sum of ordered products:

```text
OperationalTerm {
    coefficient: Int,
    product: OperationalProductKey { factors: [...] },
}
```

The integer coefficient is only the signed additive multiplicity. Every multiplicative value,
including a matrix, polynomial scalar, dynamic message bit, bounded secret, or Large public
matrix, is an ordered factor. There is no separately stored matrix-valued coefficient or carrier.

Factor roles classify products as follows:

- zero Large factors: bounded/noise term;
- one or more Large factors: signal term.

Multiple Large factors do not introduce an opaque third category. They remain signal factors
until an exact checked relation simplifies them, or until the endpoint handles the resulting
signal product. The design has no `opaqueFinite` or `notSmall` fallback.

Only structurally identical ordered factor lists may merge. Names, likely runtime equality, and
equality modulo the ring are not guessed. Ordinary uncompressed signal expressions distribute
over addition. An analysis-only bounded summary is never reopened after compression because its
individual origins have deliberately been discarded.

## Source snapshots

This audit uses the following historical snapshots as implementation references:

- `7aaf935f`, especially `lean/Mxx/Certificate/AffineNormalize.lean`,
  `lean/Mxx/Certificate/Typing.lean`,
  `lean/Mxx/Certificate/Rules/MatrixAffine.lean`,
  `lean/Mxx/Certificate/Rules/MatrixSelect.lean`, and the positional relation code in
  `lean/Mxx/Certificate/Analyzer.lean`;
- `48117fdb`, especially the former `src/simulator/poly_norm.rs` and
  `src/simulator/poly_matrix_norm.rs` deterministic metadata and worst-case branches;
- the current `lean/Mxx/Certificate/OperationalBounds.lean` numeric sequential-recurrence
  evaluator.

These commits are references, not dependencies. Historical files that no longer exist in the
current tree must not be restored as modules merely to reuse a small rule.

## Reuse classification

### Verbatim reuse

The following current recurrence machinery is retained in place rather than reimplemented:

- `OperationalBoundExpr.previous` as a typed reference to the prior numeric state;
- `OperationalBoundExpr.recurrence` as the compact recurrence result;
- `evaluateTransition`, which evaluates every transition component against the same previous
  state before producing the next state;
- `repeatTransition`, which iterates only that fixed-size numeric state.

This is the current implementation in `lean/Mxx/Certificate/OperationalBounds.lean`. Phase A
continues to analyze a sequential-loop body once. Phase B evaluates the fixed-size transition for
the concrete iteration count. It does not symbolically unroll matrix expressions.

No historical recursive analyzer module is reused verbatim.

### Minimal adaptation

The following algorithms are reused with only the representation and hard-bound domain adapted.

#### Exact merge and cancellation

Historical source: `normalizeTerm`, `mergeEntries`, `insertTerm`, `normalizeTerms`, and
`normalizeAffineForm` in `7aaf935f:lean/Mxx/Certificate/AffineNormalize.lean`.

Reused rule:

1. canonicalize the signed scalar coefficient;
2. compare complete normalized keys;
3. add coefficients only for identical keys;
4. delete a term only when the resulting integer coefficient is exactly zero.

The adaptation replaces the old coefficient/basis pair with one ordered factor-list key. It is
therefore stricter than the old basis-oriented merge: no matrix expression is moved into a
distinguished coefficient position, and no semantic equality is inferred.

#### Matrix product typing and deterministic product bound

Historical sources:

- `inferMatrixProductType` and `MatrixProductType` in
  `7aaf935f:lean/Mxx/Certificate/Typing.lean`;
- `productBound`, `multiplyAffineRight`, `multiplyAffineLeft`, and the multiplication cases in
  `7aaf935f:lean/Mxx/Certificate/Rules/MatrixAffine.lean`;
- the worst-case multiplication branch in
  `48117fdb:src/simulator/poly_matrix_norm.rs`.

Reused rule:

- infer ordinary matrix multiplication and the supported polynomial-scalar broadcast modes from
  the checked matrix types;
- multiply bounds in factor order;
- for an ordinary matrix product, use the effective inner dimension after subtracting known zero
  rows from the right factor;
- use the deterministic contraction factor from the old worst-case path, never its CLT branch;
- clear output zero-row knowledge unless an explicit deterministic transform proves it.

The current flat implementation owns its small product-mode type locally. It must not import the
retired typing/analyzer closure merely to obtain this calculation.

#### Preimage and decomposition rewrite

Historical source: `matchingRelationTarget`, `expandSignalThroughTarget`,
`rewriteAffinePreimageProduct`, and `rewritePreimageProduct?` in
`7aaf935f:lean/Mxx/Certificate/Analyzer.lean`.

Reused rule: apply a checked relation at an exact adjacent position in an ordered product. For
example, a checked relation

```text
B * K = S' * P + E  (mod R_q)
```

rewrites

```text
[prefix, B, K, suffix]
```

to the two products

```text
[prefix, S', P, suffix]
[prefix, E, suffix]
```

The relation owner, source and target identities, types, modulus, ring dimension, and layout must
match exactly. The adaptation is positional and flat; it does not reuse the old analyzer's graph
search, recursive expressions, or arena rewriting.

#### `isConstantPolynomial` and `knownZeroRows`

Historical sources: `is_const_poly` in `48117fdb:src/simulator/poly_norm.rs` and `zero_rows` in
`48117fdb:src/simulator/poly_matrix_norm.rs`.

Reused deterministic rules:

- a product is constant-polynomial only when all factors are constant-polynomial;
- a sum is constant-polynomial only when all summands are constant-polynomial;
- right-factor known zero rows reduce the effective inner dimension of an ordinary product;
- general addition clears known-zero-row information;
- a transform preserves or changes known-zero-row information only through its explicit rule.

The duplicate historical fields `is_const_poly` and `is_constant_poly` are represented by one
field, `isConstantPolynomial`.

#### Bounded addition

Historical source: `PolyNorm::add`, `PolyMatrixNorm::add`, and their assignment variants in
`48117fdb`.

Reused rule: use the deterministic triangle inequality. For signed multiplicities
`alpha_i`, the compressed noise bound is

```text
sum_i abs(alpha_i) * bound(E_i).
```

Addition clears `knownZeroRows` and preserves `isConstantPolynomial` only when every term is
constant-polynomial. Dependency sets and CLT readiness are deliberately not carried over.

#### Dynamic selection

Historical source: `maximumBounds` and `deriveMatrixSelect` in
`7aaf935f:lean/Mxx/Certificate/Rules/MatrixSelect.lean`.

Reused rule: a dynamic select uses the maximum hard bound across all possible branches, not their
sum. The exact-one selection identity remains protected until the selection rule consumes it.
Selection never licenses cancellation between different branch origins.

### New logic

The following pieces are new because the old representation cannot provide them directly:

- `OperationalFactorLeaf`, `OperationalFactorKey`, `OperationalProductKey`, and
  `OperationalTerm` as the single flat product-sum representation;
- classification by the number of Large factors, with every product containing at least one
  Large factor retained as signal;
- maximal consecutive bounded-run compression without changing factor order;
- compression of a bounded-only sum to one bounded noise summary;
- the signed-content/GCD normal form for bounded noise sums;
- typed, length-delimited provenance tokens that distinguish bounded-product compression from
  bounded-noise-sum compression;
- compression protection for relation owners, decomposition owners, exact-one indicators,
  endpoint identities, Large factors, and origin-preserving artifact identities;
- the normalization fixed point after relation application and bounded compression;
- a fail-closed term/factor cap after normalization;
- propagation of flat summaries through every currently supported IR primitive and into the
  generic checker report.
- checked exact Boolean-carrier grouping for the direct
  `Boolean input -> BoolToInt -> Select(zero, constant carrier)` executable chain. The Rust
  attachment only names the four frozen wires; Lean validates the complete chain before assigning
  the Large signal role.

These are analysis-only structures. They are not executable IR nodes, serialized user-visible
operations, or a replacement expression DAG. In particular, bounded compression is not `fold`
under another name.

## Normative normalization order

The checker must use this order:

1. canonicalize scalar integer coefficients;
2. merge exactly identical ordered factor lists;
3. remove exact zero coefficients;
4. apply checked preimage and decomposition relations;
5. compress maximal eligible consecutive bounded runs;
6. repeat exact merge and zero removal to a fixed point;
7. compress bounded-only sums, using the signed-content/GCD normal form;
8. put only commutative bounded subsets into canonical order; never reorder matrix factors;
9. sort top-level terms deterministically;
10. reject only if the normalized term/factor cap is exceeded.

Relation owners and other protected identities remain uncompressed until their consuming rule has
run. This order is required so compression cannot erase information needed for relation matching,
exact cancellation, selection, or endpoint checking.

## Prohibited reuse and imports

The operational flat checker must not import, restore, or recreate any of the following retired
machinery:

- `Mxx.Certificate.Analyzer` or an equivalent graph-wide analyzer;
- `ExpressionArena` or another global recursive expression arena;
- whole-arena substitution or rewriting;
- graph search that reconstructs operand, relation, family, or recurrence identity;
- recursive `MatrixExpr` inference as a second compiler;
- symbolic sequential-loop unrolling;
- caller-supplied loop invariants;
- CLT, square-root concentration, probabilistic independence, dependency-set, or `clt_ready`
  heuristics;
- a Rust mirror of Lean bound or recurrence evaluation;
- `opaqueFinite`, `notSmall`, or another category that hides unresolved Large factors.

Generated derivations may point to operands and checked relations, but they are untrusted. Lean
must validate node order, operand identity, relation identity, types, and all bound transitions.
A bad hint may reduce precision or be rejected; it cannot establish a bound by assertion.

## Review checklist

Before the migration is declared complete, review must establish all of the following:

- flat ordered factors preserve the meaning required by the old matrix-affine operations;
- `s * A - s * A` cancels only through exact structural factor-list equality;
- products containing multiple Large factors remain signal and distribute normally;
- bounded compression never runs before a relation, selection, cancellation, endpoint, or artifact
  identity is consumed;
- deterministic hard bounds match the old worst-case formulas, including product mode,
  `isConstantPolynomial`, and `knownZeroRows` behavior;
- dynamic selection uses a branch maximum;
- bounded factor and noise-term counts do not grow with sequential-loop iteration count;
- Phase A analyzes the loop body once and Phase B updates only fixed-size numeric state;
- no prohibited retired import or equivalent replacement exists;
- each reuse site is traceable to the source listed in this audit and each genuinely new rule has a
  focused positive and negative fixture.
