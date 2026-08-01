# Symbolic IR

`mxx-ir-symbolic` derives a non-executable symbolic view from a validated
`mxx-ir-core` graph. The executable graph remains the single runtime DAG.

## Expression arena

Every matrix wire points to a typed `SymbolicExprId` in an append-only,
hash-consed arena. The arena supports:

- typed zero and exact source atoms;
- canonical addition, integer scaling, and ordered multiplication;
- tensor product, row/column/diagonal concatenation, and selection;
- transpose, slice, reshape, and constant-coefficient extraction;
- CRT recomposition;

Local canonicalization flattens nested additions and products, combines integer
coefficients, removes exact zeros, cancels equal opposite summands, and removes
type-compatible identity matrices from products. It does not distribute an
unrelated product of sums. This keeps expressions compact while preserving the
matrix structure needed by BGG+ graphs.

`Select` stores the existing origin-aware `SelectionDomainRef`. A simulator can
therefore evaluate every occurrence of the same domain under one correlated
branch assignment instead of treating selections independently.

## Assumptions

`Mat::assume` is the only public symbolic reinterpretation operation. Ordinary
arithmetic involving a `VirtualMat` creates a small pending symbolic expression
and attaches it to an existing executable value:

```rust
let s = VirtualMat::bounded("s", s_type, secret_metadata);
let e = VirtualMat::bounded("e", e_type, error_metadata);
let c = c.assume(s * a + e)?;
```

The pending expression is evaluated and interned during elaboration. `assume`
creates no executable node. Cloning a virtual matrix preserves its identity;
constructing another virtual matrix creates another identity even when its
diagnostic name and type are equal.

## Exact identities and preimages

Source atoms carry scoped identities through subgraph calls, parallel
iterations, and artifact manifests. Recomputing a numerically equal value does
not reproduce an atom identity. Cross-production equality is obtained by
importing the originating symbolic manifest.

A preimage relation is stored separately as the exact identities of `B` and
`K`, together with the expression for `B K`. For

```text
c = s B + e
B K = S' P + E
```

the targeted rewriter derives

```text
c K = s S' P + s E + e K.
```

Only the distribution needed to expose a known preimage pair or aligned block
product is performed. In particular,

```text
concat_columns(L1, ..., Lk) * concat_rows(R1, ..., Rk)
```

rewrites to the sum of the aligned products when all block dimensions match.
No aggregate label or derived atom is introduced.

## Structural operations

Concatenation is always retained as a typed structural expression; arbitrary
row and column inputs are not rejected because they lack a common
factorization. Tensor is retained as an exact bilinear node. The noise
simulator enumerates its bounded alternatives lazily.

Generic modulus-down and modulus-up expressions are intentionally absent. The
nested-RNS circuit gadgets express their own level-switching dataflow directly
and do not require a generic symbolic conversion node.

## Manifests

A symbolic manifest contains qualified atoms, arena records, artifact roots,
preimage relations, selection domains, and the assumption digest. Import
replays every expression through the canonical constructors and preserves the
originating production identity. The current format version is checked
exactly, and import validates expression topology, artifact root types, and
relation atom references before publishing the imported state. Only the
current representation is accepted; there is no legacy reader or compatibility
alias.

## Noise analysis boundary

Symbolic IR stores source parameters and declared metadata but performs no
numerical bound calculation. `mxx-noise-simulator` recursively consumes the
arena, preserves signal and noise as separate results, and applies the existing
`PolyNorm` and `PolyMatrixNorm` rules term by term.
