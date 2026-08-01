# Noise simulator

`mxx-noise-simulator` evaluates the typed expression arena produced by
`mxx-ir-symbolic`. It owns all numerical magnitude calculations; neither the
core graph nor symbolic elaboration calculates a bound.

The public report exposes signal presence and bounded noise separately. A
Large source contributes signal, while Gaussian, preimage, decomposition, and
declared bounded sources use the established `PolyNorm` and `PolyMatrixNorm`
rules. These include the 6.5-sigma envelope, balanced digit second moment,
zero-row metadata, stable dependency sets, dependency-aware CLT eligibility,
and conservative addition.

Addition and multiplication are evaluated as lazy alternatives over the
symbolic DAG. The simulator consumes each complete bounded product immediately
instead of materializing a global sum of products. This preserves factor order,
effective inner dimensions, and dependency decisions.

Selections sharing a `SelectionDomainRef` are evaluated under one branch
assignment for the whole surrounding expression and are joined by maximum
bound. Tensor uses polynomial multiplication with no matrix inner-dimension
summation. Concatenation, reshape, and coefficient extraction are structural
aggregation boundaries.

The current generic IR has no modulus-down or modulus-up expression. Nested-RNS
level switching is analyzed with its owning circuit gadget rather than through
a generic simulator rule.
