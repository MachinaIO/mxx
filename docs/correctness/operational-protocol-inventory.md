# Operational Protocol Inventory

## Purpose

This is the normative inventory for the execution-aligned operational checker. Every executable
Rust IR variant and every nested operation variant has one row below. Rust-to-Lean emission is in
`crates/correctness/src/emit_lean.rs`; the exhaustive Lean transfer classification and the actual
transfer functions are in `lean/Mxx/Certificate/OperationalBounds.lean`.

The compiler-enforced coverage boundary has three parts:

1. Rust emission matches `NodeKind` exhaustively and has no catch-all arm.
2. Lean `operationalTransferClass` matches every `NodeKind` exhaustively.
3. Nested enums are matched by the exhaustive `classifyIntBinary`, `classifyIntCompare`,
   `classifyRealBinary`, `classifyConcatAxis`, `classifyHashVariant`, and
   `classifyLoopInputMode` functions.

Adding a constructor to any of those enums therefore breaks compilation until an explicit
operational-checker decision is made. Structural nodes are evaluated by `evaluateCheckedScope`;
ordinary scalar and matrix nodes are evaluated by `genericNodeFact`. A valid emitted node must
reach one of those paths or a documented normal rejection, never an implicit default fact.

All matrix equalities used by relations are equalities in `R_q`, not integer equalities. `B` is
the shared source public matrix in a preimage relation: branch-specific targets and preimages obey
`B * K_d = P_d (mod R_q)` and do not create branch-specific copies of `B`.

## Scalar and control operations

| Variant | Inputs | Exact operational result | Bound or interval | Identity/relation handling | Normal rejection | Source |
|---|---|---|---|---|---|---|
| `Input` | Protocol-bound input contract | Contract fact with root protocol identity | Declared interval, cutoff, or conservative matrix cap | Preserves protocol input/family/artifact identity | Missing or mismatched contract | Current contract path |
| `ConstantInt` | none | Exact integer | `[v,v]` | Fresh executable scalar origin | Non-integer output or operands present | Current IR semantics |
| `EvaluateInt` | none | Parameter expression | Exact minimum/maximum over declared loop domains | Parameter provenance remains contextual | Non-integer output or unevaluable expression | Current IR semantics |
| `ConstantReal` | none | Exact real syntax | No noise bound | No matrix identity | Non-real output or operands present | Current IR semantics |
| `ConstantBool` | none | Exact Boolean | Boolean fact | No matrix identity | Non-Boolean output or operands present | Current IR semantics |
| `BoolToInt` | Boolean | Executable conversion | `[0,1]` | No matrix identity | Wrong type or arity | Current IR semantics |
| `IntToReal` | integer | Executable conversion | Real fact | No matrix identity | Wrong type or arity | Current IR semantics |
| `IntBinary.Add` | two integers | `x + y` | `[lx+ly, ux+uy]` | Scalar origin from node | Wrong type/arity | Historical scalar interval rule |
| `IntBinary.Subtract` | two integers | `x - y` | `[lx-uy, ux-ly]` | Scalar origin from node | Wrong type/arity | Historical scalar interval rule |
| `IntBinary.Multiply` | two integers | `x * y` | Min/max of all four endpoint products | Scalar origin from node | Wrong type/arity | Historical scalar interval rule |
| `IntBinary.Divide` | two integers | Rust/Lean integer division | Endpoint envelope when divisor interval excludes zero | Scalar origin from node | Possible zero divisor | Current IR evaluator |
| `IntBinary.Remainder` | two integers | Rust/Lean remainder | Conservative signed endpoint envelope | Scalar origin from node | Possible zero divisor | Current IR evaluator |
| `IntCompare.Equal` | two integers | Exact comparison expression | Boolean fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `IntCompare.Less` | two integers | Exact comparison expression | Boolean fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `IntCompare.LessEqual` | two integers | Exact comparison expression | Boolean fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `BitExtract` | integer | Exact bit expression | Boolean fact | No matrix identity | Negative position, wrong type/arity | Current IR evaluator |
| `RealBinary.Add` | two reals | Exact real expression | Real fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `RealBinary.Subtract` | two reals | Exact real expression | Real fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `RealBinary.Multiply` | two reals | Exact real expression | Real fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `RealBinary.Divide` | two reals | Exact real expression | Real fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `RealSqrt` | real | Exact real expression | Real fact | No matrix identity | Wrong type/arity | Current IR evaluator |
| `ExtractCoefficient` | scalar matrix | Exact canonical coefficient | `[0, canonical_upper-1]` | Scalar result does not inherit matrix relation | Invalid position/type/arity | Current IR evaluator |
| `LiftIntegerToConstantPolynomial` | integer and scalar matrix type | Bounded constant-polynomial factor | Maximum absolute integer bound; constant-polynomial metadata | Fresh output identity; no matrix relation is inferred | Non-scalar type, nonpositive modulus/ring dimension, or wrong arity | Explicit fail-closed transform rule |

## Deterministic matrix leaves and samplers

| Variant | Inputs | Exact operational terms | Hard bound and metadata | Identity/relation handling | Normal rejection | Source |
|---|---|---|---|---|---|---|
| `ConstantMatrix.Zero` | none | Empty polynomial | `0`; constant polynomial | Exact zero has no Large factor | Type/shape mismatch | Deterministic leaf rule |
| `ConstantMatrix.Identity` | none | One bounded identity factor | `1`; constant polynomial | Structural identity factor | Type/shape mismatch | Deterministic leaf rule |
| `ConstantMatrix.UnitRow` | none | One bounded exact factor | `1`; constant polynomial | Exact origin retained | Invalid index/type | Deterministic leaf rule |
| `ConstantMatrix.UnitColumn` | none | One bounded exact factor | `1`; constant polynomial | Exact origin retained | Invalid index/type | Deterministic leaf rule |
| `ConstantMatrix.Gadget` | none | One Large public factor | Centered cap | Gadget identity from explicit layout | Missing/mismatched layout | Historical gadget rule |
| `ConstantMatrix.Gadget(small)` | none | One Large public factor | Centered cap | Small-gadget identity from explicit layout | Missing/mismatched layout | Historical gadget rule |
| `ConstantMatrix.PowerOfBase` | none | One Large exact factor | Centered cap | Exact node origin | Invalid base/type | Deterministic leaf rule |
| `ConstantMatrix.Rotation` | none | One bounded exact factor | `1` | Exact node origin | Invalid exponent/type | Deterministic leaf rule |
| `ConstantMatrix.Polynomial` | none | One bounded exact factor, or zero | Maximum absolute coefficient; constant-polynomial derived from positions | Exact node origin | Unevaluable coefficient/type | Deterministic leaf rule |
| `GadgetTrapdoor` | none | Port 0 is one Large gadget-public factor; port 1 is a trapdoor fact | Public uses centered cap; trapdoor uses explicit base bound | Both ports share one public identity | Missing/mismatched layout/type/output | Historical gadget rule |
| `TrapdoorSample` | none | Port 0 is one Large sampled public factor; port 1 is a trapdoor fact | Public uses centered cap; trapdoor uses explicit nonnegative cutoff | Both ports share one sampled public identity | Invalid cutoff/type/output | Sampler contract |
| `TrapdoorPublic` | trapdoor | One Large public factor | Centered cap | Recovers the trapdoor's exact public identity | Missing trapdoor identity/type | Sampler contract |
| `UniformResidueSample` | none | One Large sampled factor | Centered cap | Fresh sampled origin | Invalid matrix parameters | Sampler contract |
| `UniformIntervalSample` | none | One bounded sampled factor | `max(abs(min),abs(max))`; canonical range only when nonnegative | Fresh sampled origin | Invalid interval/type | Historical deterministic support rule |
| `GaussianSample` | none | One bounded sampled factor | Explicit nonnegative cutoff | Fresh sampled origin | Negative cutoff/type | Sampler contract |
| `HashSample.Plain` | key and ordered integer tags | One Large deterministic factor | Centered cap | Complete hash-query identity | Bad key/tag/type | Historical hash identity rule |
| `HashSample.Decomposed` | key and ordered integer tags | One bounded decomposition factor | Gadget decomposition hard bound | Relation snapshot points to the matching plain hash query | Missing base/count/layout or bad query/type | Historical decomposition rule |
| `HashSample.SmallDecomposed` | key and ordered integer tags | One bounded decomposition factor | Small decomposition hard bound | Relation snapshot points to the matching plain hash query | Missing base/count/layout or bad query/type | Historical decomposition rule |
| `PreimageSample` | public matrix, matching trapdoor, target | One bounded preimage factor | Explicit nonnegative cutoff | Owns `B*K=target (mod R_q)`; `B` identity must match trapdoor | Wrong arity/identity/type/cutoff | Sampler contract and historical rewrite |
| `GadgetDecompose(regular)` | target matrix | One bounded decomposition factor | Regular gadget-decomposition bound | Owns `G*D=target (mod R_q)` with target snapshot | Missing/mismatched layout/base/count/type | Historical decomposition rule |
| `GadgetDecompose(small)` | target matrix | One bounded decomposition factor | Small gadget-decomposition bound | Owns `G_small*D=target (mod R_q)` | Missing/mismatched layout/base/count/type | Historical decomposition rule |

## Matrix arithmetic and transforms

| Variant | Inputs | Exact operational terms | Hard bound and metadata | Identity/relation handling | Normal rejection | Source |
|---|---|---|---|---|---|---|
| `MatrixBinary.Add` | two same-type matrices | Concatenate, merge identical products, compress bounded sum | Triangle inequality; constant only if both constant; zero rows cleared | Relations remain only on untouched bare factors | Type/arity mismatch or analysis cap | Historical affine normalization |
| `MatrixBinary.Subtract` | two same-type matrices | Negate right coefficients, merge, cancel, compress | Triangle inequality after exact cancellation | Structural identity controls cancellation | Type/arity mismatch or analysis cap | Historical affine normalization |
| `MatrixBinary.Multiply` | two compatible matrices | Distributive ordered-product cross product | Deterministic `productBound`; right zero rows reduce contraction | Exhaustive adjacent preimage/decomposition rewrite before compression | Type/arity/mode mismatch or analysis cap | Historical matrix affine/product rule |
| `MatrixNegate` | matrix | Negate additive coefficients | Absolute hard bound unchanged | Complete factor identity retained | Type/arity mismatch | Historical affine normalization |
| `MatrixScale` | matrix and static scalar expression | Multiply additive coefficients; nonuniform dynamic value becomes a factor | Absolute scalar times bound | Identity retained for uniform static scale | Unevaluable scalar/type/arity | Historical affine normalization |
| `Transpose` | matrix | Reverse factor order and transpose every factor | Recomputed typed product bound/metadata | Relations are not transported through transforms | Unsupported transformed type/arity | Historical transform rule |
| `Slice` | matrix | Row slice transforms left boundary; column slice transforms right boundary | Recomputed boundary-factor bound; canonical range retained | Relations are not transported through transforms | Invalid range/type/arity or unsupported mixed transform | Historical transform rule |
| `Tensor` | two matrices | Explicit distributive tensor product | Tensor-specific deterministic bound | Relation ownership is not guessed across tensor | Type/arity mismatch or unsupported typed tensor | Current evaluator plus historical bound arithmetic |
| `Concat.Rows` | one or more matrices | Embed each term's left boundary and combine | Bounded contributions use triangle inequality | Factor order and signal identities retained | Empty/type/shape mismatch or unsupported embedding | Historical transform rule |
| `Concat.Columns` | one or more matrices | Embed each term's right boundary and combine | Bounded contributions use triangle inequality | Factor order and signal identities retained | Empty/type/shape mismatch or unsupported embedding | Historical transform rule |
| `Concat.Diagonal` | one or more matrices | Embed both boundaries and combine | Bounded contributions use triangle inequality | Factor order and signal identities retained | Empty/type/shape mismatch or unsupported embedding | Historical transform rule |
| `ThresholdDecode(bool)` | scalar matrix | Boolean outputs from canonical coefficients | Output facts are Boolean; decoder obligation uses input noise | Does not transport matrix identity | Invalid p/q/count/type/output count | Current evaluator and generic threshold obligation |
| `ThresholdDecode(int)` | scalar matrix | Integer outputs from canonical coefficients | Each result in `[0,p-1]`; decoder obligation uses input noise | Does not transport matrix identity | Invalid p/q/count/type/output count | Current evaluator and generic threshold obligation |
| `CrtRecompose` | equal one-row matrices and positional CRT metadata | Weighted sum of input polynomials | Deterministic scaled-sum hard bound | Preserves exact factor provenance in each summand | Empty/mismatched metadata/type/modulus/coefficient | Current IR semantics |
| `PackPolynomialCoefficients` | exact Boolean family | One exact Large reconstructed residue polynomial | Centered cap; canonical residue range | Fresh exact output identity | Wrong family size/type/bit width/output shape | Current IR semantics |

## Families, selection, and nested execution

| Variant | Inputs | Exact operational result | Bound/metadata behavior | Identity/relation handling | Normal rejection | Source |
|---|---|---|---|---|---|---|
| `FamilyPack` | ordered values | Packed heterogeneous family | Per-element facts retained | Exact element identities retained | Count/type mismatch | Current structural evaluator |
| `FamilyGetStatic` | family | Exact indexed element | Exact selected bound | Preserves selected element relations/identity | Invalid index/family/type | Current structural evaluator |
| `FamilyGetDynamic` | family and integer index | Uniform-family selection or conservative packed join, conditional on successful runtime access | Branch maximum for supported families | Dynamic selection identity retained; ambiguous packed relation dropped | Empty family, wholly out-of-range interval, or unsupported heterogeneous join | Historical selection rule |
| `Select` | integer index and branches | Indicator-tagged branch terms or scalar join | Exact-one branch maximum for noise | Production/scope-namespaced selection domain | Out-of-range index/type/count mismatch | Historical selection rule |
| `SubgraphCall` | ordered arguments | Checked child outputs rebound to caller | Child transfer bounds | Call occurrence namespaces fresh identities; input origins transport exactly | Missing definition/binding/output/type | Current checked scope evaluator |
| `ParallelLoop` | loop arguments | Uniform family of one checked body template | Body analyzed once; numeric bound reused by lane | Binder and lane instantiation preserve/freshen identities as appropriate | Bad count/binding/mode/body schema | Current checked scope evaluator |
| `ParallelLoop.Broadcast` | invariant argument | Same fact in every lane | Same bound | Exact invariant identity retained | Mode/type mismatch | Current checked scope evaluator |
| `ParallelLoop.Zip` | uniform family | Lane element template | Per-element bound | Lane identity instantiated from binder | Count/type mismatch; packed heterogeneous family rejected | Current checked scope evaluator |
| `ParallelLoop.ZipOffset` | uniform family | Offset lane element template | Per-element bound | Lane identity instantiated with checked offset | Range/count/type mismatch | Current checked scope evaluator |
| `SequentialLoop` | carried values and invariants | Final carried facts from one abstract body transfer | Fixed-size numeric recurrence, simultaneous updates | Fixed recurrence-result identity replaces iteration history | Bad count/binding/schema, relation escape, nonuniform bound | Current recurrence evaluator |

## Generic acceptance

### Exact Boolean protocol carriers

The `mxx-correctness.protocol-boolean-signal-grouping` attachment is an untrusted structural hint
for the exact executable chain

```text
Boolean protocol input -> BoolToInt -> Select(zero, nonzero constant matrix)
```

Lean verifies every named wire, direct operand, node kind, matrix type, and the nonzero constant
carrier before representing the selected value as one exact Large signal factor. The attachment
contains no role or bound accepted by assertion. A wrong selector, zero branch, carrier, or selected
output is rejected as `invalidDerivationAttachment`. This lets Toy form and anchor the executable
residual `ciphertext - encoded_message`; the selected carrier cancels before the Gaussian term is
measured as noise.

The generic decoder obligation is exactly

```text
2 * plaintext_modulus * noise_bound < ciphertext_modulus
```

and is evaluated with exact integers. Equality at the boundary rejects. The report retains output
facts, obligations, acceptance, and a stable rejection reason. Protocol-specific search code may
parallelize independent candidate reports, but it must not duplicate or replace the bound formula.

This operational checker is a parameter-search and runtime-validation component. Its local and
end-to-end Lean soundness proofs are tracked separately; passing this checker alone is not yet a
proved end-to-end correctness theorem.
