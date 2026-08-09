# Gadget decomposition and gadget-trapdoor semantics

## 1. Status

This specification replaces the incomplete gadget-decomposition transport currently used by the
Lean IR. It is a prerequisite for universal operational noise analysis. It does not complete the
generic checker or the Diamond WE correctness theorem.

Backward compatibility is intentionally out of scope. The removed implicit digit-count form must
not remain as a deprecated constructor, reader, alias, or fallback.

## 2. Problem statement

The superseded executable Rust IR represented decomposition with:

```text
GadgetDecompose {
  base,
  small,
  digit_count: Option<IntExpr>,
}
```

The runtime uses `small` to choose between `decompose()` and `small_decompose()`. These operations
have different digit layouts and different hard digit bounds. The Lean emitter currently drops
`small`, rejects `digit_count = None`, and emits only `(matrix_type, base, digit_count)`. The Lean
sampler contract consequently describes one operation even though the runtime has two.

The optional count is also unsound as a cross-layer abstraction. Rust graph validation derives an
implicit count by repeatedly multiplying `base` until it reaches the full modulus. The polynomial
backend derives the regular count as the sum of the per-CRT-tower digit counts and the small count
from one CRT tower. These formulas are not generally identical. A graph can therefore validate and
then fail the backend layout check.

The same distinction affects `PreimageSample`. A sampled trapdoor invokes the probabilistic
preimage sampler. A `GadgetTrapdoor` invokes deterministic gadget decomposition. Lean values
currently retain only the trapdoor public matrix, so Lean execution cannot select the runtime
branch.

## 3. Goals

The revised representation must:

1. preserve the runtime `small` choice without inference;
2. make every decomposition digit count explicit;
3. use the same resolved `(base, digit_count, small)` tuple in validation, runtime execution,
   generated Lean, operational analysis, and local soundness statements;
4. model gadget decomposition as deterministic but partial on backend-invalid layouts;
5. use the mode-aware public gadget matrix selected by the same backend tuple and retain only the
   corresponding modular equation in `R_q`;
6. use the mode-specific hard digit bound;
7. distinguish sampled and gadget trapdoors in Lean execution; and
8. avoid a second Lean implementation of the backend's CRT-layout selection algorithm.

## 4. Non-goals

- Do not change the numeric implementation of `decompose()` or `small_decompose()`.
- Do not add an implicit-count resolver to Lean.
- Do not add a compatibility reader for serialized nodes with an absent count.
- Do not claim integer equality for a gadget equation.
- Do not add probabilistic or CLT bounds.
- Do not introduce a new crate or a second noise simulator.
- Do not generalize `GadgetTrapdoor` to the small mode. The current executable node is regular
  decomposition only.
- Do not change decomposed `HashSample` variants. They already carry their mode in `HashVariant`
  and are covered separately by the universal checker plan.

## 5. Normative terminology

For an input matrix `X` over `R_q`, positive base `b`, positive digit count `k`, and a backend-valid
layout, define:

```text
D_regular(X; b, k) = the existing runtime decompose() result
D_small(X; b, k)   = the existing runtime small_decompose() result
G_regular(b, k)    = the existing runtime regular public gadget matrix
G_small(b, k)      = the existing runtime small public gadget matrix
```

The two public matrices are not interchangeable. At CRT depth greater than one, regular mode uses
the tower-aware gadget layout while small mode uses the compact one-tower layout. Regular mode
satisfies its quotient-ring relation for every input in its valid layout:

```text
G_regular(b, k) * D_regular(X; b, k) = X  in R_q.
```

Small mode satisfies the corresponding relation only when every input coefficient's unsigned
canonical representative in `[0, q)` is strictly less than the smallest CRT modulus:

```text
max_canonical_coefficient(X) < min_crt_modulus
  implies
G_small(b, k) * D_small(X; b, k) = X  in R_q.
```

Small decomposition remains executable and deterministic without this premise, and its output
digit bound remains valid. Only the reconstruction relation is unavailable for an unrestricted
input. `max_canonical_coefficient` is an exact range property, not a noise norm: a stored or
centered value `-1` has canonical representative `q - 1` and therefore does not satisfy the premise
merely because its centered norm is one. The smallest CRT modulus is backend-owned metadata; Lean
must not reconstruct it from a CRT basis.

This statement does not imply equality of stored integer coefficient vectors.

The hard centered-coefficient bounds are:

```text
regular_digit_bound(b) = max(floor(|b| / 2), 1)
small_digit_bound(b)   = |b| - 1
```

The regular implementation uses balanced digits. The small implementation uses unsigned digits.
The general centered residue cap remains available separately; it must not be used to replace the
tighter digit bounds above.

The word `small` in this specification means the existing compact, unsigned decomposition. It does
not mean that its output is interchangeable with regular balanced decomposition.

## 6. Rust IR changes

### 6.1 Node shape

Change the node to:

```rust
GadgetDecompose {
    base: IntExpr,
    small: bool,
    digit_count: IntExpr,
}
```

`digit_count` is mandatory. Delete `Option<IntExpr>` and all `None` handling. All current DSL
construction sites already supply the count explicitly.

Do not replace `small` with a newly layered mode hierarchy. A Boolean exactly mirrors the current
runtime API and is sufficient for the closed two-mode operation.

### 6.2 Validation

For every `GadgetDecompose` node, graph validation must check:

1. exactly one matrix argument;
2. `base` evaluates successfully and is strictly greater than one;
3. `digit_count` evaluates successfully and is strictly positive;
4. output modulus, ring dimension, and column count equal the input values; and
5. output rows equal `input_rows * digit_count`, using checked multiplication.

Validation must not apply `abs()` to the declared base. A negative base is invalid because the
runtime passes the signed value to `validate_gadget_layout`. Apply the same signed-base rule to
`GadgetTrapdoor`; its current validator has the same `abs()` mismatch.

Delete the implicit `decomposition_digits` fallback if it has no remaining caller after this
change.

The backend retains the final check that the declared base and count match the concrete DCRT
parameters for the selected `small` mode. Lean does not regenerate a CRT basis or gadget matrix; it
only checks the elementary count arithmetic carried by the per-request descriptor. Every execution
path, including GPU estimation, must perform or reuse the backend check before calling
decomposition. Backend-invalid layouts are ordinary deterministic failure, not members of the
decomposition contract's domain.

### 6.3 Runtime execution

Runtime execution evaluates the explicit count once and passes the same tuple to layout validation:

```text
(input_type, base, digit_count, small)
```

After successful validation it calls exactly one of:

```text
small = false -> value.decompose()
small = true  -> value.small_decompose()
```

Remove output-row-ratio inference from the executor. The already validated output type is checked
for consistency but is not a source of missing metadata.

### 6.4 DSL and serialization

The public DSL continues to require `base` and `digit_count`. The existing `Mat::decompose`
operation emits `small = false`. Add the symmetric `Mat::small_decompose(base, digit_count)` DSL
operation, emitting the same node with `small = true`. Both methods share one private node-building
helper so their output-type construction cannot diverge.

Canonical serialization includes all three fields. Old serialized nodes without `digit_count`
are rejected rather than upgraded.

## 7. Lean IR changes

### 7.0 Per-request backend layout descriptors

Each operational-checker request carries one descriptor for every concrete DCRT parameter set used
by that candidate:

```lean
structure GadgetLayoutDescriptor where
  paramsId : SamplerParamsId
  ringDimension : Nat
  crtModuli : List Nat
  crtBits : Nat
  baseBits : Nat
  base : Int
  regularDigitCount : Nat
  smallDigitCount : Nat
  smallestCrtModulus : Nat
```

`SamplerParamsId` is the hash of the versioned canonical serialization of exactly
`(ringDimension, crtModuli, crtBits, baseBits)`. The CRT moduli are ordered exactly as in the
candidate's actual `DCRTPolyParams`; they are not regenerated by Lean. `base` is checked to equal
`2^baseBits`, `smallestCrtModulus` to equal `min(crtModuli)`, and the full modulus used by Lean
`SamplerParams` to equal `product(crtModuli)`. With `digitsPerTower = ceilDiv(crtBits, baseBits)`,
Lean checks `smallDigitCount = digitsPerTower` and
`regularDigitCount = digitsPerTower * crtModuli.length`. A descriptor matches a Lean
`SamplerParams` exactly when that modulus and ring dimension agree; row/column shape and sampler
cutoff remain node-local fields and are not part of the layout identity. The request rejects two
descriptors with the same `RingKey = (modulus, ringDimension)`, even if their `paramsId` values
differ, so Phase A resolution from evaluated matrix parameters is unique. These integer checks do
not reproduce basis generation or matrix construction.

The Rust request adapter derives descriptors directly from the same concrete `DCRTPolyParams`
instances used to construct the candidate runtime/compiler configuration. Callers cannot provide a
separate descriptor override. This requires extending the checker request beyond the current flat
17 scalar arguments; it does not require regenerating or rebuilding Lean source per candidate.

Descriptors are canonically sorted by `paramsId` and included in a versioned `requestHash` recorded
verbatim in the checker report and cache key. Generated-source, derivation, and protocol hashes
remain candidate-independent. Duplicate parameter IDs, empty CRT bases, nonpositive fields,
`base <= 1`, nonmatching products/minima/counts, or a node whose explicit tuple disagrees with the
unique matching descriptor are rejected with distinct diagnostics. There is no descriptor override
in a derivation instruction or external input contract.

The descriptor is request-scoped operational metadata, not a semantic matrix and not a proof
supplied by Rust. Phase A uses it to check layout membership and construct opaque symbolic
identities such as
`gadgetPublic(paramsId, size, small, digitCount)`; it never executes `gadgetPublicMatrix`,
`gadgetDecompose`, or a runtime matrix operation. The full correctness theorem connects a validated
descriptor to the concrete sampler family through fields of `MxxBoundedSamplerContract`. A forged
descriptor may make the operational filter reject or report a useless result, but cannot satisfy
that soundness contract.

The current line-oriented Diamond checker has no persistent report store. Its server response is
therefore `accepted requestHash`; the Rust session rejects a response whose echoed hash differs
from the request it sent. This is the concrete report/cache identity for the operational stage.

### 7.1 Node representation

Lean must mirror the Rust node exactly:

```lean
| gadgetDecompose
    (matrixType : MatrixTypeExpr)
    (base : IntExpr)
    (small : Bool)
    (digitCount : IntExpr)
```

The emitter must transport `small` verbatim and must have an exhaustive Rust match. Emission of a
valid `GadgetDecompose` node may not return `UnsupportedNode`.

### 7.2 Deterministic partial operations

Change gadget decomposition in `MxxSamplerFamily` from a list-valued sampler to deterministic
partial functions:

```lean
gadgetPublicMatrix :
  SamplerParamsId -> SamplerParams -> Nat -> Int -> Bool -> Nat -> Option Matrix

gadgetDecompose :
  SamplerParamsId -> SamplerParams -> Int -> Bool -> Nat -> Matrix -> Option Matrix

smallDecompositionInputLimit :
  SamplerParamsId -> SamplerParams -> Option Nat
```

The shared Lean matrix semantics also defines:

```lean
maxCanonicalCoefficient : Matrix -> Nat
```

It computes the maximum unsigned canonical representative in `[0, q)` from the matrix value. It
is not supplied by Rust and is not interchangeable with `maxCenteredCoefficientNorm`.

The additional `Nat` argument is the original matrix size. Modulus and ring dimension come from
`SamplerParams`; a successful result has shape
`size x (size * digitCount)`. The size is never inferred from `params.rows` because those
parameters may describe either the decomposition output or the public matrix at different call
sites. Direct decomposition passes `input.rows`; constant gadget and `GadgetTrapdoor` nodes pass
their public-matrix row count.

The exact argument order may follow existing Lean style, but every use must include the uniquely
resolved `SamplerParamsId`, `size`, and `small`.
`gadgetPublicMatrix` is the single source of the mode-aware public matrix for constant gadget
nodes, `GadgetTrapdoor`, direct decomposition relations, and gadget-origin `PreimageSample`.
Neither the generic Lean IR nor an owner crate reimplements the CRT-tower layout.
`smallDecompositionInputLimit` returns the concrete backend's smallest CRT modulus for supported
DCRT parameters and `none` when that metadata is unavailable. It is used only to discharge the
small-mode reconstruction premise; it does not control whether decomposition executes.

Both operations return `none` for a backend-invalid `(params, base, small, digitCount)` tuple.
They must reject `base <= 1` and a zero count before attempting the concrete layout operation.
`evaluateNode` returns the unique output on `some` and an invalid result on `none`; it never returns
one result per member of an abstract support list.

Gaussian and sampled-preimage operations remain list-valued because they are genuinely sampled.

### 7.3 Shared bound function

Define one Lean function used by the contract, analyzer, operational checker, and reports:

```lean
def gadgetDecompositionBound (base : Int) (small : Bool) : Nat :=
  if small then base.natAbs - 1 else max (base.natAbs / 2) 1
```

No caller may reproduce this conditional locally.

### 7.4 Bounded sampler contract

Replace the existing decomposition fields with mode-aware deterministic fields equivalent to the
following conditional statements. The contract makes no claim outside the partial operation's
domain:

```lean
gadgetDecomposeRelation :
  forall paramsId params base small digitCount input public digits,
    samplers.gadgetPublicMatrix paramsId params input.rows base small digitCount = some public ->
    samplers.gadgetDecompose paramsId params base small digitCount input = some digits ->
    (small = false \/
      exists limit,
        samplers.smallDecompositionInputLimit paramsId params = some limit /\
        maxCanonicalCoefficient input < limit) ->
    MatrixModEq
      (matrixMul public (digits.withSamplerParams params))
      input

gadgetDecomposeBound :
  forall paramsId params base small digitCount input digits,
    samplers.gadgetDecompose paramsId params base small digitCount input = some digits ->
      maxCenteredCoefficientNorm (digits.withSamplerParams params) <=
      gadgetDecompositionBound base small

gadgetDecomposeCongruent :
  forall paramsId params base small digitCount leftInput rightInput leftDigits rightDigits,
    samplers.gadgetDecompose paramsId params base small digitCount leftInput = some leftDigits ->
    samplers.gadgetDecompose paramsId params base small digitCount rightInput = some rightDigits ->
    MatrixModEq leftInput rightInput ->
      leftDigits.withSamplerParams params = rightDigits.withSamplerParams params
```

The congruence theorem requires identical `paramsId`, `params`, `base`, `small`, and `digitCount`.
It gives no cross-layout or cross-mode equality.

The small-input premise applies only to `gadgetDecomposeRelation`. It does not weaken
`gadgetDecomposeBound` or deterministic same-mode congruence. The analyzer may discharge it only
from an exact canonical unsigned range fact proving `maxCanonicalCoefficient input < limit`.
A centered noise bound, even one smaller than `limit`, is insufficient because it does not exclude
negative residues. If the exact range fact is unavailable, the analyzer produces no reconstruction
relation. It must not assume that a runtime value has a small canonical representative merely
because the node selected small decomposition.

The contract remains an explicit theorem hypothesis about the concrete backend. Do not add a Lean
CRT decomposition algorithm merely to prove these backend facts inside the generic IR package.
Graph validation and runtime layout validation establish that executable nodes lie in the partial
domain; the local Lean rule must still pattern-match on `some` and fail closed on `none`.

Add a descriptor-agreement field to the same contract. Given an actual runtime layout whose
canonical `SamplerParamsId` equals the request descriptor ID and whose visible modulus/ring
dimension match the corresponding Lean `SamplerParams`, it states for each validated mode/count
that `gadgetPublicMatrix` and `gadgetDecompose` return `some` with the declared layout, and that
`smallDecompositionInputLimit` returns the descriptor's `smallestCrtModulus`. This field is used by
local soundness proofs, not by Phase A execution.

## 8. Trapdoor provenance and preimage dispatch

### 8.1 Lean value representation

Replace the matrix-only trapdoor value with explicit provenance:

```lean
inductive TrapdoorOrigin where
  | sampled
  | gadget
      (paramsId : SamplerParamsId)
      (base : Int)
      (small : Bool)
      (digitCount : Nat)

structure TrapdoorValue where
  publicMatrix : Matrix
  origin : TrapdoorOrigin

| trapdoor (value : TrapdoorValue)
```

This is execution provenance, not secret trapdoor material. Lean never receives or models the
private trapdoor bytes.

### 8.2 Producers

- `TrapdoorSample` produces origin `sampled`.
- A trapdoor artifact input is `sampled`; the current artifact format always contains sampled
  secret material and has no gadget-trapdoor artifact form.
- `GadgetTrapdoor` evaluates its explicit base and matrix type, resolves the request's unique
  `paramsId`, obtains the regular public matrix from
  `gadgetPublicMatrix paramsId params publicRows base false digitCount`, and produces origin
  `gadget paramsId base false digitCount`. A `none` result is invalid.
- `GadgetTrapdoor` obtains `digitCount` from its validated matrix shape
  `columns / rows`; zero rows or a non-divisible shape is invalid.
- `TrapdoorPublic` returns `TrapdoorValue.publicMatrix` unchanged.

### 8.3 `PreimageSample`

`PreimageSample` always evaluates `max_coefficient_bound` and rejects a negative value before
examining trapdoor origin. This keeps scalar and batch execution domain behavior identical.

It then dispatches as follows:

```text
sampled origin:
  invoke samplePreimage(public, target)
  use max_coefficient_bound as the hard output bound

gadget origin (paramsId, base, small, digitCount):
  require the public input to equal the provenance public matrix
  invoke partial deterministic gadgetDecompose(paramsId, params, base, small, digitCount, target)
  reject if it returns none
  ignore max_coefficient_bound for the resulting norm, after having validated it
  use gadgetDecompositionBound(base, small)
  if small, retain reconstruction only after proving the target canonical-range premise
```

The sampled branch and the current regular gadget branch retain only:

```text
public * output = target  in R_q.
```

For the current `GadgetTrapdoor`, `small` is always false, so its reconstruction relation is
unconditional after successful layout validation. Keeping `small` in provenance makes the dispatch
type match the deterministic decomposition contract without adding a second special operation.
Keeping `paramsId` prevents a trapdoor from being reinterpreted under a different CRT layout with
the same visible ring key. A future small gadget-trapdoor producer would have to discharge the same
strict target-bound premise; this specification does not add such a producer.

## 9. Analyzer and operational-checker behavior

For a direct `GadgetDecompose` node, Phase A looks up the node's request-scoped
`GadgetLayoutDescriptor`, checks exact base/mode/count agreement, and constructs an opaque
mode-aware public-matrix identity. It does not evaluate either partial semantic function. It always
records:

- the exact input identity;
- the exact resolved base, mode, and count;
- `Finite(gadgetDecompositionBound(base, small))`; and
- deterministic dependency provenance inherited from the input.

It records the modular gadget relation unconditionally for regular mode. For small mode it records
the relation only after the existing input fact proves an exact unsigned canonical coefficient
range strictly below the descriptor's `smallestCrtModulus`. A centered noise bound cannot discharge
this premise. If the exact range proof is unavailable, decomposition and its output bound remain
analyzable, but relation-dependent rewriting rejects with a specific missing-premise diagnostic.

Execution and local soundness separately evaluate the partial sampler-family functions and use the
descriptor-agreement contract to connect their concrete results to the opaque identities created by
Phase A. This separation prevents the operational analyzer from pretending it owns runtime values.

For a gadget-origin `PreimageSample`, it records the same decomposition fact. For a sampled-origin
preimage it records the serialized preimage cutoff and fresh sampler dependency.

If base/count evaluation, type consistency, provenance, or public-matrix identity cannot be
established, the graph is invalid or the checker rejects with a specific diagnostic. It must not
fall back to the centered modulus cap while still claiming a decomposition relation.

Regular and small decomposition results with otherwise identical inputs are different symbolic
atoms. Cancellation is permitted only when mode, base, count, input identity, and instantiation
path all agree.

## 10. Emitter and derivation rules

The mechanical derivation rule remains one `gadgetDecompose` rule. The checked frozen node carries
the mode and count, so the untrusted derivation instruction does not repeat them.

The following emitter failures must disappear:

- `implicit-digit GadgetDecompose`;
- loss of `small`; and
- a wildcard derivation match for this node.

Generated source and derivation hashes must change when `small` or `digit_count` changes.

## 11. Implementation order

1. Remove optional digit count from Rust `NodeKind`, validation, runtime, DSL, and estimator.
2. Extend the reusable checker request/report with the candidate-derived gadget-layout descriptor
   table and `requestHash`; do not regenerate Lean source per candidate.
3. Add mode and explicit count to Lean `NodeKind` and the Rust-to-Lean emitter.
4. Add the independent unsigned `CanonicalRange` fact and its closed transfer-rule table.
5. Add the mode-aware partial `gadgetPublicMatrix` and `gadgetDecompose` operations and update Lean
   execution facts.
6. Replace the decomposition fields of `MxxBoundedSamplerContract`, including descriptor agreement.
7. Add trapdoor provenance and update trapdoor producers/projection.
8. Update `PreimageSample` execution and local soundness for both origins.
9. Update frozen-slice, recurrence, pointwise, analyzer, and operational-bound consumers
   mechanically; do not redesign their public abstractions.
10. Regenerate checked-in Lean programs.
11. Run the focused tests and full Lean/Rust gates below.

Each step must compile before beginning the next. Do not temporarily map `small = true` to regular
decomposition to keep downstream proofs compiling.

## 12. Required tests

### 12.1 Rust IR and runtime

- canonical serialization distinguishes regular and small nodes;
- descriptor generation is canonical, rejects duplicate/mismatched entries, and is covered by the
  per-candidate request/report hash;
- two descriptor IDs resolving to the same `(modulus, ringDimension)` key are rejected before node
  analysis;
- a missing digit count cannot be represented;
- negative and unit bases are rejected for both decomposition and gadget trapdoors without
  applying absolute value;
- zero digit count is rejected;
- output rows unequal to `input_rows * digit_count` are rejected;
- regular mode calls `decompose()` and small mode calls `small_decompose()`;
- the estimator performs the same backend layout check as runtime execution;
- backend layout mismatch is rejected before decomposition; and
- CPU runtime outputs satisfy the existing trusted regular and small gadget reconstruction tests;
- at CRT depth greater than one, the runtime regular and small public gadget matrices are distinct;
  and
- each runtime public matrix reconstructs its corresponding decomposition, while cross-mode
  pairing is not accepted as the node's relation; and
- small-mode reconstruction fixtures sample inputs strictly below the smallest CRT modulus, plus
  an unrestricted-input fixture that still decomposes but is not granted a reconstruction fact;
  and
- canonical `q - 1` and an equivalent stored `-1` fixture do not receive the small reconstruction
  relation merely from centered norm one.
- changing CRT depth, ordered CRT basis, CRT bits, or base bits while retaining the same
  graph-visible ring dimension/modulus data changes `SamplerParamsId` and `requestHash`; a cached
  report with the old request hash is not reusable.
- trapdoor provenance resolved under one `SamplerParamsId` cannot dispatch decomposition under a
  different ID with the same visible ring key.

### 12.2 Lean

- emitted regular and small nodes parse and differ in their Boolean mode;
- cross-language fixtures at CRT depth greater than one show that Lean relations use the exact
  runtime regular or small public gadget matrix and that the two matrices are distinct;
- both modes derive the correct modular gadget equation only with their corresponding public
  matrix, with the small equation conditional on the strict unsigned canonical-range premise;
- regular mode derives `max(floor(|b| / 2), 1)`;
- small mode derives `|b| - 1`;
- quotient-equal inputs have equal decomposition only through the same-mode congruence theorem;
- sampled `PreimageSample` uses the serialized cutoff;
- gadget-origin `PreimageSample` uses the digit bound after still validating the serialized cutoff;
- public/trapdoor mismatch is rejected; and
- an invalid layout makes both partial operations fail and produces no decomposition relation or
  bound fact; and
- an unrestricted small-mode input still gets the digit bound but no reconstruction relation; and
- canonical nonnegative coefficients below the smallest CRT modulus receive the small relation,
  while `q - 1` and stored `-1` do not; and
- every canonical-range source/transfer in the universal plan has a preservation or fail-closed
  dropping fixture; and
- no decomposition theorem states integer-vector equality in place of `MatrixModEq`.

### 12.3 Regression and trust gates

Run:

```text
cargo +nightly fmt --all
cargo test -p mxx-ir-core -p mxx-dsl -p mxx-runtime -p mxx-correctness --lib
cd lean
lake build Mxx
```

Because the runtime branch reaches GPU decomposition, also compile the relevant GPU unit-test
targets and run the focused regular/small decomposition tests according to `GPU.md` when the
implementation changes GPU-facing execution.

Regenerate toy and Diamond Lean sources and verify freshness. Scan active Lean modules for
`sorry`, `admit`, and new axiom declarations.

## 13. Completion criteria

This correction is complete only when:

1. Rust and Lean carry identical explicit `(base, small, digit_count)` data;
2. `GadgetDecompose` has no optional digit-count representation;
3. Lean execution distinguishes partial deterministic decomposition from sampled preimage
   generation;
4. regular and small hard bounds are mode-correct;
5. all gadget equations are stated only in `R_q`, use the exact mode-aware runtime public matrix,
   and enforce the small-mode unsigned canonical-range premise;
6. valid decomposition nodes no longer fail Lean emission;
7. generated workflows are fresh;
8. every checker report identifies the exact candidate layout through `requestHash`;
9. required focused tests and builds pass; and
10. no compatibility path, proof hole, or new axiom is introduced.

After these criteria hold, implementation resumes at Stage A of
`docs/universal-ir-noise-checker-plan.md`.
