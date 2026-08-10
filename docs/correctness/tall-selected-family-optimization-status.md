# Tall Selected-Family Operational Checker Status

## Status

This report records an auditable but incomplete optimization snapshot of the Lean operational
noise checker used by the Tall BGG+ nested-RNS parameter simulation. It does not claim that the
Tall parameter simulation completes or that a valid parameter set has been found.

The implementation snapshot reviewed by this report is commit `58a168a8` (`Optimize compact
selected-family noise evaluation`), based on `d299eb73`.

## Workload

The measured depth-one candidate used:

- ring dimension: 8;
- CRT depth: 1;
- CRT modulus bits: 10;
- gadget base bits: 5;
- scale: 1024;
- multiplication count: 1;
- parameter-simulation parallelism: 1;
- required security bits: 0, so this run measured the graph-driven noise path rather than lattice
  security rejection.

The emitted graph contained 31 lookup tables. Their preprocessing plan contained 30,720 lookup
preimages and 48 slot-operation preimages, for 30,768 preimages in total. The graph reported 184
artifacts and the following top-level gate counts:

| Gate | Count |
| --- | ---: |
| SubCircuitOutput | 21 |
| Input | 21 |
| LargeScalarMul | 12 |
| Add | 11 |
| Sub | 3 |
| SlotTransfer | 1 |

## Implemented Optimizations

### Array-backed packed families

The previous recursive `familyPackedNil` / `familyPackedCons` representation was replaced by one
`familyPacked (Array OperationalFact)`. Packed-family construction, lookup, traversal, identity
transport, decoder residual collection, and tests now use this array representation.

This removes recursive list reconstruction from large family transport. It does not by itself
remove the cost of evaluating every branch.

### Compact selected matrices

Relation-bearing dynamic family selection no longer expands immediately into indicator-weighted
sum-of-products terms or discards branch-local relations. It produces:

```text
selectedMatrices(selectionIdentity, Array OperationalMatrixFact)
```

The representation preserves the executable selection identity and one exact fact per branch.
Dynamic selection of a relation-free family continues to use the existing exact-one signal and
branch-wise maximum rule.

### Selection-aware primitive operations

The operational evaluator propagates compact selection through the matrix operations needed by the
current graph surface:

- addition and subtraction;
- matrix multiplication and relation rewriting;
- negation and matrix scaling;
- gadget decomposition and preimage sampling;
- transpose, slice, reshape, constant-coefficient extraction, and supported structural transforms;
- concat and tensor;
- CRT recomposition;
- parallel-loop family construction, dynamic selection, broadcast, zip, and zip-offset.

An ordinary matrix input is broadcast across selected branches. Two selected inputs are zipped only
when their exact selection identities and branch counts agree. Different selections are rejected
instead of being positionally zipped or expanded into a Cartesian product.

Selected preimage and decomposition relations remain attached to the corresponding branch identity.
Decoder noise is evaluated for each complete branch and then combined by maximum, not by summing
mutually exclusive branch bounds.

`ExtractCoefficient` and `ThresholdDecode` currently reject selected matrix inputs because the
checker has no compact selected-scalar representation. Nested selections are also rejected. These
are explicit unsupported boundaries rather than silent conservative conversions.

### One parallel-loop body evaluation

A packed relation-bearing parallel-loop input is represented as a compact selection in the child
body. The body is evaluated once, and a selected child output is converted back into a packed
family. This preserves lane-aligned relations without evaluating the complete Lean scope separately
for every lane.

### Prepared workflow reuse across requests

Operational requests with identical parameter environments and gadget layouts are grouped. The
Lean workflow is evaluated once per group, and each decoder threshold request consumes the shared
stage outputs. The runner now reports source decode time and operational evaluation time separately.

### Diagnostics

Operational and derivation errors derive `Repr`, and generated runners print the concrete error
instead of only a request number. The Tall test logs emitted source sizes, cold and warm preparation
times, and per-request decode/evaluation times.

## Measurements

The depth-one Tall run emitted:

| Source | Bytes |
| --- | ---: |
| Operational IR | 38,804,763 |
| Proof IR | 30,057,136 |
| Derivation IR | 29,976,992 |

Preparation measurements were:

| Stage | Time |
| --- | ---: |
| Cold operational IR and derivation preparation | 6.765 s |
| Warm cached preparation | 0.996 s |

Graph construction took approximately 20.5 seconds before source emission. Operational evaluation
did not complete after more than ten minutes and was interrupted after the bottleneck was confirmed.
No decoder bound or acceptance result was produced by this run.

Earlier eager-SOP and exact-per-lane variants also failed to complete promptly. The compact selected
representation prevents premature indicator expansion and fixes relation alignment, but the current
implementation still materializes and repeatedly traverses large branch arrays through downstream
operations. Therefore the optimization is semantically useful but not yet sufficient for the Tall
workload.

## Remaining Performance Work

The next optimization must preserve branch-local noise semantics while avoiding repeated copies of
common symbolic products:

1. Extract the longest exactly identical common factor prefix and suffix from corresponding selected
   products. Factor equality must use exact symbolic identity and preserve noncommutative order.
2. For equally shaped sums, align corresponding terms and factor each term independently. If the
   structure is irregular, retain the compact unfactored selected representation.
3. Keep branch-local bounded expressions paired until each complete branch bound is evaluated, then
   take the maximum. Do not independently maximize subterms from different branches unless using an
   explicitly documented conservative fallback.
4. Cache reusable branch summaries and avoid rebuilding complete `OperationalMatrixFact` arrays at
   every unary or binary operation.
5. Add timing around selection creation, relation rewriting, polynomial normalization, and decoder
   branch aggregation to identify the dominant remaining traversal.

The likely target representation is an internal factored selected product containing a common
prefix, a selection identity, varying branch products, and a common suffix. It is analysis-only
structural sharing, not a semantic IR node and not a restoration of the removed `fold` operation.

## Remaining Correctness and Coverage Work

- Add direct fixtures proving selected preimage relation alignment through multiplication and
  rewrite, not only through dynamic extraction and zip-offset transport.
- Add structural factoring tests for common prefix/suffix, sum-term alignment, noncommutative order,
  identity mismatch, and irregular-branch fallback.
- Confirm statically that Tall selected values cannot reach `ExtractCoefficient` or
  `ThresholdDecode`; otherwise introduce a compact selected-scalar fact rather than special-casing
  one failure.
- Complete the depth-one Tall operational evaluation and compare its final bound with the previous
  graph-driven implementation on the same graph and request.
- Run the requested CRT-depth search only after one request completes in practical time.

## Validation Completed

The implementation snapshot passed:

```text
lake build Mxx.Certificate.OperationalBounds
cargo test -p mxx-correctness --lib operational_runner::tests
cargo test -p mxx-correctness --lib \
  operational_runner::tests::runs_generated_toy_workflow -- \
  --ignored --exact --nocapture
cargo +nightly fmt --all
git diff --check
```

The Tall integration test reached graph construction, source emission, cold preparation, warm cache
reuse, and operational evaluation. It did not finish operational evaluation, so the Tall test is not
reported as passing.

## Audit Boundaries

This snapshot changes operational analysis representation and checker execution strategy. It does
not change the executable Rust/CUDA Tall protocol graph or its runtime cryptographic operations.
The generated correctness hashes were regenerated with the repository emitter. The existing
Diamond binary-transport timing fixture is included in the implementation snapshot because it
exercises the same prepared operational checker cache path.

The operational checker remains a graph-derived parameter filter. Passing it would establish the
implemented deterministic bound calculation for a candidate; this report records no successful
Tall candidate and no end-to-end runtime validation.
