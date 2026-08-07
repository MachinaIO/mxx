# Dynamic Family Index Resolution Design Plan

## 1. Status and scope

This plan resolves the current `loop-relative-family-index` failure in the Lean analyzer. It is a
focused prerequisite for Stage E of `docs/universal-ir-noise-checker-plan.md`; it does not replace
that broader plan.

The immediate failing case occurs in the generated Diamond WE graph. A `FamilyGetDynamic` inside a
loop receives an integer produced through another family and a formal body input. The integer is
not syntactically the current loop index, even when it was ultimately computed from one. The
current normalizer rejects the expression before using the interval already carried by its
`IntegerFact`.

This design must support both of the following cases:

1. an index that is exactly the current loop index plus a statically known nonnegative offset; and
2. any other runtime integer expression whose analyzer-derived inclusive interval proves that all
   possible values are valid indices.

The change is analyzer- and checker-side only. It introduces no executable IR node, Rust bound
formula, certificate field, protocol-specific matcher, or family unrolling. The full correctness
theorem remains deferred as specified by the universal checker plan, but the retained evidence
must be sufficient to prove this rule later without changing its operational semantics.

## 2. Current failure and required behavior

### 2.1 Current data flow

The relevant generated shape is:

```text
outer parallel loop
  -> integer family F, where each lane may compute an integer expression
  -> inner loop receives F through Zip or another family transport
  -> body formal integer input has an IntegerFact
  -> FamilyGetDynamic(target_family, formal_integer_input)
```

The body input may denote any of these examples:

```text
i
i + constant_offset
select(predicate, i, i + 1)
F[i]
circuit_left_source[layer * width + gate]
```

`seedParallelInputs` already constructs an `IntegerFact` for the formal input. That fact contains:

- an exact analyzer-owned `RuntimeExpr .integer`; and
- inclusive lower and upper bounds as `IntBoundExpr`.

`applyFamilyGet` interns the exact expression, then calls `normalizeLoopRelativeIntExpr`. The
normalizer recognizes only a small loop-affine grammar. It returns an error for `familyElement`,
`select`, `boolToInt`, and other valid integer expressions. `applyFamilyGet` currently converts
every such error into `invalidExpressionReference "loop-relative-family-index"`.

### 2.2 Correct acceptance rule

For an index value `x` and family length `N`, the access is valid exactly when:

```text
0 <= x and x < N.
```

The analyzer may establish this in one of two ways:

- **Loop-offset rule:** if `x = i + k`, where `i` ranges over a loop of count `L` and `k` is a
  nonnegative literal, require `L = 0` or `L - 1 + k < N`.
- **Interval rule:** if `lower <= x <= upper` follows from the `IntegerFact`, require
  `0 <= lower` and `upper < N`.

The loop-offset rule is a precision optimization. The interval rule is the complete fallback and
is the rule that admits nested family values, selections, input-dependent circuit wiring, and any
future supported integer primitive without adding expression-specific family logic.

If neither rule produces a Phase-B-evaluable obligation, analysis must fail with a specific
range-evidence diagnostic. It must never accept the access merely because the expression is not
recognized as loop-affine.

## 3. Trust boundary

### 3.1 Analyzer-owned inputs only

The range rule consumes only:

- the frozen `FamilyGetDynamic` node and its actual two argument wires;
- the `IntegerFact` obtained by `requireInteger` from the analyzer state;
- the target `FamilyFact` and its declared count;
- the current analyzer-owned `StructuralLoopContextStack`;
- the analyzer-owned `ExpressionArena`; and
- the analyzer-owned `ParallelFamilyDerivationSource` already retained when a family is built.

No part of the rule may be supplied by a Rust certificate, protocol annotation, semantic anchor,
node name, generated artifact name, or caller-provided invariant.

### 3.2 No self-reported expression equality

The implementation must not add an API that says that an arbitrary expression is equal to
`i + k`. The sharp loop-offset classification is derived only by resolving the exact arena entry
and structurally normalizing it. All other expressions use their mechanically inferred interval.

### 3.3 No runtime trace in Phase A or Phase B

Phase A analyzes the frozen graph and constructs obligations. Phase B evaluates only parameter and
recurrence bound expressions. Neither phase executes a parallel loop or consumes a runtime trace.

Execution evidence is required only by the later Lean soundness theorem. The proof bridge must
replay the same analyzer-owned family derivation against the real execution trace; it must not
change the checker algorithm or add a second runtime evaluator.

## 4. Type-level design

### 4.1 Replace error-based loop normalization with a classification

Keep `RelativeIndexNF` for provenance normalization, but add a family-access-specific result:

```lean
inductive DynamicFamilyIndexClassification where
  | loopOffset
      (context : StructuralLoopContext)
      (offset : Nat)
  | interval
      (lower upper : IntBoundExpr)
```

The classification is analyzer-internal. It is not serialized and is not added to
`ClosedProtocolBundle`.

Define the context-aware classification function in `Analyzer.lean`, where
`StructuralLoopContextStack` is owned:

```lean
classifyDynamicFamilyIndex
    (arena : ExpressionArena)
    (contexts : StructuralLoopContextStack)
    (indexRef : RuntimeExprRef .integer)
    (fact : IntegerFact) :
    Except DynamicFamilyIndexError DynamicFamilyIndexClassification
```

Its algorithm is:

1. Check that `arena.lookupInteger indexRef = some fact.expression`. A missing or differently
   typed reference is an integrity error, not an interval fallback.
2. If there is no enclosing loop, return `.interval fact.lower fact.upper`.
3. Try `normalizeLoopRelativeIntExpr` against the innermost context.
4. Return `.loopOffset context offset` only for `.ok (.loopOffset offset)`.
5. Return `.interval fact.lower fact.upper` for `.ok (.invariant _)` and for failures that mean
   only "not in the supported loop-affine grammar", including `unsupportedArithmetic`,
   `unsupportedIndex`, and a different loop binder.
6. Propagate arena-integrity failures such as `missingExpression`, cyclic references, an escaped
   analyzer placeholder, or a wrong expression type. These failures must not be converted into an
   interval result.

The distinction between an unsupported normal form and malformed analyzer state must be explicit
in `DynamicFamilyIndexError`; do not inspect rendered error strings.

`RecurrenceBasisAlignment.lean` must not import or mention `StructuralLoopContext`: it is a lower
layer already imported by the analyzer, so doing so would create an import cycle. That module only
provides a context-neutral normalization primitive which receives the selected loop binder and
slot explicitly and returns either a normalized `RelativeIndexNF`, an unsupported-grammar result,
or a malformed-reference error. `Analyzer.lean` supplies the selected context, converts a sharp
`.loopOffset` result into `DynamicFamilyIndexClassification.loopOffset`, and applies the interval
fallback. Thus the public module dependency remains one-way:

```text
RecurrenceBasisAlignment (arena-level normalization)
    -> Analyzer (loop context selection and family-index classification)
```

### 4.2 Add one general static obligation

Extend `StaticObligation` with:

```lean
| dynamicFamilyIndexInRange
    (lower upper : IntBoundExpr)
    (familyCount : IntExpr)
```

Its exact proposition is:

```text
evaluate lower = l
evaluate upper = u
evaluate familyCount = n
0 <= l
l <= u
u < n
```

`l <= u` is checked here even though ordinary integer-fact construction also emits ordered-bound
obligations. Keeping it in the family-access obligation makes this safety check self-contained and
prevents a future caller from reusing the rule with an unordered interval.

The existing `loopFamilyAccessInRange` obligation remains unchanged for the precise loop-offset
case.

Phase B evaluates both obligations with the existing exact `IntBoundExpr` and `IntExpr`
evaluators. Evaluation failure, negative family length, unordered bounds, or a violated endpoint
returns the existing deterministic unsatisfied-obligation error. No `f64` conversion is allowed.

This static obligation may be emitted only when neither bound contains a sequential
`.carriedInput`. A raw carried placeholder is not evaluable in the top-level Phase-B environment
and must never escape in `AnalysisResult.staticObligations`.

### 4.3 Check carried intervals at every sequential step

An index in a sequential body may have bounds that depend on the previous carried state. Checking
only the final recurrence result would be unsound: an access can be out of range at iteration `t`
and return to a valid interval at the final iteration.

Add a schema-indexed step obligation alongside the numeric recurrence transition:

```lean
structure SequentialBodyRangeObligation where
  site : CoreNodeRef
  lower upper : IntBoundExpr
  familyCount : IntExpr

structure RecurrenceStepRangeObligation
    (previous : List CarriedBoundSchema) where
  site : CoreNodeRef
  lower upper : IntBoundTransitionExpr previous
  familyCount : IntExpr
```

Extend `SymbolicRecurrenceTransfer` with:

```lean
stepRangeObligations :
  List (RecurrenceStepRangeObligation
    (carriedSchemas.map CarriedValueSchema.boundSchema))
```

Extend the analyzer-owned `SequentialRecurrenceSource` with:

```lean
bodyRangeObligations : List SequentialBodyRangeObligation
```

In the active analyzer path this field is populated only from the exact body-analysis suffix
described below. It is not a protocol field or serialized certificate value. Closed Lean fixtures
may construct the structure directly, but no soundness theorem may conclude execution truth from
the field without the analyzer-origin validation described below.

These obligations are not certificate input. While analyzing the body, the sequential analyzer
collects the obligations appended by the exact body analysis. It partitions them as follows:

- obligations without `carriedInput` remain ordinary static obligations;
- a carried-dependent `dynamicFamilyIndexInRange` is translated with the same typed path
  conversion used by `translateIntegerBound`, producing `IntBoundTransitionExpr.previousState`;
- any other carried-dependent static obligation is rejected with a specific unsupported
  recurrence-obligation error until it has an explicit typed transition representation.

The source for this partition is exactly the body-local suffix:

```text
bodyState.staticObligations.drop baseState.staticObligations.length
```

It is not a list supplied to `SymbolicRecurrenceTransfer.build` by a caller. The builder reads the
analyzer-owned `SequentialRecurrenceSource.bodyRangeObligations`, translates every bound through
typed state paths, and stores a `stepRangeValidation` equality showing that the retained typed
list is exactly the result of that translation, following the same pattern as
`schemaValidation`.

Phase B replaces `iterateCarriedBoundTransition` for a transfer with a checked iterator whose
order for each iteration is exactly:

1. evaluate every `stepRangeObligation` against the immutable previous numeric state;
2. require `0 <= lower <= upper < familyCount`;
3. evaluate every output component of `boundTransition` against that same previous state; and
4. update all carried components simultaneously.

For a recurrence count of zero, no step obligation is evaluated because the body never executes.
External recurrence references are closed before iteration for both the transition and the step
obligations. A failure reports the recurrence identity, iteration number, and originating
`FamilyGet` site. Symbolic matrix expressions remain unrolled; only the existing fixed-size numeric
state and a fixed list of step obligations are evaluated `count` times.

### 4.4 Preserve the exact index fact in the resulting family element

`applyFamilyGet` continues to instantiate the selected element using the exact `indexRef` and
`fact.expression`. It must not replace a general expression with its lower bound, upper bound, or
loop index.

The resulting `ValueInstanceRef.familyElement` identity therefore remains tied to the original
arena reference. Two accesses cancel or compare as the same symbolic object only when their
aggregate identity and exact index reference are the same. Equal intervals do not imply equal
indices.

## 5. Analyzer algorithm

Change `applyFamilyGet` in the following order:

1. Resolve the family argument and require a `FamilyFact` exactly as today.
2. Resolve the dynamic index argument with `requireInteger`.
3. Intern `indexFact.expression` in the current arena and verify the returned reference resolves
   to that exact expression.
4. Call `classifyDynamicFamilyIndex`.
5. Append exactly one range obligation:
   - `.loopFamilyAccessInRange context.count offset family.count` for `.loopOffset`; or
   - `.dynamicFamilyIndexInRange lower upper family.count` for `.interval`.
6. Instantiate the family element using the existing aggregate-specific path:
   `instantiateFamilyElement`, `instantiateProtocolFamilyElement`,
   `instantiateStructuralFamilyElement`, or `instantiateRecurrenceFamilyElement`.
7. Retain all existing uniqueness, schema, relation-bearing carried-family, and origin checks.

For `FamilyGetStatic`, construct an exact temporary `IntegerFact` whose expression is
`.parameter value` and whose bounds are both `.integer value`, then route it through the same
classifier. This is necessary because a static node's `IntExpr` may still contain a loop index.
Inside the corresponding loop, `i + k` therefore uses the loop-offset rule. A loop-independent
expression uses `[value, value]`; if it is not evaluable in Phase B, checking fails closed. A
negative static index or `value >= familyCount` must be rejected by Phase B, not left to an
unrelated later failure.

Do not recursively expand a `RuntimeExpr.familyElement` into its producer body merely to prove an
index bound. Its `IntegerFact` is already the uniform summary computed from that producer. This is
what keeps analysis compact and avoids turning a parallel family into an unrolled expression DAG.

## 6. Parallel input-substitution provenance

The operational range rule needs only the transported `IntegerFact`, but the future theorem must
show that this fact describes the actual body input. Reuse the existing
`ParallelFamilyDerivationSource`; do not introduce a second graph-search layer.

### 6.1 Reconstruct rather than self-report bindings

Move the pure, deterministic part of `seedParallelInputs` into an internal function that returns a
typed record for each formal input:

```lean
structure ParallelInputSeedBinding where
  slot : Nat
  destination : CoreWireRef
  source : CoreWireRef
  mode : Mxx.Ir.LoopInputMode
  seededFact : ScopedWireFact
```

The constructor remains private to the analyzer. The binding list is reconstructed from:

- `body.inputNames`;
- `argumentRefs`;
- `modes`;
- the exact pre-loop fact table;
- the loop's analyzer-owned index reference and expression; and
- existing family derivations and input contracts.

`seedParallelInputs` then projects the arena and `seededFact` list from this result. The same pure
function is used by the later proof bridge to establish that retained `seededFacts` were not
independently asserted.

Do not add the binding list to serialized analysis output. It may be retained inside
`ParallelFamilyDerivationSource` only if reconstructing it repeatedly is measurably expensive; if
retained, add an internal `MatchesSeedBindings` proposition equating it to the deterministic
reconstruction.

### 6.2 Mode semantics

The binding record has the following exact meaning at lane `i`:

- `Broadcast`: the destination receives the source value unchanged.
- `Zip`: the destination receives source family element `i`.
- `ZipOffset k`: the destination receives source family element `i + k`.

The initial implementation must add `ZipOffset` support rather than leaving it as
`UnsupportedNode`, because the universal checker plan requires all loop input modes. Its range
obligation is the existing loop-family rule with offset `k`. Peak analysis state remains constant
in the evaluated loop count.

## 7. Future semantic soundness bridge

The operational checker is complete without executing the loop in Phase A, but the retained
evidence must
support a theorem with this shape:

```lean
theorem dynamicFamilyGet_index_in_range
    (indexFactHolds : IntegerFact.Holds ... indexFact actualIndex)
    (obligationHolds : StaticObligation.Holds parameters rangeObligation)
    (classificationMatches : ... ) :
    0 <= actualIndex ∧ actualIndex < evaluatedFamilyCount
```

The proof splits on the analyzer-owned classification:

- In the interval case, `IntegerFact.Holds` gives
  `lower <= actualIndex <= upper`; Phase-B obligation soundness gives
  `0 <= lower <= upper < familyCount`.
- In the loop-offset case, exact expression denotation and the actual loop trace give
  `actualIndex = lane + offset` and `lane < loopCount`; Phase-B obligation soundness gives the
  upper endpoint.

For carried-dependent intervals, prove the analogous statement at every actual sequential step.
The checked numeric iterator's induction is synchronized with the real `SequentialIterationsTrace`:
the previous numeric state soundly bounds the actual previous carried values, the step obligation
proves the current access safe, and local body-rule soundness constructs the next bound state.
There is no theorem that infers all-step safety from the final recurrence state.

For a formal input inside a parallel body, derive `indexFactHolds` from the exact actual child
arguments using:

- the frozen loop node and `ParallelFamilyDerivationSource`;
- `Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace`;
- the existing `ParallelLaneExecutionSource` or equivalent exact lane edge;
- the deterministic `ParallelInputSeedBinding`; and
- the local rule soundness that produced the integer family template.

The theorem must use the same workflow runner and the actual child trace. It must not accept a
caller-supplied `runChild`, lane value, input substitution, or fact truth assertion. Analyzer arena
references remain analyzer-owned; execution evidence relates them to concrete lane coordinates
without placing those references in execution-only structures.

This bridge belongs beside the generic parallel-family rules, not in `MxxWe` and not in a
Diamond-specific proof file.

## 8. Diagnostics

Replace the current catch-all diagnostic with distinct cases:

```text
invalid-dynamic-family-index-reference
unevaluable-dynamic-family-index-bound
dynamic-family-index-out-of-range
ambiguous-family-producer
invalid-family-element-schema
```

The Lean executable report and Rust-side checker wrapper must preserve the category. Parameter
search may treat `dynamic-family-index-out-of-range` as a normal rejected candidate. Missing
references, ambiguous producers, malformed schemas, and checker process failures remain
infrastructure or graph errors and must stop the search rather than causing unbounded modulus
growth.

Retain `Command::env_clear()`: independently of the runaway search, Cargo's inherited Lean
environment may itself exceed the platform's `execve` argument-and-environment limit. It is valid
process-boundary sanitization and does not alter checker semantics. Separately, candidate search
must have an explicit finite search limit and return a typed exhaustion error, so deterministic
candidate rejection cannot grow the modulus without bound.

## 9. File-by-file implementation order

1. `lean/Mxx/Certificate/RecurrenceBasisAlignment.lean`
   - separate malformed-reference errors from unsupported loop-affine normalization;
   - expose only the context-neutral normalization primitive and its closed examples;
   - do not depend on analyzer-owned loop-context types.
2. `lean/Mxx/Certificate/Workflow.lean`
   - add `dynamicFamilyIndexInRange` to `StaticObligation`.
3. `lean/Mxx/Certificate/Checker.lean`
   - define its exact proposition, checker branch, and soundness theorem;
   - add boundary fixtures before changing the analyzer.
4. `lean/Mxx/Certificate/Analyzer.lean`
   - define `DynamicFamilyIndexClassification` and the context-aware
     `classifyDynamicFamilyIndex` wrapper;
   - refactor the pure parallel seed-binding construction;
   - support `ZipOffset`;
   - update `applyFamilyGet` to emit one of the two range obligations;
   - partition closed and carried-dependent body obligations without allowing a raw carried
     placeholder into the global static list;
   - preserve the exact expression reference in element identity.
5. `lean/Mxx/Certificate/RecurrenceSchema.lean`,
   `lean/Mxx/Certificate/SymbolicRecurrenceConstruction.lean`, and
   `lean/Mxx/Certificate/SymbolicRecurrence.lean`
   - add schema-indexed `RecurrenceStepRangeObligation`;
   - translate carried bounds into `IntBoundTransitionExpr` using typed state paths;
   - close external recurrence references;
   - check obligations before every simultaneous numeric transition.
6. `lean/Mxx/Certificate/Rules/ParallelFamilyAnalysis.lean`
   - connect deterministic seed bindings to the existing analyzer-owned family derivation.
7. `lean/Mxx/Certificate/Rules/DynamicFamilyIndex.lean` (new only if the theorem does not fit
   cleanly in `ParallelFamilyAnalysis.lean`)
   - prove the generic interval and loop-offset range lemmas;
   - add the exact parallel-trace bridge without application-specific nodes;
   - connect recurrence-step range checking to the actual sequential trace.
8. `crates/we/lean/MxxWe/AnalysisFacts.lean` and the generic checker report code
   - preserve the new diagnostic categories and obligation details.
9. `crates/we/src/diamond/parameter_search.rs`
   - retain the sanitized checker environment;
   - distinguish rejected candidates, search exhaustion, and checker infrastructure errors;
   - enforce a finite candidate range supplied by the search configuration.

No step may add a Rust implementation of the range or bound calculation.

## 10. Tests

### 10.1 Closed Lean classification tests

Add examples for:

- direct current-loop index -> `loopOffset 0`;
- current-loop index plus `1` -> `loopOffset 1`;
- parameter-only integer -> interval fallback;
- `familyElement`, `select`, `boolToInt`, and coefficient extraction -> interval fallback;
- missing arena reference -> integrity error, not interval fallback;
- escaped `carriedInput` outside a sequential template -> integrity error;
- expression tied to a different loop -> interval fallback using its fact bounds;
- negative literal offset -> interval fallback, followed by range rejection if its interval permits
  a negative value.

### 10.2 Phase-B boundary tests

For a family of count `N`, verify:

- `[0, N - 1]` is accepted;
- `[1, N - 1]` is accepted;
- `[-1, N - 1]` is rejected;
- `[0, N]` is rejected;
- unordered `[2, 1]` is rejected;
- `N = 0` rejects every nonempty interval;
- a zero-count loop satisfies the loop-offset obligation without requiring an element access;
- a one-count loop with offset `N - 1` is accepted and offset `N` is rejected.

Add recurrence-step fixtures in which:

- an index interval stays in range for every iteration and is accepted;
- an intermediate iteration is out of range but the final interval is valid, and is rejected at
  the intermediate iteration;
- two carried values update simultaneously and the range check reads only the immutable previous
  state;
- count zero skips all step obligations; and
- an external earlier recurrence bound is closed before the current recurrence is iterated.

Every test comment must show the inclusive interval arithmetic explicitly.

### 10.3 Analyzer graph fixtures

Construct small frozen graphs covering:

- protocol-input integer family used as a dynamic index under its declared range;
- integer family produced by a parallel loop and consumed by a second parallel loop;
- `Zip`, `Broadcast`, and `ZipOffset` transport;
- a selected integer index with safe branch bounds;
- an out-of-range input contract rejected by Phase B;
- an ambiguous or missing producer rejected before Phase B;
- equal intervals with distinct expression references producing distinct family-element identities.

No fixture may manually construct `ParallelFamilyDerivationSource` or claim that a fact holds.

### 10.4 Existing protocol gates

After the focused fixtures pass:

1. regenerate the generated Lean modules;
2. build `MxxWe.AnalysisFacts` and the generic checker binary;
3. run `mxx_analysis_facts` and require Diamond Phase A to succeed without unsupported nodes;
4. run the focused WE parameter-search test and require a finite selected parameter set;
5. run the explicitly requested small CPU Diamond WE runtime round-trip test for both messages and
   require decryption to equal the encrypted message.

The runtime round-trip validates execution, while the checker tests validate the derived hard
bounds. Neither result may be presented as the deferred end-to-end correctness theorem.

## 11. Performance requirements

- Analyze every loop body once; never iterate over the concrete loop count in Phase A.
- Emit one family-index range obligation per `FamilyGet` node, not per lane.
- Do not expand `RuntimeExpr.familyElement` through producer bodies for ordinary interval checking.
- Reuse the existing expression arena and hash-consed references.
- Phase B evaluates each closed interval once per candidate parameter set.
- A carried-dependent interval is evaluated once per numeric recurrence step, immediately before
  the existing transition; this is the only count-dependent work added by this design.
- Semantic proof replay may select one arbitrary trace lane; it must not construct all lanes.

## 12. Completion criteria

This design change is complete only when:

1. valid `FamilyGetStatic` and `FamilyGetDynamic` nodes never fail solely because their index is not
   in the loop-affine grammar;
2. every family access has a checked static range obligation;
3. malformed arena references and missing/ambiguous family provenance still fail closed;
4. arbitrary runtime indices are accepted only through analyzer-derived intervals;
5. no Rust or certificate-supplied bound/range formula is introduced;
6. `Broadcast`, `Zip`, and `ZipOffset` have exact compact analyzer semantics;
7. the future theorem can recover formal-input truth from the actual parallel trace and the
   analyzer-owned derivation, without a caller-supplied runner or invariant;
8. the focused Lean fixtures, Diamond Phase A, WE parameter search, and the requested runtime
   round-trip all pass; and
9. every carried-dependent family access is checked at every numeric recurrence step, with no raw
   `carriedInput` escaping into top-level static obligations; and
10. active Lean code contains no new axiom, `sorry`, or `admit`.
