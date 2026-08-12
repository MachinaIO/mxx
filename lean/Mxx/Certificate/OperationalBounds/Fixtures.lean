import Mxx.Certificate.OperationalBounds.Evaluation

namespace Mxx.Certificate

open Mxx.Ir

private def operationalGatherFixtureWire (node : Nat) : GatherLookupWire := {
  scope := .root (.standalone 0)
  node
  port := 0
}

private def operationalGatherFixtureOwner (node : Nat) : GatherLookupOwner := {
  indices := operationalGatherFixtureWire (node + 1)
}

private def operationalFixtureGather (node : Nat) (source position : IndexExpr) : IndexExpr :=
  let sourceCount := match source with
    | .variable binder => binder.count
    | _ => .constant 1
  .gather (operationalGatherFixtureOwner node) sourceCount position

private def fixtureDirectScalarFact
    (arena : OperationalExprArena)
    (fact : OperationalFact)
    (indices : IndexValueEnvironment := []) : Except OperationalError OperationalScalarFact := do
  let root ← match fact.payload with
    | .directValue root => pure root
    | .matrix root | .scalar root => throw (.unsupportedOperationalExpr root)
  arena.direct.scalarFactAt [] indices root (arena.direct.values.size + 1)

/-- The same protocol input has one root identity across workflow stages even though each stage
binds it to a different local subject wire. -/
example : (show Except OperationalError Bool from do
    let input : ProtocolInputId := ⟨"shared-key"⟩
    let (arena, left) ← contractFact {} (.root (.workflowStage ⟨"left"⟩))
      { node := 0, port := 0 }
      input (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let (arena, right) ← contractFact arena (.root (.workflowStage ⟨"right"⟩))
      { node := 7, port := 0 }
      input (.bytes (.constant 32)) (.bytes (.constant 32)) []
    match left, right with
    | left@{ payload := .directValue _, .. },
        right@{ payload := .directValue _, .. } =>
        match ← fixtureDirectScalarFact arena left, ← fixtureDirectScalarFact arena right with
        | .bytes left, .bytes right => pure (left.origin == right.origin)
        | _, _ => pure false
    | _, _ => pure false) = .ok true := by
  native_decide

/-- Equal-looking values from different protocol inputs remain distinct. -/
example : (do
    let (arena, left) ← contractFact {} (.root (.workflowStage ⟨"left"⟩))
      { node := 0, port := 0 }
      ⟨"left-key"⟩ (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let (arena, right) ← contractFact arena (.root (.workflowStage ⟨"right"⟩))
      { node := 0, port := 0 }
      ⟨"right-key"⟩ (.bytes (.constant 32)) (.bytes (.constant 32)) []
    match left, right with
    | left@{ payload := .directValue _, .. },
        right@{ payload := .directValue _, .. } =>
        match ← fixtureDirectScalarFact arena left, ← fixtureDirectScalarFact arena right with
        | .bytes left, .bytes right => pure (left.origin != right.origin)
        | _, _ => pure false
    | _, _ => pure false) = .ok true := by
  native_decide

/-- Static elements of one external family retain the root input identity and the selected index. -/
example : (do
    let (arena, family) ← contractFact {} (.root (.workflowStage ⟨"stage"⟩))
      { node := 0, port := 0 }
      ⟨"keys"⟩ (.indexedFamily (.bytes (.constant 32)) (.constant 2))
      (.family (.constant 2) (.bytes (.constant 32))) []
    match family with
    | expression@{ payload := .directValue _, .. } =>
        let binder : IndexVariable ← match expression.context.binders[0]? with
          | some binder => pure binder
          | none => throw (OperationalError.unsupportedOperationalExpr 0)
        let firstMap ← match closedStaticIndexMap [] expression.context binder 0 with
          | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 0)
        let secondMap ← match closedStaticIndexMap [] expression.context binder 1 with
          | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 0)
        let (arena, first) ← arena.reindexDirectMatrixFact firstMap expression
        let (arena, second) ← arena.reindexDirectMatrixFact secondMap expression
        let first ← fixtureDirectScalarFact arena first
        let second ← fixtureDirectScalarFact arena second
        match first, second with
        | .bytes first, .bytes second => pure (first.origin != second.origin)
        | _, _ => pure false
    | _ => pure false) = .ok true := by
  native_decide

/-- Repeating one direct static family map preserves the selected external value identity. -/
example : (do
    let (arena, family) ← contractFact {} (.root (.workflowStage ⟨"stage"⟩))
      { node := 0, port := 0 }
      ⟨"keys"⟩ (.indexedFamily (.bytes (.constant 32)) (.constant 2))
      (.family (.constant 2) (.bytes (.constant 32))) []
    let binder : IndexVariable ← match family.context.binders[0]? with
      | some binder => pure binder
      | none => throw (OperationalError.unsupportedOperationalExpr 0)
    let map ← match closedStaticIndexMap [] family.context binder 1 with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 0)
    let (arena, first) ← arena.reindexDirectMatrixFact map family
    let (arena, second) ← arena.reindexDirectMatrixFact map family
    match first, second with
    | first@{ payload := .directValue _, .. },
        second@{ payload := .directValue _, .. } =>
        match ← fixtureDirectScalarFact arena first, ← fixtureDirectScalarFact arena second with
        | .bytes first, .bytes second => pure (first.origin == second.origin)
        | _, _ => pure false
    | _, _ => pure false) = .ok true := by
  native_decide

private def fixtureType : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 1, columns := .constant 1
}

private def fixtureParams : Mxx.SamplerParams := {
  maxCoefficientBound := 8
  modulus := 17
  ringDimension := 1
  rows := 1
  columns := 1
}

private def knownZeroLeftType : MatrixTypeExpr := {
  modulus := .constant 97, ringDimension := .constant 2,
  rows := .constant 1, columns := .constant 3
}

private def knownZeroRightType : MatrixTypeExpr := {
  modulus := .constant 97, ringDimension := .constant 2,
  rows := .constant 3, columns := .constant 1
}

private def knownZeroOutputType : MatrixTypeExpr := {
  modulus := .constant 97, ringDimension := .constant 2,
  rows := .constant 1, columns := .constant 1
}

/-- Multiplication consumes `knownZeroRows` only after checking the concrete certified count.
The ordinary, constant-polynomial, and unit-column cases respectively use `k*N*L*R`,
`k*L*R`, and the reduced effective inner dimension one. -/
private def knownZeroRowsBoundFixture : Bool :=
  match (do
    let left ← classifiedMatrixFact 900 0 knownZeroLeftType [] 2 false
    let ordinaryRight ← classifiedMatrixFact 901 0 knownZeroRightType [] 1 false
    let constantRight := { ordinaryRight with metadata := { isConstantPolynomial := true } }
      |>.refreshPrimitivePolynomial
    let unitRight := { constantRight with
      metadata := { isConstantPolynomial := true, knownZeroRows := some (.constant 2) } }
      |>.refreshPrimitivePolynomial
    let ordinary ← multiplyConcreteMatrixFacts 902 0 knownZeroOutputType
      .matrixMultiplyBound { node := 901, port := 0 } [] left ordinaryRight
    let constant ← multiplyConcreteMatrixFacts 903 0 knownZeroOutputType
      .matrixMultiplyBound { node := 901, port := 0 } [] left constantRight
    let effective ← multiplyConcreteMatrixFacts 904 0 knownZeroOutputType
      .matrixMultiplyBound { node := 901, port := 0 } [] left unitRight
    let ordinaryBound ← ordinary.evaluateNoiseHardBound []
    let constantBound ← constant.evaluateNoiseHardBound []
    let effectiveBound ← effective.evaluateNoiseHardBound []
    pure (ordinaryBound == 12 && constantBound == 6 && effectiveBound == 2)) with
  | .ok value => value
  | .error _ => false

/-- Summary row witnesses follow the actual product output, including scalar broadcasts. -/
private def boundedSummaryRowCountFixture : Bool :=
  match (do
    let scalarType : MatrixTypeExpr := {
      modulus := .constant 97, ringDimension := .constant 2,
      rows := .constant 1, columns := .constant 1
    }
    let scalar ← classifiedMatrixFact 905 0 scalarType [] 2 false
    let matrix ← classifiedMatrixFact 906 0 knownZeroRightType [] 1 false
    let leftBroadcast ← multiplyConcreteMatrixFacts 907 0 knownZeroRightType
      .matrixMultiplyBound { node := 906, port := 0 } [] scalar matrix
    let rightBroadcast ← multiplyConcreteMatrixFacts 908 0 knownZeroRightType
      .matrixMultiplyBound { node := 905, port := 0 } [] matrix scalar
    let rowCount (fact : OperationalMatrixFact) :=
      fact.polynomial.head?.bind fun term => term.product.factors.head?.bind fun factor =>
        factor.boundedSummary.map OperationalBoundedFactorSummary.rowCount
    pure (rowCount leftBroadcast == some 3 && rowCount rightBroadcast == some 3)) with
  | .ok value => value
  | .error _ => false

/-- Sparse-row metadata is checked at consumption, so forged negative, oversized, and symbolic
counts cannot turn a product into a negative or unjustified bound. -/
private def knownZeroRowsRejectedFixture : Bool :=
  match (do
    let left ← classifiedMatrixFact 910 0 knownZeroLeftType [] 2 false
    let right ← classifiedMatrixFact 911 0 knownZeroRightType [] 1 false
    let rejected (zeroRows : IntExpr) :=
      multiplyConcreteMatrixFacts 912 0 knownZeroOutputType .matrixMultiplyBound
        { node := 911, port := 0 } [] left
        ({ right with metadata := { knownZeroRows := some zeroRows } }.refreshPrimitivePolynomial)
    let negative := match rejected (.constant (-1)) with
      | .error (.flat 912 .invalidKnownZeroRows) => true
      | _ => false
    let oversized := match rejected (.constant 4) with
      | .error (.flat 912 .invalidKnownZeroRows) => true
      | _ => false
    let symbolic := match rejected (.parameter "unproved") with
      | .error (.flat 912 .invalidKnownZeroRows) => true
      | _ => false
    pure (negative && oversized && symbolic)) with
  | .ok value => value
  | .error _ => false

/-- Transform transfer invalidates sparse-row metadata; a transformed unit-shaped factor therefore
uses the ordinary constant-polynomial dimension rather than retaining an unjustified reduction. -/
private def knownZeroRowsTransformInvalidationFixture : Bool :=
  match (do
    let left ← classifiedMatrixFact 920 0 knownZeroLeftType [] 2 false
    let right ← classifiedMatrixFact 921 0 knownZeroRightType [] 1 false
    let sparse := { right with metadata := {
      isConstantPolynomial := true, knownZeroRows := some (.constant 2) } }.refreshPrimitivePolynomial
    let transformed ← transformConcreteMatrixFact 922 0 knownZeroRightType .negate [] sparse
    let product ← multiplyConcreteMatrixFacts 923 0 knownZeroOutputType .matrixMultiplyBound
      { node := 922, port := 0 } [] left transformed
    let bound ← product.evaluateNoiseHardBound []
    pure (transformed.metadata.knownZeroRows.isNone && bound == 6)) with
  | .ok value => value
  | .error _ => false

/-- Exact empty zero polynomials remain zero products independently of sparse-row metadata. -/
private def zeroMatrixExactProductFixture : Bool :=
  match (do
    let left ← classifiedMatrixFact 930 0 knownZeroLeftType [] 2 false
    let zero ← polynomialMatrixFact 931 0 knownZeroRightType [] []
    let product ← multiplyConcreteMatrixFacts 932 0 knownZeroOutputType .matrixMultiplyBound
      { node := 931, port := 0 } [] left zero
    let bound ← product.evaluateNoiseHardBound []
    pure (product.polynomial.isEmpty && bound == 0)) with
  | .ok value => value
  | .error _ => false

private def knownZeroRowsUnitColumnScope : Scope := {
  nodes := #[
    { kind := .gaussianSample knownZeroLeftType (.constant 2), arguments := [],
      outputTypes := [.matrix knownZeroLeftType] },
    { kind := .unitColumnMatrix knownZeroRightType (.constant 1), arguments := [],
      outputTypes := [.matrix knownZeroRightType] },
    { kind := .matrixMultiply, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix knownZeroOutputType] }
  ]
  outputs := [("result", { node := 2, port := 0 })]
  inputNames := []
}

private def knownZeroRowsUnitColumnDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .unitColumnMatrix, arguments := [] },
  { sourceNode := 2, rule := .matrixMultiplyBound,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
] }

/-- The actual Graph IR unit-column producer validates its position and seeds `k - 1` certified
zero rows, giving effective contraction dimension one. -/
private def knownZeroRowsUnitColumnGraphFixture : Bool :=
  match (do
    let facts ← evaluateScopeOperationalWithLayouts knownZeroRowsUnitColumnScope
      knownZeroRowsUnitColumnDerivation [] []
    let unit ← derivedMatrixFactAt 2 facts { node := 1, port := 0 }
    let bound ← matrixMaximum 2 { node := 2, port := 0 } facts []
    pure (unit.metadata.knownZeroRows == some (IntExpr.constant 2) && bound == 2)) with
  | .ok value => value
  | .error _ => false

/-- A unit-column node in a parallel-body context may use the loop coordinate itself.  Every
physical assignment is checked: all three in-range lanes share the same `k - 1` certificate,
while the fourth lane is rejected. -/
private def knownZeroRowsLoopVaryingUnitColumnFixture : Bool :=
  let inRange : Node := {
    kind := .unitColumnMatrix knownZeroRightType (.loopIndex 0)
    arguments := []
    outputTypes := [.matrix knownZeroRightType]
  }
  let domains : List OperationalParameterDomain := [.loopIndex 0 3]
  let accepts := match genericNodeMatrixFactConcrete (.parallelBody (.root (.standalone 0)) 940)
      0 inRange .unitColumnMatrix 0 (.matrix knownZeroRightType) {} [] domains [] with
    | .ok fact => fact.metadata.knownZeroRows == some (IntExpr.constant 2)
    | .error _ => false
  let rejects := match genericNodeMatrixFactConcrete (.parallelBody (.root (.standalone 0)) 941)
      0 inRange .unitColumnMatrix 0 (.matrix knownZeroRightType) {} []
        [.loopIndex 0 4] [] with
    | .error (.invalidMatrixParameters 0) => true
    | _ => false
  accepts && rejects

example : knownZeroRowsBoundFixture = true := by native_decide
example : boundedSummaryRowCountFixture = true := by native_decide
example : knownZeroRowsRejectedFixture = true := by native_decide
example : knownZeroRowsTransformInvalidationFixture = true := by native_decide
example : zeroMatrixExactProductFixture = true := by native_decide
example : knownZeroRowsUnitColumnGraphFixture = true := by native_decide
example : knownZeroRowsLoopVaryingUnitColumnFixture = true := by native_decide

private def interningFixtureFactor (node : Nat) : OperationalFactorKey := {
  leaf := .primitive (.matrix (.value temporaryScope { node, port := 0 }))
  inputType := fixtureType
  outputType := fixtureType
  role := .large
}

private def interningFixtureProduct (factors : List OperationalFactorKey) : OperationalProductKey := {
  factors
  modes := List.replicate (factors.length - 1) .ordinaryMatrixProduct
  outputType := fixtureType
}

/-- Factor fingerprints are candidate indices only: equal factors reuse an ID, while two unequal
factors deliberately sharing this coarse fingerprint remain distinct. -/
example :
    let first := interningFixtureFactor 10
    let different := interningFixtureFactor 11
    let (arena, firstId) := internOperationalFactor {} first
    let (arena, repeatedId) := internOperationalFactor arena first
    let (arena, differentId) := internOperationalFactor arena different
    firstId = repeatedId ∧ firstId ≠ differentId ∧
      arena.factorHits = 1 ∧ arena.factorMisses = 2 := by
  native_decide

/-- Ordered products receive stable request-local IDs and cancellation retains deterministic
first-occurrence order. -/
example :
    let first := interningFixtureFactor 20
    let second := interningFixtureFactor 21
    let forward := interningFixtureProduct [first, second]
    let reverse := interningFixtureProduct [second, first]
    let (arena, forwardId) := internOperationalProduct {} forward
    let (arena, repeatedId) := internOperationalProduct arena forward
    let (_, reverseId) := internOperationalProduct arena reverse
    let normalized := normalizeOperationalTerms [
      { coefficient := 3, product := forward },
      { coefficient := 5, product := reverse },
      { coefficient := -3, product := forward }]
    (forwardId == repeatedId) = true ∧ (forwardId == reverseId) = false ∧
      (normalized == [{ coefficient := 5, product := reverse }]) = true := by
  native_decide

/-- An exact external matrix is not a zero matrix. Without an explicit bounded contract it keeps
the conservative centered-residue cap and a Large primitive factor. -/
example : (do
    let (arena, fact) ← contractFact {} (.root (.workflowStage ⟨"stage"⟩))
      { node := 0, port := 0 }
      ⟨"matrix"⟩ (.matrix fixtureType) (.matrixExact fixtureType none false) []
    match fact with
    | expression@{ context := { binders := #[] }, payload := .directValue _, .. } =>
      let matrix ← arena.directValueRepresentativeFactAt [] expression
        if matrix.polynomial.any operationalTermIsSignal then
          matrix.totalHardBound.evaluate [] #[]
        else pure (-1)
    | _ => pure (-1)) = .ok 8 := by
  native_decide

private def fixtureFamilyBinder : FamilyTemplateBinder := {
  owner := .root (.standalone 7)
  producerNode := 4
  binderSlot := 0
}

private def fixtureSampledIdentity : PublicMatrixIdentity :=
  .sampledTrapdoor (.parallelBody (.root (.standalone 7)) 4) { node := 0, port := 0 }

private def fixturePublicMatrixFact : OperationalMatrixFact := ({
  subject := { node := 0, port := 0 }
  origin := .value (.parallelBody (.root (.standalone 7)) 4) { node := 0, port := 0 }
  matrixType := fixtureType
  matrixParams := fixtureParams
  totalHardBound := .closedInt (.constant 8)
  identity := some fixtureSampledIdentity
} : OperationalMatrixFact).initializePrimitivePolynomial .large

private def fixturePublicFact
    (arena : OperationalExprArena) : Except OperationalError (OperationalExprArena × OperationalFact) :=
  arena.liftConcreteMatrixFact fixturePublicMatrixFact

/-- Sequential schemas ignore alternate encodings of bounded-only zero signal, but retain the
ordered structure of every Large-bearing term. -/
private def carriedSignalSchemaFixture : Bool :=
  let base : OperationalMatrixFact := ({
    subject := { node := 0, port := 0 }
    origin := .value temporaryScope { node := 0, port := 0 }
    matrixType := fixtureType
    matrixParams := fixtureParams
    totalHardBound := .closedInt (.constant 8)
  } : OperationalMatrixFact).initializePrimitivePolynomial .bounded
  let withBoundedZero := { base with polynomial := base.polynomial ++
    (base.polynomial.map fun term => { term with coefficient := 0 }) }
  let large := base.initializePrimitivePolynomial .large
  let changedLarge := { large with polynomial := large.polynomial.map fun term => { term with
    product := { term.product with factors := term.product.factors.map fun factor =>
      { factor with transforms := [.transpose] } } } }
  match (do
    let (arena, base) ← ({} : OperationalExprArena).liftConcreteMatrixFact base
    let (arena, withBoundedZero) ← arena.liftConcreteMatrixFact withBoundedZero
    let (arena, large) ← arena.liftConcreteMatrixFact large
    let (arena, changedLarge) ← arena.liftConcreteMatrixFact changedLarge
    pure (sameCarriedSchema arena base withBoundedZero &&
      !sameCarriedSchema arena large changedLarge)) with
  | .ok value => value
  | .error _ => false

example : carriedSignalSchemaFixture = true := by
  native_decide

/-- Sequential carried-bound rewriting maps the checked Shared representative in place.  The
logical selection context and Shared storage remain attached to the indexed result. -/
private def indexedCarriedRecurrenceFixture : Bool :=
  match (do
    let selection := DynamicSelectionIdentity.fromOrigin
      (.local temporaryScope { node := 744, port := 0 }) 2
    let base : OperationalMatrixFact := ({
      subject := { node := 744, port := 0 }
      origin := .value temporaryScope { node := 744, port := 0 }
      matrixType := fixtureType
      matrixParams := fixtureParams
      totalHardBound := .closedInt (.constant 3)
    } : OperationalMatrixFact).initializePrimitivePolynomial .bounded
    let (arena, representative) := ({} : OperationalExprArena).pushConcrete base
    let (arena, root) ← arena.pushSharedSelection selection 2 representative
      (selectedMatrixSummary #[base])
    let indexed ← arena.indexedExpr root
    let arena ← arena.rememberIndexedExpr indexed
    let fact : OperationalFact := indexed
    let (arena, abstract) ← abstractCarriedMaximum 0 arena fact
    let (arena, recurrent) ← setFactRecurrenceState 2 [(.matrixMaximum 0 0)]
      [.closedInt (.constant 3)] [.previous (.matrixMaximum 0 0)] 0 [] arena abstract
    match recurrent with
    | mapped@{ payload := .matrix mappedRoot, .. } =>
        let bounds ← arena.foldMatrixConcreteLeaves mappedRoot ([] : List OperationalBoundExpr)
          (fun values leaf => pure (values ++ [leaf.totalHardBound]))
        pure (mapped.context == indexed.context && mapped.storage == IndexedStorage.sharedTemplate &&
          bounds.all fun bound => match bound with
            | .recurrenceState 2 _ _ _ (.matrixMaximum 0 0) => true
            | _ => false)
    | _ => pure false) with
  | .ok value => value
  | .error _ => false

example : indexedCarriedRecurrenceFixture = true := by
  native_decide

private def fixtureTrapdoorFact : OperationalScalarFact := .trapdoor {
  subject := { node := 0, port := 1 }
  matrixType := fixtureType
  matrixParams := fixtureParams
  maximum := .closedInt (.constant 3)
  preimageCutoff := some (.contextual .maximum [] [] (.constant 3))
  publicIdentity := fixtureSampledIdentity
}

private def sharedPreimageBaseScope : Scope := {
  nodes := #[
    {
      kind := .trapdoorSample fixtureType (.constant 3)
      arguments := []
      outputCount := 2
      outputTypes := [
        .matrix fixtureType,
        .trapdoor fixtureType (.rational 1) (.constant 2) (.constant 1) (.constant 3)
      ]
    },
    {
      kind := .gaussianSample fixtureType (.constant 2)
      arguments := []
      outputTypes := [.matrix fixtureType]
    },
    {
      kind := .identityMatrix fixtureType
      arguments := []
      outputTypes := [.matrix fixtureType]
    },
    {
      kind := .preimageSample fixtureType (.constant 3)
      arguments := [
        { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
      ]
      outputTypes := [.preimage fixtureType]
    },
    {
      kind := .preimageSample fixtureType (.constant 3)
      arguments := [
        { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 2, port := 0 }
      ]
      outputTypes := [.preimage fixtureType]
    }
  ]
  outputs := [
    ("first", { node := 3, port := 0 }),
    ("second", { node := 4, port := 0 })
  ]
  inputNames := []
}

private def sharedPreimageBaseDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .trapdoorSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .identityMatrix, arguments := [] },
  { sourceNode := 3, rule := .preimageSample,
    arguments := [
      { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
    ] },
  { sourceNode := 4, rule := .preimageSample,
    arguments := [
      { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 2, port := 0 }
    ] }
] }

/-- Branch-specific targets create distinct preimages and target snapshots, but both relations
retain the one source public matrix identity. This is the Diamond transition shape
`B*K_d = P_d (mod R_q)`: the digit changes `K_d` and `P_d`, never `B`. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts sharedPreimageBaseScope
      sharedPreimageBaseDerivation [] []
    let first ← derivedMatrixFactAt 4 facts { node := 3, port := 0 }
    let second ← derivedMatrixFactAt 4 facts { node := 4, port := 0 }
    match first.relations, second.relations with
    | [.preimage left], [.preimage right] =>
        pure (left.publicIdentity == right.publicIdentity &&
          left.targetOrigin != right.targetOrigin && left.producer != right.producer)
    | _, _ => pure false) = .ok true := by
  native_decide

/-- The exact same family and exact same executable index wire preserve the public/private pair. -/
example : (do
    let selection := DynamicSelectionIdentity.fromOrigin
      (.local (.root (.standalone 7)) { node := 3, port := 0 }) 2
    let (arena, trapdoorInput) ← pushIndexedScalarFact {} fixtureTrapdoorFact
    let (arena, publicInput) ← fixturePublicFact arena
    let (arena, publicFact) ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 5, port := 0 } arena publicInput
    let (arena, trapdoor) ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 6, port := 0 } arena trapdoorInput
    match publicFact, trapdoor with
    | publicFact@{ payload := .matrix _, .. }, trapdoor@{ payload := .scalar _, .. } =>
        match ← arena.concreteFact publicFact.payload, ← arena.concreteIndexedScalar trapdoor with
        | publicFact, .trapdoor trapdoor => pure (publicFact.identity == some trapdoor.publicIdentity)
        | _, _ => pure false
    | _, _ => pure false) = .ok true := by
  native_decide

/-- Merely equal-looking selections from different executable index wires do not compare equal. -/
example : (do
    let publicSelection := DynamicSelectionIdentity.fromOrigin
      (.local (.root (.standalone 7)) { node := 3, port := 0 }) 2
    let trapdoorSelection := DynamicSelectionIdentity.fromOrigin
      (.local (.root (.standalone 7)) { node := 4, port := 0 }) 2
    let (arena, trapdoorInput) ← pushIndexedScalarFact {} fixtureTrapdoorFact
    let (arena, publicInput) ← fixturePublicFact arena
    let (arena, publicFact) ← selectDynamicUniformFact fixtureFamilyBinder publicSelection
      { node := 5, port := 0 } arena publicInput
    let (arena, trapdoor) ← selectDynamicUniformFact fixtureFamilyBinder trapdoorSelection
      { node := 6, port := 0 } arena trapdoorInput
    match publicFact, trapdoor with
    | publicFact@{ payload := .matrix _, .. }, trapdoor@{ payload := .scalar _, .. } =>
        match ← arena.concreteFact publicFact.payload, ← arena.concreteIndexedScalar trapdoor with
        | publicFact, .trapdoor trapdoor => pure (!(publicFact.identity == some trapdoor.publicIdentity))
        | _, _ => pure false
    | _, _ => pure false) = .ok true := by
  native_decide

/-- The flat polynomial, not merely the outer fact, preserves dynamic-selection identity. -/
example : (do
    let selection := DynamicSelectionIdentity.fromOrigin
      (.local (.root (.standalone 7)) { node := 3, port := 0 }) 2
    let differentSelection := DynamicSelectionIdentity.fromOrigin
      (.local (.root (.standalone 7)) { node := 4, port := 0 }) 2
    let (arena, publicInput) ← fixturePublicFact {}
    let (arena, first) ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 5, port := 0 } arena publicInput
    let (arena, publicInput) ← fixturePublicFact arena
    let (arena, same) ← selectDynamicUniformFact fixtureFamilyBinder selection
      { node := 6, port := 0 } arena publicInput
    let (arena, publicInput) ← fixturePublicFact arena
    let (arena, different) ← selectDynamicUniformFact fixtureFamilyBinder differentSelection
      { node := 7, port := 0 } arena publicInput
    match first, same, different with
    | first@{ payload := .matrix _, .. }, same@{ payload := .matrix _, .. },
        different@{ payload := .matrix _, .. } =>
        let first ← arena.concreteFact first.payload
        let same ← arena.concreteFact same.payload
        let different ← arena.concreteFact different.payload
        pure (first.polynomial == same.polynomial && first.polynomial != different.polynomial)
    | _, _, _ => pure false) = .ok true := by
  native_decide

private def fixtureScope : Scope := {
  nodes := #[
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 2, port := 0 })], inputNames := []
}

private def fixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd, arguments := [{ node := 0, port := 0 },
    { node := 1, port := 0 }] }
] }

example : (do
    let facts ← evaluateScopeOperationalWithLayouts fixtureScope fixtureDerivation [] []
    matrixMaximum 2 { node := 2, port := 0 } facts []) = .ok 3 := by
  native_decide

/-- A fresh sample produced by one parallel-body template denotes a different source in each
lane, so subtraction across distinct lanes cannot cancel structurally. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts fixtureScope fixtureDerivation [] []
    let sample ← lookupFact 2 facts { node := 1, port := 0 }
    let (arena, first) ← instantiateFactLoopIndex 0 0 facts.arena sample
    let (arena, second) ← instantiateFactLoopIndex 0 1 arena sample
    match first, second with
    | first@{ context := { binders := #[] }, payload := .matrix _, .. },
        second@{ context := { binders := #[] }, payload := .matrix _, .. } =>
        let first ← arena.concreteFact first.payload
        let second ← arena.concreteFact second.payload
        pure (!(subtractOperationalPolynomials first.polynomial second.polynomial).isEmpty)
    | first@{ context := { binders := #[] }, payload := .directValue _, .. },
        second@{ context := { binders := #[] }, payload := .directValue _, .. } =>
        let first ← arena.directValueFactAt [] first
        let second ← arena.directValueFactAt [] second
        pure (!(subtractOperationalPolynomials first.polynomial second.polynomial).isEmpty)
    | _, _ => pure false) = .ok true := by
  native_decide

private def scaledNoiseScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixScale (.constant 2), arguments := [{ node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := []
}

private def scaledNoiseDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .matrixScale, arguments := [{ node := 0, port := 0 }] }
] }

/-- The additive coefficient outside a compressed bounded product remains part of its bound. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let fact ← derivedMatrixFactAt 1 facts { node := 1, port := 0 }
    fact.evaluateNoiseHardBound []) = .ok 6 := by
  native_decide

private def mixedSignalNoiseScope : Scope := {
  nodes := #[
    { kind := .uniformResidueSample fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 2, port := 0 })]
  inputNames := []
}

private def mixedSignalNoiseDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .uniformResidueSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
] }

/-- A mixed signal/noise value keeps an unconditional whole-value cap while exposing noise
separately for the endpoint inequality. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts mixedSignalNoiseScope
      mixedSignalNoiseDerivation [] []
    let fact ← derivedMatrixFactAt 2 facts { node := 2, port := 0 }
    let total ← fact.totalHardBound.evaluate [] #[]
    let noise ← fact.evaluateNoiseHardBound []
    pure (total, noise)) = .ok (8, 3) := by
  native_decide

private def flatCancellationScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixSubtract,
      arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := []
}

private def flatCancellationDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .matrixSubtract,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] }
] }

/-- Exact factor-list equality, rather than equality of numeric bounds, eliminates `E-E`. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts flatCancellationScope
      flatCancellationDerivation [] []
    let result ← derivedMatrixFactAt 1 facts { node := 1, port := 0 }
    pure result.polynomial.isEmpty) = .ok true := by
  native_decide

private def flatNoiseOrderScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixSubtract,
      arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 4, port := 0 })]
  inputNames := []
}

private def flatNoiseOrderDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .matrixAdd,
    arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 4, rule := .matrixSubtract,
    arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }] }
] }

/-- Canonical bounded-noise provenance is independent of additive construction order. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts flatNoiseOrderScope
      flatNoiseOrderDerivation [] []
    let result ← derivedMatrixFactAt 4 facts { node := 4, port := 0 }
    pure result.polynomial.isEmpty) = .ok true := by
  native_decide

private def flatMultiLargeScope : Scope := {
  nodes := #[
    { kind := .uniformResidueSample fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .uniformResidueSample fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 2, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 3, port := 0 })]
  inputNames := []
}

private def flatMultiLargeDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .uniformResidueSample, arguments := [] },
  { sourceNode := 1, rule := .uniformResidueSample, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .matrixMultiplyBound,
    arguments := [{ node := 2, port := 0 }, { node := 2, port := 0 }] }
] }

/-- Multiplication distributes over signal sums; two Large factors remain signal, not opaque. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts flatMultiLargeScope
      flatMultiLargeDerivation [] []
    let result ← derivedMatrixFactAt 3 facts { node := 3, port := 0 }
    pure (result.polynomial.length, result.polynomial.all fun term =>
      operationalLargeFactorCount term = 2)) = .ok (4, true) := by
  native_decide

example : checkScopeDerivation fixtureScope { steps := #[
  { sourceNode := 1, rule := .gaussianSample, arguments := [] }
] } = .error (.sourceNodeMismatch 0 1) := by
  native_decide

private def gadgetFixtureScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] }
  ],
  outputs := [("result", { node := 1, port := 0 })], inputNames := []
}

private def gadgetFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] }
] }

/-- A request cannot silently derive a CRT layout from a graph-visible modulus. -/
example : (match evaluateScopeOperationalWithLayouts gadgetFixtureScope gadgetFixtureDerivation [] [] with
    | .error (.missingGadgetLayout 1) => true
    | _ => false) = true := by
  native_decide

private def fixtureLayout : Mxx.GadgetLayoutDescriptor := {
  paramsId := "fixture"
  ringDimension := 1
  crtModuli := [17]
  crtBits := 1
  baseBits := 1
  base := 2
  regularDigitCount := 1
  smallDigitCount := 1
  smallestCrtModulus := 17
}

private def fixtureRows2Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 2, columns := .constant 1
}

private def fixtureRows4Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 4, columns := .constant 1
}

private def fixtureColumns2Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 1, columns := .constant 2
}

private def fixtureSquare2Type : MatrixTypeExpr := {
  modulus := .constant 17, ringDimension := .constant 1,
  rows := .constant 2, columns := .constant 2
}

private def matrixTransformCoverageScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .identityMatrix fixtureType, arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixSubtract,
      arguments := [{ node := 2, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 1, port := 0 }, { node := 3, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixNegate, arguments := [{ node := 4, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixScale (.constant (-2)), arguments := [{ node := 5, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .transpose, arguments := [{ node := 6, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .slice none none, arguments := [{ node := 7, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .extractCoefficient (.constant 0), arguments := [{ node := 0, port := 0 }],
      outputTypes := [.integer] },
    { kind := .liftIntegerToConstantPolynomial fixtureType, arguments := [{ node := 9, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .tensor, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .concat .rows, arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureRows2Type] },
    { kind := .transpose, arguments := [{ node := 12, port := 0 }],
      outputTypes := [.matrix fixtureColumns2Type] },
    { kind := .concat .columns,
      arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureColumns2Type] },
    { kind := .concat .diagonal,
      arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureSquare2Type] },
    { kind := .slice (some (.constant 0, .constant 1))
        (some (.constant 0, .constant 1)), arguments := [{ node := 15, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .uniformIntervalSample fixtureType (.constant (-2)) (.constant 4), arguments := [],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 17, port := 0 })]
  inputNames := []
}

private def matrixTransformCoverageDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .identityMatrix, arguments := [] },
  { sourceNode := 2, rule := .matrixAdd,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .matrixSubtract,
    arguments := [{ node := 2, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 4, rule := .matrixMultiplyBound,
    arguments := [{ node := 1, port := 0 }, { node := 3, port := 0 }] },
  { sourceNode := 5, rule := .matrixNegate, arguments := [{ node := 4, port := 0 }] },
  { sourceNode := 6, rule := .matrixScale, arguments := [{ node := 5, port := 0 }] },
  { sourceNode := 7, rule := .transpose, arguments := [{ node := 6, port := 0 }] },
  { sourceNode := 8, rule := .slice, arguments := [{ node := 7, port := 0 }] },
  { sourceNode := 9, rule := .extractCoefficient, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 10, rule := .liftIntegerToConstantPolynomial, arguments := [{ node := 9, port := 0 }] },
  { sourceNode := 11, rule := .tensor,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 12, rule := .concat,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 13, rule := .transpose, arguments := [{ node := 12, port := 0 }] },
  { sourceNode := 14, rule := .concat,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 15, rule := .concat,
    arguments := [{ node := 0, port := 0 }, { node := 0, port := 0 }] },
  { sourceNode := 16, rule := .slice, arguments := [{ node := 15, port := 0 }] },
  { sourceNode := 17, rule := .uniformIntervalSample, arguments := [] }
] }

/-- Every non-relation matrix arithmetic/transform variant reaches an explicit operational
transfer. The equalities below also pin conservative inter-node bounded-summary subtraction,
centered-cap scaling, coefficient selection, tensor-with-identity, and interval sampling. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts matrixTransformCoverageScope
      matrixTransformCoverageDerivation [] []
    let afterCancellation ← matrixMaximum 17 { node := 3, port := 0 } facts []
    let afterScale ← matrixMaximum 17 { node := 6, port := 0 } facts []
    let coefficient ← matrixMaximum 17 { node := 10, port := 0 } facts []
    let tensor ← matrixMaximum 17 { node := 11, port := 0 } facts []
    let interval ← matrixMaximum 17 { node := 17, port := 0 } facts []
    pure (afterCancellation, afterScale, coefficient, tensor, interval)) =
      .ok (5, 8, 8, 3, 4) := by
  native_decide


private def samplerAndDecodeCoverageScope : Scope := {
  nodes := #[
    { kind := .trapdoorSample fixtureType (.constant 3), arguments := [], outputCount := 2,
      outputTypes := [
        .matrix fixtureType,
        .trapdoor fixtureType (.rational 1) (.constant 2) (.constant 1) (.constant 3)
      ] },
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .preimageSample fixtureType (.constant 3),
      arguments := [
        { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
      ], outputTypes := [.preimage fixtureType] },
    { kind := .trapdoorPublic, arguments := [{ node := 0, port := 1 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .thresholdDecodeBool (.constant 17) (.constant 2) (.constant 1),
      arguments := [{ node := 1, port := 0 }], outputTypes := [.boolean] },
    { kind := .thresholdDecodeInt (.constant 17) (.constant 3) (.constant 1),
      arguments := [{ node := 1, port := 0 }], outputTypes := [.integer] },
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .identityMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .crtRecompose [.constant 2, .constant 3] [.constant 9, .constant 6],
      arguments := [{ node := 6, port := 0 }, { node := 7, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .constantBool false, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool false, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .constantBool false, arguments := [], outputTypes := [.boolean] },
    { kind := .familyPack,
      arguments := [
        { node := 9, port := 0 }, { node := 10, port := 0 }, { node := 11, port := 0 },
        { node := 12, port := 0 }, { node := 13, port := 0 }
      ], outputTypes := [.indexedFamily .boolean (.constant 5)] },
    { kind := .packPolynomialCoefficients fixtureType (.constant 5),
      arguments := [{ node := 14, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("packed", { node := 15, port := 0 })]
  inputNames := []
}

private def samplerAndDecodeCoverageDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .trapdoorSample, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .preimageSample,
    arguments := [
      { node := 0, port := 0 }, { node := 0, port := 1 }, { node := 1, port := 0 }
    ] },
  { sourceNode := 3, rule := .trapdoorPublic, arguments := [{ node := 0, port := 1 }] },
  { sourceNode := 4, rule := .thresholdDecodeBool, arguments := [{ node := 1, port := 0 }] },
  { sourceNode := 5, rule := .thresholdDecodeInt, arguments := [{ node := 1, port := 0 }] },
  { sourceNode := 6, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 7, rule := .identityMatrix, arguments := [] },
  { sourceNode := 8, rule := .crtRecompose,
    arguments := [{ node := 6, port := 0 }, { node := 7, port := 0 }] },
  { sourceNode := 9, rule := .constantBool, arguments := [] },
  { sourceNode := 10, rule := .constantBool, arguments := [] },
  { sourceNode := 11, rule := .constantBool, arguments := [] },
  { sourceNode := 12, rule := .constantBool, arguments := [] },
  { sourceNode := 13, rule := .constantBool, arguments := [] },
  { sourceNode := 14, rule := .familyPack,
    arguments := [
      { node := 9, port := 0 }, { node := 10, port := 0 }, { node := 11, port := 0 },
      { node := 12, port := 0 }, { node := 13, port := 0 }
    ] },
  { sourceNode := 15, rule := .packPolynomialCoefficients,
    arguments := [{ node := 14, port := 0 }] }
] }

/-- Sampler pairing, preimage ownership, threshold outputs, CRT recomposition, Boolean-family
packing, and residue reconstruction all reach explicit transfers in one closed fixture. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts samplerAndDecodeCoverageScope
      samplerAndDecodeCoverageDerivation [] []
    let publicFact ← derivedMatrixFactAt 15 facts { node := 0, port := 0 }
    let recovered ← derivedMatrixFactAt 15 facts { node := 3, port := 0 }
    let preimage ← derivedMatrixFactAt 15 facts { node := 2, port := 0 }
    let decoded ← integerFactAt 15 facts { node := 5, port := 0 }
    let packed ← derivedMatrixFactAt 15 facts { node := 15, port := 0 }
    pure (publicFact.identity == recovered.identity, preimage.relations.length,
      decoded.lower, decoded.upper, packed.polynomial.any operationalTermIsSignal)) =
      .ok (true, 1, 0, 2, true) := by
  native_decide

private def hashIdentityFixtureScope : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.constant 7] [] [] none none,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] },
    { kind := .hashSample fixtureType .decomposed [109, 120, 120] [.constant 7] [] []
        (some (.constant 2)) (some (.constant 1)),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] }
  ]
  outputs := [("plain", { node := 1, port := 0 }), ("decomposed", { node := 2, port := 0 })]
  inputNames := ["key"]
}

private def hashIdentityFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .hashSample, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 2, rule := .hashSample, arguments := [{ node := 0, port := 0 }] }
] }

/-- Plain and decomposed modes of the same fully evaluated hash query share the target identity. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts hashIdentityFixtureScope
      hashIdentityFixtureDerivation [] [fixtureLayout]
    let plain ← derivedMatrixFactAt 2 facts { node := 1, port := 0 }
    let decomposed ← derivedMatrixFactAt 2 facts { node := 2, port := 0 }
    match decomposed.relations with
    | [.decomposition relation] => pure (plain.origin == relation.inputOrigin)
    | _ => pure false) = .ok true := by
  native_decide

private def trailingHashIdentityFixtureScope : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .constantInt 9, arguments := [], outputTypes := [.integer] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.constant 7] [] [] none none,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .hashSample fixtureType .decomposed [109, 120, 120] [.constant 7] [] []
        (some (.constant 2)) (some (.constant 1)),
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.preimage fixtureType] }
  ]
  outputs := [("plain", { node := 2, port := 0 }), ("decomposed", { node := 3, port := 0 })]
  inputNames := ["key"]
}

private def trailingHashIdentityFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .constantInt, arguments := [] },
  { sourceNode := 2, rule := .hashSample,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .hashSample,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
] }

/-- A trailing integer operand participates in the plain/decomposed query identity in exact
argument order rather than being silently discarded. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts trailingHashIdentityFixtureScope
      trailingHashIdentityFixtureDerivation [] [fixtureLayout]
    let plain ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let decomposed ← derivedMatrixFactAt 3 facts { node := 3, port := 0 }
    match decomposed.relations with
    | [.decomposition relation] => pure (plain.origin == relation.inputOrigin)
    | _ => pure false) = .ok true := by
  native_decide

/-- Two stages hashing the same protocol key with the same complete query receive one semantic
hash origin even though their formal input and output wires are separately namespaced. -/
example : (do
    let input : ProtocolInputId := ⟨"shared-key"⟩
    let leftScope := ScopeTemplateKey.root (.workflowStage ⟨"left"⟩)
    let rightScope := ScopeTemplateKey.root (.workflowStage ⟨"right"⟩)
    let (leftArena, leftInput) ← contractFact {} leftScope { node := 0, port := 0 } input
      (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let (rightArena, rightInput) ← contractFact {} rightScope { node := 0, port := 0 } input
      (.bytes (.constant 32)) (.bytes (.constant 32)) []
    let leftFacts ← evaluateScopeOperationalWithKey leftScope hashIdentityFixtureScope
      hashIdentityFixtureDerivation [] [fixtureLayout] [leftInput] leftArena
    let rightFacts ← evaluateScopeOperationalWithKey rightScope hashIdentityFixtureScope
      hashIdentityFixtureDerivation [] [fixtureLayout] [rightInput] rightArena
    let left ← derivedMatrixFactAt 2 leftFacts { node := 1, port := 0 }
    let right ← derivedMatrixFactAt 2 rightFacts { node := 1, port := 0 }
    pure (left.origin == right.origin)) = .ok true := by
  native_decide

private def scalarIntervalFixtureScope : Scope := {
  nodes := #[
    { kind := .constantInt (-2), arguments := [], outputTypes := [.integer] },
    { kind := .constantInt 3, arguments := [], outputTypes := [.integer] },
    { kind := .intBinary .multiply,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.integer] },
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .boolToInt, arguments := [{ node := 3, port := 0 }], outputTypes := [.integer] }
  ]
  outputs := [
    ("product", { node := 2, port := 0 }),
    ("bit", { node := 4, port := 0 })
  ]
  inputNames := []
}

private def scalarIntervalFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .constantInt, arguments := [] },
  { sourceNode := 2, rule := .intBinary,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
  { sourceNode := 3, rule := .constantBool, arguments := [] },
  { sourceNode := 4, rule := .boolToInt, arguments := [{ node := 3, port := 0 }] }
] }

/-- Scalar facts are derived from executable semantics rather than the former `[0, 0]`
fallback. Signed multiplication and Boolean conversion retain sound intervals. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scalarIntervalFixtureScope
      scalarIntervalFixtureDerivation [] []
    let product ← integerFactAt 5 facts { node := 2, port := 0 }
    let bit ← integerFactAt 5 facts { node := 4, port := 0 }
    pure (product.lower, product.upper, bit.lower, bit.upper)) = .ok (-6, -6, 0, 1) := by
  native_decide

private def malformedScalarOutputScope : Scope := {
  nodes := #[
    { kind := .constantInt 1, arguments := [], outputTypes := [.boolean] }
  ]
  outputs := []
  inputNames := []
}

private def malformedScalarOutputDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] }
] }

/-- A derivation cannot disguise an integer producer as a Boolean output. -/
example : (match evaluateScopeOperationalWithLayouts malformedScalarOutputScope
    malformedScalarOutputDerivation [] [] with
  | .error (.outputTypeMismatch 0) => true
  | _ => false) = true := by
  native_decide

private def negativeBitScope : Scope := {
  nodes := #[
    { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
    { kind := .bitExtract (.constant (-1)), arguments := [{ node := 0, port := 0 }],
      outputTypes := [.boolean] }
  ]
  outputs := []
  inputNames := []
}

private def negativeBitDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .bitExtract, arguments := [{ node := 0, port := 0 }] }
] }

/-- A negative bit position is rejected rather than coerced to a natural number. -/
example : (match evaluateScopeOperationalWithLayouts negativeBitScope negativeBitDerivation [] [] with
  | .error (.invalidCount 1 (-1)) => true
  | _ => false) = true := by
  native_decide

private def scalarTypeMismatchScope : Scope := {
  nodes := #[
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .boolToInt, arguments := [{ node := 0, port := 0 }], outputTypes := [.integer] }
  ]
  outputs := []
  inputNames := []
}

private def scalarTypeMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .boolToInt, arguments := [{ node := 0, port := 0 }] }
] }

/-- Scalar transfer rules reject operands of a different executable wire type. -/
example : (match evaluateScopeOperationalWithLayouts scalarTypeMismatchScope
    scalarTypeMismatchDerivation [] [] with
  | .error (.operandNotBoolean 1 { node := 0, port := 0 }) => true
  | _ => false) = true := by
  native_decide

private def selectRangeMismatchScope : Scope := {
  nodes := #[
    { kind := .constantInt 2, arguments := [], outputTypes := [.integer] },
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .identityMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .select,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
        { node := 2, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := []
  inputNames := []
}

private def selectRangeMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 2, rule := .identityMatrix, arguments := [] },
  { sourceNode := 3, rule := .select,
    arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
      { node := 2, port := 0 }] }
] }

/-- A dynamic selector must be proved inside the executable branch range. -/
example : (match evaluateScopeOperationalWithLayouts selectRangeMismatchScope
    selectRangeMismatchDerivation [] [] with
  | .error (.invalidCount 3 2) => true
  | _ => false) = true := by
  native_decide

private def crtMetadataMismatchScope : Scope := {
  nodes := #[
    { kind := .zeroMatrix fixtureType, arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .crtRecompose [.constant 2, .constant 3] [.constant 1, .constant 1],
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := []
  inputNames := []
}

private def crtMetadataMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
  { sourceNode := 1, rule := .crtRecompose, arguments := [{ node := 0, port := 0 }] }
] }

/-- CRT metadata is positional and must have exactly one entry for every operand. -/
example : (match evaluateScopeOperationalWithLayouts crtMetadataMismatchScope
    crtMetadataMismatchDerivation [] [] with
  | .error (.unsupportedOutputArity 1 1) => true
  | _ => false) = true := by
  native_decide

private def packedPolynomialInputMismatchScope : Scope := {
  nodes := #[
    { kind := .constantBool true, arguments := [], outputTypes := [.boolean] },
    { kind := .packPolynomialCoefficients fixtureType (.constant 5),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := []
  inputNames := []
}

private def packedPolynomialInputMismatchDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantBool, arguments := [] },
  { sourceNode := 1, rule := .packPolynomialCoefficients,
    arguments := [{ node := 0, port := 0 }] }
] }

/-- Polynomial reconstruction accepts only the exact Boolean family shape required by the IR. -/
example : (match evaluateScopeOperationalWithLayouts packedPolynomialInputMismatchScope
    packedPolynomialInputMismatchDerivation [] [] with
  | .error (.loopInputModeMismatch 1 0) => true
  | _ => false) = true := by
  native_decide

private def loopHashBody : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.loopIndex 0] [] [] none none,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := ["key"]
}

private def loopHashProgram : Prog := {
  root := {
    nodes := #[
      { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
      { kind := .parallelLoop "body" (.constant 2) 0 [] [.broadcast],
        arguments := [{ node := 0, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .familyGetStatic (.constant 0), arguments := [{ node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyGetStatic (.constant 1), arguments := [{ node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("first", { node := 2, port := 0 }), ("second", { node := 3, port := 0 })]
    inputNames := ["key"]
  }
  definitions := [("body", loopHashBody)]
}

private def loopHashDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .parallelLoop, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 2, rule := .familyGetStatic, arguments := [{ node := 1, port := 0 }] },
    { sourceNode := 3, rule := .familyGetStatic, arguments := [{ node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .hashSample, arguments := [{ node := 0, port := 0 }] }
  ] })]
}



/-- Static extraction instantiates the loop-dependent hash query, so two lanes cannot acquire the
same deterministic source identity merely because the body was analyzed once. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts loopHashProgram loopHashDerivation [] []
    let first ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let second ← derivedMatrixFactAt 3 facts { node := 3, port := 0 }
    pure (first.origin != second.origin)) = .ok true := by
  native_decide

private def aliasedHashBody : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .hashSample fixtureType .plain [109, 120, 120] [.parameter "tag"] [] [] none none,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := ["key"]
}

private def aliasedLoopBody : Scope := {
  nodes := #[
    { kind := .input "key", arguments := [], outputTypes := [.bytes (.constant 32)] },
    { kind := .subgraphCall "hash" [("tag", .loopIndex 0)],
      arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 1, port := 0 })]
  inputNames := ["key"]
}

private def aliasedLoopHashProgram : Prog := {
  root := loopHashProgram.root
  definitions := [("body", aliasedLoopBody), ("hash", aliasedHashBody)]
}

private def aliasedLoopHashDerivation : ProgramDerivation := {
  root := loopHashDerivation.root
  definitions := [
    ("body", { steps := #[
      { sourceNode := 0, rule := .input, arguments := [] },
      { sourceNode := 1, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] }
    ] }),
    ("hash", { steps := #[
      { sourceNode := 0, rule := .input, arguments := [] },
      { sourceNode := 1, rule := .hashSample, arguments := [{ node := 0, port := 0 }] }
    ] })
  ]
}



/-- A child parameter bound to an enclosing loop index retains that binding frame in the hash
identity. Flattening the child environment at template index zero would make these origins equal. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts aliasedLoopHashProgram
      aliasedLoopHashDerivation [] []
    let first ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let second ← derivedMatrixFactAt 3 facts { node := 3, port := 0 }
    pure (first.origin != second.origin)) = .ok true := by
  native_decide

private def relationFixtureScope : Scope := {
  nodes := #[
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 3, port := 0 })], inputNames := []
}

private def relationFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
  { sourceNode := 2, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 3, rule := .matrixMultiplyRelation { node := 2, port := 0 },
    arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

private def wrongRelationFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .gaussianSample, arguments := [] },
  { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
  { sourceNode := 2, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
  { sourceNode := 3, rule := .matrixMultiplyRelation { node := 1, port := 0 },
    arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

example : checkScopeDerivation relationFixtureScope wrongRelationFixtureDerivation =
    .error (.invalidRelationOperand 3 { node := 1, port := 0 }) := by
  native_decide

private def childRelationScope : Scope := {
  nodes := #[
    { kind := .input "target", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] }
  ],
  outputs := [("preimage", { node := 1, port := 0 })], inputNames := ["target"]
}

private def childRelationDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] }
] }

private def subgraphRelationProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .subgraphCall "decompose" [], arguments := [{ node := 0, port := 0 }],
        outputTypes := [.preimage fixtureType] },
      { kind := .matrixMultiply,
        arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ],
    outputs := [("result", { node := 3, port := 0 })], inputNames := []
  }
  definitions := [("decompose", childRelationScope)]
}

private def subgraphRelationDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
    { sourceNode := 2, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 3, rule := .matrixMultiplyRelation { node := 2, port := 0 },
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
  ] }
  definitions := [("decompose", childRelationDerivation)]
}

example : (do
    let facts ← evaluateProgramOperationalWithLayouts subgraphRelationProgram
      subgraphRelationDerivation [] [fixtureLayout]
    matrixMaximum 3 { node := 3, port := 0 } facts []) = .ok 3 := by
  native_decide

private def distinctCallIdentityProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .subgraphCall "decompose" [], arguments := [{ node := 0, port := 0 }],
        outputTypes := [.preimage fixtureType] },
      { kind := .subgraphCall "decompose" [], arguments := [{ node := 0, port := 0 }],
        outputTypes := [.preimage fixtureType] }
    ]
    outputs := [("left", { node := 1, port := 0 }), ("right", { node := 2, port := 0 })]
    inputNames := []
  }
  definitions := [("decompose", childRelationScope)]
}

private def distinctCallIdentityDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 2, rule := .subgraphCall, arguments := [{ node := 0, port := 0 }] }
  ] }
  definitions := [("decompose", childRelationDerivation)]
}

/-- Equal local node/port numbers in two call instances are not the same sampled/derived event. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts distinctCallIdentityProgram
      distinctCallIdentityDerivation [] [fixtureLayout]
    let left ← derivedMatrixFactAt 2 facts { node := 1, port := 0 }
    let right ← derivedMatrixFactAt 2 facts { node := 2, port := 0 }
    pure (left.origin != right.origin)) = .ok true := by
  native_decide

private def packedFamilyFixtureScope : Scope := {
  nodes := relationFixtureScope.nodes ++ #[
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] },
    { kind := .familyPack,
      arguments := [{ node := 2, port := 0 }, { node := 4, port := 0 }],
      outputTypes := [.indexedFamily (.preimage fixtureType) (.constant 2)] },
    { kind := .familyGetStatic (.constant 0), arguments := [{ node := 5, port := 0 }],
      outputTypes := [.preimage fixtureType] },
    { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
    { kind := .familyGetDynamic,
      arguments := [{ node := 5, port := 0 }, { node := 7, port := 0 }],
      outputTypes := [.preimage fixtureType] },
    { kind := .familyPack,
      arguments := [{ node := 1, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .familyGetDynamic,
      arguments := [{ node := 9, port := 0 }, { node := 7, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 10, port := 0 }, { node := 8, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("static", { node := 6, port := 0 }), ("dynamic", { node := 8, port := 0 }),
    ("rewritten", { node := 11, port := 0 })]
  inputNames := []
}

private def packedFamilyFixtureDerivation : ScopeDerivation := {
  steps := relationFixtureDerivation.steps ++ #[
    { sourceNode := 4, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 5, rule := .familyPack,
      arguments := [{ node := 2, port := 0 }, { node := 4, port := 0 }] },
    { sourceNode := 6, rule := .familyGetStatic, arguments := [{ node := 5, port := 0 }] },
    { sourceNode := 7, rule := .constantInt, arguments := [] },
    { sourceNode := 8, rule := .familyGetDynamic,
      arguments := [{ node := 5, port := 0 }, { node := 7, port := 0 }] },
    { sourceNode := 9, rule := .familyPack,
      arguments := [{ node := 1, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 10, rule := .familyGetDynamic,
      arguments := [{ node := 9, port := 0 }, { node := 7, port := 0 }] },
    { sourceNode := 11, rule := .matrixMultiplyRelation { node := 8, port := 0 },
      arguments := [{ node := 10, port := 0 }, { node := 8, port := 0 }] }
  ]
}

/-- Dynamic extraction keeps relation-bearing branches aligned under one compact expression
selection. Relation-consuming multiplication then rewrites every branch independently and agrees
with the explicitly unrolled bound. -/
private def exactRelationSelectionFixtureResult : Except OperationalError Bool := do
    let facts ← evaluateScopeOperationalWithLayouts packedFamilyFixtureScope
      packedFamilyFixtureDerivation [] [fixtureLayout]
    let dynamicOk ← factHasRelation facts.arena (← lookupFact 8 facts { node := 8, port := 0 })
    let rewritten ← lookupFact 11 facts { node := 11, port := 0 }
    let rewrittenSchema ← derivedMatrixFactAt 11 facts { node := 11, port := 0 }
    let (rewrittenBound, _) ← operationalNoiseBoundForFact facts.arena rewritten []
    let representative : OperationalMatrixFact := {
      subject := { node := 20, port := 0 }
      origin := .value temporaryScope { node := 20, port := 0 }
      matrixType := fixtureType
      matrixParams := fixtureParams
      totalHardBound := .closedInt (.constant 7)
    }
    let summary := selectedMatrixSummary #[representative]
    let (envelopeArena, representativeId) :=
      ({} : OperationalExprArena).pushConcrete representative
    let envelopeSelection := DynamicSelectionIdentity.fromOrigin
      (.local temporaryScope { node := 21, port := 0 }) 30720
    let (envelopeArena, envelopeRoot) ← envelopeArena.pushSharedSelection envelopeSelection
      30720 representativeId summary
    let (envelopeBound, _) ← evaluateCompleteBound envelopeArena [] envelopeRoot
      (OperationalExprEvaluationState.empty envelopeArena)
    let staleRepresentative := { representative with
      totalHardBound := OperationalBoundExpr.closedInt (.constant 8) }
    let (staleArena, staleId) := ({} : OperationalExprArena).pushConcrete staleRepresentative
    let staleRejected := match staleArena.pushSharedSelection envelopeSelection 2 staleId summary with
      | .error (.unsupportedOperationalExpr _) => true
      | _ => false
    let report ← decoderNoiseCheckReportForFact [] facts.arena rewritten [] 2 25
    pure (dynamicOk && !matrixFactHasRelation rewrittenSchema && rewrittenBound == 3 &&
      envelopeArena.nodes.size == 2 &&
      envelopeBound == 7 && staleRejected &&
      report.obligations == [.decoderThreshold 2 25 3])

/-- Matrix family packing retains its lane context in the direct carrier; static and known dynamic
access both apply an `IndexMap` and recover the selected lane's production bound. -/
private def familyPackPreservesDomainFixtureResult : Except OperationalError Bool := do
  let facts ← evaluateScopeOperationalWithLayouts packedFamilyFixtureScope
    packedFamilyFixtureDerivation [] [fixtureLayout]
  let family ← lookupFact 10 facts { node := 9, port := 0 }
  let dynamic ← lookupFact 10 facts { node := 10, port := 0 }
  let expectedMaximum ← matrixMaximum 10 { node := 1, port := 0 } facts []
  let dynamicMaximum ← matrixMaximum 10 { node := 10, port := 0 } facts []
  match family, dynamic with
  | { context := familyContext, payload := .directValue _, .. },
      { context := dynamicContext, payload := .directValue _, .. } =>
      pure (!familyContext.binders.isEmpty && dynamicContext.binders.isEmpty &&
        dynamicMaximum == expectedMaximum)
  | _, _ => pure false

private def selectFixtureScope : Scope := {
  nodes := #[
    { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .select, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 },
      { node := 2, port := 0 }], outputTypes := [.matrix fixtureType] }
  ],
  outputs := [("result", { node := 3, port := 0 })], inputNames := []
}

private def selectFixtureDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .constantInt, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .gaussianSample, arguments := [] },
  { sourceNode := 3, rule := .select, arguments := [{ node := 0, port := 0 },
    { node := 1, port := 0 }, { node := 2, port := 0 }] }
] }

example : (do
    let facts ← evaluateScopeOperationalWithLayouts selectFixtureScope selectFixtureDerivation [] []
    matrixMaximum 3 { node := 3, port := 0 } facts []) = .ok 5 := by
  native_decide

/-- Executable matrix `select` is represented as one direct ordered family table followed by an
`IndexMap`.  A singleton selector is a static substitution; non-singleton selectors use their
executable selector variable.  The output does not name a legacy choice node. -/
private def directMatrixSelectFixtureResult : Except OperationalError Bool := do
  let facts ← evaluateScopeOperationalWithLayouts selectFixtureScope selectFixtureDerivation [] []
  let selection ← integerFactAt 3 facts { node := 0, port := 0 }
  let output ← lookupFact 3 facts { node := 3, port := 0 }
  match output with
  | expression@{ payload := .directValue root, .. } => do
      let value ← match facts.arena.direct.valueAt? root with
        | some value => pure value
        | none => throw (.invalidOperationalExprRef root)
      let lanes ← facts.arena.directValueFactsAt [] expression
      pure (expression.context == emptyContext &&
        (match value.payload with
        | .mapped (.matrix outputType) _ map =>
            outputType == fixtureType && map.destination == expression.context &&
              map.assignments == #[.constant selection.lower.toNat]
        | _ => false) &&
        lanes.length == 1 && lanes.all (fun lane => lane.subject == { node := 3, port := 0 }))
  | _ => pure false

example : directMatrixSelectFixtureResult = .ok true := by
  native_decide

/-- Selecting two direct matrix families stays entirely in the direct carrier.  The static
selection substitutes both the branch and lane binders, while the bounded input selector retains
the selected branch dimension and the output family lane dimension. -/
private def directFamilySelectScope : Scope := {
  nodes := #[
    { kind := .input "selector", arguments := [], outputTypes := [.integer] },
    { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyPack, arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 7), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyPack, arguments := [{ node := 5, port := 0 }, { node := 6, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .select, arguments := [{ node := 1, port := 0 }, { node := 4, port := 0 },
        { node := 7, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .select, arguments := [{ node := 0, port := 0 }, { node := 4, port := 0 },
        { node := 7, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .familyGetStatic (.constant 1), arguments := [{ node := 8, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyGetDynamic, arguments := [{ node := 9, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("static", { node := 10, port := 0 }), ("dynamic", { node := 11, port := 0 })]
  inputNames := ["selector"]
}

private def directFamilySelectDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .constantInt, arguments := [] },
  { sourceNode := 2, rule := .gaussianSample, arguments := [] },
  { sourceNode := 3, rule := .gaussianSample, arguments := [] },
  { sourceNode := 4, rule := .familyPack, arguments := [{ node := 2, port := 0 },
    { node := 3, port := 0 }] },
  { sourceNode := 5, rule := .gaussianSample, arguments := [] },
  { sourceNode := 6, rule := .gaussianSample, arguments := [] },
  { sourceNode := 7, rule := .familyPack, arguments := [{ node := 5, port := 0 },
    { node := 6, port := 0 }] },
  { sourceNode := 8, rule := .select, arguments := [{ node := 1, port := 0 },
    { node := 4, port := 0 }, { node := 7, port := 0 }] },
  { sourceNode := 9, rule := .select, arguments := [{ node := 0, port := 0 },
    { node := 4, port := 0 }, { node := 7, port := 0 }] },
  { sourceNode := 10, rule := .familyGetStatic, arguments := [{ node := 8, port := 0 }] },
  { sourceNode := 11, rule := .familyGetDynamic, arguments := [{ node := 9, port := 0 },
    { node := 0, port := 0 }] }
] }

private def directFamilySelectFixture : Except OperationalError Bool := do
  let scopeKey : ScopeTemplateKey := .root (.standalone 801)
  let (arena, selector) ← contractFact {} scopeKey { node := 0, port := 0 } ⟨"selector"⟩
    .integer (.integerRange (.constant 0) (.constant 1)) []
  let facts ← evaluateScopeOperationalWithKey scopeKey directFamilySelectScope directFamilySelectDerivation
    [] [] [selector] arena
  let staticFamily ← lookupFact 11 facts { node := 8, port := 0 }
  let dynamicFamily ← lookupFact 11 facts { node := 9, port := 0 }
  let staticOutput ← lookupFact 11 facts { node := 10, port := 0 }
  let dynamicOutput ← lookupFact 11 facts { node := 11, port := 0 }
  let staticBound ← matrixMaximum 11 { node := 10, port := 0 } facts []
  let dynamicBound ← matrixMaximum 11 { node := 11, port := 0 } facts []
  let noLegacySelect := facts.arena.nodes.all fun expression => match expression.node with
    | .select .. => false
    | _ => true
  pure (match staticFamily, dynamicFamily, staticOutput, dynamicOutput with
    | { context := staticContext, payload := .directValue _, .. },
        { context := dynamicContext, payload := .directValue _, .. },
        { context := staticOutputContext, payload := .directValue _, .. },
        { context := dynamicOutputContext, payload := .directValue _, .. } =>
          !staticContext.binders.isEmpty && dynamicContext.binders.size == 2 &&
            /- Dynamic family access consumes the selected lane but retains both independent
            selectors introduced by the nested direct table and branch selection. -/
            staticOutputContext == emptyContext && dynamicOutputContext.binders.size == 2 &&
            staticBound == 7 && dynamicBound == 7 && noLegacySelect
    | _, _, _, _ => false)

example : directFamilySelectFixture = .ok true := by
  native_decide







/-- A dynamic selector whose declared range exceeds a two-lane direct matrix family is rejected
at `familyGetDynamic`; the evaluator must not truncate or canonicalize the selector range. -/
private def outOfRangeDirectFamilyGetScope : Scope := {
  nodes := #[
    { kind := .input "selector", arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyPack, arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .familyGetDynamic, arguments := [{ node := 3, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 4, port := 0 })]
  inputNames := ["selector"]
}

private def outOfRangeDirectFamilyGetDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .gaussianSample, arguments := [] },
  { sourceNode := 2, rule := .gaussianSample, arguments := [] },
  { sourceNode := 3, rule := .familyPack, arguments := [{ node := 1, port := 0 },
    { node := 2, port := 0 }] },
  { sourceNode := 4, rule := .familyGetDynamic, arguments := [{ node := 3, port := 0 },
    { node := 0, port := 0 }] }
] }

private def outOfRangeDirectFamilyGetFixture : Bool :=
  match (do
    let scopeKey : ScopeTemplateKey := .root (.standalone 803)
    let (arena, selector) ← contractFact {} scopeKey { node := 0, port := 0 } ⟨"selector"⟩
      .integer (.integerRange (.constant 0) (.constant 3)) []
    evaluateScopeOperationalWithKey scopeKey outOfRangeDirectFamilyGetScope
      outOfRangeDirectFamilyGetDerivation [] [] [selector] arena) with
  | .error (.invalidCount 4 3) => true
  | _ => false

/-- A symbolic family count survives direct Select and subsequent static and dynamic extraction
without allocating a legacy selection node. -/
private def symbolicFamilySelectScope : Scope := {
  nodes := #[
    { kind := .input "selector", arguments := [], outputTypes := [.integer] },
    { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyPack, arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.parameter "lane_count")] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 7), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyPack, arguments := [{ node := 5, port := 0 }, { node := 6, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.parameter "lane_count")] },
    { kind := .select, arguments := [{ node := 1, port := 0 }, { node := 4, port := 0 },
        { node := 7, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.parameter "lane_count")] },
    { kind := .select, arguments := [{ node := 0, port := 0 }, { node := 4, port := 0 },
        { node := 7, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.parameter "lane_count")] },
    { kind := .familyGetStatic (.constant 1), arguments := [{ node := 8, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyGetDynamic, arguments := [{ node := 9, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("static", { node := 10, port := 0 }), ("dynamic", { node := 11, port := 0 })]
  inputNames := ["selector"]
}

private def symbolicFamilySelectFixture : Except OperationalError Bool := do
  let environment : ParamEnvironment := [("lane_count", .integer 2)]
  let scopeKey : ScopeTemplateKey := .root (.standalone 802)
  let (arena, selector) ← contractFact {} scopeKey { node := 0, port := 0 } ⟨"selector"⟩
    .integer (.integerRange (.constant 0) (.constant 1)) environment
  let facts ← evaluateScopeOperationalWithKey scopeKey symbolicFamilySelectScope directFamilySelectDerivation
    environment [] [selector] arena
  let staticOutput ← lookupFact 11 facts { node := 10, port := 0 }
  let dynamicOutput ← lookupFact 11 facts { node := 11, port := 0 }
  let staticBound ← matrixMaximum 11 { node := 10, port := 0 } facts environment
  let dynamicBound ← matrixMaximum 11 { node := 11, port := 0 } facts environment
  let noLegacySelect := facts.arena.nodes.all fun expression => match expression.node with
    | .select .. => false
    | _ => true
  pure (match staticOutput, dynamicOutput with
    | { context := staticContext, payload := .directValue _, .. },
        { context := dynamicContext, payload := .directValue _, .. } =>
          /- The symbolic path preserves the same two independent direct selectors as the
          closed fixture; only the statically selected lane is consumed. -/
          staticContext == emptyContext && dynamicContext.binders.size == 2 &&
            staticBound == 7 && dynamicBound == 7 && noLegacySelect
    | _, _ => false)

example : symbolicFamilySelectFixture = .ok true := by
  native_decide




private def loopBoundBody : Scope := {
  nodes := #[{
    kind := .gaussianSample fixtureType (.parameter "lane_bound")
    arguments := []
    outputTypes := [.matrix fixtureType]
  }]
  outputs := [("result", { node := 0, port := 0 })]
  inputNames := []
}

private def loopBoundProgram : Prog := {
  root := {
    nodes := #[{
      kind := .parallelLoop "body" (.constant 4) 0
        [("lane_bound", .add (.loopIndex 0) (.constant 1))] []
      arguments := []
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 4)]
    }, {
      kind := .familyGetStatic (.constant 2)
      arguments := [{ node := 0, port := 0 }]
      outputTypes := [.matrix fixtureType]
    }]
    outputs := [("results", { node := 0, port := 0 }), ("selected", { node := 1, port := 0 })]
    inputNames := []
  }
  definitions := [("body", loopBoundBody)]
}

private def loopBoundDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .parallelLoop, arguments := [] },
    { sourceNode := 1, rule := .familyGetStatic, arguments := [{ node := 0, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] }
  ] })]
}

/-- A loop-dependent child parameter is evaluated numerically over all four indices while the
body graph itself is evaluated once. The resulting indexed expression stores the exact maximum 4. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts loopBoundProgram loopBoundDerivation [] []
    matrixMaximum 1 { node := 0, port := 0 } facts []) = .ok 4 := by
  native_decide

example : (do
    let facts ← evaluateProgramOperationalWithLayouts loopBoundProgram loopBoundDerivation [] []
    matrixMaximum 2 { node := 1, port := 0 } facts []) = .ok 3 := by
  native_decide

private def packedSelectionLoopBody : Scope := {
  nodes := #[{
    kind := .input "value"
    arguments := []
    outputTypes := [.matrix fixtureType]
  }]
  outputs := [("result", { node := 0, port := 0 })]
  inputNames := ["value"]
}

/-- Production gather path: a two-lane integer family is zipped into a parallel body while a
three-lane matrix family is broadcast intact.  The body dynamically gathers from the latter, so
the gather codomain and lookup-position domains are deliberately unequal. -/
private def productionGatherLoopBody : Scope := {
  nodes := #[
    { kind := .input "indices", arguments := [], outputTypes := [.integer] },
    { kind := .input "b", arguments := [], outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 3)] },
    { kind := .input "k", arguments := [], outputTypes := [.indexedFamily (.preimage fixtureType) (.constant 3)] },
    { kind := .familyGetDynamic, arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyGetDynamic, arguments := [{ node := 2, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply, arguments := [{ node := 3, port := 0 }, { node := 4, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 5, port := 0 })]
  inputNames := ["indices", "b", "k"]
}

private def productionGatherLoopProgram : Prog := {
  root := {
    nodes := #[
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
      { kind := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily .integer (.constant 2)] },
      { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .familyPack, arguments := [{ node := 3, port := 0 }, { node := 3, port := 0 }, { node := 3, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 3)] },
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 5), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 7), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
        arguments := [{ node := 5, port := 0 }], outputTypes := [.preimage fixtureType] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
        arguments := [{ node := 6, port := 0 }], outputTypes := [.preimage fixtureType] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
        arguments := [{ node := 7, port := 0 }], outputTypes := [.preimage fixtureType] },
      { kind := .familyPack, arguments := [{ node := 8, port := 0 }, { node := 9, port := 0 }, { node := 10, port := 0 }],
        outputTypes := [.indexedFamily (.preimage fixtureType) (.constant 3)] },
      { kind := .parallelLoop "gather" (.constant 2) 0 [] [.zip, .broadcast, .broadcast],
        arguments := [{ node := 2, port := 0 }, { node := 4, port := 0 }, { node := 11, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] }
    ]
    outputs := [("result", { node := 12, port := 0 })]
    inputNames := []
  }
  definitions := [("gather", productionGatherLoopBody)]
}

private def productionGatherLoopDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .constantInt, arguments := [] },
    { sourceNode := 1, rule := .constantInt, arguments := [] },
    { sourceNode := 2, rule := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .gadgetMatrix, arguments := [] },
    { sourceNode := 4, rule := .familyPack, arguments := [{ node := 3, port := 0 }, { node := 3, port := 0 }, { node := 3, port := 0 }] },
    { sourceNode := 5, rule := .gaussianSample, arguments := [] },
    { sourceNode := 6, rule := .gaussianSample, arguments := [] },
    { sourceNode := 7, rule := .gaussianSample, arguments := [] },
    { sourceNode := 8, rule := .gadgetDecompose, arguments := [{ node := 5, port := 0 }] },
    { sourceNode := 9, rule := .gadgetDecompose, arguments := [{ node := 6, port := 0 }] },
    { sourceNode := 10, rule := .gadgetDecompose, arguments := [{ node := 7, port := 0 }] },
    { sourceNode := 11, rule := .familyPack, arguments := [{ node := 8, port := 0 }, { node := 9, port := 0 }, { node := 10, port := 0 }] },
    { sourceNode := 12, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }, { node := 4, port := 0 }, { node := 11, port := 0 }] }
  ] }
  definitions := [("gather", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .input, arguments := [] },
    { sourceNode := 3, rule := .familyGetDynamic, arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }] },
    { sourceNode := 4, rule := .familyGetDynamic, arguments := [{ node := 2, port := 0 }, { node := 0, port := 0 }] },
    { sourceNode := 5, rule := .matrixMultiplyRelation { node := 4, port := 0 },
      arguments := [{ node := 3, port := 0 }, { node := 4, port := 0 }] }
  ] })]
}

private def productionGatherLoopFixture : Except OperationalError Bool := do
  let facts ← evaluateProgramOperationalWithLayouts productionGatherLoopProgram productionGatherLoopDerivation [] [fixtureLayout]
  let output ← lookupFact 13 facts { node := 12, port := 0 }
  let entries ← facts.arena.reducedDirectValueFactsAt [] output
  let maximum ← matrixMaximum 13 { node := 12, port := 0 } facts []
  let operationalScope : ScopeTemplateKey := .parallelBody (.root (.standalone 0)) 12
  let scope : GatherScopeTemplateKey := operationalScope.toGatherScopeTemplateKey
  let owner : GatherLookupOwner := {
    indices := { scope, node := 0, port := 0 }
  }
  pure (maximum == 7 && entries.length == 3 && entries.all fun entry =>
    (match entry.key with
    | some (.gather actualOwner (.constant 3) (.variable position)) =>
        actualOwner == owner && position.count == .constant 2
    | _ => false) && entry.ordinal < 3 && !matrixFactHasRelation entry.fact)

example : productionGatherLoopFixture = .ok true := by native_decide

/-- The interval of a zipped executable integer family is validated against the gathered source
family before lowering.  A lane value equal to the source count is rejected, never truncated. -/
private def productionGatherOutOfRangeProgram : Prog := {
  productionGatherLoopProgram with
  root := {
    productionGatherLoopProgram.root with
    nodes := productionGatherLoopProgram.root.nodes.set! 1 {
      kind := .constantInt 3
      arguments := []
      outputTypes := [.integer]
    }
  }
}

private def productionGatherOutOfRangeFixture : Bool :=
  match evaluateProgramOperationalWithLayouts productionGatherOutOfRangeProgram
      productionGatherLoopDerivation [] [fixtureLayout] with
  | .error (.inScope (.parallelBody (.root (.standalone 0)) 12) (.invalidCount 3 3)) => true
  | _ => false

example : productionGatherOutOfRangeFixture = true := by native_decide

/-- Two different executable index-family producers may have the same lane values, but their
gathers are not correlated.  The Graph IR lowering must therefore reject the relation rewrite
instead of pairing B and K by ordinal alone. -/
private def productionDistinctIndexLoopBody : Scope := {
  nodes := #[
    { kind := .input "bIndices", arguments := [], outputTypes := [.integer] },
    { kind := .input "kIndices", arguments := [], outputTypes := [.integer] },
    { kind := .input "b", arguments := [], outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 3)] },
    { kind := .input "k", arguments := [], outputTypes := [.indexedFamily (.preimage fixtureType) (.constant 3)] },
    { kind := .familyGetDynamic, arguments := [{ node := 2, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .familyGetDynamic, arguments := [{ node := 3, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply, arguments := [{ node := 4, port := 0 }, { node := 5, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  /- Export the two gathered operands separately.  The root relation below is then reduced only
  by this fixture, so a rejection cannot be mistaken for an unrelated loop-closing failure. -/
  outputs := [("b", { node := 4, port := 0 }), ("k", { node := 5, port := 0 })]
  inputNames := ["bIndices", "kIndices", "b", "k"]
}

private def productionDistinctIndexProgram : Prog := {
  root := {
    nodes := #[
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
      { kind := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily .integer (.constant 2)] },
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
      { kind := .familyPack, arguments := [{ node := 3, port := 0 }, { node := 4, port := 0 }],
        outputTypes := [.indexedFamily .integer (.constant 2)] },
      { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .familyPack, arguments := [{ node := 6, port := 0 }, { node := 6, port := 0 }, { node := 6, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 3)] },
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 5), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 7), arguments := [], outputTypes := [.matrix fixtureType] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1), arguments := [{ node := 8, port := 0 }], outputTypes := [.preimage fixtureType] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1), arguments := [{ node := 9, port := 0 }], outputTypes := [.preimage fixtureType] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1), arguments := [{ node := 10, port := 0 }], outputTypes := [.preimage fixtureType] },
      { kind := .familyPack, arguments := [{ node := 11, port := 0 }, { node := 12, port := 0 }, { node := 13, port := 0 }], outputTypes := [.indexedFamily (.preimage fixtureType) (.constant 3)] },
      { kind := .parallelLoop "distinct" (.constant 2) 0 [] [.zip, .zip, .broadcast, .broadcast], arguments := [{ node := 2, port := 0 }, { node := 5, port := 0 }, { node := 7, port := 0 }, { node := 14, port := 0 }], outputCount := 2, outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2), .indexedFamily (.preimage fixtureType) (.constant 2)] },
      { kind := .matrixMultiply, arguments := [{ node := 15, port := 0 }, { node := 15, port := 1 }], outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 16, port := 0 })]
    inputNames := []
  }
  definitions := [("distinct", productionDistinctIndexLoopBody)]
}

private def productionDistinctIndexDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .constantInt, arguments := [] }, { sourceNode := 1, rule := .constantInt, arguments := [] },
    { sourceNode := 2, rule := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .constantInt, arguments := [] }, { sourceNode := 4, rule := .constantInt, arguments := [] },
    { sourceNode := 5, rule := .familyPack, arguments := [{ node := 3, port := 0 }, { node := 4, port := 0 }] },
    { sourceNode := 6, rule := .gadgetMatrix, arguments := [] },
    { sourceNode := 7, rule := .familyPack, arguments := [{ node := 6, port := 0 }, { node := 6, port := 0 }, { node := 6, port := 0 }] },
    { sourceNode := 8, rule := .gaussianSample, arguments := [] }, { sourceNode := 9, rule := .gaussianSample, arguments := [] }, { sourceNode := 10, rule := .gaussianSample, arguments := [] },
    { sourceNode := 11, rule := .gadgetDecompose, arguments := [{ node := 8, port := 0 }] }, { sourceNode := 12, rule := .gadgetDecompose, arguments := [{ node := 9, port := 0 }] }, { sourceNode := 13, rule := .gadgetDecompose, arguments := [{ node := 10, port := 0 }] },
    { sourceNode := 14, rule := .familyPack, arguments := [{ node := 11, port := 0 }, { node := 12, port := 0 }, { node := 13, port := 0 }] },
    { sourceNode := 15, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }, { node := 5, port := 0 }, { node := 7, port := 0 }, { node := 14, port := 0 }] },
    { sourceNode := 16, rule := .matrixMultiplyRelation { node := 15, port := 1 }, arguments := [{ node := 15, port := 0 }, { node := 15, port := 1 }] }
  ] }
  definitions := [("distinct", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }, { sourceNode := 1, rule := .input, arguments := [] }, { sourceNode := 2, rule := .input, arguments := [] }, { sourceNode := 3, rule := .input, arguments := [] },
    { sourceNode := 4, rule := .familyGetDynamic, arguments := [{ node := 2, port := 0 }, { node := 0, port := 0 }] }, { sourceNode := 5, rule := .familyGetDynamic, arguments := [{ node := 3, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 6, rule := .matrixMultiplyRelation { node := 5, port := 0 }, arguments := [{ node := 4, port := 0 }, { node := 5, port := 0 }] }
  ] })]
}

private def productionDistinctIndexFixtureResult : Except OperationalError
    (Bool × Except OperationalError (List ReducedDirectMatrixFact)) := do
    let facts ← evaluateProgramOperationalWithLayouts productionDistinctIndexProgram
      productionDistinctIndexDerivation [] [fixtureLayout]
    let relation ← lookupFact 17 facts { node := 16, port := 0 }
    let relationRoot ← match relation.payload with
      | .directValue root => pure root
      | .matrix root | .scalar root => throw (OperationalError.unsupportedOperationalExpr root)
    let relationLowered := match facts.arena.direct.valueAt? relationRoot with
      | some { payload := .pointwise (.matrix _) (.matrix operation) inputs, .. } =>
          match operation.kind with
          | .multiply (.matrixMultiplyRelation rightWire) _ =>
              operation.ownerNode == 16 && rightWire == { node := 15, port := 1 } && inputs.size == 2
          | _ => false
      | _ => false
    let reduction := facts.arena.reducedDirectValueFactsAt [] relation
    pure (relationLowered, reduction)

private def productionDistinctIndexFixture : Bool :=
  match productionDistinctIndexFixtureResult with
  | .ok (true, .error (.unsupportedOperationalExpr _)) => true
  | _ => false

example : productionDistinctIndexFixture = true := by native_decide

private def packedSelectionLoopProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyPack,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .parallelLoop "body" (.constant 2) 0 [] [.zip],
        arguments := [{ node := 2, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .familyGetStatic (.constant 1), arguments := [{ node := 3, port := 0 }],
        outputTypes := [.matrix fixtureType] },
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .familyGetDynamic,
        arguments := [{ node := 3, port := 0 }, { node := 5, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("family", { node := 3, port := 0 }),
      ("static", { node := 4, port := 0 }), ("dynamic", { node := 6, port := 0 })]
    inputNames := []
  }
  definitions := [("body", packedSelectionLoopBody)]
}

private def packedSelectionLoopDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .familyPack,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }] },
    { sourceNode := 4, rule := .familyGetStatic, arguments := [{ node := 3, port := 0 }] },
    { sourceNode := 5, rule := .constantInt, arguments := [] },
    { sourceNode := 6, rule := .familyGetDynamic,
      arguments := [{ node := 3, port := 0 }, { node := 5, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }
  ] })]
}

/- A packed matrix family crosses a one-evaluation parallel body as one direct indexed value. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts packedSelectionLoopProgram
      packedSelectionLoopDerivation [] []
    let staticMaximum ← matrixMaximum 7 { node := 4, port := 0 } facts []
    let dynamic ← lookupFact 7 facts { node := 6, port := 0 }
    let report ← decoderNoiseCheckReportForFact [] facts.arena dynamic [] 2 25
    let familyIsDirectIndexed ← match ← lookupFact 7 facts { node := 3, port := 0 } with
      | { context, payload := .directValue _, .. } => pure (!context.binders.isEmpty)
      | _ => pure false
    pure (staticMaximum, report.obligations, familyIsDirectIndexed)) =
    .ok (5, [.decoderThreshold 2 25 3], true) := by
  native_decide

/-- Parallel-loop input modes keep direct matrix values in the direct carrier: Broadcast lifts a
constant value over the loop binder, while Zip and ZipOffset reindex a packed direct family. -/
private def directLoopInputProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .gaussianSample fixtureType (.constant 7), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .parallelLoop "body" (.constant 2) 0 [] [.broadcast],
        arguments := [{ node := 3, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .parallelLoop "body" (.constant 2) 0 [] [.zip],
        arguments := [{ node := 2, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .parallelLoop "body" (.constant 1) 0 [] [.zipOffset 1],
        arguments := [{ node := 2, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 1)] }
    ]
    outputs := [("broadcast", { node := 4, port := 0 }), ("zip", { node := 5, port := 0 }),
      ("zipOffset", { node := 6, port := 0 })]
    inputNames := []
  }
  definitions := [("body", packedSelectionLoopBody)]
}

private def directLoopInputDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .familyPack, arguments := [{ node := 0, port := 0 },
      { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .gaussianSample, arguments := [] },
    { sourceNode := 4, rule := .parallelLoop, arguments := [{ node := 3, port := 0 }] },
    { sourceNode := 5, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }] },
    { sourceNode := 6, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }
  ] })]
}

private def directLoopInputFixture : Except OperationalError Bool := do
  let facts ← evaluateProgramOperationalWithLayouts directLoopInputProgram directLoopInputDerivation [] []
  let broadcast ← lookupFact 7 facts { node := 4, port := 0 }
  let zipped ← lookupFact 7 facts { node := 5, port := 0 }
  let offset ← lookupFact 7 facts { node := 6, port := 0 }
  let broadcastBound ← matrixMaximum 7 { node := 4, port := 0 } facts []
  let zipBound ← matrixMaximum 7 { node := 5, port := 0 } facts []
  let offsetBound ← matrixMaximum 7 { node := 6, port := 0 } facts []
  let noLegacySelect := facts.arena.nodes.all fun expression => match expression.node with
    | .select .. => false
    | _ => true
  pure (match broadcast, zipped, offset with
    | { context := broadcastContext, payload := .directValue _, .. },
        { context := zipContext, payload := .directValue _, .. },
        { context := offsetContext, payload := .directValue _, .. } =>
          broadcastContext.binders.size == 1 && zipContext.binders.size == 1 &&
            offsetContext.binders.size == 1 && broadcastBound == 7 && zipBound == 5 &&
            offsetBound == 5 && noLegacySelect
    | _, _, _ => false)

/-- A parameter-valued `familyPack` count is resolved from the production environment before a
zipped loop reindexes its exact direct lane binder. -/
private def symbolicCountDirectZipProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.parameter "lane_count")] },
      { kind := .parallelLoop "body" (.parameter "lane_count") 0 [] [.zip],
        arguments := [{ node := 2, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.parameter "lane_count")] }
    ]
    outputs := [("result", { node := 3, port := 0 })]
    inputNames := []
  }
  definitions := [("body", packedSelectionLoopBody)]
}

private def symbolicCountDirectZipDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .familyPack, arguments := [{ node := 0, port := 0 },
      { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .parallelLoop, arguments := [{ node := 2, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }
  ] })]
}

private def symbolicCountDirectZipFixture : Except OperationalError Bool := do
  let environment : ParamEnvironment := [("lane_count", .integer 2)]
  let facts ← evaluateProgramOperationalWithLayouts symbolicCountDirectZipProgram
    symbolicCountDirectZipDerivation environment []
  let output ← lookupFact 4 facts { node := 3, port := 0 }
  let bound ← matrixMaximum 4 { node := 3, port := 0 } facts environment
  let noLegacySelect := facts.arena.nodes.all fun expression => match expression.node with
    | .select .. => false
    | _ => true
  pure (match output with
    | { context, payload := .directValue _, .. } =>
        context.binders.size == 1 && bound == 5 && noLegacySelect
    | _ => false)

private def selectedSequentialBody : Scope := {
  nodes := #[
    { kind := .input "state", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .input "selector", arguments := [], outputTypes := [.integer] },
    { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 5), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .select,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 },
        { node := 3, port := 0 }], outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 4, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("state", { node := 5, port := 0 })]
  inputNames := ["state", "selector"]
}

private def selectedSequentialProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 1), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .constantInt 0, arguments := [], outputTypes := [.integer] },
      { kind := .sequentialLoop "body" (.constant 2) 0 [] 1,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 2, port := 0 })]
    inputNames := []
  }
  definitions := [("body", selectedSequentialBody)]
}

private def selectedSequentialDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .constantInt, arguments := [] },
    { sourceNode := 2, rule := .sequentialLoop,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .gaussianSample, arguments := [] },
    { sourceNode := 3, rule := .gaussianSample, arguments := [] },
    { sourceNode := 4, rule := .select,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 },
        { node := 3, port := 0 }] },
    { sourceNode := 5, rule := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 4, port := 0 }] }
  ] })]
}

/-- A sequential body may contain a selection.  Here its executable selector is statically zero,
so concrete-index reduction selects the bound-three branch before constructing the numeric
recurrence.  Two iterations therefore evaluate `min(q/2, previous + 3)` without retaining a
spurious dynamic alternative. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts selectedSequentialProgram
      selectedSequentialDerivation [] []
    matrixMaximum 3 { node := 2, port := 0 } facts []) = .ok 7 := by
  native_decide

private def sequentialRelationBody : Scope := {
  nodes := #[
    { kind := .input "target", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .input "public", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
      arguments := [{ node := 0, port := 0 }], outputTypes := [.preimage fixtureType] },
    { kind := .matrixMultiply,
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("result", { node := 3, port := 0 })]
  inputNames := ["target", "public"]
}

private def sequentialRelationProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gadgetMatrix fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .sequentialLoop "body" (.constant 3) 0 [] 1,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 2, port := 0 })]
    inputNames := []
  }
  definitions := [("body", sequentialRelationBody)]
}

private def sequentialRelationDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gadgetMatrix, arguments := [] },
    { sourceNode := 2, rule := .sequentialLoop,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .gadgetDecompose, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 3, rule := .matrixMultiplyRelation { node := 2, port := 0 },
      arguments := [{ node := 1, port := 0 }, { node := 2, port := 0 }] }
  ] })]
}

/-- A relation may depend on the previous carried bound when it is created and consumed inside
one body execution. Only the resulting relation-free target fact becomes the next carried state. -/
example : (do
    let facts ← evaluateProgramOperationalWithLayouts sequentialRelationProgram
      sequentialRelationDerivation [] [fixtureLayout]
    matrixMaximum 2 { node := 2, port := 0 } facts []) = .ok 2 := by
  native_decide

private def relationCarryBody : Scope := {
  nodes := #[{ kind := .input "carried", arguments := [], outputTypes := [.preimage fixtureType] }]
  outputs := [("result", { node := 0, port := 0 })]
  inputNames := ["carried"]
}

private def relationCarryProgram : Prog := {
  root := {
    nodes := relationFixtureScope.nodes.take 3 ++ #[{
      kind := .sequentialLoop "body" (.constant 1) 0 [] 1
      arguments := [{ node := 2, port := 0 }]
      outputTypes := [.preimage fixtureType]
    }]
    outputs := [("result", { node := 3, port := 0 })]
    inputNames := []
  }
  definitions := [("body", relationCarryBody)]
}

private def relationCarryDerivation : ProgramDerivation := {
  root := { steps := relationFixtureDerivation.steps.take 3 ++ #[{
    sourceNode := 3
    rule := .sequentialLoop
    arguments := [{ node := 2, port := 0 }]
  }] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] }
  ] })]
}

/-- Relations are body-local tokens; carrying one across iterations rejects before abstraction. -/
example : (match evaluateProgramOperationalWithLayouts relationCarryProgram relationCarryDerivation
    [] [fixtureLayout] with
  | .error (.relationBearingCarriedValue (.root (.standalone 0)) 3 0) => true
  | _ => false) = true := by
  native_decide

/-- A production sequential body updates two carried matrix slots from the same previous-state
vector.  Slot one reads slot zero's previous value, not the already-updated slot-zero result.
This exercises direct-carrier abstraction, schema closure, and recurrence-state installation
through the actual Graph IR evaluator. -/
private def directSimultaneousSequentialBody : Scope := {
  nodes := #[
    { kind := .input "left", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .input "right", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .gaussianSample fixtureType (.constant 1), arguments := [],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 0, port := 0 }, { node := 2, port := 0 }],
      outputTypes := [.matrix fixtureType] },
    { kind := .matrixAdd, arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("left", { node := 3, port := 0 }), ("right", { node := 4, port := 0 })]
  inputNames := ["left", "right"]
}

private def directSimultaneousSequentialProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 1), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .sequentialLoop "body" (.constant 2) 0 [] 2,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputCount := 2,
        outputTypes := [.matrix fixtureType, .matrix fixtureType] }
    ]
    outputs := [("left", { node := 2, port := 0 }), ("right", { node := 2, port := 1 })]
    inputNames := []
  }
  definitions := [("body", directSimultaneousSequentialBody)]
}

private def directSimultaneousSequentialDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .sequentialLoop,
      arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
  ] }
  definitions := [("body", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .gaussianSample, arguments := [] },
    { sourceNode := 3, rule := .matrixAdd,
      arguments := [{ node := 0, port := 0 }, { node := 2, port := 0 }] },
    { sourceNode := 4, rule := .matrixAdd,
      arguments := [{ node := 1, port := 0 }, { node := 0, port := 0 }] }
  ] })]
}

example : (do
    let facts ← evaluateProgramOperationalWithLayouts directSimultaneousSequentialProgram
      directSimultaneousSequentialDerivation [] []
    let left ← matrixMaximum 3 { node := 2, port := 0 } facts []
    let right ← matrixMaximum 3 { node := 2, port := 1 } facts []
    pure (left == 3 && right == 5)) = .ok true := by
  native_decide

/-- A direct integer carrier crosses a parallel family boundary, is selected concretely, and then
is carried through a sequential loop.  Its lower and upper interval components are independently
installed as recurrence-state paths; this guards against treating an integer carrier as a matrix
maximum or silently dropping either endpoint. -/
private def directIntegerParallelBody : Scope := {
  nodes := #[{ kind := .input "value", arguments := [], outputTypes := [.integer] }]
  outputs := [("value", { node := 0, port := 0 })]
  inputNames := ["value"]
}

private def directIntegerSequentialBody : Scope := {
  nodes := #[{ kind := .input "value", arguments := [], outputTypes := [.integer] }]
  outputs := [("value", { node := 0, port := 0 })]
  inputNames := ["value"]
}

private def directIntegerSequentialProgram : Prog := {
  root := {
    nodes := #[
      { kind := .constantInt 4, arguments := [], outputTypes := [.integer] },
      { kind := .parallelLoop "parallel" (.constant 2) 0 [] [.broadcast],
        arguments := [{ node := 0, port := 0 }],
        outputTypes := [.indexedFamily .integer (.constant 2)] },
      { kind := .familyGetStatic (.constant 1), arguments := [{ node := 1, port := 0 }],
        outputTypes := [.integer] },
      { kind := .sequentialLoop "sequential" (.constant 2) 0 [] 1,
        arguments := [{ node := 2, port := 0 }], outputTypes := [.integer] }
    ]
    outputs := [("value", { node := 3, port := 0 })]
    inputNames := []
  }
  definitions := [("parallel", directIntegerParallelBody),
    ("sequential", directIntegerSequentialBody)]
}

private def directIntegerSequentialDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .constantInt, arguments := [] },
    { sourceNode := 1, rule := .parallelLoop, arguments := [{ node := 0, port := 0 }] },
    { sourceNode := 2, rule := .familyGetStatic, arguments := [{ node := 1, port := 0 }] },
    { sourceNode := 3, rule := .sequentialLoop, arguments := [{ node := 2, port := 0 }] }
  ] }
  definitions := [
    ("parallel", { steps := #[{ sourceNode := 0, rule := .input, arguments := [] }] }),
    ("sequential", { steps := #[{ sourceNode := 0, rule := .input, arguments := [] }] })
  ]
}

example : (do
    let facts ← evaluateProgramOperationalWithLayouts directIntegerSequentialProgram
      directIntegerSequentialDerivation [] []
    let result ← lookupFact 4 facts { node := 3, port := 0 }
    let integer ← facts.arena.directValueScalarFactAt [] result
    pure (match result, integer with
      | { payload := .directValue _, .. }, .integer value =>
          value.lower == 4 && value.upper == 4 &&
            match value.lowerExpression, value.upperExpression with
            | .recurrenceState 2 paths _ _ (OperationalBoundPath.integerLower 0 0),
                .recurrenceState 2 upperPaths _ _ (OperationalBoundPath.integerUpper 0 0) =>
                paths == [(OperationalBoundPath.integerLower 0 0),
                  (OperationalBoundPath.integerUpper 0 0)] && upperPaths == paths
            | _, _ => false
      | _, _ => false)) = .ok true := by
  native_decide

/-- The exact lower/upper recurrence payload is checked with an asymmetric direct input interval.
This uses the ordinary Graph IR sequential node and a direct-carrier input, rather than a
hand-written recurrence expression: swapping or dropping one endpoint changes the result. -/
private def directIntegerIntervalSequentialScope : Scope := {
  nodes := #[
    { kind := .input "value", arguments := [], outputTypes := [.integer] },
    { kind := .sequentialLoop "sequential" (.constant 2) 0 [] 1,
      arguments := [{ node := 0, port := 0 }], outputTypes := [.integer] }
  ]
  outputs := [("value", { node := 1, port := 0 })]
  inputNames := ["value"]
}

private def directIntegerIntervalSequentialDerivation : ScopeDerivation := { steps := #[
  { sourceNode := 0, rule := .input, arguments := [] },
  { sourceNode := 1, rule := .sequentialLoop, arguments := [{ node := 0, port := 0 }] }
] }

private def directIntegerIntervalSequentialProgram : Prog := {
  root := directIntegerIntervalSequentialScope
  definitions := [("sequential", directIntegerSequentialBody)]
}

private def directIntegerIntervalSequentialProgramDerivation : ProgramDerivation := {
  root := directIntegerIntervalSequentialDerivation
  definitions := [("sequential", { steps := #[{ sourceNode := 0, rule := .input, arguments := [] }] })]
}

private def directIntegerIntervalSequentialFixture : Except OperationalError Bool := do
  let initial ← integerFactWithExpressions 0 0 2 5
    (.closedInt (.constant 2)) (.closedInt (.constant 5))
  let (arena, input) ← ({} : OperationalExprArena).promoteConcreteScalarFact initial
  let prepared ← prepareProgramOperational directIntegerIntervalSequentialProgram
    directIntegerIntervalSequentialProgramDerivation
  let facts ← evaluatePreparedScope prepared.definitions [] (.root (.standalone 61))
    (prepared.definitions.size + 1) prepared.root [] [] arena [input]
  let result ← lookupFact 2 facts { node := 1, port := 0 }
  match result, ← facts.arena.directValueScalarFactAt [] result with
  | { payload := .directValue _, .. }, .integer value =>
      pure <| value.lower == 2 && value.upper == 5 &&
        value.lowerExpression == .recurrenceState 2
          [(.integerLower 0 0), (.integerUpper 0 0)]
          [.closedInt (.constant 2), .closedInt (.constant 5)]
          [.previous (.integerLower 0 0), .previous (.integerUpper 0 0)]
          (.integerLower 0 0) &&
        value.upperExpression == .recurrenceState 2
          [(.integerLower 0 0), (.integerUpper 0 0)]
          [.closedInt (.constant 2), .closedInt (.constant 5)]
          [.previous (.integerLower 0 0), .previous (.integerUpper 0 0)]
          (.integerUpper 0 0)
  | _, _ => pure false

example : directIntegerIntervalSequentialFixture = .ok true := by
  native_decide

/-- An inner parallel loop is evaluated under a sequential body that reuses index slot zero.
The result is selected back to one carried matrix value.  The two loop coordinates must remain
scope-owned and distinct, while the selected inner lane remains a valid direct carried result. -/
private def nestedParallelSequentialInnerBody : Scope := {
  nodes := #[{ kind := .input "state", arguments := [], outputTypes := [.matrix fixtureType] }]
  outputs := [("state", { node := 0, port := 0 })]
  inputNames := ["state"]
}

private def nestedParallelSequentialOuterBody : Scope := {
  nodes := #[
    { kind := .input "state", arguments := [], outputTypes := [.matrix fixtureType] },
    { kind := .parallelLoop "inner" (.constant 2) 0 [] [.broadcast],
      arguments := [{ node := 0, port := 0 }],
      outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
    { kind := .familyGetStatic (.constant 1), arguments := [{ node := 1, port := 0 }],
      outputTypes := [.matrix fixtureType] }
  ]
  outputs := [("state", { node := 2, port := 0 })]
  inputNames := ["state"]
}

private def nestedParallelSequentialProgram : Prog := {
  root := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .sequentialLoop "outer" (.constant 2) 0 [] 1,
        arguments := [{ node := 0, port := 0 }], outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("state", { node := 1, port := 0 })]
    inputNames := []
  }
  definitions := [("outer", nestedParallelSequentialOuterBody),
    ("inner", nestedParallelSequentialInnerBody)]
}

private def nestedParallelSequentialDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .sequentialLoop, arguments := [{ node := 0, port := 0 }] }
  ] }
  definitions := [
    ("outer", { steps := #[
      { sourceNode := 0, rule := .input, arguments := [] },
      { sourceNode := 1, rule := .parallelLoop, arguments := [{ node := 0, port := 0 }] },
      { sourceNode := 2, rule := .familyGetStatic, arguments := [{ node := 1, port := 0 }] }
    ] }),
    ("inner", { steps := #[{ sourceNode := 0, rule := .input, arguments := [] }] })
  ]
}

private def simultaneousRecurrence (slot : Nat) : OperationalBoundExpr :=
  .recurrence 2 [
      .closedInt (.constant 2),
      .closedInt (.constant 5)
    ] [
      .add (.previous (.matrixMaximum 0 0)) (.closedInt (.constant 3)),
      .add (.previous (.matrixMaximum 0 1)) (.previous (.matrixMaximum 0 0))
    ] slot

/-- All carried slots read the previous state. The second slot must not observe the first slot's
new value from the same iteration. -/
example : (simultaneousRecurrence 0).evaluate [] #[] = .ok 8 := by
  native_decide

example : (simultaneousRecurrence 1).evaluate [] #[] = .ok 12 := by
  native_decide

private def nestedRecurrence : OperationalBoundExpr :=
  .recurrence 2 [.closedInt (.constant 2)] [
    .recurrence 2 [.previous (.matrixMaximum 0 0)] [
      .add (.previous (.matrixMaximum 0 0)) (.previous (.matrixMaximum 1 0))
    ] 0
  ] 0

/-- The inner depth zero denotes the inner state and depth one denotes the enclosing state. -/
example : nestedRecurrence.evaluate [] #[] = .ok 18 := by
  native_decide

/-- A zero-count recurrence returns the initial slot without evaluating its transition. -/
example : (.recurrence 0 [.closedInt (.constant 7)]
    [.previous (.matrixMaximum 0 99)] 0 : OperationalBoundExpr).evaluate [] #[] = .ok 7 := by
  native_decide

/-- A typed carried placeholder has no meaning outside recurrence evaluation. -/
example : (.previous (.matrixMaximum 0 0) : OperationalBoundExpr).evaluate [] #[] =
    .error (.invalidPreviousPath (.matrixMaximum 0 0)) := by
  native_decide

private def sampledLoopIdentity : PublicMatrixIdentity :=
  .sampledTrapdoor (.parallelBody (.root (.standalone 0)) 4) { node := 2, port := 0 }

/-- Independent samples produced at one body wire receive distinct concrete loop identities. -/
example : instantiatePublicIdentityLoopIndex 0 0 sampledLoopIdentity !=
    instantiatePublicIdentityLoopIndex 0 1 sampledLoopIdentity := by
  native_decide

/-- Nested loop instantiation retains both concrete selections. -/
example : instantiatePublicIdentityLoopIndex 1 3
    (instantiatePublicIdentityLoopIndex 0 2 sampledLoopIdentity) =
    .loopInstance 0 (.constant 2) (.loopInstance 1 (.constant 3) sampledLoopIdentity) := by
  native_decide

/-- Deterministic gadget matrices are not spuriously made lane-local. -/
example : instantiatePublicIdentityLoopIndex 0 7
    (.gadget "fixture" fixtureParams 1 2 false 3) =
    (.gadget "fixture" fixtureParams 1 2 false 3) := by
  native_decide

private def mismatchedFixtureType : MatrixTypeExpr :=
  { fixtureType with rows := .constant 2 }

/-- A frozen leaf cannot claim an output matrix type different from the type it executes. -/
example : (match evaluateScopeOperationalWithLayouts {
    nodes := #[{
      kind := .zeroMatrix fixtureType
      arguments := []
      outputTypes := [.matrix mismatchedFixtureType]
    }]
    outputs := [("result", { node := 0, port := 0 })]
    inputNames := []
  } {
    steps := #[{ sourceNode := 0, rule := .zeroMatrix, arguments := [] }]
  } [] [] with
  | .error (.outputTypeMismatch 0) => true
  | _ => false) = true := by
  native_decide

/-- Arithmetic operands must have the exact declared output matrix type. -/
example : (match evaluateScopeOperationalWithLayouts {
    nodes := #[
      { kind := .zeroMatrix fixtureType, arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .zeroMatrix mismatchedFixtureType, arguments := [],
        outputTypes := [.matrix mismatchedFixtureType] },
      { kind := .matrixAdd,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.matrix fixtureType] }
    ]
    outputs := [("result", { node := 2, port := 0 })]
    inputNames := []
  } {
    steps := #[
      { sourceNode := 0, rule := .zeroMatrix, arguments := [] },
      { sourceNode := 1, rule := .zeroMatrix, arguments := [] },
      { sourceNode := 2, rule := .matrixAdd,
        arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] }
    ]
  } [] [] with
  | .error (.outputTypeMismatch 2) => true
  | _ => false) = true := by
  native_decide

/-- Output arity is checked before any operational fact is constructed. -/
example : (match evaluateScopeOperationalWithLayouts {
    nodes := #[{
      kind := .zeroMatrix fixtureType
      arguments := []
      outputCount := 2
      outputTypes := [.matrix fixtureType]
    }]
    outputs := [("result", { node := 0, port := 0 })]
    inputNames := []
  } {
    steps := #[{ sourceNode := 0, rule := .zeroMatrix, arguments := [] }]
  } [] [] with
  | .error (.unsupportedOutputArity 0 2) => true
  | _ => false) = true := by
  native_decide

/-- The generic decoder obligation uses the exact strict product inequality. At noise three and
plaintext modulus two, ciphertext modulus thirteen passes while the boundary value twelve fails. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let residual ← derivedMatrixFactAt 1 facts { node := 1, port := 0 }
    let accepted ← decoderNoiseCheckReport [] residual [] 2 25
    let rejected ← decoderNoiseCheckReport [] residual [] 2 24
    pure (accepted.accepted, accepted.rejection, rejected.accepted, rejected.rejection)) =
    .ok (true, none, false, some (.decoderThresholdNotMet 2 24 6)) := by
  native_decide

/-- An invalid plaintext modulus is rejected by the generic report rather than interpreted by an
application-specific checker. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let residual ← derivedMatrixFactAt 1 facts { node := 1, port := 0 }
    let report ← decoderNoiseCheckReport [] residual [] 1 100
    pure (report.accepted, report.rejection)) =
    .ok (false, some (.invalidPlaintextModulus 1)) := by
  native_decide

/-- Exact indexed residual families inspect every member rather than using a representative lane. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let first ← derivedMatrixFactAt 2 facts { node := 0, port := 0 }
    let second ← derivedMatrixFactAt 2 facts { node := 1, port := 0 }
    let (fixed, firstReference) := ({} : FixedOperationalPayloadArena).pushMatrix first
    let (fixed, secondReference) := fixed.pushMatrix second
    let direct : DirectOperationalIndexedArena := { fixed }
    let binder := directCarrierFixtureBinder 820
    let (direct, residual) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[firstReference, secondReference] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let arena : OperationalExprArena := { direct }
    let report ← decoderNoiseCheckReportForFact [] arena {
      context := { binders := #[binder] }
      payload := .directValue residual
      storage := .explicitTable
    } [] 2 25
    pure report.obligations) = .ok [.decoderThreshold 2 25 6] := by
  native_decide

/-- A checked Shared indexed family stores one direct template independently of its logical count. -/
example : (do
    let facts ← evaluateScopeOperationalWithLayouts scaledNoiseScope scaledNoiseDerivation [] []
    let template ← derivedMatrixFactAt 2 facts { node := 1, port := 0 }
    let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushMatrix template
    let direct : DirectOperationalIndexedArena := { fixed }
    let binder := { directCarrierFixtureBinder 821 with count := .constant 100 }
    let (direct, residual) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) reference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let arena : OperationalExprArena := { direct }
    let report ← decoderNoiseCheckReportForFact [] arena {
      context := { binders := #[binder] }
      payload := .directValue residual
      storage := .sharedTemplate
    } [] 2 25
    pure (direct.values.size, report.obligations)) = .ok (1, [.decoderThreshold 2 25 6]) := by
  native_decide

/-- Empty residual families are rejected instead of being assigned a zero bound. -/
example :
    let binder := directCarrierFixtureBinder 822
    let fact : OperationalMatrixFact := {
      subject := { node := 823, port := 0 }
      origin := .value temporaryScope { node := 823, port := 0 }
      matrixType := fixtureType
      matrixParams := fixtureParams
      totalHardBound := .closedInt (.constant 3)
    }
    let (fixed, _) := ({} : FixedOperationalPayloadArena).pushMatrix fact
    let direct : DirectOperationalIndexedArena := { fixed }
    direct.pushExplicit [] { binders := #[binder] } binder (.matrix fixtureType) #[] = none := by
  native_decide

/-! The expression-arena fixtures use `decide`, not `native_decide`: request-local IDs and memo
statistics are ordinary checker data and do not enlarge the trusted evaluation base. -/

private def operationalExprFixtureFact (node : Nat) (bound : Int) : OperationalMatrixFact := {
  subject := { node, port := 0 }
  origin := .value temporaryScope { node, port := 0 }
  matrixType := fixtureType
  matrixParams := fixtureParams
  totalHardBound := .closedInt (.constant bound)
}

private def boundedOperationalExprFixtureFact
    (node : Nat)
    (bound : Int) : OperationalMatrixFact :=
  (operationalExprFixtureFact node bound).initializePrimitivePolynomial .bounded

/-- Evaluate the sequential body itself, then inspect its inner parallel carrier before the
outer recurrence abstracts it.  Reusing index slot zero is safe only when the inner binder and
selection retain the nested scope identity. -/
example : (do
    let prepared ← prepareProgramOperational nestedParallelSequentialProgram
      nestedParallelSequentialDerivation
    let outer ← match prepared.definitions.find?
        (fun entry : String × PreparedOperationalScope => entry.1 == "outer") with
      | some (_, scope) => pure scope
      | none => throw (OperationalError.unsupportedOperationalExpr 0)
    let (arena, input) ← ({} : OperationalExprArena).promoteConcreteMatrixFact
      (boundedOperationalExprFixtureFact 961 3)
    let outerScope : ScopeTemplateKey := .sequentialBody (.root (.standalone 0)) 1
    /- The parallel loop is owned by this sequential body, rather than by the root scope. -/
    let expectedScope : ScopeTemplateKey := outerScope
    let expectedBinder := parallelLoopFamilyBinder expectedScope 1 0
    let expectedBinderVariable ← parallelLoopLaneBinder expectedScope 1 0 (IntExpr.constant 2)
    let outerBinderVariable ← parallelLoopLaneBinder (.root (.standalone 0)) 1 0 (IntExpr.constant 2)
    let expectedContext : IndexContext := { binders := #[expectedBinderVariable] }
    let outerContext : IndexContext := { binders := #[outerBinderVariable] }
    let facts ← evaluatePreparedScope prepared.definitions [] outerScope
      (prepared.definitions.size + 1) outer [] [] arena [input]
    let innerFamily ← lookupFact 2 facts { node := 1, port := 0 }
    pure (match innerFamily.payload with
      | .directValue _ =>
          innerFamily.context == expectedContext && expectedContext != outerContext &&
            expectedBinder.owner == expectedScope
      | _ => false)) = .ok true := by
  native_decide

/-- Direct ordinary matrix operations are request-owned values: concrete leaves are promoted once,
then add and multiply allocate only delayed direct nodes and execute the fixed-assignment kernels.
No legacy expression root is available on the resulting facts. -/
private def directOrdinaryMatrixPipelineFixture : Bool :=
  match (show Except OperationalError Bool from do
    let (arena, left) ← ({} : OperationalExprArena).promoteConcreteMatrixFact
      (boundedOperationalExprFixtureFact 801 2)
    let (arena, right) ← arena.promoteConcreteMatrixFact (boundedOperationalExprFixtureFact 802 3)
    let add : PrimitiveOperation := {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 803,
      outputPort := 0, parameterEnvironment := [] }
    let (arena, sum) ← arena.pushDirectMatrixPointwise add left right
    let multiply : PrimitiveOperation := {
      kind := .multiply .matrixMultiplyBound { node := 803, port := 0 }, outputType := fixtureType,
      ownerScope := none, ownerNode := 804, outputPort := 0, parameterEnvironment := [] }
    let (arena, product) ← arena.pushDirectMatrixPointwise multiply sum right
    let sumId ← match sum.payload with
      | .directValue id => pure id
      | _ => throw (OperationalError.unsupportedOperationalExpr 0)
    let productId ← match product.payload with
      | .directValue id => pure id
      | _ => throw (OperationalError.unsupportedOperationalExpr 0)
    let sumFact ← arena.direct.matrixFactAt [] [] sumId (arena.direct.values.size + 1)
    let productFact ← arena.direct.matrixFactAt [] [] productId (arena.direct.values.size + 1)
    let (_, diagnostics) ← operationalNoiseBoundForFact arena product []
    pure (arena.nodes.isEmpty && arena.direct.values.size == 4 && sum.context == emptyContext &&
      product.context == emptyContext && sumFact.evaluateNoiseHardBound [] == Except.ok 5 &&
      productFact.evaluateNoiseHardBound [] == Except.ok 15 && diagnostics.relationRewriteCount == 0)) with
  | Except.ok value => value
  | Except.error _ => false

/-- Shared selector binders remain one dimension, while independent binders remain two dimensions
inside one delayed direct operation.  Neither case allocates an exact-choice Cartesian table. -/
private def directValueContextCorrelationFixture : Bool :=
  let shared := directCarrierFixtureBinder 805
  let independent := directCarrierFixtureBinder 806
  let leftFact := boundedOperationalExprFixtureFact 807 2
  let rightFact := boundedOperationalExprFixtureFact 808 3
  let (fixed, leftRef) := ({} : FixedOperationalPayloadArena).pushMatrix leftFact
  let (fixed, rightRef) := fixed.pushMatrix rightFact
  let direct : DirectOperationalIndexedArena := { fixed }
  let operation := OperationalIndexedPointwiseOperation.matrix {
    kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 809,
    outputPort := 0, parameterEnvironment := [] }
  match direct.pushShared { binders := #[shared] } (.matrix fixtureType) leftRef with
  | none => false
  | some (direct, left) => match direct.pushShared { binders := #[shared] }
      (.matrix fixtureType) rightRef with
    | none => false
    | some (direct, same) => match direct.pushPointwise operation #[left, same] with
      | none => false
      | some (direct, correlated) => match direct.pushShared { binders := #[independent] }
          (.matrix fixtureType) rightRef with
        | none => false
        | some (direct, other) => match direct.pushPointwise operation #[correlated, other] with
          | none => false
          | some (direct, independentResult) =>
              direct.values.size == 5 &&
                direct.values[correlated]?.any (fun value =>
                  value.context == { binders := #[shared] }) &&
                direct.values[independentResult]?.any (fun value =>
                  value.context == { binders := #[shared, independent] })

/-- Matrix-to-scalar direct kernels retain the full direct context. A caller asking for one scalar
without assigning that context fails closed; it cannot select lane zero as a representative. -/
private def directValueScalarContextFixture : Bool :=
  let binder := directCarrierFixtureBinder 810
  let fact := { boundedOperationalExprFixtureFact 811 2 with canonicalRange := .below 9 }
  let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushMatrix fact
  let direct : DirectOperationalIndexedArena := { fixed }
  match direct.pushShared { binders := #[binder] } (.matrix fixtureType) reference with
  | none => false
  | some (direct, matrix) =>
      let operation : DirectValueScalarOperation := {
        kind := .extractCoefficient (.constant 0), ownerScope := none, ownerNode := 812, outputPort := 0,
        parameterEnvironment := [] }
      match direct.pushPointwise (.matrixToScalar operation) #[matrix] with
      | none => false
      | some (direct, scalar) =>
          direct.values[scalar]?.any (fun value =>
            value.context == { binders := #[binder] } &&
              match direct.scalarFactAt [] [] scalar (direct.values.size + 1) with
              | .error _ => true
              | .ok _ => false)

/-- Direct preimage producers preserve a common public/trapdoor as shared storage, pair explicit
targets only with the same selector, and retain the exact target relation per physical lane. -/
private def directRelationProducerFixture : Bool :=
  match (do
    let target0 := boundedOperationalExprFixtureFact 820 2
    let target1 := boundedOperationalExprFixtureFact 821 3
    let binder := { directCarrierFixtureBinder 822 with count := .constant 2 }
    let other := { directCarrierFixtureBinder 823 with count := .constant 2 }
    let (fixed, publicRef) := ({} : FixedOperationalPayloadArena).pushMatrix fixturePublicMatrixFact
    let (fixed, target0Ref) := fixed.pushMatrix target0
    let (fixed, target1Ref) := fixed.pushMatrix target1
    let (fixed, trapdoorRef) := fixed.pushScalar fixtureTrapdoorFact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, publicValue) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) publicRef with | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 820)
    let (direct, trapdoor) ← match direct.pushShared { binders := #[binder] }
        (.scalar (.trapdoor fixtureType)) trapdoorRef with | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 821)
    let (direct, target) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[target0Ref, target1Ref] with | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 822)
    let (direct, wrongTarget) ← match direct.pushExplicit [] { binders := #[other] } other
        (.matrix fixtureType) #[target0Ref, target1Ref] with | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 823)
    let arena : OperationalExprArena := { direct }
    let wrap (context : IndexContext) (root : OperationalIndexedValueId) : OperationalFact :=
      { context, payload := .directValue root, storage := .explicitTable }
    let operation : DirectRelationOperation := {
      kind := .preimage (.constant 3) [], outputType := fixtureType, ownerScope := none,
      ownerNode := 824, outputPort := 0, parameterEnvironment := [] }
    let (arena, accepted) ← arena.pushDirectRelationPointwise operation
      #[wrap { binders := #[binder] } publicValue, wrap { binders := #[binder] } trapdoor,
        wrap { binders := #[binder] } target]
    let acceptedEntries ← arena.reducedDirectValueFactsAt [] accepted
    let (arena, independentlySelected) ← arena.pushDirectRelationPointwise operation
      #[wrap { binders := #[binder] } publicValue, wrap { binders := #[binder] } trapdoor,
        wrap { binders := #[other] } wrongTarget]
    let independentlySelected ← arena.reducedDirectValueFactsAt [] independentlySelected
    let acceptedOk := acceptedEntries.length == 2 && acceptedEntries.all fun entry =>
      entry.key == some (IndexExpr.variable binder) && entry.fact.relations.any fun relation =>
        match relation with | .preimage value => value.producer == entry.fact.origin | _ => false
    let independentlySelectedOk := independentlySelected.length == 2 && independentlySelected.all fun entry =>
      entry.key == some (IndexExpr.variable other)
    pure (acceptedOk && independentlySelectedOk)) with
  | .ok value => value
  | .error _ => false

example : directRelationProducerFixture = true := by native_decide

/-- When two or more relation operands are lane-dependent, the direct zipper accepts only an
identical selector/ordinal and rejects a Cartesian pairing with another selector. -/
private def directRelationLaneAlignmentFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 825 with count := .constant 2 }
    let other := { directCarrierFixtureBinder 826 with count := .constant 2 }
    let target := boundedOperationalExprFixtureFact 827 2
    let (fixed, publicRef) := ({} : FixedOperationalPayloadArena).pushMatrix fixturePublicMatrixFact
    let (fixed, targetRef) := fixed.pushMatrix target
    let (fixed, trapdoorRef) := fixed.pushScalar fixtureTrapdoorFact
    let direct : DirectOperationalIndexedArena := { fixed }
    let table schema refs b := direct.pushExplicit [] { binders := #[b] } b schema refs
    let (direct, publicValue) ← match table (.matrix fixtureType) #[publicRef, publicRef] binder with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 825)
    let (direct, trapdoor) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.scalar (.trapdoor fixtureType)) #[trapdoorRef, trapdoorRef] with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 826)
    let (direct, target) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[targetRef, targetRef] with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 827)
    let (direct, wrong) ← match direct.pushExplicit [] { binders := #[other] } other
        (.matrix fixtureType) #[targetRef, targetRef] with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 828)
    let arena : OperationalExprArena := { direct }
    let fact (binderValue : IndexVariable) (valueId : OperationalIndexedValueId) : OperationalFact := {
      context := { binders := #[binderValue] }
      payload := .directValue valueId
      storage := .explicitTable
    }
    let op : DirectRelationOperation := {
      kind := .preimage (.constant 3) []
      outputType := fixtureType
      ownerScope := none
      ownerNode := 829
      outputPort := 0
      parameterEnvironment := []
    }
    let (arena, same) ← arena.pushDirectRelationPointwise op #[fact binder publicValue, fact binder trapdoor, fact binder target]
    let same ← arena.reducedDirectValueFactsAt [] same
    let (arena, different) ← arena.pushDirectRelationPointwise op #[fact binder publicValue, fact binder trapdoor, fact other wrong]
    let different := arena.reducedDirectValueFactsAt [] different
    pure (same.length == 2 && match different with | .error (.unsupportedOperationalExpr _) => true | _ => false)) with
  | .ok value => value | .error _ => false

example : directRelationLaneAlignmentFixture = true := by native_decide

/-- Forged direct relation descriptors fail at construction: no lane reduction may repair a
public/trapdoor mismatch, target-product mismatch, digit-expansion mismatch, or negative bound. -/
private def directRelationClosedSchemaFixture : Bool :=
  let result : Except OperationalError Bool := do
    let fact := boundedOperationalExprFixtureFact 829 2
    let rows2Fact := { fact with matrixType := fixtureRows2Type, matrixParams :=
      { fact.matrixParams with rows := 2 } }.refreshPrimitivePolynomial
    let wrongTrapdoorFact := match fixtureTrapdoorFact with
      | .trapdoor value => .trapdoor {
          value with
          matrixType := fixtureRows2Type
          matrixParams := { value.matrixParams with rows := 2 }
        }
      | _ => .trapdoor {
          subject := { node := 0, port := 1 }
          matrixType := fixtureRows2Type
          matrixParams := { fixtureParams with rows := 2 }
          maximum := .closedInt (.constant 3)
          publicIdentity := fixtureSampledIdentity
        }
    let (fixed, publicRef) := ({} : FixedOperationalPayloadArena).pushMatrix fixturePublicMatrixFact
    let (fixed, targetRef) := fixed.pushMatrix fact
    let (fixed, wrongTargetRef) := fixed.pushMatrix rows2Fact
    let (fixed, trapdoorRef) := fixed.pushScalar fixtureTrapdoorFact
    let (fixed, wrongTrapdoorRef) := fixed.pushScalar wrongTrapdoorFact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, publicValue) ← match direct.pushShared emptyContext (.matrix fixtureType) publicRef with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 829)
    let (direct, trapdoor) ← match direct.pushShared emptyContext (.scalar (.trapdoor fixtureType)) trapdoorRef with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 830)
    let (direct, wrongTrapdoor) ← match direct.pushShared emptyContext
        (.scalar (.trapdoor fixtureRows2Type)) wrongTrapdoorRef with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 831)
    let (direct, target) ← match direct.pushShared emptyContext (.matrix fixtureType) targetRef with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 832)
    let (direct, wrongTarget) ← match direct.pushShared emptyContext (.matrix fixtureRows2Type) wrongTargetRef with
      | some value => pure value | none => throw (OperationalError.unsupportedOperationalExpr 833)
    let arena : OperationalExprArena := { direct }
    let wrap (value : OperationalIndexedValueId) : OperationalFact :=
      { context := emptyContext, payload := .directValue value, storage := .sharedTemplate }
    let preimage (maximum : IntExpr) (outputType : MatrixTypeExpr) : DirectRelationOperation := {
      kind := .preimage maximum [], outputType, ownerScope := none, ownerNode := 834,
      outputPort := 0, parameterEnvironment := [] }
    let decomposition : DirectRelationOperation := {
      kind := .decomposition fixtureType (.constant 2) false (.constant 1) [] [fixtureLayout]
      outputType := fixtureRows2Type, ownerScope := none, ownerNode := 835,
      outputPort := 0, parameterEnvironment := [] }
    let rejected (result : Except OperationalError (OperationalExprArena × OperationalFact)) :=
      match result with | .error _ => true | .ok _ => false
    let trapdoorRejected := rejected (arena.pushDirectRelationPointwise (preimage (.constant 3) fixtureType)
      #[wrap publicValue, wrap wrongTrapdoor, wrap target])
    let targetRejected := rejected (arena.pushDirectRelationPointwise (preimage (.constant 3) fixtureType)
      #[wrap publicValue, wrap trapdoor, wrap wrongTarget])
    let negativeRejected := rejected (arena.pushDirectRelationPointwise (preimage (.constant (-1)) fixtureType)
      #[wrap publicValue, wrap trapdoor, wrap target])
    let decompositionRejected := rejected (arena.pushDirectRelationPointwise decomposition #[wrap target])
    pure (trapdoorRejected && targetRejected && negativeRejected && decompositionRejected)
  match result with
  | Except.ok value => value
  | Except.error _ => false

example : directRelationClosedSchemaFixture = true := by native_decide

/-- Canonical coefficient representatives are not stable under negation or nonidentity scaling.
For modulus 17, a fact in `[0, 2)` may become 16 after negation, so a subsequent small
decomposition must not receive the original range authorization.  The direct pointwise kernel is
also the fixed-assignment production path for delayed indexed values. -/
private def canonicalRangeTransformFixture : Bool :=
  match (do
    let input := { boundedOperationalExprFixtureFact 845 2 with canonicalRange := .below 2 }
    let negate : PrimitiveOperation := {
      kind := .transform .negate, outputType := fixtureType, ownerScope := none, ownerNode := 846,
      outputPort := 0, parameterEnvironment := [] }
    let scale (node : Nat) (scalar : Int) : PrimitiveOperation := {
      kind := .scale (.constant scalar) [scalar] [], outputType := fixtureType, ownerScope := none,
      ownerNode := node, outputPort := 0, parameterEnvironment := [] }
    let negated ← applyDirectMatrixPointwiseOperation negate fixtureType #[input]
    let scaled ← applyDirectMatrixPointwiseOperation (scale 847 2) fixtureType #[input]
    let identity ← applyDirectMatrixPointwiseOperation (scale 848 1) fixtureType #[input]
    let concreteNegated ← transformConcreteMatrixFact 852 0 fixtureType .negate [] input
    let concreteScaled ← scaleConcreteMatrixFact 853 0 fixtureType (.constant 2) [2] [] [] input
    let zero := { input with canonicalRange := .below 1 }
    let zeroNegated ← applyDirectMatrixPointwiseOperation negate fixtureType #[zero]
    let zeroScaled ← applyDirectMatrixPointwiseOperation (scale 854 2) fixtureType #[zero]
    let decompose (node : Nat) (fact : OperationalMatrixFact) :
        Except OperationalError ReconstructionStatus := do
      let operation : DirectRelationOperation := {
        kind := .decomposition fixtureType (.constant 2) true (.constant 1) [] [fixtureLayout]
        outputType := fixtureType, ownerScope := none, ownerNode := node, outputPort := 0,
        parameterEnvironment := [] }
      let output ← applyDirectRelationProducer operation fixtureType #[.matrix fact]
      match output.relations with
      | [.decomposition relation] => pure relation.status
      | _ => throw (OperationalError.unsupportedOperationalExpr node)
    let negatedStatus ← decompose 849 negated
    let scaledStatus ← decompose 850 scaled
    let identityStatus ← decompose 851 identity
    pure (negated.canonicalRange == CanonicalRange.unknown &&
      scaled.canonicalRange == CanonicalRange.unknown &&
      identity.canonicalRange == CanonicalRange.below 2 &&
      concreteNegated.canonicalRange == CanonicalRange.unknown &&
      concreteScaled.canonicalRange == CanonicalRange.unknown &&
      zeroNegated.canonicalRange == CanonicalRange.below 1 &&
      zeroScaled.canonicalRange == CanonicalRange.below 1 &&
      negatedStatus == ReconstructionStatus.smallRangeMissing 17 &&
      scaledStatus == ReconstructionStatus.smallRangeMissing 17 &&
      identityStatus == ReconstructionStatus.available)) with
  | .ok value => value
  | .error _ => false

example : canonicalRangeTransformFixture = true := by native_decide

/-- Direct decomposition retains the selected source lane as its relation input, rather than
collapsing a delayed family to a representative matrix fact. -/
private def directDecompositionFamilyFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 830 with count := .constant 2 }
    let sourceFamily : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 830, binderSlot := 0 }
    let sourceSelection : DynamicSelectionIdentity := {
      index := .local temporaryScope { node := 830, port := 0 }
      expression := .variable binder
    }
    let input0 := indexMatrixFact sourceFamily sourceSelection { node := 831, port := 0 }
      { boundedOperationalExprFixtureFact 831 2 with canonicalRange := .below 17 }
    let input1 := indexMatrixFact sourceFamily sourceSelection { node := 832, port := 0 }
      { boundedOperationalExprFixtureFact 832 3 with canonicalRange := .below 17 }
    let (fixed, input0Ref) := ({} : FixedOperationalPayloadArena).pushMatrix input0
    let (fixed, input1Ref) := fixed.pushMatrix input1
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, input) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[input0Ref, input1Ref] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 830)
    let arena : OperationalExprArena := { direct }
    let operation : DirectRelationOperation := {
      kind := .decomposition fixtureType (.constant 2) false (.constant 1) [] [fixtureLayout]
      outputType := fixtureType
      ownerScope := none
      ownerNode := 833
      outputPort := 0
      parameterEnvironment := []
    }
    let inputFact : OperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue input
      storage := .explicitTable
    }
    let (arena, output) ← arena.pushDirectRelationPointwise operation #[inputFact]
    let entries ← arena.reducedDirectValueFactsAt [] output
    let selector := { directCarrierFixtureBinder 834 with count := .constant 2 }
    let map ← match dynamicIndexMap output.context binder (IndexExpr.variable selector) with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 834)
    let (arena, mappedOutput) ← arena.reindexDirectMatrixFact map output
    let mappedEntries ← arena.reducedDirectValueFactsAt [] mappedOutput
    let gatherPosition := { directCarrierFixtureBinder 835 with count := .constant 3 }
    let gathered := operationalFixtureGather 835 (IndexExpr.variable selector)
      (IndexExpr.variable gatherPosition)
    let gatherMap ← match dynamicIndexMap output.context binder gathered with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 835)
    let (arena, gatheredOutput) ← arena.reindexDirectMatrixFact gatherMap output
    let gatheredEntries ← arena.reducedDirectValueFactsAt [] gatheredOutput
    let sourceOk := entries.length == 2 && entries.all fun entry =>
      entry.key == some (IndexExpr.variable binder) && entry.fact.relations.any fun relation =>
        match relation with
        | .decomposition value => value.producer == entry.fact.origin &&
            value.inputOrigin != entry.fact.origin && value.status == ReconstructionStatus.available
        | _ => false
    let mappedOk := mappedEntries.length == 2 && mappedEntries.all fun entry =>
      entry.key == some (IndexExpr.variable selector) && entry.fact.relations.any fun relation =>
        match relation with
        | .decomposition value => value.producer == entry.fact.origin &&
            value.inputOrigin != entry.fact.origin
        | _ => false
    let gatheredOk := gatheredEntries.length == 2 && gatheredEntries.all fun entry =>
      entry.key == some gathered && entry.fact.relations.any fun relation =>
        match relation with
        | .decomposition value =>
            value.producer == entry.fact.origin &&
            value.inputOrigin == MatrixOriginIdentity.indexed sourceFamily gathered
              (match entry.ordinal with
              | 0 => (boundedOperationalExprFixtureFact 831 2).origin
              | _ => (boundedOperationalExprFixtureFact 832 3).origin) &&
            value.inputSummary.origin == value.inputOrigin
        | _ => false
    pure (sourceOk && mappedOk && gatheredOk)) with
  | .ok value => value
  | .error _ => false

example : directDecompositionFamilyFixture = true := by native_decide

/-- A regular two-digit layout expands one input row into the declared two-row output.  The
declared decomposition type is the output type, not the input type. -/
private def directDecompositionDigitExpansionFixture : Bool :=
  match (do
    let layout := { fixtureLayout with crtBits := 2, regularDigitCount := 2, smallDigitCount := 2 }
    let input := { boundedOperationalExprFixtureFact 836 2 with canonicalRange := .below 17 }
    let (fixed, inputRef) := ({} : FixedOperationalPayloadArena).pushMatrix input
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, inputValue) ← match direct.pushShared emptyContext (.matrix fixtureType) inputRef with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 836)
    let arena : OperationalExprArena := { direct }
    let operation : DirectRelationOperation := {
      kind := .decomposition fixtureRows2Type (.constant 2) false (.constant 2) [] [layout]
      outputType := fixtureRows2Type
      ownerScope := none
      ownerNode := 837
      outputPort := 0
      parameterEnvironment := []
    }
    let inputFact : OperationalFact := {
      context := emptyContext
      payload := .directValue inputValue
      storage := .sharedTemplate
    }
    let (arena, output) ← arena.pushDirectRelationPointwise operation #[inputFact]
    let entries ← arena.reducedDirectValueFactsAt [] output
    pure (entries.length == 1 && entries[0]?.any fun (entry : ReducedDirectMatrixFact) =>
      entry.fact.matrixParams.rows == 2 && entry.fact.matrixParams.columns == 1 &&
        entry.fact.relations.any fun relation => match relation with
          | .decomposition value => value.digitCount == 2 && value.inputSummary.matrixParams.rows == 1
          | _ => false)) with
  | .ok value => value
  | .error _ => false

example : directDecompositionDigitExpansionFixture = true := by native_decide

/-- A shared 30,720-lane relation has one physical producer invocation, whereas explicit
relation tables retain one reduced producer result per physical lane. -/
private def directRelationPhysicalCardinalityFixture : Bool :=
  let operation : DirectRelationOperation := {
    kind := .preimage (.constant 3) []
    outputType := fixtureType
    ownerScope := none
    ownerNode := 835
    outputPort := 0
    parameterEnvironment := []
  }
  let sharedCase : Except OperationalError (Nat × Nat) := do
    let binder := { directCarrierFixtureBinder 836 with count := .constant 30720 }
    let target := boundedOperationalExprFixtureFact 837 2
    let (fixed, publicRef) := ({} : FixedOperationalPayloadArena).pushMatrix fixturePublicMatrixFact
    let (fixed, targetRef) := fixed.pushMatrix target
    let (fixed, trapdoorRef) := fixed.pushScalar fixtureTrapdoorFact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, publicValue) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) publicRef with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 836)
    let (direct, trapdoor) ← match direct.pushShared { binders := #[binder] }
        (.scalar (.trapdoor fixtureType)) trapdoorRef with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 837)
    let (direct, targetValue) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) targetRef with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 838)
    let arena : OperationalExprArena := { direct }
    let wrap (value : OperationalIndexedValueId) : OperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue value
      storage := .sharedTemplate
    }
    let (arena, output) ← arena.pushDirectRelationPointwise operation
      #[wrap publicValue, wrap trapdoor, wrap targetValue]
    let entries ← arena.reducedDirectValueFactsAt [] output
    pure (arena.direct.values.size, entries.length)
  let explicitCase (count : Nat) : Except OperationalError (Nat × Nat) := do
    let binder := { directCarrierFixtureBinder (840 + count) with count := .constant count }
    let target := boundedOperationalExprFixtureFact (850 + count) 2
    let (fixed, publicRef) := ({} : FixedOperationalPayloadArena).pushMatrix fixturePublicMatrixFact
    let (fixed, targetRef) := fixed.pushMatrix target
    let (fixed, trapdoorRef) := fixed.pushScalar fixtureTrapdoorFact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, publicValue) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) (Array.replicate count publicRef) with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 839)
    let (direct, trapdoor) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.scalar (.trapdoor fixtureType)) (Array.replicate count trapdoorRef) with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 840)
    let (direct, targetValue) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) (Array.replicate count targetRef) with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 841)
    let arena : OperationalExprArena := { direct }
    let wrap (value : OperationalIndexedValueId) : OperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue value
      storage := .explicitTable
    }
    let (arena, output) ← arena.pushDirectRelationPointwise operation
      #[wrap publicValue, wrap trapdoor, wrap targetValue]
    let entries ← arena.reducedDirectValueFactsAt [] output
    pure (arena.direct.values.size, entries.length)
  match sharedCase, explicitCase 2, explicitCase 3 with
  | .ok (4, 1), .ok (4, 2), .ok (4, 3) => true
  | _, _, _ => false

example : directRelationPhysicalCardinalityFixture = true := by native_decide

/-- A family-packed direct operand stays in Graph IR through decomposition: the relation root
retains its lane context and scope evaluation does not reintroduce legacy expression nodes. -/
private def directScopeRelationGraphIRFixture : Bool :=
  let scope : Scope := {
    nodes := #[
      { kind := .gaussianSample fixtureType (.constant 2), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .gaussianSample fixtureType (.constant 3), arguments := [],
        outputTypes := [.matrix fixtureType] },
      { kind := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily (.matrix fixtureType) (.constant 2)] },
      { kind := .gadgetDecompose fixtureType (.constant 2) false (.constant 1),
        arguments := [{ node := 2, port := 0 }], outputTypes := [.preimage fixtureType] }
    ]
    outputs := [("result", { node := 3, port := 0 })]
    inputNames := []
  }
  let derivation : ScopeDerivation := { steps := #[
    { sourceNode := 0, rule := .gaussianSample, arguments := [] },
    { sourceNode := 1, rule := .gaussianSample, arguments := [] },
    { sourceNode := 2, rule := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .gadgetDecompose, arguments := [{ node := 2, port := 0 }] }
  ] }
  match evaluateScopeOperationalWithLayouts scope derivation [] [fixtureLayout] with
  | .error _ => false
  | .ok facts => match lookupFact 4 facts { node := 3, port := 0 } with
    | .error _ => false
    | .ok { context, payload := .directValue root, .. } =>
        !context.binders.isEmpty && facts.arena.nodes.isEmpty && (facts.arena.direct.valueAt? root).any fun value =>
          match value.payload with
          | .pointwise (.matrix _) (.relation { kind := .decomposition .., .. }) _ => true
          | _ => false
    | .ok _ => false

example : directScopeRelationGraphIRFixture = true := by native_decide

/-- Direct threshold kernels use the same fixed-assignment validation and exact output intervals
as their executable node handlers. The closed schema registry rejects a scalar input. -/
private def directValueScalarKernelFixture : Bool :=
  let fact := { boundedOperationalExprFixtureFact 813 2 with canonicalRange := .below 9 }
  let (fixed, matrixReference) := ({} : FixedOperationalPayloadArena).pushMatrix fact
  let (fixed, scalarReference) := fixed.pushScalar .real
  let direct : DirectOperationalIndexedArena := { fixed }
  let boolOperation : DirectValueScalarOperation := {
    kind := .thresholdDecodeBool (.constant 17) (.constant 2) (.constant 1), ownerScope := none, ownerNode := 814,
    outputPort := 0, parameterEnvironment := [] }
  let intOperation : DirectValueScalarOperation := {
    kind := .thresholdDecodeInt (.constant 17) (.constant 3) (.constant 1), ownerScope := none, ownerNode := 815,
    outputPort := 0, parameterEnvironment := [] }
  match direct.pushShared emptyContext (.matrix fixtureType) matrixReference with
  | none => false
  | some (direct, matrix) =>
      match direct.pushPointwise (.matrixToScalar boolOperation) #[matrix] with
      | none => false
      | some (direct, boolOutput) =>
          match direct.pushPointwise (.matrixToScalar intOperation) #[matrix] with
          | none => false
          | some (direct, intOutput) =>
              match direct.pushShared emptyContext (.scalar .real) scalarReference with
              | none => false
              | some (direct, scalar) =>
                  let boolValid := match direct.scalarFactAt [] [] boolOutput
                      (direct.values.size + 1) with
                    | .ok .boolean => true
                    | _ => false
                  let intValid := match direct.scalarFactAt [] [] intOutput
                      (direct.values.size + 1) with
                    | .ok (OperationalScalarFact.integer fact) =>
                        fact.lower == 0 && fact.upper == 2
                    | _ => false
                  boolValid && intValid &&
                    (direct.pushPointwise (.matrixToScalar intOperation) #[scalar]).isNone

/-- The scalar-to-matrix lift uses the same direct pointwise carrier, checks the exact scalar
schema and declared matrix type, and retains a family binder rather than collapsing to a lane. -/
private def directIntegerLiftFixture : Bool :=
  let binder := directCarrierFixtureBinder 816
  let integerFact : OperationalIntegerFact := {
    subject := { node := 816, port := 0 }
    origin := .local temporaryScope { node := 816, port := 0 }
    lower := -3
    upper := 5
    lowerExpression := .closedInt (.constant (-3))
    upperExpression := .closedInt (.constant 5)
  }
  let integer : OperationalScalarFact := .integer integerFact
  let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushScalar integer
  let direct : DirectOperationalIndexedArena := { fixed }
  let operation : DirectValueMatrixOperation := {
    kind := .liftIntegerToConstantPolynomial fixtureType
    ownerScope := none
    ownerNode := 817
    outputPort := 0
    parameterEnvironment := []
  }
  match direct.pushShared { binders := #[binder] } (.scalar .integer) reference with
  | none => false
  | some (direct, input) =>
      match direct.pushPointwise (.matrixFromScalar operation) #[input] with
      | none => false
      | some (direct, output) =>
          direct.values[output]?.any fun value =>
            value.context == { binders := #[binder] } &&
              match direct.matrixFactAt [] [(.variable binder, 0)] output (direct.values.size + 1) with
              | .ok fact => fact.metadata.isConstantPolynomial &&
                  fact.totalHardBound == .minimum (.closedInt (.constant 8))
                    (.maximum (.negate integerFact.lowerExpression) integerFact.upperExpression)
              | .error _ => false

example : directIntegerLiftFixture = true := by
  native_decide

/-- Direct tensor rejects a forged output ring even when the Kronecker shape happens to match. -/
private def directTensorOutputRingFixture : Bool :=
  let left := boundedOperationalExprFixtureFact 830 2
  let right := boundedOperationalExprFixtureFact 831 3
  let (fixed, leftReference) := ({} : FixedOperationalPayloadArena).pushMatrix left
  let (fixed, rightReference) := fixed.pushMatrix right
  let direct : DirectOperationalIndexedArena := { fixed }
  let forgedModulus : PrimitiveOperation := {
    kind := .tensor, outputType := { fixtureType with modulus := .constant 19 }, ownerScope := none,
    ownerNode := 832, outputPort := 0, parameterEnvironment := [] }
  let forgedDimension : PrimitiveOperation := {
    kind := .tensor, outputType := { fixtureType with ringDimension := .constant 2 }, ownerScope := none,
    ownerNode := 833, outputPort := 0, parameterEnvironment := [] }
  match direct.pushShared emptyContext (.matrix fixtureType) leftReference with
  | none => false
  | some (direct, left) => match direct.pushShared emptyContext (.matrix fixtureType) rightReference with
    | none => false
    | some (direct, right) =>
        (direct.pushPointwise (.matrix forgedModulus) #[left, right]).isNone &&
          (direct.pushPointwise (.matrix forgedDimension) #[left, right]).isNone

/-- A direct pointwise output receives its executable owner namespace at fixed evaluation while
the primitive factors retain the input identities that entered the operation. -/
private def directPointwiseOutputNamespaceFixture : Bool :=
  let scope : ScopeTemplateKey := .root (.standalone 834)
  let left := boundedOperationalExprFixtureFact 835 2
  let right := boundedOperationalExprFixtureFact 836 3
  let (fixed, leftReference) := ({} : FixedOperationalPayloadArena).pushMatrix left
  let (fixed, rightReference) := fixed.pushMatrix right
  let direct : DirectOperationalIndexedArena := { fixed }
  let operation : PrimitiveOperation := {
    kind := .add false, outputType := fixtureType, ownerScope := some scope, ownerNode := 837,
    outputPort := 1, parameterEnvironment := [] }
  match direct.pushShared emptyContext (.matrix fixtureType) leftReference with
  | none => false
  | some (direct, leftValue) => match direct.pushShared emptyContext (.matrix fixtureType) rightReference with
    | none => false
    | some (direct, rightValue) => match direct.pushPointwise (.matrix operation) #[leftValue, rightValue] with
      | none => false
      | some (direct, output) => match direct.matrixFactAt [] [] output (direct.values.size + 1) with
        | .error _ => false
        | .ok fact =>
            fact.subject == { node := 837, port := 1 } &&
              fact.origin == .value scope { node := 837, port := 1 } &&
              fact.polynomial.any fun term => term.product.factors.any fun factor =>
                match factor.leaf with
                | .primitive (.matrix origin) => origin == left.origin
                | .boundedSummary origin _ => origin.tokens.any fun token =>
                    match token with
                    | .primitive (.matrix inputOrigin) => inputOrigin == left.origin
                    | _ => false
                | _ => false

example : directTensorOutputRingFixture = true ∧ directPointwiseOutputNamespaceFixture = true := by
  native_decide

/-- Constant-polynomial matrix multiplication preserves the strict nonnegative range needed by a
following coefficient extraction and LUT selection. General polynomial inputs remain unknown
because negacyclic reduction can map a negative coefficient close to the modulus. -/
private def constantPolynomialProductCanonicalRangeFixture : Bool :=
  let constant (node upper : Nat) := {
    operationalExprFixtureFact node (Int.ofNat upper) with
    metadata := { isConstantPolynomial := true }
    canonicalRange := .below upper
  }
  constantPolynomialProductCanonicalRange (constant 20 4) (constant 21 5) == .below 13 &&
    constantPolynomialProductCanonicalRange
      { constant 20 4 with metadata := {} } (constant 21 5) == .unknown

/-- Matrix-product compatibility uses evaluated dimensions, accepts equivalent symbolic syntax
and every product mode supported by `inferOperationalProductMode`, retains the declared canonical
output type, and still rejects a genuinely incompatible product. -/
private def equivalentProductDimensionFixture : Bool :=
  let leftType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .constant 1, columns := .constant 2
  }
  let rightType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .multiply (.constant 1) (.constant 2), columns := .constant 3
  }
  let incompatibleRightType : MatrixTypeExpr := {
    rightType with rows := .constant 3
  }
  let outputType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .constant 1, columns := .constant 3
  }
  let scalarType : MatrixTypeExpr := {
    modulus := .constant 17, ringDimension := .constant 1,
    rows := .constant 1, columns := .constant 1
  }
  match (do
    let base := boundedOperationalExprFixtureFact 22 2
    let (arena, baseId) := ({} : OperationalExprArena).pushConcrete base
    let (arena, left) := arena.pushPrimitive 23 0 leftType [] (.add false) #[baseId, baseId]
    let (arena, right) := arena.pushPrimitive 24 0 rightType [] (.add false) #[baseId, baseId]
    let (arena, accepted) ← multiplyOperationalExprIds 25 0 outputType .matrixMultiplyBound
      { node := 24, port := 0 } [] deriveOperationalSchemaFact arena left right
      (arena.nodes.size + 1)
    let acceptedType ← match arena.get? accepted with
      | some expression => pure expression.matrixType
      | none => throw (OperationalError.invalidOperationalExprRef accepted)
    let (arena, incompatible) := arena.pushPrimitive 26 0 incompatibleRightType [] (.add false)
      #[baseId, baseId]
    let rejected := match multiplyOperationalExprIds 27 0 outputType .matrixMultiplyBound
        { node := 26, port := 0 } [] deriveOperationalSchemaFact arena left incompatible
        (arena.nodes.size + 1) with
      | .error (.operationalExprTypeMismatch _ _) => true
      | _ => false
    pure (acceptedType == outputType && rejected &&
      concreteMatrixProductMatches scalarType leftType leftType [] &&
      concreteMatrixProductMatches leftType scalarType leftType [] &&
      concreteMatrixProductMatches leftType leftType leftType [])) with
  | .ok value => value
  | .error _ => false

private def endpointIdentityFixtureFact
    (node : Nat)
    (identity : PublicMatrixIdentity) : OperationalMatrixFact :=
  ({ operationalExprFixtureFact node 8 with identity := some identity })
    |>.initializePrimitivePolynomial .large

/-- Endpoint identity validation is universal over complete selected alternatives: two matching
branches pass, while replacing only one branch with a different public identity rejects the whole
endpoint. -/
private def oneBadEndpointIdentityRejectsFixture : Bool :=
  match (do
    let expected := fixtureSampledIdentity
    let matchingFirst := endpointIdentityFixtureFact 70 expected
    let matchingSecond := endpointIdentityFixtureFact 71 expected
    let different : PublicMatrixIdentity :=
      .sampledTrapdoor (.root (.standalone 8)) { node := 0, port := 0 }
    let mismatching := endpointIdentityFixtureFact 72 different
    let binder := { directCarrierFixtureBinder 73 with count := .constant 2 }
    let (fixed, goodFirst) := ({} : FixedOperationalPayloadArena).pushMatrix matchingFirst
    let (fixed, goodSecond) := fixed.pushMatrix matchingSecond
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, goodRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[goodFirst, goodSecond] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let goodArena : OperationalExprArena := { direct }
    let goodExpression : OperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue goodRoot
      storage := .explicitTable
    }
    requireOperationalBoundaryPublicIdentity goodArena [] 74 expected goodExpression
    let (fixed, badFirst) := ({} : FixedOperationalPayloadArena).pushMatrix matchingFirst
    let (fixed, badSecond) := fixed.pushMatrix mismatching
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, badRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[badFirst, badSecond] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let badArena : OperationalExprArena := { direct }
    let badExpression : OperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue badRoot
      storage := .explicitTable
    }
    let rejected := match requireOperationalBoundaryPublicIdentity badArena [] 74 expected
        badExpression with
      | .error (.publicIdentityMismatch 74) => true
      | _ => false
    pure rejected) with
  | .ok result => result
  | .error _ => false

/-- A relation selected under one executable identity cannot be consumed by a public matrix
selected under another identity, even when their underlying unselected gadget identities match. -/
private def crossSelectionRelationMismatchFixtureResult := do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let publicMatrix ← derivedMatrixFactAt 3 facts { node := 1, port := 0 }
    let preimage ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let binder : FamilyTemplateBinder := {
      owner := temporaryScope, producerNode := 80, binderSlot := 0
    }
    let leftSelection := DynamicSelectionIdentity.fromOrigin
      (.local temporaryScope { node := 81, port := 0 }) 2
    let rightSelection := DynamicSelectionIdentity.fromOrigin
      (.local temporaryScope { node := 82, port := 0 }) 2
    let selectedPublic := indexMatrixFact binder leftSelection { node := 83, port := 0 }
      publicMatrix
    let matchingPreimage := indexMatrixFact binder leftSelection { node := 84, port := 0 }
      preimage
    let mismatchingPreimage := indexMatrixFact binder rightSelection
      { node := 85, port := 0 } preimage
    let _ ← multiplyConcreteMatrixFacts 86 0 fixtureType
      (.matrixMultiplyRelation { node := 84, port := 0 }) { node := 84, port := 0 } []
      selectedPublic matchingPreimage
    let rejected := match multiplyConcreteMatrixFacts 87 0 fixtureType
        (.matrixMultiplyRelation { node := 85, port := 0 }) { node := 85, port := 0 } []
        selectedPublic mismatchingPreimage with
      | .error (.missingRelation 87 { node := 85, port := 0 }) => true
      | _ => false
    pure (true, rejected)

private def crossSelectionRelationMismatchFixture : Bool :=
  match crossSelectionRelationMismatchFixtureResult with
  | .ok (positive, rejected) => positive && rejected
  | .error _ => false

/-- Exact adjacent relation matching rejects every independently forged target boundary field;
the accepted relation may splice a multi-term snapshot, whose equal products normalize together. -/
private def parameterizedSnapshotType
    (matrixType : MatrixTypeExpr)
    (snapshot : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
  snapshot.map fun (term : RelationSnapshotTerm) =>
    let factors := term.product.factors.map fun (factor : RelationSnapshotFactor) => {
      factor with
      inputType := matrixType
      outputType := matrixType
      boundedSummary := factor.boundedSummary.map fun summary => { summary with matrixType }
    }
    { term with product := { term.product with factors, outputType := matrixType } }

private def exactAdjacentRelationMatcherFixture : Bool :=
  match (do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let publicMatrix ← derivedMatrixFactAt 3 facts { node := 1, port := 0 }
    let preimage ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let publicFactor ← match publicMatrix.polynomial with
      | [{ product := { factors := [factor], .. }, .. }] => pure factor
      | _ => throw (OperationalError.malformedRelation 165)
    let preimageFactor ← match preimage.polynomial with
      | [{ product := { factors := [factor], .. }, .. }] => pure factor
      | _ => throw (OperationalError.malformedRelation 165)
    let matched := (matchingFactorRelation? [] publicFactor preimageFactor).isSome
    let mapRelation (map : OperationalMatrixRelation → OperationalMatrixRelation) :=
      { preimageFactor with relations := preimageFactor.relations.map map }
    let rejects (map : OperationalMatrixRelation → OperationalMatrixRelation) :=
      (matchingFactorRelation? [] publicFactor (mapRelation map)).isNone
    let forgedType : MatrixTypeExpr := { fixtureType with rows := .constant 2 }
    let forgedModulus : MatrixTypeExpr := { fixtureType with modulus := .constant 19 }
    let forgedRing : MatrixTypeExpr := { fixtureType with ringDimension := .constant 2 }
    let rewriteTarget (target : RelationTargetSummary) := { target with matrixType := forgedType }
    let forgeSnapshot (snapshot : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
      snapshot.map fun (term : RelationSnapshotTerm) =>
        { term with product := { term.product with outputType := forgedType } }
    let forgeModes (snapshot : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
      snapshot.map fun (term : RelationSnapshotTerm) =>
        { term with product := { term.product with modes := [.ordinaryMatrixProduct] } }
    let forgeMalformed (snapshot : RelationSnapshotPolynomial) : RelationSnapshotPolynomial :=
      snapshot.map fun (term : RelationSnapshotTerm) =>
        { term with product := { term.product with factors := [] } }
    let parameterType : MatrixTypeExpr := { fixtureType with modulus := .parameter "fixture_modulus" }
    let parameterizedFactor := mapRelation fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary := { value.inputSummary with
          matrixType := parameterType
          polynomial := parameterizedSnapshotType parameterType value.inputSummary.polynomial } }
      | .preimage value => .preimage { value with targetSummary := { value.targetSummary with
          matrixType := parameterType
          polynomial := parameterizedSnapshotType parameterType value.targetSummary.polynomial } }
    let parameterizedAccepted := (matchingFactorRelation? [("fixture_modulus", .integer 17)]
      publicFactor parameterizedFactor).isSome
    let parameterizedMissingRejected := (matchingFactorRelation? [] publicFactor parameterizedFactor).isNone
    let parameterizedWrongRejected := (matchingFactorRelation? [("fixture_modulus", .integer 19)]
      publicFactor parameterizedFactor).isNone
    let typeRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary := rewriteTarget value.inputSummary }
      | .preimage value => .preimage { value with targetSummary := rewriteTarget value.targetSummary }
    let modulusRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with matrixType := forgedModulus } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with matrixType := forgedModulus } }
    let ringRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with matrixType := forgedRing } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with matrixType := forgedRing } }
    let publicRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with publicIdentity := fixtureSampledIdentity }
      | .preimage value => .preimage { value with publicIdentity := fixtureSampledIdentity }
    let producer : MatrixOriginIdentity := .value temporaryScope { node := 997, port := 0 }
    let producerRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with producer }
      | .preimage value => .preimage { value with producer }
    let targetOrigin : MatrixOriginIdentity := .value temporaryScope { node := 998, port := 0 }
    let originRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputOrigin := targetOrigin }
      | .preimage value => .preimage { value with targetOrigin }
    let paramsRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with matrixParams := { value.inputSummary.matrixParams with modulus := 19 } } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with matrixParams := { value.targetSummary.matrixParams with modulus := 19 } } }
    let layoutRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with polynomial := forgeSnapshot value.inputSummary.polynomial } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with polynomial := forgeSnapshot value.targetSummary.polynomial } }
    let modesRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with polynomial := forgeModes value.inputSummary.polynomial } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with polynomial := forgeModes value.targetSummary.polynomial } }
    let malformedRejected := rejects fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with polynomial := forgeMalformed value.inputSummary.polynomial } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with polynomial := forgeMalformed value.targetSummary.polynomial } }
    let multiTermFactor := mapRelation fun relation => match relation with
      | .decomposition value => .decomposition { value with inputSummary :=
          { value.inputSummary with polynomial := value.inputSummary.polynomial ++ value.inputSummary.polynomial } }
      | .preimage value => .preimage { value with targetSummary :=
          { value.targetSummary with polynomial := value.targetSummary.polynomial ++ value.targetSummary.polynomial } }
    let product ← operationalProductFromFactors [publicFactor, multiTermFactor]
      |>.mapError (flatErrorAt 165)
    let rewritten ← rewriteOperationalRelations 165 [] [{ coefficient := 1, product }]
    pure (matched && parameterizedAccepted && parameterizedMissingRejected && parameterizedWrongRejected &&
      typeRejected && modulusRejected && ringRejected && publicRejected &&
      producerRejected && originRejected && paramsRejected && layoutRejected && modesRejected &&
      malformedRejected &&
      rewritten.length == 1 && rewritten.head?.any fun (term : OperationalTerm) =>
        term.coefficient == 2)) with
  | .ok value => value
  | .error _ => false

example : exactAdjacentRelationMatcherFixture = true := by native_decide

/-- Complementary column/row concat layouts retain every physical partition snapshot and expose
the relation-bearing pairwise products before the ordinary relation rewrite. -/
private def complementaryBlockContractFixture : Bool :=
  match (do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let publicMatrix ← derivedMatrixFactAt 3 facts { node := 1, port := 0 }
    let preimage ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let left ← concatConcreteMatrixFacts 170 0 .columns fixtureColumns2Type [] #[publicMatrix, publicMatrix]
    let right ← concatConcreteMatrixFacts 171 0 .rows fixtureRows2Type [] #[preimage, preimage]
    let output ← multiplyConcreteMatrixFacts 172 0 fixtureType
      (.matrixMultiplyRelation { node := 2, port := 0 }) { node := 2, port := 0 } [] left right
    let leftLayoutOk := left.blockLayout.any fun (layout : OperationalBlockLayout) =>
      layout.axis == ConcatAxis.columns && layout.partitions.size == 2
    let rightLayoutOk := right.blockLayout.any fun (layout : OperationalBlockLayout) =>
      layout.axis == ConcatAxis.rows && layout.partitions.size == 2
    let embedsRemoved := output.polynomial.all fun term => term.product.factors.all fun factor =>
      !factor.transforms.any fun transform => match transform with
        | .columnEmbed .columns _ | .rowEmbed .rows _ => true
        | _ => false
    pure (leftLayoutOk && rightLayoutOk && !matrixFactHasRelation output && embedsRemoved)) with
  | .ok value => value
  | .error _ => false

/-- Non-complementary concat axes remain on the compact ordinary product path; they are never
mistaken for a blockwise sum. -/
private def reverseBlockLayoutPreservedFixture : Bool :=
  match (do
    let fact := boundedOperationalExprFixtureFact 173 2
    let left ← concatConcreteMatrixFacts 174 0 .rows fixtureRows2Type [] #[fact, fact]
    let right ← concatConcreteMatrixFacts 175 0 .columns fixtureColumns2Type [] #[fact, fact]
    let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial |>.mapError (flatErrorAt 176)
    let preserved ← contractComplementaryBlocks 176 fixtureSquare2Type left right raw
    pure (!raw.isEmpty && preserved == raw)) with
  | .ok true => true
  | _ => false

/-- Reverse and diagonal concat layouts carry nonempty raw products unchanged; neither is a
complementary column/row contraction. -/
private def diagonalBlockLayoutPreservedFixture : Bool :=
  match (do
    let fact := boundedOperationalExprFixtureFact 176 2
    let left ← concatConcreteMatrixFacts 177 0 .diagonal fixtureSquare2Type [] #[fact, fact]
    let right ← concatConcreteMatrixFacts 178 0 .diagonal fixtureSquare2Type [] #[fact, fact]
    let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial |>.mapError (flatErrorAt 179)
    let preserved ← contractComplementaryBlocks 179 fixtureSquare2Type left right raw
    pure (!raw.isEmpty && preserved == raw)) with
  | .ok true => true
  | _ => false

/-- A syntactically forged layout owner or output shape is rejected before any snapshot product
is considered. -/
private def forgedComplementaryBlockLayoutRejectedFixture : Bool :=
  match (do
    let fact := boundedOperationalExprFixtureFact 180 2
    let left ← concatConcreteMatrixFacts 181 0 .columns fixtureColumns2Type [] #[fact, fact]
    let right ← concatConcreteMatrixFacts 182 0 .rows fixtureRows2Type [] #[fact, fact]
    let forgedOwner := { left with matrixType := fixtureType }
    let ownerRejected := match contractComplementaryBlocks 183 fixtureType forgedOwner right [] with
      | .error (.malformedRelation 183) => true
      | _ => false
    let outputRejected := match contractComplementaryBlocks 184 fixtureColumns2Type left right [] with
      | .error (.malformedRelation 184) => true
      | _ => false
    pure (ownerRejected && outputRejected)) with
  | .ok value => value
  | .error _ => false

/-- An actual zero block has no polynomial terms but remains the second ordered physical layout
partition, so contraction does not infer count or ordering from visible terms. -/
private def zeroComplementaryBlockPartitionFixture : Bool :=
  match (do
    let fact := boundedOperationalExprFixtureFact 185 2
    let zero := { fact with polynomial := [] }
    let left ← concatConcreteMatrixFacts 186 0 .columns fixtureColumns2Type [] #[fact, zero]
    let right ← concatConcreteMatrixFacts 187 0 .rows fixtureRows2Type [] #[fact, zero]
    let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial |>.mapError (flatErrorAt 188)
    let contracted ← contractComplementaryBlocks 188 fixtureType left right raw
    let leftLayout ← match left.blockLayout with
      | some layout => pure layout
      | none => throw (OperationalError.malformedRelation 188)
    let rightLayout ← match right.blockLayout with
      | some layout => pure layout
      | none => throw (OperationalError.malformedRelation 188)
    pure (leftLayout.partitions.size == 2 && rightLayout.partitions.size == 2 &&
      leftLayout.partitions[1]?.any fun (partition : OperationalBlockPartition) =>
        partition.polynomial.isEmpty &&
      rightLayout.partitions[1]?.any fun (partition : OperationalBlockPartition) =>
        partition.polynomial.isEmpty && !contracted.isEmpty)) with
  | .ok value => value
  | .error _ => false

/-- Complementary layouts must have exactly the same ordered physical partition count. -/
private def complementaryBlockCountMismatchFixture : Bool :=
  match (do
    let fact := boundedOperationalExprFixtureFact 177 2
    let left ← concatConcreteMatrixFacts 178 0 .columns fixtureColumns2Type [] #[fact, fact]
    let right ← concatConcreteMatrixFacts 179 0 .rows fixtureType [] #[fact]
    contractComplementaryBlocks 180 fixtureType left right []) with
  | .error (.malformedRelation 180) => true
  | _ => false

/-- Complementary layouts also validate the inner type at every partition boundary. -/
private def complementaryBlockBoundaryMismatchFixture : Bool :=
  match (do
    let fact := boundedOperationalExprFixtureFact 181 2
    let rows2 := { fact with matrixType := fixtureRows2Type, matrixParams :=
      { fact.matrixParams with rows := 2 } }.refreshPrimitivePolynomial
    let left ← concatConcreteMatrixFacts 182 0 .columns fixtureColumns2Type [] #[fact, fact]
    let right ← concatConcreteMatrixFacts 183 0 .rows fixtureRows4Type [] #[rows2, rows2]
    contractComplementaryBlocks 184 fixtureType left right []) with
  | .error (.malformedRelation 184) => true
  | _ => false

/-- A singleton complementary layout is still contracted through its authoritative snapshot. -/
private def singletonComplementaryBlockFixture : Bool :=
  match (do
    let left := boundedOperationalExprFixtureFact 185 2
    let right := boundedOperationalExprFixtureFact 186 3
    let leftLayout ← concatConcreteMatrixFacts 187 0 .columns fixtureType [] #[left]
    let rightLayout ← concatConcreteMatrixFacts 188 0 .rows fixtureType [] #[right]
    let raw ← multiplyOperationalPolynomials left.polynomial right.polynomial |>.mapError (flatErrorAt 189)
    let contracted ← contractComplementaryBlocks 189 fixtureType leftLayout rightLayout raw
    pure (!contracted.isEmpty && leftLayout.blockLayout.any fun (layout : OperationalBlockLayout) =>
      layout.partitions.size == 1 && rightLayout.blockLayout.any fun (rightLayout : OperationalBlockLayout) =>
        rightLayout.partitions.size == 1)) with
  | .ok value => value
  | .error _ => false

/-- Direct family reduction contracts the matching column/row layouts lane-by-lane when both
explicit tables carry the same selector. A different selector is rejected before any pairing. -/
private def directFamilyComplementaryBlockFixture : Bool :=
  match (do
    let facts ← evaluateScopeOperationalWithLayouts relationFixtureScope
      relationFixtureDerivation [] [fixtureLayout]
    let publicMatrix ← derivedMatrixFactAt 3 facts { node := 1, port := 0 }
    let preimage ← derivedMatrixFactAt 3 facts { node := 2, port := 0 }
    let leftFact ← concatConcreteMatrixFacts 190 0 .columns fixtureColumns2Type []
      #[publicMatrix, publicMatrix]
    let rightFact ← concatConcreteMatrixFacts 191 0 .rows fixtureRows2Type [] #[preimage, preimage]
    let binder := { directCarrierFixtureBinder 192 with count := .constant 2 }
    let otherBinder := { directCarrierFixtureBinder 193 with count := .constant 2 }
    let (fixed, leftReference) := ({} : FixedOperationalPayloadArena).pushMatrix leftFact
    let (fixed, rightReference) := fixed.pushMatrix rightFact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, leftRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureColumns2Type) #[leftReference, leftReference] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let (direct, rightRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureRows2Type) #[rightReference, rightReference] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let (direct, otherRoot) ← match direct.pushExplicit [] { binders := #[otherBinder] } otherBinder
        (.matrix fixtureRows2Type) #[rightReference, rightReference] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let arena : OperationalExprArena := { direct }
    let left : IndexedOperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue leftRoot
      storage := .explicitTable
    }
    let right : IndexedOperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue rightRoot
      storage := .explicitTable
    }
    let other : IndexedOperationalFact := {
      context := { binders := #[otherBinder] }
      payload := .directValue otherRoot
      storage := .explicitTable
    }
    let relationWire : WireRef := { node := 2, port := 0 }
    let relationRule : DerivationRule := .matrixMultiplyRelation relationWire
    let operation : PrimitiveOperation := {
      kind := .multiply relationRule relationWire
      outputType := fixtureType
      ownerScope := none
      ownerNode := 194
      outputPort := 0
      parameterEnvironment := []
    }
    let (arena, accepted) ← arena.pushDirectMatrixPointwise operation left right
    let (_, acceptedRewriteEvents) ←
      arena.reducedDirectValueFactsAtWithRelationRewriteEvents [] accepted
    let (_, repeatedRewriteEvents) ←
      arena.reducedDirectValueFactsAtWithRelationRewriteEvents [] accepted
    let (_, diagnostics) ← operationalNoiseBoundForFact arena accepted []
    let directStaticMap ← match closedStaticIndexMap [] accepted.context binder 0 with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 198)
    let (arena, directStatic) ← arena.reindexDirectMatrixFact directStaticMap accepted
    let middle := { directCarrierFixtureBinder 199 with count := .constant 2 }
    let firstMap ← match dynamicIndexMap accepted.context binder (IndexExpr.variable middle) with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 199)
    let (arena, nestedMiddle) ← arena.reindexDirectMatrixFact firstMap accepted
    let finalMap ← match closedStaticIndexMap [] nestedMiddle.context middle 0 with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 200)
    let (arena, nestedStatic) ← arena.reindexDirectMatrixFact finalMap nestedMiddle
    let eventJoin : PrimitiveOperation := {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 201,
      outputPort := 0, parameterEnvironment := [] }
    let (arena, equivalentJoined) ← arena.pushDirectMatrixPointwise eventJoin directStatic nestedStatic
    let (_, equivalentEvents) ←
      arena.reducedDirectValueFactsAtWithRelationRewriteEvents [] equivalentJoined
    let distinctStaticMap ← match closedStaticIndexMap [] accepted.context binder 1 with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 202)
    let (arena, distinctStatic) ← arena.reindexDirectMatrixFact distinctStaticMap accepted
    let (arena, distinctJoined) ← arena.pushDirectMatrixPointwise eventJoin directStatic distinctStatic
    let (_, distinctEvents) ← arena.reducedDirectValueFactsAtWithRelationRewriteEvents [] distinctJoined
    let accepted ← arena.reducedDirectValueFactsAt [] accepted
    let binderThree := { directCarrierFixtureBinder 198 with count := .constant 3 }
    let (directThree, leftThreeRoot) ← match arena.direct.pushExplicit [] { binders := #[binderThree] }
        binderThree (.matrix fixtureColumns2Type) #[leftReference, leftReference, leftReference] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr 198)
    let (directThree, rightThreeRoot) ← match directThree.pushExplicit [] { binders := #[binderThree] }
        binderThree (.matrix fixtureRows2Type) #[rightReference, rightReference, rightReference] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr 199)
    let arenaThree := { arena with direct := directThree }
    let leftThree : IndexedOperationalFact := {
      context := { binders := #[binderThree] }, payload := .directValue leftThreeRoot,
      storage := .explicitTable }
    let rightThree : IndexedOperationalFact := {
      context := { binders := #[binderThree] }, payload := .directValue rightThreeRoot,
      storage := .explicitTable }
    let (arenaThree, acceptedThree) ← arenaThree.pushDirectMatrixPointwise operation leftThree rightThree
    let (acceptedThreeEntries, acceptedThreeEvents) ←
      arenaThree.reducedDirectValueFactsAtWithRelationRewriteEvents [] acceptedThree
    let (_, diagnosticsThree) ← operationalNoiseBoundForFact arenaThree acceptedThree []
    let (arena, rejected) ← arena.pushDirectMatrixPointwise operation left other
    let rejected := arena.reducedDirectValueFactsAt [] rejected
    let acceptedOk := accepted.length == 2 && accepted.all fun (entry : ReducedDirectMatrixFact) =>
      entry.key == some (IndexExpr.variable binder) && !matrixFactHasRelation entry.fact
    let selector := { directCarrierFixtureBinder 195 with count := .constant 2 }
    let map ← match dynamicIndexMap left.context binder (IndexExpr.variable selector) with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 195)
    let (arena, reindexedLeft) ← arena.reindexDirectMatrixFact map left
    let (arena, reindexedRight) ← arena.reindexDirectMatrixFact map right
    let (arena, reindexedAccepted) ← arena.pushDirectMatrixPointwise operation reindexedLeft reindexedRight
    let reindexedAccepted ← arena.reducedDirectValueFactsAt [] reindexedAccepted
    let reindexedOk := reindexedAccepted.length == 2 &&
      reindexedAccepted.all fun (entry : ReducedDirectMatrixFact) =>
        entry.key == some (IndexExpr.variable selector) && !matrixFactHasRelation entry.fact
    let gatherPosition := { directCarrierFixtureBinder 196 with count := .constant 3 }
    let gathered := operationalFixtureGather 196 (IndexExpr.variable selector)
      (IndexExpr.variable gatherPosition)
    let gatherMap ← match dynamicIndexMap left.context binder gathered with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 196)
    let (arena, gatheredLeft) ← arena.reindexDirectMatrixFact gatherMap left
    let (arena, gatheredRight) ← arena.reindexDirectMatrixFact gatherMap right
    let (arena, gatheredAccepted) ← arena.pushDirectMatrixPointwise operation gatheredLeft gatheredRight
    let gatheredAccepted ← arena.reducedDirectValueFactsAt [] gatheredAccepted
    let distinctPosition := { directCarrierFixtureBinder 197 with count := .constant 3 }
    let distinctGathered :=
      operationalFixtureGather 197 (IndexExpr.variable selector) (IndexExpr.variable distinctPosition)
    let distinctGatherMap ← match dynamicIndexMap right.context binder distinctGathered with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 197)
    let (arena, distinctGatheredRight) ← arena.reindexDirectMatrixFact distinctGatherMap right
    let (arena, gatheredRejected) ←
      arena.pushDirectMatrixPointwise operation gatheredLeft distinctGatheredRight
    let gatheredRejected := arena.reducedDirectValueFactsAt [] gatheredRejected
    let gatheredOk := gatheredAccepted.length == 2 &&
      gatheredAccepted.all fun (entry : ReducedDirectMatrixFact) =>
        entry.key == some gathered && !matrixFactHasRelation entry.fact
    let gatheredRejectedOk := match gatheredRejected with
      | .error (.unsupportedOperationalExpr _) => true
      | _ => false
    let rejectedOk := match rejected with
      | .error (.unsupportedOperationalExpr _) => true
      | _ => false
    pure (acceptedOk && acceptedRewriteEvents.length == 2 &&
      repeatedRewriteEvents == acceptedRewriteEvents && diagnostics.relationRewriteCount == 2 &&
      equivalentEvents.length == 1 && distinctEvents.length == 2 &&
      acceptedThreeEntries.length == 3 && acceptedThreeEvents.length == 3 &&
      diagnosticsThree.relationRewriteCount == 3 &&
      reindexedOk && gatheredOk && gatheredRejectedOk && rejectedOk)) with
  | .ok value => value
  | .error _ => false

example : complementaryBlockContractFixture = true := by native_decide
example : reverseBlockLayoutPreservedFixture = true := by native_decide
example : diagonalBlockLayoutPreservedFixture = true := by native_decide
example : forgedComplementaryBlockLayoutRejectedFixture = true := by native_decide
example : zeroComplementaryBlockPartitionFixture = true := by native_decide
example : complementaryBlockCountMismatchFixture = true := by native_decide
example : complementaryBlockBoundaryMismatchFixture = true := by native_decide
example : singletonComplementaryBlockFixture = true := by native_decide
example : directFamilyComplementaryBlockFixture = true := by native_decide

/-- The summary-transfer registry has an explicit fail-closed row for every operation category
used by the Tall inventory; no registered category falls through to the unregistered behavior. -/
private def summaryTransferRegistryCoverageFixture : Bool :=
  let representative := boundedOperationalExprFixtureFact 96 3
  let source := selectedMatrixSummary #[representative]
  let registered := #[
    EnvelopeSummaryTransferOperation.instantiationMap,
    .recurrenceBoundShift, .addSubtract, .multiplyOrdinary, .tensor,
    .concat, .transform,
    .scale, .bggGrouping]
  registered.all (fun operation =>
    (transferSelectedMatrixSummary operation #[source] representative).isSome) &&
    (transferSelectedMatrixSummary .unregistered #[source] representative).isNone

private def indexedScalarFixtureInteger (node lower upper : Nat) : OperationalScalarFact :=
  .integer {
    subject := { node, port := 0 }
    origin := .local temporaryScope { node, port := 0 }
    lower := Int.ofNat lower
    upper := Int.ofNat upper
    lowerExpression := .closedInt (.constant (Int.ofNat lower))
    upperExpression := .closedInt (.constant (Int.ofNat upper))
  }

/-- Direct packing consumes only the explicit coefficient-family binder.  The residual direct
selector remains on the packed matrix carrier, without constructing a legacy scalar selection. -/
private def nestedDirectPackContextFixture : Bool :=
  match (do
    let inner := { directCarrierFixtureBinder 782 with count := .constant 3 }
    let outer := { directCarrierFixtureBinder 783 with count := .constant 5 }
    let (fixed, booleanReference) := ({} : FixedOperationalPayloadArena).pushScalar .boolean
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, innerRoot) ← match direct.pushShared { binders := #[inner] }
        (.scalar .boolean) booleanReference with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr 782)
    let (direct, outerRoot) ← match direct.pushExplicitValues [] outer
        #[innerRoot, innerRoot, innerRoot, innerRoot, innerRoot] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr 783)
    let arena : OperationalExprArena := { direct }
    let outer : IndexedOperationalFact := {
      context := { binders := #[inner, outer] }
      payload := .directValue outerRoot
      storage := .explicitTable
    }
    let node : Node := {
      kind := .packPolynomialCoefficients fixtureType (.constant 5)
      arguments := [{ node := 0, port := 0 }]
      outputTypes := [.matrix fixtureType]
    }
    let facts : OperationalScopeFacts := { values := #[#[outer]], arena }
    let (arena, output) ← genericNodeFact temporaryScope 1 node
      .packPolynomialCoefficients 0 (.matrix fixtureType) facts [] [] []
      deriveOperationalSchemaFact
    let expression ← match output with
      | expression@{ payload := .directValue _, .. } => pure expression
      | _ => throw (OperationalError.outputTypeMismatch 1)
    let value ← match arena.direct.valueAt? expression.payload.root with
      | some value => pure value
      | none => throw (OperationalError.invalidOperationalExprRef expression.payload.root)
    let firstMap ← match closedStaticIndexMap [] expression.context inner 0 with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 784)
    let secondMap ← match closedStaticIndexMap [] expression.context inner 1 with
      | some map => pure map
      | none => throw (OperationalError.unsupportedOperationalExpr 785)
    let (arena, first) ← arena.reindexDirectMatrixFact firstMap expression
    let (arena, second) ← arena.reindexDirectMatrixFact secondMap expression
    let firstEntries ← arena.reducedDirectValueFactsAt [] first
    let secondEntries ← arena.reducedDirectValueFactsAt [] second
    let subtract : PrimitiveOperation := {
      kind := .add true, outputType := fixtureType, ownerScope := none, ownerNode := 786,
      outputPort := 0, parameterEnvironment := [] }
    let (arena, residual) ← arena.pushDirectMatrixPointwise subtract first second
    let residualEntries ← arena.reducedDirectValueFactsAt [] residual
    match value.payload, firstEntries, secondEntries, residualEntries with
    | .shared (.matrix matrixType) (.matrix _), [firstEntry], [secondEntry], [residualEntry] =>
        pure (operationalMatrixTypeEqual matrixType fixtureType &&
          expression.context.binders == #[inner] && firstEntry.fact.origin != secondEntry.fact.origin &&
          !(subtractOperationalPolynomials firstEntry.fact.polynomial secondEntry.fact.polynomial).isEmpty &&
          !residualEntry.fact.polynomial.isEmpty)
    | _, _, _, _ => pure false) with
  | .ok value => value
  | .error _ => false

/-- Build one exact direct table whose lane bounds and subjects expose its physical ordering. -/
private def reducerExactTable
    (node count : Nat) (boundOffset : Int) :
    Except OperationalError (OperationalExprArena × IndexedOperationalFact × IndexVariable) := do
  let binder := { directCarrierFixtureBinder node with count := .constant count }
  let mut fixed : FixedOperationalPayloadArena := {}
  let mut references : Array FixedOperationalPayloadRef := #[]
  for lane in [:count] do
    let fact := boundedOperationalExprFixtureFact (node + lane + 1) (boundOffset + lane)
    let (nextFixed, reference) := fixed.pushMatrix fact
    fixed := nextFixed
    references := references.push reference
  let direct : DirectOperationalIndexedArena := { fixed }
  let (direct, root) ← match direct.pushExplicit [] { binders := #[binder] } binder
      (.matrix fixtureType) references with
    | some result => pure result
    | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
  pure ({ direct }, {
    context := { binders := #[binder] }
    payload := .directValue root
    storage := .explicitTable
  }, binder)

/-- Shared logical families reduce to their one stored representative, regardless of declared
cardinality.  This guards the aggregate path against domain enumeration. -/
private def reducedSharedLogicalCountsFixture : Bool :=
  let check (count : Nat) : Except OperationalError Bool := do
    let binder := { directCarrierFixtureBinder (900 + count % 97) with count := .constant count }
    let fact := boundedOperationalExprFixtureFact (1000 + count % 97) 7
    let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushMatrix fact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, root) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) reference with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let arena : OperationalExprArena := { direct }
    let expression : IndexedOperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue root
      storage := .sharedTemplate
    }
    let entries ← arena.reducedDirectValueFactsAt [] expression
    pure (entries.length == 1 && entries[0]?.any fun (entry : ReducedDirectMatrixFact) =>
      entry.key.isNone && entry.ordinal == 0 &&
        entry.fact.totalHardBound == OperationalBoundExpr.closedInt (.constant 7))
  match check 2, check 1024, check 30720 with
  | .ok true, .ok true, .ok true => true
  | _, _, _ => false

/-- Exact direct tables retain every physical lane in table order, while the older materializing
API remains exhaustively assignment-driven. -/
private def reducedExplicitTableFixture : Bool :=
  match (do
    let (arena, expression, binder) ← reducerExactTable 1100 3 4
    let reduced ← arena.reducedDirectValueFactsAt [] expression
    let exhaustive ← arena.directValueFactsAt [] expression
    pure (reduced.length == 3 && exhaustive.length == 3 &&
      reduced.map (fun (entry : ReducedDirectMatrixFact) =>
        (entry.key, entry.ordinal, entry.fact.subject.node, entry.fact.totalHardBound)) == [
          (some (IndexExpr.variable binder), 0, 1101, OperationalBoundExpr.closedInt (.constant 4)),
          (some (IndexExpr.variable binder), 1, 1102, OperationalBoundExpr.closedInt (.constant 5)),
          (some (IndexExpr.variable binder), 2, 1103,
            OperationalBoundExpr.closedInt (.constant 6))] &&
      exhaustive.map (fun (fact : OperationalMatrixFact) =>
        (fact.subject.node, fact.totalHardBound)) == [
        (1101, OperationalBoundExpr.closedInt (.constant 4)),
        (1102, OperationalBoundExpr.closedInt (.constant 5)),
        (1103, OperationalBoundExpr.closedInt (.constant 6))])) with
  | .ok value => value
  | .error _ => false

/-- Equal selector tables reduce pointwise lane-by-lane; independent tables fail before any
Cartesian pairing can occur. -/
private def reducedPointwiseCorrelationFixture : Bool :=
  match (do
    let (arena, left, binder) ← reducerExactTable 1200 3 1
    let mut direct := arena.direct
    let mut rightReferences : Array FixedOperationalPayloadRef := #[]
    for lane in [:3] do
      let (fixed, reference) := direct.fixed.pushMatrix
        (boundedOperationalExprFixtureFact (1301 + lane) (10 + lane))
      direct := { direct with fixed }
      rightReferences := rightReferences.push reference
    let (rightDirect, rightRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) rightReferences with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr direct.values.size)
    let arena := { arena with direct := rightDirect }
    let right : IndexedOperationalFact := {
      context := { binders := #[binder] }
      payload := .directValue rightRoot
      storage := .explicitTable
    }
    let operation : PrimitiveOperation := {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 1400,
      outputPort := 0, parameterEnvironment := [] }
    let (arena, output) ← arena.pushDirectMatrixPointwise operation left right
    let reduced ← arena.reducedDirectValueFactsAt [] output
    let (fixed, sharedReference) := arena.direct.fixed.pushMatrix
      (boundedOperationalExprFixtureFact 1401 7)
    let sharedDirect := { arena.direct with fixed }
    let (sharedDirect, sharedRoot) ← match sharedDirect.pushShared emptyContext (.matrix fixtureType) sharedReference with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr sharedDirect.values.size)
    let arena := { arena with direct := sharedDirect }
    let shared : IndexedOperationalFact := {
      context := emptyContext, payload := .directValue sharedRoot, storage := .sharedTemplate }
    let (arena, sharedFirst) ← arena.pushDirectMatrixPointwise operation shared left
    let (arena, sharedLast) ← arena.pushDirectMatrixPointwise operation left shared
    let sharedFirstReduced ← arena.reducedDirectValueFactsAt [] sharedFirst
    let sharedLastReduced ← arena.reducedDirectValueFactsAt [] sharedLast
    let independentBinder := { directCarrierFixtureBinder 1500 with count := .constant 3 }
    let mut independentDirect := arena.direct
    let mut independentReferences : Array FixedOperationalPayloadRef := #[]
    for lane in [:3] do
      let (fixed, reference) := independentDirect.fixed.pushMatrix
        (boundedOperationalExprFixtureFact (1501 + lane) (20 + lane))
      independentDirect := { independentDirect with fixed }
      independentReferences := independentReferences.push reference
    let (completedIndependentDirect, independentRoot) ← match independentDirect.pushExplicit []
        { binders := #[independentBinder] }
        independentBinder (.matrix fixtureType) independentReferences with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr independentDirect.values.size)
    let independentArena := { arena with direct := completedIndependentDirect }
    let independent : IndexedOperationalFact := {
      context := { binders := #[independentBinder] }, payload := .directValue independentRoot,
      storage := .explicitTable }
    let (independentArena, rejected) ←
      independentArena.pushDirectMatrixPointwise operation left independent
    let rejectedResult := independentArena.reducedDirectValueFactsAt [] rejected
    let reducedOk := reduced.map (fun (entry : ReducedDirectMatrixFact) => (entry.key, entry.ordinal)) == [
        (some (IndexExpr.variable binder), 0), (some (IndexExpr.variable binder), 1),
        (some (IndexExpr.variable binder), 2)]
    let sharedFirstOk := sharedFirstReduced.map (fun (entry : ReducedDirectMatrixFact) =>
        (entry.key, entry.ordinal, entry.fact.evaluateNoiseHardBound [])) == [
        (some (IndexExpr.variable binder), 0, Except.ok 8),
        (some (IndexExpr.variable binder), 1, Except.ok 9),
        (some (IndexExpr.variable binder), 2, Except.ok 10)]
    let sharedLastOk := sharedLastReduced.map (fun (entry : ReducedDirectMatrixFact) =>
        (entry.key, entry.ordinal, entry.fact.evaluateNoiseHardBound [])) == [
        (some (IndexExpr.variable binder), 0, Except.ok 8),
        (some (IndexExpr.variable binder), 1, Except.ok 9),
        (some (IndexExpr.variable binder), 2, Except.ok 10)]
    let rejectedOk := match rejectedResult with
      | .error (.unsupportedOperationalExpr _) => independentArena.nodes.isEmpty
      | _ => false
    pure (reducedOk && sharedFirstOk && sharedLastOk && rejectedOk)) with
  | .ok value => value
  | .error _ => false

/-- Static access consumes the selected lane. Dynamic, offset, and dependent gather maps retain
the exact owner-bearing key and source physical ordinal.  Gather keeps one source-table entry per
source lane rather than expanding it once per lookup position. -/
private def reducedMappedDirectFixture : Bool :=
  match (do
    let (arena, expression, binder) ← reducerExactTable 1600 3 4
    let staticMap ← match closedStaticIndexMap [] expression.context binder 1 with
      | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 1600)
    let (arena, staticOutput) ← arena.reindexDirectMatrixFact staticMap expression
    let staticEntries ← arena.reducedDirectValueFactsAt [] staticOutput
    let selector := { directCarrierFixtureBinder 1700 with count := .constant 3 }
    let dynamicMap ← match dynamicIndexMap expression.context binder
        (IndexExpr.variable selector) with
      | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 1601)
    let (arena, dynamicOutput) ← arena.reindexDirectMatrixFact dynamicMap expression
    let dynamicEntries ← arena.reducedDirectValueFactsAt [] dynamicOutput
    let offsetSelector := { directCarrierFixtureBinder 1701 with count := .constant 2 }
    let offsetMap ← match dynamicIndexMap expression.context binder
        (IndexExpr.offset (IndexExpr.variable offsetSelector) 1) with
      | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 1602)
    let (arena, offsetOutput) ← arena.reindexDirectMatrixFact offsetMap expression
    let offsetEntries ← arena.reducedDirectValueFactsAt [] offsetOutput
    let gatherPosition := { directCarrierFixtureBinder 1702 with count := .constant 4 }
    let gathered := operationalFixtureGather 1702 (IndexExpr.variable selector)
      (IndexExpr.variable gatherPosition)
    let gatherMap ← match dynamicIndexMap expression.context binder gathered with
      | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 1603)
    let (arena, gatherOutput) ← arena.reindexDirectMatrixFact gatherMap expression
    let gatherEntries ← arena.reducedDirectValueFactsAt [] gatherOutput
    let (arena, gatherPointwise) ← arena.pushDirectMatrixPointwise {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 1604,
      outputPort := 0, parameterEnvironment := [] } gatherOutput gatherOutput
    let gatherPointwiseEntries ← arena.reducedDirectValueFactsAt [] gatherPointwise
    let distinctPosition := { directCarrierFixtureBinder 1703 with count := .constant 4 }
    let distinctGathered :=
      operationalFixtureGather 1703 (IndexExpr.variable selector) (IndexExpr.variable distinctPosition)
    let distinctGatherMap ← match dynamicIndexMap expression.context binder distinctGathered with
      | some map => pure map | none => throw (OperationalError.unsupportedOperationalExpr 1605)
    let (arena, distinctGatherOutput) ← arena.reindexDirectMatrixFact distinctGatherMap expression
    let (arena, rejectedGatherPointwise) ← arena.pushDirectMatrixPointwise {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 1606,
      outputPort := 0, parameterEnvironment := [] } gatherOutput distinctGatherOutput
    let rejectedGatherResult := arena.reducedDirectValueFactsAt [] rejectedGatherPointwise
    pure (staticEntries.map (fun (entry : ReducedDirectMatrixFact) =>
        (entry.key, entry.ordinal, entry.fact.subject.node)) ==
        [(none, 0, 1602)] &&
      dynamicEntries.map (fun (entry : ReducedDirectMatrixFact) => (entry.key, entry.ordinal)) == [
        (some (IndexExpr.variable selector), 0), (some (IndexExpr.variable selector), 1),
        (some (IndexExpr.variable selector), 2)] &&
      offsetEntries.map (fun (entry : ReducedDirectMatrixFact) => (entry.key, entry.ordinal)) == [
        (some (IndexExpr.variable offsetSelector), 0), (some (IndexExpr.variable offsetSelector), 1)] &&
      gatherEntries.map (fun (entry : ReducedDirectMatrixFact) => (entry.key, entry.ordinal)) == [
        (some gathered, 0), (some gathered, 1), (some gathered, 2)] &&
      gatherPointwiseEntries.map (fun (entry : ReducedDirectMatrixFact) => (entry.key, entry.ordinal)) == [
        (some gathered, 0), (some gathered, 1), (some gathered, 2)] &&
      match rejectedGatherResult with
      | .error (.unsupportedOperationalExpr _) => true
      | _ => false)) with
  | .ok value => value
  | .error _ => false

example : reducedSharedLogicalCountsFixture = true := by native_decide
example : reducedExplicitTableFixture = true := by native_decide
example : reducedPointwiseCorrelationFixture = true := by native_decide

/-- Nested direct tables retain both independent physical assignments.  Reusing the nested
table in a pointwise primitive zips exact composite environments rather than expanding them. -/
private def reducedNestedExplicitValuesCorrelationFixture : Bool :=
  match (do
    let (arena, child, inner) ← reducerExactTable 1550 2 3
    let outer := { directCarrierFixtureBinder 1551 with count := .constant 2 }
    let (direct, root) ← match arena.direct.pushExplicitValues [] outer #[child.payload.root, child.payload.root] with
      | some result => pure result
      | none => throw (OperationalError.unsupportedOperationalExpr 1551)
    let arena : OperationalExprArena := { arena with direct }
    let nested : IndexedOperationalFact := {
      context := { binders := #[inner, outer] }, payload := .directValue root, storage := .explicitTable }
    let entries ← arena.reducedDirectValueFactsAt [] nested
    let (arena, pointwise) ← arena.pushDirectMatrixPointwise {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 1552,
      outputPort := 0, parameterEnvironment := [] } nested nested
    let zipped ← arena.reducedDirectValueFactsAt [] pointwise
    pure (entries.map (fun (entry : ReducedDirectMatrixFact) => entry.correlation) == [
      [(IndexExpr.variable outer, 0), (IndexExpr.variable inner, 0)],
      [(IndexExpr.variable outer, 0), (IndexExpr.variable inner, 1)],
      [(IndexExpr.variable outer, 1), (IndexExpr.variable inner, 0)],
      [(IndexExpr.variable outer, 1), (IndexExpr.variable inner, 1)]] &&
      zipped.map (fun (entry : ReducedDirectMatrixFact) => entry.correlation) ==
        entries.map (fun (entry : ReducedDirectMatrixFact) => entry.correlation))) with
  | .ok value => value
  | .error _ => false

example : reducedNestedExplicitValuesCorrelationFixture = true := by native_decide
example : reducedMappedDirectFixture = true := by native_decide

/-- Direct storage diagnostics count physical allocations separately from logical lanes.  A
three-lane Shared matrix family has one carrier node and one fixed leaf, not three copied leaves;
its direct reduction nevertheless records the normalized one-term result. -/
private def directStorageDiagnosticsFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 1750 with count := .constant 3 }
    let fact := boundedOperationalExprFixtureFact 1751 7
    let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushMatrix fact
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, root) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) reference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr (1750 : Nat))
    let arena : OperationalExprArena := { direct }
    let residual : IndexedOperationalFact := {
      context := { binders := #[binder] }, payload := .directValue root, storage := .sharedTemplate }
    let (bound, diagnostics) ← operationalNoiseBoundForFact arena residual []
    pure (bound == 7 && diagnostics.expressionNodeCount == 2 &&
      diagnostics.envelopeLogicalBranchCount == 3 && diagnostics.envelopeStoredBranchCount == 1 &&
      diagnostics.maximumPolynomialTerms == 1 && diagnostics.peakMemoEntries == 0)) with
  | Except.ok true => true
  | _ => false

example : directStorageDiagnosticsFixture = true := by native_decide

/-- Diagnostics are arena inventory telemetry: an unrelated scalar allocation remains visible
alongside the selected matrix family instead of being silently pruned as unreachable. -/
private def arenaInventoryDiagnosticsFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 1754 with count := .constant 3 }
    let fact := boundedOperationalExprFixtureFact 1755 7
    let (fixed, matrixReference) := ({} : FixedOperationalPayloadArena).pushMatrix fact
    let (fixed, scalarReference) := fixed.pushScalar .boolean
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, _) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) matrixReference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1754)
    let (direct, _) ← match direct.pushShared emptyContext (.scalar .boolean) scalarReference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1755)
    let arena : OperationalExprArena := { direct }
    let diagnostics := operationalAnalysisDiagnostics arena
    pure (diagnostics.expressionNodeCount == 4 && diagnostics.envelopeLogicalBranchCount == 4 &&
      diagnostics.envelopeStoredBranchCount == 2) : Except OperationalError Bool) with
  | Except.ok true => true
  | _ => false

example : arenaInventoryDiagnosticsFixture = true := by native_decide

/-- Direct reduction retains widths reached by children even when a later normalization cancels
them.  The child has two Large terms; subtracting it from itself produces the zero residual, so
the final fact has no terms while reporting must still retain the intermediate width two. -/
private def directIntermediatePolynomialWidthFixture : Bool :=
  match (do
    let first := (operationalExprFixtureFact 1760 1).initializePrimitivePolynomial .large
    let second := (operationalExprFixtureFact 1761 1).initializePrimitivePolynomial .large
    let (fixed, firstReference) := ({} : FixedOperationalPayloadArena).pushMatrix first
    let (fixed, secondReference) := fixed.pushMatrix second
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, firstRoot) ← match direct.pushShared emptyContext (.matrix fixtureType) firstReference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1760)
    let (direct, secondRoot) ← match direct.pushShared emptyContext (.matrix fixtureType) secondReference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1761)
    let arena : OperationalExprArena := { direct }
    let firstFact : IndexedOperationalFact := {
      context := emptyContext, payload := .directValue firstRoot, storage := .sharedTemplate }
    let secondFact : IndexedOperationalFact := {
      context := emptyContext, payload := .directValue secondRoot, storage := .sharedTemplate }
    let add : PrimitiveOperation := {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 1762,
      outputPort := 0, parameterEnvironment := [] }
    let subtract : PrimitiveOperation := { add with kind := .add true, ownerNode := 1763 }
    let (arena, child) ← arena.pushDirectMatrixPointwise add firstFact secondFact
    let (arena, residual) ← arena.pushDirectMatrixPointwise subtract child child
    let (bound, diagnostics) ← operationalNoiseBoundForFact arena residual []
    pure (bound == 0 && diagnostics.maximumPolynomialTerms == 2)) with
  | .ok true => true
  | _ => false

example : directIntermediatePolynomialWidthFixture = true := by native_decide

/-- `explicitValues` must retain its child's reporting width while it rebases the physical lane
key.  Both outer lanes are the cancelled zero residual from the preceding construction. -/
private def explicitValuesIntermediatePolynomialWidthFixture : Bool :=
  match (do
    let first := (operationalExprFixtureFact 1770 1).initializePrimitivePolynomial .large
    let second := (operationalExprFixtureFact 1771 1).initializePrimitivePolynomial .large
    let (fixed, firstReference) := ({} : FixedOperationalPayloadArena).pushMatrix first
    let (fixed, secondReference) := fixed.pushMatrix second
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, firstRoot) ← match direct.pushShared emptyContext (.matrix fixtureType) firstReference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1770)
    let (direct, secondRoot) ← match direct.pushShared emptyContext (.matrix fixtureType) secondReference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1771)
    let arena : OperationalExprArena := { direct }
    let firstFact : IndexedOperationalFact := {
      context := emptyContext, payload := .directValue firstRoot, storage := .sharedTemplate }
    let secondFact : IndexedOperationalFact := {
      context := emptyContext, payload := .directValue secondRoot, storage := .sharedTemplate }
    let add : PrimitiveOperation := {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 1772,
      outputPort := 0, parameterEnvironment := [] }
    let subtract : PrimitiveOperation := { add with kind := .add true, ownerNode := 1773 }
    let (arena, child) ← arena.pushDirectMatrixPointwise add firstFact secondFact
    let binder := { directCarrierFixtureBinder 1774 with count := .constant 2 }
    let childRoot := child.payload.root
    let (direct, wrappedRoot) ← match arena.direct.pushExplicitValues [] binder #[childRoot, childRoot] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 1774)
    let arena := { arena with direct }
    let wrapped : IndexedOperationalFact := {
      context := { binders := #[binder] }, payload := .directValue wrappedRoot, storage := .explicitTable }
    let (arena, residual) ← arena.pushDirectMatrixPointwise subtract wrapped wrapped
    let (_, _, directMaximumPolynomialTerms) ←
      arena.reducedDirectValueFactsAtWithDiagnostics [] residual
    let (bound, diagnostics) ← operationalNoiseBoundForFact arena residual []
    pure (bound == 0 && directMaximumPolynomialTerms == 2 &&
      diagnostics.maximumPolynomialTerms == 2)) with
  | .ok true => true
  | _ => false

example : explicitValuesIntermediatePolynomialWidthFixture = true := by native_decide

/-- Decoder-bound evaluation rejects a pure Large residual and a normalized mixed
Large-plus-bounded residual before looking at their bounded-only summaries. -/
private def residualLargeSingleAndMixedFixture : Bool :=
  let large := (operationalExprFixtureFact 5001 8).initializePrimitivePolynomial .large
  let bounded := boundedOperationalExprFixtureFact 5002 3
  match (do
    let (arena, largeRoot) := ({} : OperationalExprArena).pushConcrete large
    let largeResidual ← arena.indexedExpr largeRoot
    let (arena, boundedRoot) := arena.pushConcrete bounded
    let (arena, mixedRoot) ← addOperationalExprIds 5003 0 fixtureType false []
      deriveOperationalSchemaFact arena largeRoot boundedRoot (arena.nodes.size + 1)
    let mixedResidual ← arena.indexedExpr mixedRoot
    pure (operationalNoiseBoundForFact arena largeResidual [],
      operationalNoiseBoundForFact arena mixedResidual [])) with
  | .ok (.error (.residualContainsLargeTerm 5001), .error (.residualContainsLargeTerm 5003)) => true
  | _ => false

/-- Direct indexed families retain their physical alternatives through the residual boundary.
One bounded lane cannot hide a Large lane behind a family-wide maximum. -/
private def residualLargeDirectFamilyFixture : Bool :=
  let binder := { directCarrierFixtureBinder 5030 with count := .constant 2 }
  let bounded := boundedOperationalExprFixtureFact 5031 2
  let large := (operationalExprFixtureFact 5032 8).initializePrimitivePolynomial .large
  match (show Except OperationalError (Except OperationalError (Int × OperationalAnalysisDiagnostics))
    from do
    let (fixed, boundedRef) := ({} : FixedOperationalPayloadArena).pushMatrix bounded
    let (fixed, largeRef) := fixed.pushMatrix large
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, root) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[boundedRef, largeRef] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 5030)
    let arena : OperationalExprArena := { direct }
    let residual : OperationalFact := {
      context := { binders := #[binder] }, payload := .directValue root, storage := .explicitTable }
    pure (operationalNoiseBoundForFact arena residual [])) with
  | Except.ok (Except.error (.residualContainsLargeTerm 5032)) => true
  | _ => false

/-- Exact polynomial cancellation occurs before the residual boundary, so a cancelled Large
signal leaves the zero noise residual rather than being rejected from one of its inputs. -/
private def residualLargeCancellationFixture : Bool :=
  let large := (operationalExprFixtureFact 5040 8).initializePrimitivePolynomial .large
  match (do
    let (arena, left) := ({} : OperationalExprArena).pushConcrete large
    let (arena, right) := arena.pushConcrete large
    let (arena, root) ← addOperationalExprIds 5042 0 fixtureType true []
      deriveOperationalSchemaFact arena left right (arena.nodes.size + 1)
    let residual ← arena.indexedExpr root
    pure (operationalNoiseBoundForFact arena residual [])) with
  | .ok (.ok (0, _)) => true
  | _ => false

/-- The relation fixture reaches the same boundary only after its exact relation has rewritten
away the signal term; its existing bound of three is therefore an acceptance regression test. -/
private def residualLargeRelationConsumptionFixture : Bool :=
  exactRelationSelectionFixtureResult == .ok true

example : residualLargeSingleAndMixedFixture = true := by native_decide
example : residualLargeDirectFamilyFixture = true := by native_decide
example : residualLargeCancellationFixture = true := by native_decide
example : residualLargeRelationConsumptionFixture = true := by native_decide

/-- Cutoff agreement is pointwise in its contextual domain.  In particular, equal extrema do
not permit `i` to agree with `1 - i`. -/
private def contextualCutoffFixture : Bool :=
  let domains := [OperationalParameterDomain.loopIndex 0 2]
  let nonnegative := validateContextualCutoffNonnegative 5100 [] domains (.loopIndex 0)
  let negative := validateContextualCutoffNonnegative 5101 [] domains
    (.subtract (.loopIndex 0) (.constant 1))
  let equalSyntax := validatePreimageCutoffAgreement 5102 [] domains
    (.add (.loopIndex 0) (.constant 0))
    fixtureSampledIdentity
    (some (.contextual .maximum [] domains (.loopIndex 0)))
  let equalExtrema := validatePreimageCutoffAgreement 5103 [] domains (.loopIndex 0)
    fixtureSampledIdentity
    (some (.contextual .maximum [] domains (.subtract (.constant 1) (.loopIndex 0))))
  let forwardMismatch := validatePreimageCutoffAgreement 5104 [] [] (.constant 2)
    fixtureSampledIdentity
    (some (.contextual .maximum [] [] (.constant 3)))
  let reverseMismatch := validatePreimageCutoffAgreement 5105 [] [] (.constant 3)
    fixtureSampledIdentity
    (some (.contextual .maximum [] [] (.constant 2)))
  let parameterDomain := validateContextualCutoffNonnegative 5106
    [("cutoff", .integer 3)] [] (.parameter "cutoff")
  match nonnegative, negative, equalSyntax, equalExtrema, forwardMismatch, reverseMismatch,
      parameterDomain with
  | .ok _, .error (.invalidBound 5101 (-1)), .ok (), .error (.preimageCutoffMismatch 5103),
      .error (.preimageCutoffMismatch 5104), .error (.preimageCutoffMismatch 5105), .ok _ => true
  | _, _, _, _, _, _, _ => false

/-- The direct relation reducer checks the retained trapdoor cutoff for every physical lane. -/
private def directContextualCutoffFixture : Bool :=
  let trapdoor : OperationalTrapdoorFact := match fixtureTrapdoorFact with
    | .trapdoor fact => { fact with
        preimageCutoff := some (.contextual .maximum [] [] (.constant 3)) }
    | _ => {
        subject := { node := 0, port := 1 }, matrixType := fixtureType, matrixParams := fixtureParams,
        maximum := .closedInt (.constant 3),
        preimageCutoff := some (.contextual .maximum [] [] (.constant 3)),
        publicIdentity := fixtureSampledIdentity }
  let operation (cutoff : IntExpr) : DirectRelationOperation := {
    kind := .preimage cutoff [], outputType := fixtureType, ownerScope := none,
    ownerNode := 5110, outputPort := 0, parameterEnvironment := [] }
  let target := boundedOperationalExprFixtureFact 5111 2
  match applyDirectRelationProducer (operation (.constant 3)) fixtureType
      #[.matrix fixturePublicMatrixFact, .trapdoor trapdoor, .matrix target],
      applyDirectRelationProducer (operation (.constant 2)) fixtureType
      #[.matrix fixturePublicMatrixFact, .trapdoor trapdoor, .matrix target] with
  | .ok _, .error (.preimageCutoffMismatch 5110) => true
  | _, _ => false

/-- Missing sampled-trapdoor metadata and conflicting assignment domains are rejected before a
relation can be formed; a parameterized cutoff remains comparable pointwise. -/
private def cutoffMetadataClosureFixture : Bool :=
  let loop2 := [OperationalParameterDomain.loopIndex 0 2]
  let loop3 := [OperationalParameterDomain.loopIndex 0 3]
  let parameters : ParamEnvironment := [("cutoff", .integer 3)]
  let parameterDomains := [OperationalParameterDomain.parameter "cutoff" parameters []
    (.parameter "cutoff")]
  let sampledMissing := validatePreimageCutoffAgreement 5112 [] [] (.constant 3)
    fixtureSampledIdentity none
  let conflictingDomains := validatePreimageCutoffAgreement 5113 [] loop3 (.loopIndex 0)
    fixtureSampledIdentity (some (.contextual .maximum [] loop2 (.loopIndex 0)))
  let parameterMatch := validatePreimageCutoffAgreement 5114 parameters parameterDomains
    (.parameter "cutoff") fixtureSampledIdentity
    (some (.contextual .maximum parameters parameterDomains (.parameter "cutoff")))
  let parameterMismatch := validatePreimageCutoffAgreement 5115 parameters parameterDomains
    (.add (.parameter "cutoff") (.constant 1)) fixtureSampledIdentity
    (some (.contextual .maximum parameters parameterDomains (.parameter "cutoff")))
  match sampledMissing, conflictingDomains, parameterMatch, parameterMismatch with
  | .error (.missingPreimageCutoff 5112), .error (.preimageCutoffMismatch 5113), .ok (),
      .error (.preimageCutoffMismatch 5115) => true
  | _, _, _, _ => false

/-- Direct indexed reduction reads the cutoff from each physical trapdoor lane.  A shared public
matrix and explicit trapdoor/target tables therefore accept matching lanes and reject one stale
lane without choosing a representative. -/
private def directIndexedTrapdoorCutoffFixture : Bool :=
  let run (secondCutoff : Int) : Except OperationalError (List ReducedDirectMatrixFact) := do
    let binder := { directCarrierFixtureBinder 5116 with count := .constant 2 }
    let trapdoor (cutoff : Int) : OperationalScalarFact := match fixtureTrapdoorFact with
      | .trapdoor fact => .trapdoor { fact with
          preimageCutoff := some (.contextual .maximum [] [] (.constant cutoff)) }
      | fact => fact
    let target := boundedOperationalExprFixtureFact 5117 2
    let (fixed, publicRef) := ({} : FixedOperationalPayloadArena).pushMatrix fixturePublicMatrixFact
    let (fixed, targetRef) := fixed.pushMatrix target
    let (fixed, firstTrapdoorRef) := fixed.pushScalar (trapdoor 3)
    let (fixed, secondTrapdoorRef) := fixed.pushScalar (trapdoor secondCutoff)
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, publicValue) ← match direct.pushShared { binders := #[binder] }
        (.matrix fixtureType) publicRef with
      | some value => pure value
      | none => throw (.unsupportedOperationalExpr 5116)
    let (direct, trapdoorValue) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.scalar (.trapdoor fixtureType)) #[firstTrapdoorRef, secondTrapdoorRef] with
      | some value => pure value
      | none => throw (.unsupportedOperationalExpr 5117)
    let (direct, targetValue) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[targetRef, targetRef] with
      | some value => pure value
      | none => throw (.unsupportedOperationalExpr 5118)
    let arena : OperationalExprArena := { direct }
    let wrap (value : OperationalIndexedValueId) : OperationalFact := {
      context := { binders := #[binder] }, payload := .directValue value, storage := .explicitTable }
    let operation : DirectRelationOperation := {
      kind := .preimage (.constant 3) [], outputType := fixtureType, ownerScope := none,
      ownerNode := 5119, outputPort := 0, parameterEnvironment := [] }
    let (arena, output) ← arena.pushDirectRelationPointwise operation
      #[wrap publicValue, wrap trapdoorValue, wrap targetValue]
    arena.reducedDirectValueFactsAt [] output
  match run 3, run 2 with
  | .ok matching, .error (.preimageCutoffMismatch 5119) => matching.length == 2
  | _, _ => false

example : contextualCutoffFixture = true := by native_decide
example : directContextualCutoffFixture = true := by native_decide
example : cutoffMetadataClosureFixture = true := by native_decide
example : directIndexedTrapdoorCutoffFixture = true := by native_decide

/-- Inputs and outputs of one parallel loop share one exact lane coordinate.  A nested body or
a sibling loop remains distinct even when it reuses the same numeric index slot. -/
private def parallelLoopScopeOwnedIdentityFixture : Bool :=
  let root : ScopeTemplateKey := .root (.standalone 91)
  let sameInput := parallelLoopLaneSelection root 40 0 (.constant 3)
  let sameOutput := parallelLoopLaneSelection root 40 0 (.constant 3)
  let nested := parallelLoopLaneSelection (.parallelBody root 40) 40 0 (.constant 3)
  let sibling := parallelLoopLaneSelection root 41 0 (.constant 3)
  match parallelLoopLaneBinder root 40 0 (.constant 3),
      parallelLoopLaneBinder (.parallelBody root 40) 40 0 (.constant 3) with
  | .ok sameBinder, .ok nestedBinder =>
      sameInput.expression == sameOutput.expression && sameInput.expression != nested.expression &&
        sameInput.expression != sibling.expression && sameBinder != nestedBinder
  | _, _ => false

example : parallelLoopScopeOwnedIdentityFixture = true := by native_decide

/-- Actual scalar `ParallelLoop` lowering: an integer family is zipped, one integer is
broadcast, and the body adds them through `DirectScalarOperation`.  The output remains a direct
explicit family, proving that neither loop input nor scalar primitive re-enters the legacy scalar
selection arena. -/
private def directScalarParallelBody : Scope := {
  nodes := #[
    { kind := .input "zipped", arguments := [], outputTypes := [.integer] },
    { kind := .input "broadcast", arguments := [], outputTypes := [.integer] },
    { kind := .intBinary .add, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
      outputTypes := [.integer] }
  ]
  outputs := [("result", { node := 2, port := 0 })]
  inputNames := ["zipped", "broadcast"]
}

private def directScalarParallelProgram : Prog := {
  root := {
    nodes := #[
      { kind := .constantInt 1, arguments := [], outputTypes := [.integer] },
      { kind := .constantInt 2, arguments := [], outputTypes := [.integer] },
      { kind := .familyPack, arguments := [{ node := 0, port := 0 }, { node := 1, port := 0 }],
        outputTypes := [.indexedFamily .integer (.constant 2)] },
      { kind := .constantInt 5, arguments := [], outputTypes := [.integer] },
      { kind := .parallelLoop "scalar" (.constant 2) 0 [] [.zip, .broadcast],
        arguments := [{ node := 2, port := 0 }, { node := 3, port := 0 }],
        outputTypes := [.indexedFamily .integer (.constant 2)] },
      { kind := .familyGetStatic (.constant 1), arguments := [{ node := 4, port := 0 }],
        outputTypes := [.integer] }
    ]
    outputs := [("family", { node := 4, port := 0 }), ("selected", { node := 5, port := 0 })]
    inputNames := []
  }
  definitions := [("scalar", directScalarParallelBody)]
}

private def directScalarParallelDerivation : ProgramDerivation := {
  root := { steps := #[
    { sourceNode := 0, rule := .constantInt, arguments := [] },
    { sourceNode := 1, rule := .constantInt, arguments := [] },
    { sourceNode := 2, rule := .familyPack, arguments := [{ node := 0, port := 0 },
      { node := 1, port := 0 }] },
    { sourceNode := 3, rule := .constantInt, arguments := [] },
    { sourceNode := 4, rule := .parallelLoop, arguments := [{ node := 2, port := 0 },
      { node := 3, port := 0 }] },
    { sourceNode := 5, rule := .familyGetStatic, arguments := [{ node := 4, port := 0 }] }
  ] }
  definitions := [("scalar", { steps := #[
    { sourceNode := 0, rule := .input, arguments := [] },
    { sourceNode := 1, rule := .input, arguments := [] },
    { sourceNode := 2, rule := .intBinary, arguments := [{ node := 0, port := 0 },
      { node := 1, port := 0 }] }
  ] })]
}

private def directScalarParallelFixture : Except OperationalError Bool := do
  let facts ← evaluateProgramOperationalWithLayouts directScalarParallelProgram
    directScalarParallelDerivation [] []
  let family ← lookupFact 6 facts { node := 4, port := 0 }
  let selected ← integerFactAt 6 facts { node := 5, port := 0 }
  let entries ← facts.arena.reducedDirectScalarValueFactsAt [] family
  pure (match family.payload with
    | .directValue _ => entries.length == 2 && entries.map (fun entry =>
        match entry.fact with | .integer fact => (fact.lower, fact.upper) | _ => (0, -1)) ==
          [(6, 6), (7, 7)] && selected.lower == 7 && selected.upper == 7
    | _ => false)

example : directScalarParallelFixture = .ok true := by native_decide

/-- A complete direct endpoint table is reduced only after every physical lane has formed its
full sum.  The maximum is therefore twelve, not either input-table maximum. -/
private def directEndpointNoiseMaximumFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 6200 with count := .constant 2 }
    let (fixed, left0) := ({} : FixedOperationalPayloadArena).pushMatrix
      (boundedOperationalExprFixtureFact 6201 2)
    let (fixed, left1) := fixed.pushMatrix (boundedOperationalExprFixtureFact 6202 11)
    let (fixed, right0) := fixed.pushMatrix (boundedOperationalExprFixtureFact 6203 5)
    let (fixed, right1) := fixed.pushMatrix (boundedOperationalExprFixtureFact 6204 1)
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, leftRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[left0, left1] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6200)
    let (direct, rightRoot) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.matrix fixtureType) #[right0, right1] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6200)
    let arena : OperationalExprArena := { direct }
    let wrap root : OperationalFact := {
      context := { binders := #[binder] }, payload := .directValue root, storage := .explicitTable }
    let operation : PrimitiveOperation := {
      kind := .add false, outputType := fixtureType, ownerScope := none, ownerNode := 6205,
      outputPort := 0, parameterEnvironment := [] }
    let (arena, summed) ← arena.pushDirectMatrixPointwise operation (wrap leftRoot) (wrap rightRoot)
    let lanes ← arena.reducedDirectValueFactsAt [] summed
    let (maximum, _) ← operationalNoiseBoundForFact arena summed []
    pure (lanes.map (fun (entry : ReducedDirectMatrixFact) => entry.fact.evaluateNoiseHardBound []) ==
      [Except.ok 7, Except.ok 12] &&
      maximum == 12)) with
  | .ok value => value
  | .error _ => false

private def directScalarTable
    (node : Nat) (bounds : Array Nat) : Except OperationalError
    (OperationalExprArena × OperationalFact × IndexVariable) := do
  let binder := { directCarrierFixtureBinder node with count := .constant bounds.size }
  let mut fixed : FixedOperationalPayloadArena := {}
  let mut references : Array FixedOperationalPayloadRef := #[]
  for lane in [:bounds.size] do
    let (nextFixed, reference) := fixed.pushScalar
      (indexedScalarFixtureInteger (node + lane + 1) bounds[lane]! bounds[lane]!)
    fixed := nextFixed
    references := references.push reference
  let direct : DirectOperationalIndexedArena := { fixed }
  let (direct, root) ← match direct.pushExplicit [] { binders := #[binder] } binder
      (.scalar .integer) references with
    | some value => pure value
    | none => throw (.unsupportedOperationalExpr node)
  pure ({ direct }, {
    context := { binders := #[binder] }, payload := .directValue root, storage := .explicitTable }, binder)

/-- Direct scalar ZipOffset uses the same checked map and consumes the requested half-open range
without falling back to the legacy scalar selection graph. -/
private def directScalarZipOffsetFixture : Bool :=
  match (do
    let (arena, source, binder) ← directScalarTable 6210 #[3, 5, 7]
    let selector := { directCarrierFixtureBinder 6211 with count := .constant 2 }
    let map ← match dynamicIndexMap source.context binder
        (.offset (.variable selector) 1) with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6210)
    let (arena, mapped) ← arena.reindexDirectMatrixFact map source
    let root ← match mapped.payload with
      | .directValue root => pure root
      | _ => throw (OperationalError.unsupportedOperationalExpr 6210)
    let first ← arena.direct.scalarFactAt [] [(.variable selector, 0)] root
      (arena.direct.values.size + 1)
    let second ← arena.direct.scalarFactAt [] [(.variable selector, 1)] root
      (arena.direct.values.size + 1)
    pure (mapped.context.binders == #[selector] && (match first, second with
      | .integer first, .integer second => first.lower == 5 && second.lower == 7
      | _, _ => false))) with
  | .ok value => value
  | .error _ => false

/-- Nested direct scalar tables retain both physical coordinates and zip only matching composite
assignments. -/
private def nestedDirectScalarContextFixture : Bool :=
  match (do
    let (arena, child, inner) ← directScalarTable 6220 #[2, 4]
    let outer := { directCarrierFixtureBinder 6221 with count := .constant 2 }
    let childRoot ← match child.payload with
      | .directValue root => pure root
      | _ => throw (OperationalError.unsupportedOperationalExpr 6220)
    let (direct, root) ← match arena.direct.pushExplicitValues [] outer #[childRoot, childRoot] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6221)
    let arena : OperationalExprArena := { arena with direct }
    let nested : OperationalFact := {
      context := { binders := #[inner, outer] }, payload := .directValue root, storage := .explicitTable }
    let entries ← arena.reducedDirectScalarValueFactsAt [] nested
    pure (entries.map (fun (entry : ReducedDirectScalarFact) => entry.correlation) == [
      [(IndexExpr.variable outer, 0), (IndexExpr.variable inner, 0)],
      [(IndexExpr.variable outer, 0), (IndexExpr.variable inner, 1)],
      [(IndexExpr.variable outer, 1), (IndexExpr.variable inner, 0)],
      [(IndexExpr.variable outer, 1), (IndexExpr.variable inner, 1)]])) with
  | .ok value => value
  | .error _ => false

/-- Explicit and Shared direct scalar carriers zip once per explicit lane in either operand
order, never through a Cartesian product. -/
private def directMixedScalarPointwiseFixture : Bool :=
  match (do
    let (arena, explicit, binder) ← directScalarTable 6230 #[1, 2]
    let (fixed, reference) := arena.direct.fixed.pushScalar (indexedScalarFixtureInteger 6233 10 10)
    let direct := { arena.direct with fixed }
    let (direct, root) ← match direct.pushShared emptyContext (.scalar .integer) reference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6233)
    let arena : OperationalExprArena := { arena with direct }
    let sharedFact : OperationalFact := { context := emptyContext, payload := .directValue root, storage := .sharedTemplate }
    let operation : DirectScalarOperation := {
      kind := .intBinary .add, ownerScope := none, ownerNode := 6234, outputPort := 0 }
    let (arena, first) ← arena.pushDirectScalarPointwiseN operation #[explicit, sharedFact]
    let (arena, second) ← arena.pushDirectScalarPointwiseN operation #[sharedFact, explicit]
    let lowers (value : OperationalFact) : Except OperationalError (List Int) := do
      let entries ← arena.reducedDirectScalarValueFactsAt [] value
      pure (entries.map fun (entry : ReducedDirectScalarFact) =>
        match entry.fact with | .integer fact => fact.lower | _ => -1)
    pure (first.context.binders == #[binder] && second.context.binders == #[binder] &&
      (← lowers first) == [11, 12] && (← lowers second) == [11, 12])) with
  | .ok value => value
  | .error _ => false

/-- A direct Boolean family remains indexed through `BoolToInt`; static access specializes its
selected lane and retains the result's direct producer provenance. -/
private def directBooleanToIntStaticFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 6240 with count := .constant 2 }
    let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushScalar .boolean
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, root) ← match direct.pushShared { binders := #[binder] } (.scalar .boolean) reference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6240)
    let arena : OperationalExprArena := { direct }
    let family : OperationalFact := { context := { binders := #[binder] }, payload := .directValue root, storage := .sharedTemplate }
    let operation : DirectScalarOperation := {
      kind := .boolToInt, ownerScope := some temporaryScope, ownerNode := 6241, outputPort := 0 }
    let (arena, mapped) ← arena.pushDirectScalarPointwiseN operation #[family]
    let map ← match closedStaticIndexMap [] mapped.context binder 1 with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6241)
    let (arena, selected) ← arena.reindexDirectMatrixFact map mapped
    let root ← match selected.payload with
      | .directValue root => pure root
      | _ => throw (OperationalError.unsupportedOperationalExpr 6241)
    let fact ← arena.direct.scalarFactAt [] [] root (arena.direct.values.size + 1)
    pure (selected.context == emptyContext && (match fact with
      | .integer value => value.subject == ({ node := 6241, port := 0 } : WireRef) && value.lower == 0 &&
          value.upper == 1 && value.origin == (OperationalValueOrigin.local temporaryScope { node := 6241, port := 0 })
      | _ => false))) with
  | .ok value => value
  | .error _ => false

/-- Static direct scalar reindexing substitutes loop-index bound expressions in every fixed
lane rather than retaining the removed family coordinate. -/
private def directScalarLoopIndexStaticFixture : Bool :=
  match (do
    let binder := { directCarrierFixtureBinder 6250 with count := .constant 2 }
    let scalar : OperationalScalarFact := .integer {
      subject := { node := 6251, port := 0 }, origin := .local temporaryScope { node := 6251, port := 0 },
      lower := 0, upper := 1, lowerExpression := .closedInt (.loopIndex binder.slot),
      upperExpression := .closedInt (.loopIndex binder.slot) }
    let (fixed, first) := ({} : FixedOperationalPayloadArena).pushScalar scalar
    let (fixed, second) := fixed.pushScalar scalar
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, root) ← match direct.pushExplicit [] { binders := #[binder] } binder
        (.scalar .integer) #[first, second] with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6250)
    let arena : OperationalExprArena := { direct }
    let source : OperationalFact := { context := { binders := #[binder] }, payload := .directValue root, storage := .explicitTable }
    let map ← match closedStaticIndexMap [] source.context binder 1 with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6250)
    let (arena, selected) ← arena.reindexDirectMatrixFact map source
    let root ← match selected.payload with
      | .directValue root => pure root
      | _ => throw (OperationalError.unsupportedOperationalExpr 6250)
    match ← arena.direct.scalarFactAt [] [] root (arena.direct.values.size + 1) with
    | .integer fact => pure (fact.lowerExpression == OperationalBoundExpr.closedInt (.constant 1) &&
        fact.upperExpression == OperationalBoundExpr.closedInt (.constant 1))
    | _ => pure false) with
  | .ok value => value
  | .error _ => false

example : directEndpointNoiseMaximumFixture = true := by native_decide
example : directScalarZipOffsetFixture = true := by native_decide
example : nestedDirectScalarContextFixture = true := by native_decide
example : directMixedScalarPointwiseFixture = true := by native_decide
example : directBooleanToIntStaticFixture = true := by native_decide
example : directScalarLoopIndexStaticFixture = true := by native_decide

/-- Hash provenance follows a direct transport chain only when each map owns its immediate
source and destination context.  A manually forged mapped carrier cannot substitute a key
origin across an unrelated lane binder. -/
private def malformedDirectMapProvenanceFixture : Bool :=
  match (do
    let bytes : OperationalScalarFact := .bytes {
      subject := { node := 6100, port := 0 }
      origin := .local temporaryScope { node := 6100, port := 0 }
      length := 8
    }
    let (fixed, reference) := ({} : FixedOperationalPayloadArena).pushScalar bytes
    let direct : DirectOperationalIndexedArena := { fixed }
    let (direct, source) ← match direct.pushShared emptyContext (.scalar (.bytes 8)) reference with
      | some value => pure value
      | none => throw (OperationalError.unsupportedOperationalExpr 6100)
    let binder := directCarrierFixtureBinder 6101
    let forgedMap : IndexMap := {
      source := emptyContext
      destination := emptyContext
      assignments := #[]
    }
    let (direct, mapped) := direct.pushValue { binders := #[binder] }
      (.mapped (.scalar (.bytes 8)) source forgedMap)
    let facts : OperationalScopeFacts := {
      values := #[#[{
        context := { binders := #[binder] }
        payload := .directValue mapped
        storage := .mappedTemplate
      }]]
      arena := { direct }
    }
    valueOriginAt temporaryScope 6102 facts { node := 0, port := 0}) with
  | .error (.unsupportedOperationalExpr 1) => true
  | _ => false

example : malformedDirectMapProvenanceFixture = true := by native_decide

/-! Reuse one pre-existing native fixture gate for the computationally heavy operational
fixtures.  This keeps the trusted-evaluation surface unchanged while checking the production
functions rather than duplicating their behavior in proof-only reference code. -/
example : exactRelationSelectionFixtureResult = .ok true ∧
    directOrdinaryMatrixPipelineFixture = true ∧
    directValueContextCorrelationFixture = true ∧
    directValueScalarContextFixture = true ∧
    directValueScalarKernelFixture = true ∧
    canonicalRangeTransformFixture = true ∧
    directFamilySelectFixture = .ok true ∧
    outOfRangeDirectFamilyGetFixture = true ∧
    symbolicFamilySelectFixture = .ok true ∧
    directLoopInputFixture = .ok true ∧
    symbolicCountDirectZipFixture = .ok true ∧
    familyPackPreservesDomainFixtureResult = .ok true ∧
    constantPolynomialProductCanonicalRangeFixture = true ∧
    equivalentProductDimensionFixture = true ∧
    oneBadEndpointIdentityRejectsFixture = true ∧
    crossSelectionRelationMismatchFixture = true ∧
    summaryTransferRegistryCoverageFixture = true ∧
    nestedDirectPackContextFixture = true ∧
    reducedSharedLogicalCountsFixture = true ∧
    reducedExplicitTableFixture = true ∧
    reducedPointwiseCorrelationFixture = true ∧
    reducedMappedDirectFixture = true := by
  native_decide



end Mxx.Certificate
