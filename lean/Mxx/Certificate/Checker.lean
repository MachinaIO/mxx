import Mxx.Certificate.Analyzer

namespace Mxx.Certificate

/-- A concrete static check failed. The obligation is retained for a precise diagnostic. -/
inductive CheckError where
  | unsatisfied (obligation : StaticObligation)
  | symbolicRecurrence (error : ResolveSymbolicRecurrencesError)

/-- Evidence returned by phase B. It contains no caller-provided facts: success means that the
checker evaluated every obligation derived by phase A. -/
structure CheckedStaticObligations where
  checked : List StaticObligation
  symbolicRecurrenceStates : CheckedSymbolicRecurrenceStateTable := {}

/-- Exact meaning of each parameter-only obligation. Runtime values never occur here. -/
def StaticObligation.Holds
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceStates : CheckedSymbolicRecurrenceStateTable) : StaticObligation → Prop
  | .positiveModulus value =>
      match evaluateIntExpr environment value with
      | .ok evaluated => 0 < evaluated
      | .error _ => False
  | .positiveDivisor value => 0 < value
  | .loopFamilyAccessInRange loopCount offset familyCount =>
      match evaluateIntExpr environment loopCount, evaluateIntExpr environment familyCount with
      | .ok loopValue, .ok familyValue =>
          0 ≤ loopValue ∧ 0 ≤ familyValue ∧
            (loopValue ≠ 0 → loopValue - 1 + offset < familyValue)
      | _, _ => False
  | .dynamicFamilyIndexInRange _ lower upper familyCount =>
      match lower.evaluateWithSymbolicRecurrences environment recurrenceStates,
          upper.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment familyCount with
      | .ok lowerValue, .ok upperValue, .ok familyValue =>
          0 ≤ lowerValue ∧ lowerValue ≤ upperValue ∧ upperValue < familyValue
      | _, _, _ => False
  | .matchingMatrixTypes left right =>
      match evaluateIntExpr environment left.modulus,
          evaluateIntExpr environment left.ringDimension,
          evaluateIntExpr environment left.rows,
          evaluateIntExpr environment left.columns,
          evaluateIntExpr environment right.modulus,
          evaluateIntExpr environment right.ringDimension,
          evaluateIntExpr environment right.rows,
          evaluateIntExpr environment right.columns with
      | .ok leftModulus, .ok leftRingDimension, .ok leftRows, .ok leftColumns,
          .ok rightModulus, .ok rightRingDimension, .ok rightRows, .ok rightColumns =>
          leftModulus = rightModulus ∧ leftRingDimension = rightRingDimension ∧
            leftRows = rightRows ∧ leftColumns = rightColumns
      | _, _, _, _, _, _, _, _ => False
  | .intBoundNonnegative value =>
      match value.evaluateWithSymbolicRecurrences environment recurrenceStates with
      | .ok evaluated => 0 ≤ evaluated
      | .error _ => False
  | .intBoundPositive value =>
      match value.evaluateWithSymbolicRecurrences environment recurrenceStates with
      | .ok evaluated => 0 < evaluated
      | .error _ => False
  | .intBoundsOrdered lower upper =>
      match lower.evaluateWithSymbolicRecurrences environment recurrenceStates,
          upper.evaluateWithSymbolicRecurrences environment recurrenceStates with
      | .ok lowerValue, .ok upperValue => lowerValue ≤ upperValue
      | _, _ => False
  | .thresholdNoise noise ciphertextModulus plaintextModulus =>
      match noise.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment ciphertextModulus,
          evaluateIntExpr environment plaintextModulus with
      | .ok noiseValue, .ok ciphertextValue, .ok plaintextValue =>
          0 < ciphertextValue ∧ 0 < plaintextValue ∧
            2 * plaintextValue.toNat * noiseValue < ciphertextValue.toNat
      | _, _, _ => False
  | .diamondFalseInterval noise ciphertextModulus =>
      match noise.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          4 ≤ ciphertextValue ∧
            (noiseValue : Int) < quarter ∧
            3 * quarter + noiseValue < ciphertextValue
      | _, _ => False
  | .diamondTrueInterval noise ciphertextModulus =>
      match noise.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          4 ≤ ciphertextValue ∧
            quarter + noiseValue ≤ ciphertextValue / 2 ∧
            ciphertextValue / 2 + noiseValue ≤ 3 * quarter
      | _, _ => False

private def checkOne
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceStates : CheckedSymbolicRecurrenceStateTable)
    (obligation : StaticObligation) : Except CheckError Unit :=
  match obligation with
  | .positiveModulus expression =>
      match evaluateIntExpr environment expression with
      | .ok value => if 0 < value then .ok () else .error (.unsatisfied obligation)
      | .error _ => .error (.unsatisfied obligation)
  | .positiveDivisor value =>
      if 0 < value then .ok () else .error (.unsatisfied obligation)
  | .loopFamilyAccessInRange loopCount offset familyCount =>
      match evaluateIntExpr environment loopCount, evaluateIntExpr environment familyCount with
      | .ok loopValue, .ok familyValue =>
          if 0 ≤ loopValue ∧ 0 ≤ familyValue ∧
              (loopValue = 0 ∨ loopValue - 1 + offset < familyValue) then .ok ()
          else .error (.unsatisfied obligation)
      | _, _ => .error (.unsatisfied obligation)
  | .dynamicFamilyIndexInRange _ lower upper familyCount =>
      match lower.evaluateWithSymbolicRecurrences environment recurrenceStates,
          upper.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment familyCount with
      | .ok lowerValue, .ok upperValue, .ok familyValue =>
          if 0 ≤ lowerValue ∧ lowerValue ≤ upperValue ∧ upperValue < familyValue then .ok ()
          else .error (.unsatisfied obligation)
      | _, _, _ => .error (.unsatisfied obligation)
  | .matchingMatrixTypes left right =>
      match evaluateIntExpr environment left.modulus,
          evaluateIntExpr environment left.ringDimension,
          evaluateIntExpr environment left.rows,
          evaluateIntExpr environment left.columns,
          evaluateIntExpr environment right.modulus,
          evaluateIntExpr environment right.ringDimension,
          evaluateIntExpr environment right.rows,
          evaluateIntExpr environment right.columns with
      | .ok leftModulus, .ok leftRingDimension, .ok leftRows, .ok leftColumns,
          .ok rightModulus, .ok rightRingDimension, .ok rightRows, .ok rightColumns =>
          if leftModulus = rightModulus ∧ leftRingDimension = rightRingDimension ∧
              leftRows = rightRows ∧ leftColumns = rightColumns then
            .ok ()
          else .error (.unsatisfied obligation)
      | _, _, _, _, _, _, _, _ => .error (.unsatisfied obligation)
  | .intBoundNonnegative value =>
      match value.evaluateWithSymbolicRecurrences environment recurrenceStates with
      | .ok evaluated => if 0 ≤ evaluated then .ok () else .error (.unsatisfied obligation)
      | .error _ => .error (.unsatisfied obligation)
  | .intBoundPositive value =>
      match value.evaluateWithSymbolicRecurrences environment recurrenceStates with
      | .ok evaluated => if 0 < evaluated then .ok () else .error (.unsatisfied obligation)
      | .error _ => .error (.unsatisfied obligation)
  | .intBoundsOrdered lower upper =>
      match lower.evaluateWithSymbolicRecurrences environment recurrenceStates,
          upper.evaluateWithSymbolicRecurrences environment recurrenceStates with
      | .ok lowerValue, .ok upperValue =>
          if lowerValue ≤ upperValue then .ok () else .error (.unsatisfied obligation)
      | _, _ => .error (.unsatisfied obligation)
  | .thresholdNoise noise ciphertextModulus plaintextModulus =>
      match noise.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment ciphertextModulus,
          evaluateIntExpr environment plaintextModulus with
      | .ok noiseValue, .ok ciphertextValue, .ok plaintextValue =>
          if 0 < ciphertextValue ∧ 0 < plaintextValue ∧
              2 * plaintextValue.toNat * noiseValue < ciphertextValue.toNat then
            .ok ()
          else .error (.unsatisfied obligation)
      | _, _, _ => .error (.unsatisfied obligation)
  | .diamondFalseInterval noise ciphertextModulus =>
      match noise.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          if 4 ≤ ciphertextValue ∧
              (noiseValue : Int) < quarter ∧
              3 * quarter + noiseValue < ciphertextValue then .ok ()
          else .error (.unsatisfied obligation)
      | _, _ => .error (.unsatisfied obligation)
  | .diamondTrueInterval noise ciphertextModulus =>
      match noise.evaluateWithSymbolicRecurrences environment recurrenceStates,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          if 4 ≤ ciphertextValue ∧
              quarter + noiseValue ≤ ciphertextValue / 2 ∧
              ciphertextValue / 2 + noiseValue ≤ 3 * quarter then .ok ()
          else .error (.unsatisfied obligation)
      | _, _ => .error (.unsatisfied obligation)

private theorem checkOne_sound
    {environment : Mxx.Ir.ParamEnvironment}
    {recurrenceStates : CheckedSymbolicRecurrenceStateTable}
    {obligation : StaticObligation}
    (accepted : checkOne environment recurrenceStates obligation = .ok ()) :
    obligation.Holds environment recurrenceStates := by
  cases obligation <;>
    simp only [checkOne, StaticObligation.Holds] at accepted ⊢ <;>
    split at accepted <;> simp_all

private def checkAll
    (environment : Mxx.Ir.ParamEnvironment) :
    (recurrenceStates : CheckedSymbolicRecurrenceStateTable) →
    List StaticObligation → Except CheckError Unit
  | _, [] => .ok ()
  | recurrenceStates, obligation :: tail =>
      match checkOne environment recurrenceStates obligation with
      | .error error => .error error
      | .ok () => checkAll environment recurrenceStates tail

private theorem checkAll_sound
    {environment : Mxx.Ir.ParamEnvironment}
    {recurrenceStates : CheckedSymbolicRecurrenceStateTable}
    {obligations : List StaticObligation}
    (accepted : checkAll environment recurrenceStates obligations = .ok ()) :
    ∀ obligation ∈ obligations, obligation.Holds environment recurrenceStates := by
  induction obligations with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases headAccepted : checkOne environment recurrenceStates head with
      | error error => simp [checkAll, headAccepted] at accepted
      | ok value =>
        cases value
        have tailAccepted : checkAll environment recurrenceStates tail = .ok () := by
          simpa [checkAll, headAccepted] using accepted
        intro obligation member
        simp only [List.mem_cons] at member
        rcases member with rfl | member
        · exact checkOne_sound headAccepted
        · exact inductionHypothesis tailAccepted obligation member

/-- Phase B of the verifier. All arithmetic is evaluated by Lean over `Int`/`Nat`; no floating
point conversion and no Rust-side mirror participates in acceptance. -/
def checkStaticParameters
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment) :
    Except CheckError CheckedStaticObligations :=
  match resolveSymbolicRecurrences environment analysis.symbolicRecurrences with
  | .error error => .error (.symbolicRecurrence error)
  | .ok symbolicRecurrenceStates =>
      match checkAll environment symbolicRecurrenceStates analysis.staticObligations with
      | .error error => .error error
      | .ok () => .ok {
          checked := analysis.staticObligations
          symbolicRecurrenceStates
        }

theorem checkStaticParameters_sound
    {analysis : AnalysisResult}
    {environment : Mxx.Ir.ParamEnvironment}
    {checked : CheckedStaticObligations}
    (accepted : checkStaticParameters analysis environment = .ok checked) :
    ∀ obligation ∈ analysis.staticObligations,
      obligation.Holds environment checked.symbolicRecurrenceStates := by
  cases symbolicResolved :
      resolveSymbolicRecurrences environment analysis.symbolicRecurrences with
  | error error => simp [checkStaticParameters, symbolicResolved] at accepted
  | ok symbolicRecurrenceStates =>
      cases checkedAll : checkAll environment symbolicRecurrenceStates analysis.staticObligations with
      | error error =>
          simp [checkStaticParameters, symbolicResolved, checkedAll] at accepted
      | ok value =>
          cases value
          simp only [checkStaticParameters, symbolicResolved, checkedAll, Except.ok.injEq]
            at accepted
          subst checked
          exact checkAll_sound checkedAll

private def integerRecurrenceCheckerFixture : SequentialRecurrenceSource := {
  loop := { site := { stage := ⟨"checker"⟩, scope := ⟨[]⟩, node := ⟨0⟩ } }
  count := .constant 0
  carriedArity := 1
  initial := ⟨#[{
    fact := .integer {
      expression := .intConstant 3
      lower := .integer (.constant 3)
      upper := .integer (.constant 3)
    }
    schema := .integer
  }], rfl⟩
  bodyInputs := ⟨#[{
    definition := { stage := ⟨"checker"⟩, name := "body" }
    bodyScope := ⟨[]⟩
    node := ⟨0⟩
    port := 0
  }], rfl⟩
  bodyOutputs := ⟨#[{
    fact := .integer {
      expression := .carriedInput (.integerValue 0)
      lower := .carriedInput (.lower 0)
      upper := .carriedInput (.upper 0)
    }
    schema := .integer
  }], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩
}

private def integerRecurrenceCheckerRef : SequentialRecurrenceInstanceRef := {
  recurrence := ⟨integerRecurrenceCheckerFixture.loop.site⟩
  path := []
}

private def symbolicCheckerIntegerPath :
    CarriedBoundStatePath [.integerInterval] .integerInterval :=
  .head .here

private def symbolicCheckerZeroCountTransfer : SymbolicRecurrenceTransfer where
  identity := integerRecurrenceCheckerRef
  source := integerRecurrenceCheckerFixture
  sourceIdentity := rfl
  carriedSchemas := [.integer]
  initialBounds := .cons
    (.integer (.integer (.constant 3)) (.integer (.constant 3))) .nil
  initialOutputs := .cons (.integer ⟨0⟩) .nil
  bodyOutputs := .cons (.integer ⟨0⟩) .nil
  boundTransition := .cons
    (.integer
      (.previousState symbolicCheckerIntegerPath .lower)
      (.previousState symbolicCheckerIntegerPath .upper))
    .nil
  schemaValidation := rfl

private def symbolicCheckerZeroCountAnalysis : AnalysisResult where
  facts := []
  families := []
  symbolicRecurrences := [symbolicCheckerZeroCountTransfer]
  staticObligations := []
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

private def symbolicMatrixState : CheckedSymbolicRecurrenceState where
  identity := integerRecurrenceCheckerRef
  schemas := [.matrixSummary]
  values := .cons (.matrix true 7 11 13) .nil

private def symbolicMatrixStateTable : CheckedSymbolicRecurrenceStateTable := {
  entries := [symbolicMatrixState]
}

example :
    (BoundExpr.recurrenceResult integerRecurrenceCheckerRef
      (.affineCoefficientBound 0 9)).evaluateWithSymbolicRecurrences []
        symbolicMatrixStateTable = .ok 7 := by
  rfl

private def symbolicMatrixLaneReference : SequentialRecurrenceInstanceRef :=
  integerRecurrenceCheckerRef.appendPath [
    .parallelLane { integerRecurrenceCheckerRef.recurrence.site with node := ⟨8⟩ } ⟨3⟩,
    .parallelLane { integerRecurrenceCheckerRef.recurrence.site with node := ⟨9⟩ } ⟨4⟩
  ]

/-- Phase B uses the same lane-uniform identity matcher as recurrence construction. -/
example :
    (BoundExpr.recurrenceResult symbolicMatrixLaneReference
      (.matrixTotalBound 0)).evaluateWithSymbolicRecurrences [] symbolicMatrixStateTable =
      .ok 13 := by
  rfl

/-- A sequential occurrence is never canonicalized to the lane-uniform base state. -/
example :
    let sequentialReference := integerRecurrenceCheckerRef.appendPath [
      .sequentialIteration { integerRecurrenceCheckerRef.recurrence.site with node := ⟨8⟩ } ⟨3⟩
    ]
    (BoundExpr.recurrenceResult sequentialReference
      (.matrixTotalBound 0)).evaluateWithSymbolicRecurrences [] symbolicMatrixStateTable =
      .error (.unresolvedRecurrence sequentialReference (.matrixTotalBound 0)) := by
  rfl

/-- A path whose carried slot is outside the dependent schema fails closed. -/
example :
    (BoundExpr.recurrenceResult integerRecurrenceCheckerRef
      (.matrixTotalBound 1)).evaluateWithSymbolicRecurrences [] symbolicMatrixStateTable =
      .error (.unresolvedRecurrence integerRecurrenceCheckerRef (.matrixTotalBound 1)) := by
  rfl

/-- An integer path cannot read a matrix state. -/
example :
    (IntBoundExpr.recurrenceResult integerRecurrenceCheckerRef
      (.lower 0)).evaluateWithSymbolicRecurrences [] symbolicMatrixStateTable =
      .error (.unresolvedRecurrence integerRecurrenceCheckerRef (.lower 0)) := by
  rfl

/-- Even a manually duplicated checked identity is rejected rather than choosing one state. -/
example :
    (BoundExpr.recurrenceResult integerRecurrenceCheckerRef
      (.matrixTotalBound 0)).evaluateWithSymbolicRecurrences [] {
        entries := [symbolicMatrixState, symbolicMatrixState]
      } =
      .error (.unresolvedRecurrence integerRecurrenceCheckerRef (.matrixTotalBound 0)) := by
  rfl

private def symbolicFamilyMatrixStateTable : CheckedSymbolicRecurrenceStateTable := {
  entries := [{
    identity := integerRecurrenceCheckerRef
    schemas := [.family (.constant 2) .matrixSummary]
    values := .cons (.familyEnvelope (.matrix false 5 8 13)) .nil
  }]
}

/-- A family path is accepted only by recursively matching the family-envelope schema. -/
example :
    (BoundExpr.recurrenceResult integerRecurrenceCheckerRef
      (.familyElement 0 ⟨0⟩ (.matrixTotalBound 0))).evaluateWithSymbolicRecurrences []
        symbolicFamilyMatrixStateTable = .ok 13 := by
  rfl

/-- Phase B consumes the schema-indexed symbolic recurrence list even when the legacy list is
empty. A zero-count loop returns its initial state unchanged. -/
example : checkStaticParameters symbolicCheckerZeroCountAnalysis [] = .ok {
    checked := []
    symbolicRecurrenceStates := { entries := [{
      identity := integerRecurrenceCheckerRef
      schemas := [.integerInterval]
      values := .cons (.integer 3 3) .nil
    }] }
  } := by
  rfl

private def integerRecurrenceCheckerAnalysis : AnalysisResult where
  facts := []
  families := []
  symbolicRecurrences := [symbolicCheckerZeroCountTransfer]
  staticObligations := [
    .intBoundNonnegative (.recurrenceResult
      integerRecurrenceCheckerRef (.lower 0)),
    .intBoundPositive (.recurrenceResult
      integerRecurrenceCheckerRef (.lower 0)),
    .intBoundsOrdered
      (.recurrenceResult integerRecurrenceCheckerRef (.lower 0))
      (.recurrenceResult integerRecurrenceCheckerRef (.upper 0))
  ]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

/-- Phase B checks signed endpoint obligations against the schema-indexed recurrence state. -/
example : (checkStaticParameters integerRecurrenceCheckerAnalysis []).isOk = true := rfl

private def invalidIntBoundAnalysis : AnalysisResult where
  facts := []
  families := []
  staticObligations := [.intBoundPositive (.integer (.constant 0))]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

example : checkStaticParameters invalidIntBoundAnalysis [] =
    .error (.unsatisfied (.intBoundPositive (.integer (.constant 0)))) := rfl

private def thresholdTestAnalysis (noise : Nat) : AnalysisResult where
  facts := []
  families := []
  staticObligations := [
    .thresholdNoise (.constant noise) (.constant 256) (.constant 2)
  ]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

/-- Closed Phase-B regression tests for the only dynamic-family range rule. These use the real
checker, so a successful reduction verifies the same arithmetic used by recurrence acceptance. -/
private def loopFamilyRangeTestAnalysis
    (loopCount : Int) (offset : Nat) (familyCount : Int) : AnalysisResult where
  facts := []
  families := []
  staticObligations := [
    .loopFamilyAccessInRange (.constant loopCount) offset (.constant familyCount)
  ]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

example : (checkStaticParameters (loopFamilyRangeTestAnalysis 3 0 3) []).isOk = true := rfl
example : (checkStaticParameters (loopFamilyRangeTestAnalysis 3 1 4) []).isOk = true := rfl
example : (checkStaticParameters (loopFamilyRangeTestAnalysis 0 99 0) []).isOk = true := rfl
example : (checkStaticParameters (loopFamilyRangeTestAnalysis 3 1 3) []).isOk = false := rfl
example : (checkStaticParameters (loopFamilyRangeTestAnalysis (-1) 0 3) []).isOk = false := rfl

private def dynamicFamilyRangeTestSite : CoreNodeRef := {
  stage := ⟨"checker"⟩
  scope := ⟨[]⟩
  node := ⟨0⟩
}

private def dynamicFamilyRangeTestAnalysis
    (lower upper familyCount : Int) : AnalysisResult where
  facts := []
  families := []
  staticObligations := [.dynamicFamilyIndexInRange dynamicFamilyRangeTestSite
    (.integer (.constant lower)) (.integer (.constant upper)) (.constant familyCount)]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

example : (checkStaticParameters (dynamicFamilyRangeTestAnalysis 0 2 3) []).isOk = true := rfl
example : (checkStaticParameters (dynamicFamilyRangeTestAnalysis (-1) 2 3) []).isOk = false := rfl
example : (checkStaticParameters (dynamicFamilyRangeTestAnalysis 0 3 3) []).isOk = false := rfl
example : (checkStaticParameters (dynamicFamilyRangeTestAnalysis 2 1 3) []).isOk = false := rfl

example :
    checkStaticParameters (thresholdTestAnalysis 63) [] =
      .ok { checked := [
        .thresholdNoise (.constant 63) (.constant 256) (.constant 2)
      ] } := rfl

example :
    checkStaticParameters (thresholdTestAnalysis 64) [] =
      .error (.unsatisfied
        (.thresholdNoise (.constant 64) (.constant 256) (.constant 2))) := rfl

private def diamondIntervalTestAnalysis (noise : Nat) : AnalysisResult where
  facts := []
  families := []
  staticObligations := [
    .diamondFalseInterval (.constant noise) (.constant 256),
    .diamondTrueInterval (.constant noise) (.constant 256)
  ]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

example :
    (checkStaticParameters (diamondIntervalTestAnalysis 63) []).toOption.isSome = true := rfl

example :
    checkStaticParameters (diamondIntervalTestAnalysis 64) [] =
      .error (.unsatisfied
        (.diamondFalseInterval (.constant 64) (.constant 256))) := rfl

private def oneDiamondIntervalTestAnalysis
    (trueCarrier : Bool) (noise modulus : Nat) : AnalysisResult where
  facts := []
  families := []
  staticObligations := [if trueCarrier then
    .diamondTrueInterval (.constant noise) (.constant modulus)
  else
    .diamondFalseInterval (.constant noise) (.constant modulus)]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

example : (checkStaticParameters (oneDiamondIntervalTestAnalysis false 3 17) []).isOk = true := rfl
example : (checkStaticParameters (oneDiamondIntervalTestAnalysis true 3 17) []).isOk = true := rfl
example : (checkStaticParameters (oneDiamondIntervalTestAnalysis false 3 16) []).isOk = true := rfl
example : (checkStaticParameters (oneDiamondIntervalTestAnalysis true 4 16) []).isOk = true := rfl

/-- Regression: the previous true-carrier condition accepted `(q,B)=(6,1)`, although the runtime
decoder rejects the upper endpoint `q/2+B=4` because its inclusive upper threshold is `3`. -/
example : checkStaticParameters (oneDiamondIntervalTestAnalysis true 1 6) [] =
    .error (.unsatisfied (.diamondTrueInterval (.constant 1) (.constant 6))) := rfl

example : checkStaticParameters (oneDiamondIntervalTestAnalysis false 4 17) [] =
    .error (.unsatisfied (.diamondFalseInterval (.constant 4) (.constant 17))) := rfl

end Mxx.Certificate
