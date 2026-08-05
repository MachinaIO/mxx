import Mxx.Certificate.Analyzer

namespace Mxx.Certificate

/-- A concrete static check failed. The obligation is retained for a precise diagnostic. -/
inductive CheckError where
  | unsatisfied (obligation : StaticObligation)
  | recurrence (error : ResolveRecurrenceBoundsError)

/-- Evidence returned by phase B. It contains no caller-provided facts: success means that the
checker evaluated every obligation derived by phase A. -/
structure CheckedStaticObligations where
  checked : List StaticObligation
  recurrenceBounds : CheckedRecurrenceBoundTable := {}

/-- Exact meaning of each parameter-only obligation. Runtime values never occur here. -/
def StaticObligation.Holds
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceBounds : CheckedRecurrenceBoundTable) : StaticObligation → Prop
  | .positiveModulus value =>
      match evaluateIntExpr environment value with
      | .ok evaluated => 0 < evaluated
      | .error _ => False
  | .positiveDivisor value => 0 < value
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
      match value.evaluate environment recurrenceBounds with
      | .ok evaluated => 0 ≤ evaluated
      | .error _ => False
  | .intBoundPositive value =>
      match value.evaluate environment recurrenceBounds with
      | .ok evaluated => 0 < evaluated
      | .error _ => False
  | .intBoundsOrdered lower upper =>
      match lower.evaluate environment recurrenceBounds, upper.evaluate environment recurrenceBounds with
      | .ok lowerValue, .ok upperValue => lowerValue ≤ upperValue
      | _, _ => False
  | .thresholdNoise noise ciphertextModulus plaintextModulus =>
      match noise.evaluateWithRecurrences environment recurrenceBounds,
          evaluateIntExpr environment ciphertextModulus,
          evaluateIntExpr environment plaintextModulus with
      | .ok noiseValue, .ok ciphertextValue, .ok plaintextValue =>
          0 < ciphertextValue ∧ 0 < plaintextValue ∧
            2 * plaintextValue.toNat * noiseValue < ciphertextValue.toNat
      | _, _, _ => False
  | .diamondFalseInterval noise ciphertextModulus =>
      match noise.evaluateWithRecurrences environment recurrenceBounds,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          4 ≤ ciphertextValue ∧
            (noiseValue : Int) < quarter ∧
            3 * quarter + noiseValue < ciphertextValue
      | _, _ => False
  | .diamondTrueInterval noise ciphertextModulus =>
      match noise.evaluateWithRecurrences environment recurrenceBounds,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          4 ≤ ciphertextValue ∧
            quarter + noiseValue ≤ ciphertextValue / 2 ∧
            ciphertextValue / 2 + noiseValue ≤ 3 * quarter
      | _, _ => False

private def checkOne
    (environment : Mxx.Ir.ParamEnvironment)
    (recurrenceBounds : CheckedRecurrenceBoundTable)
    (obligation : StaticObligation) : Except CheckError Unit :=
  match obligation with
  | .positiveModulus expression =>
      match evaluateIntExpr environment expression with
      | .ok value => if 0 < value then .ok () else .error (.unsatisfied obligation)
      | .error _ => .error (.unsatisfied obligation)
  | .positiveDivisor value =>
      if 0 < value then .ok () else .error (.unsatisfied obligation)
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
      match value.evaluate environment recurrenceBounds with
      | .ok evaluated => if 0 ≤ evaluated then .ok () else .error (.unsatisfied obligation)
      | .error _ => .error (.unsatisfied obligation)
  | .intBoundPositive value =>
      match value.evaluate environment recurrenceBounds with
      | .ok evaluated => if 0 < evaluated then .ok () else .error (.unsatisfied obligation)
      | .error _ => .error (.unsatisfied obligation)
  | .intBoundsOrdered lower upper =>
      match lower.evaluate environment recurrenceBounds, upper.evaluate environment recurrenceBounds with
      | .ok lowerValue, .ok upperValue =>
          if lowerValue ≤ upperValue then .ok () else .error (.unsatisfied obligation)
      | _, _ => .error (.unsatisfied obligation)
  | .thresholdNoise noise ciphertextModulus plaintextModulus =>
      match noise.evaluateWithRecurrences environment recurrenceBounds,
          evaluateIntExpr environment ciphertextModulus,
          evaluateIntExpr environment plaintextModulus with
      | .ok noiseValue, .ok ciphertextValue, .ok plaintextValue =>
          if 0 < ciphertextValue ∧ 0 < plaintextValue ∧
              2 * plaintextValue.toNat * noiseValue < ciphertextValue.toNat then
            .ok ()
          else .error (.unsatisfied obligation)
      | _, _, _ => .error (.unsatisfied obligation)
  | .diamondFalseInterval noise ciphertextModulus =>
      match noise.evaluateWithRecurrences environment recurrenceBounds,
          evaluateIntExpr environment ciphertextModulus with
      | .ok noiseValue, .ok ciphertextValue =>
          let quarter := Mxx.Ir.roundDiv (ciphertextValue - 2) 4
          if 4 ≤ ciphertextValue ∧
              (noiseValue : Int) < quarter ∧
              3 * quarter + noiseValue < ciphertextValue then .ok ()
          else .error (.unsatisfied obligation)
      | _, _ => .error (.unsatisfied obligation)
  | .diamondTrueInterval noise ciphertextModulus =>
      match noise.evaluateWithRecurrences environment recurrenceBounds,
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
    {recurrenceBounds : CheckedRecurrenceBoundTable}
    {obligation : StaticObligation}
    (accepted : checkOne environment recurrenceBounds obligation = .ok ()) :
    obligation.Holds environment recurrenceBounds := by
  cases obligation <;>
    simp only [checkOne, StaticObligation.Holds] at accepted ⊢ <;>
    split at accepted <;> simp_all

private def checkAll
    (environment : Mxx.Ir.ParamEnvironment) :
    (recurrenceBounds : CheckedRecurrenceBoundTable) →
    List StaticObligation → Except CheckError Unit
  | _, [] => .ok ()
  | recurrenceBounds, obligation :: tail =>
      match checkOne environment recurrenceBounds obligation with
      | .error error => .error error
      | .ok () => checkAll environment recurrenceBounds tail

private theorem checkAll_sound
    {environment : Mxx.Ir.ParamEnvironment}
    {recurrenceBounds : CheckedRecurrenceBoundTable}
    {obligations : List StaticObligation}
    (accepted : checkAll environment recurrenceBounds obligations = .ok ()) :
    ∀ obligation ∈ obligations, obligation.Holds environment recurrenceBounds := by
  induction obligations with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases headAccepted : checkOne environment recurrenceBounds head with
      | error error => simp [checkAll, headAccepted] at accepted
      | ok value =>
        cases value
        have tailAccepted : checkAll environment recurrenceBounds tail = .ok () := by
          simpa [checkAll, headAccepted] using accepted
        intro obligation member
        simp only [List.mem_cons] at member
        rcases member with rfl | member
        · exact checkOne_sound headAccepted
        · exact inductionHypothesis tailAccepted obligation member

private def resolveAllRecurrenceBounds
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment) :
    Except CheckError CheckedRecurrenceBoundTable := do
  let mut resolved : CheckedRecurrenceBoundTable := {}
  for (reference, _) in analysis.recurrences do
    resolved ← resolveRecurrenceBounds analysis environment resolved reference
      |>.mapError .recurrence
  return resolved

/-- Phase B of the verifier. All arithmetic is evaluated by Lean over `Int`/`Nat`; no floating
point conversion and no Rust-side mirror participates in acceptance. -/
def checkStaticParameters
    (analysis : AnalysisResult)
    (environment : Mxx.Ir.ParamEnvironment) :
    Except CheckError CheckedStaticObligations :=
  match resolveAllRecurrenceBounds analysis environment with
  | .error error => .error error
  | .ok recurrenceBounds =>
      match checkAll environment recurrenceBounds analysis.staticObligations with
      | .error error => .error error
      | .ok () => .ok { checked := analysis.staticObligations, recurrenceBounds }

theorem checkStaticParameters_sound
    {analysis : AnalysisResult}
    {environment : Mxx.Ir.ParamEnvironment}
    {checked : CheckedStaticObligations}
    (accepted : checkStaticParameters analysis environment = .ok checked) :
    ∀ obligation ∈ analysis.staticObligations,
      obligation.Holds environment checked.recurrenceBounds := by
  cases recurrenceResolved : resolveAllRecurrenceBounds analysis environment with
  | error error => simp [checkStaticParameters, recurrenceResolved] at accepted
  | ok recurrenceBounds =>
      cases checkedAll : checkAll environment recurrenceBounds analysis.staticObligations with
      | error error => simp [checkStaticParameters, recurrenceResolved, checkedAll] at accepted
      | ok value =>
          cases value
          simp only [checkStaticParameters, recurrenceResolved, checkedAll, Except.ok.injEq]
            at accepted
          subst checked
          exact checkAll_sound checkedAll

private def integerRecurrenceCheckerFixture : FactRecurrence := {
  loop := { site := { stage := ⟨"checker"⟩, scope := ⟨[]⟩, node := ⟨0⟩ } }
  count := .constant 0
  carriedArity := 1
  initial := ⟨#[.integer {
    expression := .intConstant 3
    lower := .integer (.constant 3)
    upper := .integer (.constant 3)
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

private def integerRecurrenceCheckerAnalysis : AnalysisResult where
  facts := []
  families := []
  recurrences := [({ recurrence := ⟨"checker-integer"⟩, path := [] },
    integerRecurrenceCheckerFixture)]
  staticObligations := [
    .intBoundNonnegative (.recurrenceResult
      { recurrence := ⟨"checker-integer"⟩, path := [] } (.lower 0)),
    .intBoundPositive (.recurrenceResult
      { recurrence := ⟨"checker-integer"⟩, path := [] } (.lower 0)),
    .intBoundsOrdered
      (.recurrenceResult { recurrence := ⟨"checker-integer"⟩, path := [] } (.lower 0))
      (.recurrenceResult { recurrence := ⟨"checker-integer"⟩, path := [] } (.upper 0))
  ]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

/-- Phase B derives the recurrence table internally before checking signed endpoint obligations. -/
example : (checkStaticParameters integerRecurrenceCheckerAnalysis []).isOk = true := rfl

private def invalidIntBoundAnalysis : AnalysisResult where
  facts := []
  families := []
  recurrences := []
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
  recurrences := []
  staticObligations := [
    .thresholdNoise (.constant noise) (.constant 256) (.constant 2)
  ]
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

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
  recurrences := []
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
  recurrences := []
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
