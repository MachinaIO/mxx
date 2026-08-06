import Mxx.Certificate.Semantics

namespace Mxx.Certificate

/-- Package an exact executable integer denotation and its checked interval into the fact-table
predicate.  Primitive soundness lemmas supply every premise from the executable node and the two
input facts; certificates never construct this theorem's evidence. -/
theorem integerFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {fact : IntegerFact}
    {value lower upper : Int}
    (wireLookup : environment.values (.ofCoreWire wire) = some (.integer value))
    (denotes : RuntimeIntExpr.Denotes environment fact.expression value)
    (lowerEvaluates : fact.lower.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok lower)
    (upperEvaluates : fact.upper.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok upper)
    (lowerBound : lower ≤ value)
    (upperBound : value ≤ upper) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .integer fact
    } :=
  ⟨value, lower, upper, wireLookup, denotes, lowerEvaluates, upperEvaluates, lowerBound, upperBound⟩

/-- Boolean counterpart of `integerFact_holds`. -/
theorem booleanFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {fact : BooleanFact}
    {value : Bool}
    (wireLookup : environment.values (.ofCoreWire wire) = some (.boolean value))
    (denotes : RuntimeBoolExpr.Denotes environment fact.expression value) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .boolean fact
    } :=
  ⟨value, wireLookup, denotes⟩

theorem intAddFact_holds
    {environment : FactEnvironment}
    {leftWire rightWire outputWire : CoreWireRef}
    {leftType rightType : Option MatrixTypeExpr}
    {left right : IntegerFact}
    (leftHolds : ScopedWireFact.Holds environment {
      wire := leftWire, matrixType := leftType, fact := .integer left })
    (rightHolds : ScopedWireFact.Holds environment {
      wire := rightWire, matrixType := rightType, fact := .integer right })
    (outputLookup : ∀ leftValue rightValue,
      environment.values (.ofCoreWire leftWire) = some (.integer leftValue) →
      environment.values (.ofCoreWire rightWire) = some (.integer rightValue) →
      environment.values (.ofCoreWire outputWire) = some (.integer (leftValue + rightValue))) :
    ScopedWireFact.Holds environment {
      wire := outputWire
      matrixType := none
      fact := .integer {
        expression := .intBinary .add left.expression right.expression
        lower := .add left.lower right.lower
        upper := .add left.upper right.upper
      }
    } := by
  obtain ⟨leftValue, leftLower, leftUpper, leftLookup, leftDenotes, leftLowerEvaluates,
    leftUpperEvaluates, leftLowerBound, leftUpperBound⟩ := leftHolds
  obtain ⟨rightValue, rightLower, rightUpper, rightLookup, rightDenotes,
    rightLowerEvaluates, rightUpperEvaluates, rightLowerBound, rightUpperBound⟩ := rightHolds
  apply integerFact_holds (value := leftValue + rightValue)
    (lower := leftLower + rightLower) (upper := leftUpper + rightUpper)
    (outputLookup leftValue rightValue leftLookup rightLookup)
    (.intBinary leftDenotes rightDenotes rfl)
  · simp only [IntBoundExpr.evaluateWithSymbolicRecurrences]
    rw [leftLowerEvaluates, rightLowerEvaluates]
    change Except.ok (leftLower + rightLower) = Except.ok (leftLower + rightLower)
    rfl
  · simp only [IntBoundExpr.evaluateWithSymbolicRecurrences]
    rw [leftUpperEvaluates, rightUpperEvaluates]
    change Except.ok (leftUpper + rightUpper) = Except.ok (leftUpper + rightUpper)
    rfl
  · omega
  · omega

theorem intSubtractFact_holds
    {environment : FactEnvironment}
    {leftWire rightWire outputWire : CoreWireRef}
    {leftType rightType : Option MatrixTypeExpr}
    {left right : IntegerFact}
    (leftHolds : ScopedWireFact.Holds environment {
      wire := leftWire, matrixType := leftType, fact := .integer left })
    (rightHolds : ScopedWireFact.Holds environment {
      wire := rightWire, matrixType := rightType, fact := .integer right })
    (outputLookup : ∀ leftValue rightValue,
      environment.values (.ofCoreWire leftWire) = some (.integer leftValue) →
      environment.values (.ofCoreWire rightWire) = some (.integer rightValue) →
      environment.values (.ofCoreWire outputWire) = some (.integer (leftValue - rightValue))) :
    ScopedWireFact.Holds environment {
      wire := outputWire
      matrixType := none
      fact := .integer {
        expression := .intBinary .subtract left.expression right.expression
        lower := .subtract left.lower right.upper
        upper := .subtract left.upper right.lower
      }
    } := by
  obtain ⟨leftValue, leftLower, leftUpper, leftLookup, leftDenotes, leftLowerEvaluates,
    leftUpperEvaluates, leftLowerBound, leftUpperBound⟩ := leftHolds
  obtain ⟨rightValue, rightLower, rightUpper, rightLookup, rightDenotes,
    rightLowerEvaluates, rightUpperEvaluates, rightLowerBound, rightUpperBound⟩ := rightHolds
  apply integerFact_holds (value := leftValue - rightValue)
    (lower := leftLower - rightUpper) (upper := leftUpper - rightLower)
    (outputLookup leftValue rightValue leftLookup rightLookup)
    (.intBinary leftDenotes rightDenotes rfl)
  · simp only [IntBoundExpr.evaluateWithSymbolicRecurrences]
    rw [leftLowerEvaluates, rightUpperEvaluates]
    change Except.ok (leftLower - rightUpper) = Except.ok (leftLower - rightUpper)
    rfl
  · simp only [IntBoundExpr.evaluateWithSymbolicRecurrences]
    rw [leftUpperEvaluates, rightLowerEvaluates]
    change Except.ok (leftUpper - rightLower) = Except.ok (leftUpper - rightLower)
    rfl
  · omega
  · omega

theorem intCompareFact_holds
    {environment : FactEnvironment}
    {leftWire rightWire outputWire : CoreWireRef}
    {leftType rightType : Option MatrixTypeExpr}
    {left right : IntegerFact}
    {operation : Mxx.Ir.IntCompareOp}
    (leftHolds : ScopedWireFact.Holds environment {
      wire := leftWire, matrixType := leftType, fact := .integer left })
    (rightHolds : ScopedWireFact.Holds environment {
      wire := rightWire, matrixType := rightType, fact := .integer right })
    (outputLookup : ∀ leftValue rightValue,
      environment.values (.ofCoreWire leftWire) = some (.integer leftValue) →
      environment.values (.ofCoreWire rightWire) = some (.integer rightValue) →
      environment.values (.ofCoreWire outputWire) =
        some (.boolean (Mxx.Ir.evaluateIntCompare operation leftValue rightValue))) :
    ScopedWireFact.Holds environment {
      wire := outputWire
      matrixType := none
      fact := .boolean { expression := .compare operation left.expression right.expression }
    } := by
  obtain ⟨leftValue, _, _, leftLookup, leftDenotes, _⟩ := leftHolds
  obtain ⟨rightValue, _, _, rightLookup, rightDenotes, _⟩ := rightHolds
  exact booleanFact_holds (outputLookup leftValue rightValue leftLookup rightLookup)
    (.compare leftDenotes rightDenotes)

theorem bitExtractFact_holds
    {environment : FactEnvironment}
    {inputWire outputWire : CoreWireRef}
    {inputType : Option MatrixTypeExpr}
    {input : IntegerFact}
    {position : IntExpr}
    {evaluatedPosition : Int}
    (inputHolds : ScopedWireFact.Holds environment {
      wire := inputWire, matrixType := inputType, fact := .integer input })
    (positionEvaluates : evaluateIntExpr environment.parameters position = .ok evaluatedPosition)
    (nonnegative : 0 ≤ evaluatedPosition)
    (outputLookup : ∀ inputValue,
      environment.values (.ofCoreWire inputWire) = some (.integer inputValue) →
      environment.values (.ofCoreWire outputWire) =
        some (.boolean (((inputValue / (2 ^ evaluatedPosition.toNat)) % 2) ≠ 0))) :
    ScopedWireFact.Holds environment {
      wire := outputWire
      matrixType := none
      fact := .boolean { expression := .bitExtract input.expression position }
    } := by
  obtain ⟨inputValue, _, _, inputLookup, inputDenotes, _⟩ := inputHolds
  exact booleanFact_holds (outputLookup inputValue inputLookup)
    (.bitExtract inputDenotes positionEvaluates nonnegative)

/-- The exact integer constant fact emitted by `inferScalarOrSelect` is valid for the selected
runtime output.  No interval or denotation evidence is supplied by a certificate. -/
theorem constantIntFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {value : Int}
    (wireLookup : environment.values (.ofCoreWire wire) = some (.integer value)) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .integer {
        expression := .intConstant value
        lower := .integer (.constant value)
        upper := .integer (.constant value)
      }
    } := by
  refine ⟨value, value, value, wireLookup, .intConstant value, ?_, ?_, le_rfl, le_rfl⟩
  · simp [IntBoundExpr.evaluateWithSymbolicRecurrences, evaluateIntExpr, Except.mapError]
  · simp [IntBoundExpr.evaluateWithSymbolicRecurrences, evaluateIntExpr, Except.mapError]

/-- A parameter expression gets an exact singleton interval only after the shared IR evaluator
has produced the same value. -/
theorem evaluateIntFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {expression : IntExpr}
    {value : Int}
    (evaluates : evaluateIntExpr environment.parameters expression = .ok value)
    (wireLookup : environment.values (.ofCoreWire wire) = some (.integer value)) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .integer {
        expression := .parameter expression
        lower := .integer expression
        upper := .integer expression
      }
    } := by
  refine ⟨value, value, value, wireLookup, .parameter evaluates, ?_, ?_, le_rfl, le_rfl⟩
  · simp [IntBoundExpr.evaluateWithSymbolicRecurrences, evaluates, Except.mapError]
  · simp [IntBoundExpr.evaluateWithSymbolicRecurrences, evaluates, Except.mapError]

/-- The exact Boolean constant fact emitted by `inferScalarOrSelect`. -/
theorem constantBoolFact_holds
    {environment : FactEnvironment}
    {wire : CoreWireRef}
    {value : Bool}
    (wireLookup : environment.values (.ofCoreWire wire) = some (.boolean value)) :
    ScopedWireFact.Holds environment {
      wire
      matrixType := none
      fact := .boolean { expression := .boolConstant value }
    } := by
  exact ⟨value, wireLookup, .boolConstant value⟩

end Mxx.Certificate
