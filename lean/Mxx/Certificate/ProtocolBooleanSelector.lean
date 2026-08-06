import Mxx.Certificate.SymbolicEvaluation

namespace Mxx.Certificate

/-! # Closed protocol-Boolean selector semantics

This module proves the semantic bridge for the one closed coefficient form that retains a
dynamic protocol Boolean as provenance.  The selector is analyzer-owned syntax: neither its
`ProtocolInputId` nor its zero/identity branches are supplied as semantic claims by a
certificate.
-/

/-- Equality of closed selector expressions preserves both the protocol-input identity and the
checked selector type. -/
theorem protocolBooleanSelectorExpression_injective
    {leftInput rightInput : ProtocolInputId}
    {leftType rightType : MatrixTypeExpr}
    (equal :
      protocolBooleanSelectorExpression leftInput leftType =
        protocolBooleanSelectorExpression rightInput rightType) :
    leftInput = rightInput ∧ leftType = rightType := by
  simp only [protocolBooleanSelectorExpression] at equal
  cases equal
  exact ⟨rfl, rfl⟩

/-- Matching the closed selector identity is exactly matching the corresponding selector
expression; there is no name- or value-based fallback. -/
theorem protocolBooleanSelectorIdentity_matches_iff
    (input : ProtocolInputId)
    (selectorType : MatrixTypeExpr)
    (expression : MatrixExpr) :
    (SignalCoefficientIdentity.protocolBooleanSelector input selectorType).Matches expression ↔
      expression = protocolBooleanSelectorExpression input selectorType :=
  Iff.rfl

/-- The false protocol input selects the typed zero coefficient. -/
theorem protocolBooleanSelector_denotes_false
    {environment : FactEnvironment}
    {input : ProtocolInputId}
    {selectorType : MatrixTypeExpr}
    {selectorParams : Mxx.SamplerParams}
    (inputLookup :
      environment.values (.protocolInput input) = some (.boolean false))
    (typeEvaluates :
      selectorType.evaluate environment.parameters = some selectorParams) :
    MatrixExpr.Denotes environment
      (protocolBooleanSelectorExpression input selectorType)
      (zeroConstantOutput selectorParams) := by
  exact .select
    (.boolToInt (.boolWire inputLookup))
    (by omega)
    rfl
    (.zero typeEvaluates)

/-- The true protocol input selects the typed identity coefficient. -/
theorem protocolBooleanSelector_denotes_true
    {environment : FactEnvironment}
    {input : ProtocolInputId}
    {selectorType : MatrixTypeExpr}
    {selectorParams : Mxx.SamplerParams}
    (inputLookup :
      environment.values (.protocolInput input) = some (.boolean true))
    (typeEvaluates :
      selectorType.evaluate environment.parameters = some selectorParams) :
    MatrixExpr.Denotes environment
      (protocolBooleanSelectorExpression input selectorType)
      (identityConstantOutput selectorParams) := by
  exact .select
    (.boolToInt (.boolWire inputLookup))
    (by omega)
    rfl
    (.identity typeEvaluates)

/-- Uniform denotation of the closed selector.  The selected matrix is definitionally zero or
identity according to the value bound to the exact `ProtocolInputId`. -/
theorem protocolBooleanSelector_denotes
    {environment : FactEnvironment}
    {input : ProtocolInputId}
    {selectorType : MatrixTypeExpr}
    {selectorParams : Mxx.SamplerParams}
    (message : Bool)
    (inputLookup :
      environment.values (.protocolInput input) = some (.boolean message))
    (typeEvaluates :
      selectorType.evaluate environment.parameters = some selectorParams) :
    MatrixExpr.Denotes environment
      (protocolBooleanSelectorExpression input selectorType)
      (if message then identityConstantOutput selectorParams
        else zeroConstantOutput selectorParams) := by
  cases message
  · exact protocolBooleanSelector_denotes_false inputLookup typeEvaluates
  · exact protocolBooleanSelector_denotes_true inputLookup typeEvaluates

/-- The selected zero-or-identity coefficient has hard norm at most one. -/
theorem protocolBooleanSelector_norm_le_one
    (selectorParams : Mxx.SamplerParams)
    (message : Bool)
    (modulusPositive : 0 < selectorParams.modulus) :
    Mxx.maxCenteredCoefficientNorm
        (if message then identityConstantOutput selectorParams
          else zeroConstantOutput selectorParams) ≤ 1 := by
  cases message
  · simpa using (zeroConstant_norm_eq_zero selectorParams).le.trans (Nat.zero_le 1)
  · exact identityConstant_norm_le_one selectorParams modulusPositive

/-- The denotation and norm proofs combine into the bounded coefficient fact consumed by signal
term semantics. -/
theorem protocolBooleanSelector_bounded_holds
    {environment : FactEnvironment}
    {input : ProtocolInputId}
    {selectorType : MatrixTypeExpr}
    {selectorParams : Mxx.SamplerParams}
    (message : Bool)
    (inputLookup :
      environment.values (.protocolInput input) = some (.boolean message))
    (typeEvaluates :
      selectorType.evaluate environment.parameters = some selectorParams)
    (modulusPositive : 0 < selectorParams.modulus) :
    BoundedMatrixExpr.Holds environment {
      expression := protocolBooleanSelectorExpression input selectorType
      normBound := .constant 1
    } (if message then identityConstantOutput selectorParams
      else zeroConstantOutput selectorParams) := by
  refine ⟨protocolBooleanSelector_denotes message inputLookup typeEvaluates, 1, rfl, ?_⟩
  exact protocolBooleanSelector_norm_le_one selectorParams message modulusPositive

/-- Closed construction of one evaluated selector term.  The stored coefficient identity carries
the exact protocol input used by the denotation proof, while the two runtime cases remain the
ordinary zero and identity matrices. -/
def evaluatedProtocolBooleanSelectorTerm
    (environment : FactEnvironment)
    (input : ProtocolInputId)
    (selectorType : MatrixTypeExpr)
    (selectorParams : Mxx.SamplerParams)
    (message : Bool)
    (inputLookup :
      environment.values (.protocolInput input) = some (.boolean message))
    (typeEvaluates :
      selectorType.evaluate environment.parameters = some selectorParams)
    (modulusPositive : 0 < selectorParams.modulus)
    (basis : MatrixExpr)
    (basisValue : Mxx.Matrix)
    (basisDenotes : MatrixExpr.Denotes environment basis basisValue)
    (mode : SignalProductMode) : EvaluatedSignalTerm environment :=
  let coefficientValue :=
    if message then identityConstantOutput selectorParams else zeroConstantOutput selectorParams
  let coefficient : BoundedMatrixExpr := {
    expression := protocolBooleanSelectorExpression input selectorType
    normBound := .constant 1
  }
  let symbolic : SignalTerm := { coefficient, basis, mode }
  let coefficientHolds := protocolBooleanSelector_bounded_holds message inputLookup
    typeEvaluates modulusPositive
  {
    symbolic
    coefficientIdentity := .protocolBooleanSelector input selectorType
    coefficientBound := 1
    carrierValue := basisValue
    termValue := Mxx.matrixMultiply coefficientValue basisValue
    identityMatches := rfl
    coefficientBoundEvaluates := rfl
    carrierDenotes := basisDenotes
    termDenotes := .product coefficientHolds basisDenotes
  }

@[simp] theorem evaluatedProtocolBooleanSelectorTerm_coefficientIdentity
    (environment : FactEnvironment)
    (input : ProtocolInputId)
    (selectorType : MatrixTypeExpr)
    (selectorParams : Mxx.SamplerParams)
    (message : Bool)
    (inputLookup :
      environment.values (.protocolInput input) = some (.boolean message))
    (typeEvaluates :
      selectorType.evaluate environment.parameters = some selectorParams)
    (modulusPositive : 0 < selectorParams.modulus)
    (basis : MatrixExpr)
    (basisValue : Mxx.Matrix)
    (basisDenotes : MatrixExpr.Denotes environment basis basisValue)
    (mode : SignalProductMode) :
    (evaluatedProtocolBooleanSelectorTerm environment input selectorType selectorParams message
      inputLookup typeEvaluates modulusPositive basis basisValue basisDenotes mode).coefficientIdentity =
        .protocolBooleanSelector input selectorType :=
  rfl

end Mxx.Certificate
