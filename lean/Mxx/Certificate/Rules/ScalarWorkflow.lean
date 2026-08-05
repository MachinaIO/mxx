import Mxx.Certificate.Semantics

namespace Mxx.Certificate

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
  · simp [IntBoundExpr.evaluate, evaluateIntExpr, Except.mapError]
  · simp [IntBoundExpr.evaluate, evaluateIntExpr, Except.mapError]

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
  · simp [IntBoundExpr.evaluate, evaluates, Except.mapError]
  · simp [IntBoundExpr.evaluate, evaluates, Except.mapError]

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
