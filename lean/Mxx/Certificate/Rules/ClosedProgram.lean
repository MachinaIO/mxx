import Mxx.Certificate.Soundness
import Mxx.Certificate.Rules.ScalarWorkflow

namespace Mxx.Certificate

/-! # Closed primitive-program soundness

This module starts the executable-path induction used by whole-program soundness. Its public
theorem accepts only an actual `EvaluatesNodesPath`, analyzer acceptance, and already-proved input
facts. In particular it has no `headSound`, `suffixSound`, or certificate-provided invariant.
-/

/-- The initial closed scalar-leaf fragment. This is a syntactic restriction on the executable
program, not a semantic callback. -/
inductive ClosedPrimitiveNode (params : Mxx.Ir.ParamEnvironment) : Mxx.Ir.Node → Prop where
  | input (name : String) (outputCount : Nat) (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .input name
        arguments := []
        outputCount
        outputTypes
      }
  | constantInt (value : Int) (outputCount : Nat)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .constantInt value
        arguments := []
        outputCount
        outputTypes
      }
  | evaluateInt (expression : IntExpr) (value : Int)
      (evaluates : expression.evaluate params = some value) (outputCount : Nat)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .evaluateInt expression
        arguments := []
        outputCount
        outputTypes
      }
  | constantBool (value : Bool) (outputCount : Nat)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .constantBool value
        arguments := []
        outputCount
        outputTypes
      }
  | zeroMatrix (matrixType : MatrixTypeExpr) (matrixParams : Mxx.SamplerParams)
      (typeEvaluates : matrixType.evaluate params = some matrixParams)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .zeroMatrix matrixType
        arguments := []
        outputCount := 1
        outputTypes
      }
  | identityMatrix (matrixType : MatrixTypeExpr) (matrixParams : Mxx.SamplerParams)
      (typeEvaluates : matrixType.evaluate params = some matrixParams)
      (modulusPositive : 0 < matrixParams.modulus)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .identityMatrix matrixType
        arguments := []
        outputCount := 1
        outputTypes
      }
  | constantMatrix (matrixType : MatrixTypeExpr) (coefficients : List IntExpr)
      (matrixParams : Mxx.SamplerParams) (evaluated : List Int)
      (typeEvaluates : matrixType.evaluate params = some matrixParams)
      (coefficientsEvaluate : coefficients.mapM (Mxx.Ir.IntExpr.evaluate params) = some evaluated)
      (modulusPositive : 0 < matrixParams.modulus)
      (bound : Nat)
      (boundEvaluates :
        (BoundExpr.floorDivide (.absolute matrixType.modulus) 2).evaluate params = .ok bound)
      (boundExact : bound = matrixParams.modulus.natAbs / 2)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .constantMatrix matrixType coefficients
        arguments := []
        outputCount := 1
        outputTypes
      }
  | gadgetMatrix (matrixType : MatrixTypeExpr) (base : IntExpr)
      (matrixParams : Mxx.SamplerParams) (evaluatedBase : Int)
      (typeEvaluates : matrixType.evaluate params = some matrixParams)
      (baseEvaluates : base.evaluate params = some evaluatedBase)
      (modulusPositive : 0 < matrixParams.modulus)
      (bound : Nat)
      (boundEvaluates :
        (BoundExpr.floorDivide (.absolute matrixType.modulus) 2).evaluate params = .ok bound)
      (boundExact : bound = matrixParams.modulus.natAbs / 2)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .gadgetMatrix matrixType base
        arguments := []
        outputCount := 1
        outputTypes
      }
  | gaussianSample (matrixType : MatrixTypeExpr) (cutoff : IntExpr)
      (matrixParams : Mxx.SamplerParams)
      (typeEvaluates : matrixType.evaluate params cutoff = some matrixParams)
      (boundEvaluates : (BoundExpr.parameter cutoff).evaluate params =
        .ok matrixParams.maxCoefficientBound)
      (outputTypes : List Mxx.Ir.WireTypeExpr) :
      ClosedPrimitiveNode params {
        kind := .gaussianSample matrixType cutoff
        arguments := []
        outputCount := 1
        outputTypes
      }

private theorem futureWireFresh_afterBinding
    (nodeId : Nat)
    (state : Mxx.Ir.WireEnvironment)
    (values : List Mxx.Ir.Value)
    (fresh : ∀ target port, nodeId ≤ target →
      Mxx.Ir.lookupWire ⟨target, port⟩ state = none) :
    ∀ target port, nodeId + 1 ≤ target →
      Mxx.Ir.lookupWire ⟨target, port⟩
        (state ++ Mxx.Ir.bindOutputs nodeId values) = none := by
  intro target port after
  rw [Mxx.Ir.lookupWire_append_of_eq_none (fresh target port (by omega))]
  exact Mxx.Ir.lookupWire_bindOutputs_of_node_ne nodeId target port values (by omega)

private theorem singletonOutputLookup
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {current : Nat}
    {tail : List Mxx.Ir.Node}
    {state output : Mxx.Ir.WireEnvironment}
    {values : List Mxx.Ir.Value}
    (tailPath : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs (current + 1) tail
      (state ++ Mxx.Ir.bindOutputs current values) output)
    (fresh : Mxx.Ir.lookupWire ⟨current, 0⟩ state = none)
    (value : Mxx.Ir.Value)
    (valuesExact : values = [value]) :
    Mxx.Ir.lookupWire ⟨current, 0⟩ output = some value := by
  apply tailPath.lookupWire_preserved
  rw [Mxx.Ir.lookupWire_append_bindOutputs fresh (by simp [valuesExact])]
  simp [valuesExact]

private theorem inferRulesFrom_sound_closedPrimitives_aux
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {stage : StageId}
    {nodeId : Nat}
    {nodes : List Mxx.Ir.Node}
    {state final : Mxx.Ir.WireEnvironment}
    {initial result : ScopedWireFactTable}
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs nodeId nodes state final)
    (closed : List.Forall (ClosedPrimitiveNode params) nodes)
    (fresh : ∀ target port, nodeId ≤ target →
      Mxx.Ir.lookupWire ⟨target, port⟩ state = none)
    (accepted : inferRulesFrom stage ⟨[]⟩ nodeId nodes initial = .ok result)
    (initialHolds : initial.Holds
      (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ final)) :
    result.Holds (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ final) := by
  induction path generalizing initial result with
  | nil =>
      simp only [inferRulesFrom] at accepted
      injection accepted with equality
      rw [← equality]
      exact initialHolds
  | @cons current node tail currentState values output valuesMember tailPath induction =>
      have closedParts := (List.forall_cons (ClosedPrimitiveNode params) node tail).mp closed
      have nodeClosed : ClosedPrimitiveNode params node := closedParts.1
      have tailClosed : List.Forall (ClosedPrimitiveNode params) tail := closedParts.2
      show result.Holds (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ output)
      have compose : result.Holds
          (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ output) := by
        simp only [inferRulesFrom] at accepted
        cases inferred : inferNodeFacts stage ⟨[]⟩ current node initial with
        | error error =>
            rw [inferred] at accepted
            contradiction
        | ok next =>
            rw [inferred] at accepted
            have nextFresh := futureWireFresh_afterBinding current currentState values fresh
            have nextHolds : next.Holds
                (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ output) := by
              cases nodeClosed with
              | input name outputCount outputTypes =>
                  rw [inferNodeFacts_input] at inferred
                  injection inferred with equality
                  rw [← equality]
                  exact initialHolds
              | constantInt value outputCount outputTypes =>
                  have valuesExact := constantIntNode_local_sound runChild samplers params inputs
                    currentState value outputCount valuesMember
                  have outputInBounds : 0 < values.length := by simp [valuesExact]
                  have outputLookup :
                      Mxx.Ir.lookupWire ⟨current, 0⟩ output = some (.integer value) := by
                    apply tailPath.lookupWire_preserved
                    rw [Mxx.Ir.lookupWire_append_bindOutputs
                      (fresh current 0 (by omega)) outputInBounds]
                    simp [valuesExact]
                  rw [inferNodeFacts_constantInt] at inferred
                  injection inferred with equality
                  rw [← equality]
                  apply initialHolds.append_singleton
                  apply constantIntFact_holds
                  simpa [ValueInstanceRef.ofCoreWire, FactEnvironment.ofWireEnvironment]
                    using outputLookup
              | evaluateInt expression value evaluates outputCount outputTypes =>
                  have valuesExact := evaluateIntNode_local_sound runChild samplers params inputs
                    currentState expression value outputCount evaluates valuesMember
                  have outputInBounds : 0 < values.length := by simp [valuesExact]
                  have outputLookup :
                      Mxx.Ir.lookupWire ⟨current, 0⟩ output = some (.integer value) := by
                    apply tailPath.lookupWire_preserved
                    rw [Mxx.Ir.lookupWire_append_bindOutputs
                      (fresh current 0 (by omega)) outputInBounds]
                    simp [valuesExact]
                  rw [inferNodeFacts_evaluateInt] at inferred
                  injection inferred with equality
                  rw [← equality]
                  apply initialHolds.append_singleton
                  apply evaluateIntFact_holds
                  · exact evaluateIntExpr_ok_of_ir_evaluate params expression value evaluates
                  · simpa [ValueInstanceRef.ofCoreWire, FactEnvironment.ofWireEnvironment]
                      using outputLookup
              | constantBool value outputCount outputTypes =>
                  have valuesExact := constantBoolNode_local_sound runChild samplers params inputs
                    currentState value outputCount valuesMember
                  have outputInBounds : 0 < values.length := by simp [valuesExact]
                  have outputLookup :
                      Mxx.Ir.lookupWire ⟨current, 0⟩ output = some (.boolean value) := by
                    apply tailPath.lookupWire_preserved
                    rw [Mxx.Ir.lookupWire_append_bindOutputs
                      (fresh current 0 (by omega)) outputInBounds]
                    simp [valuesExact]
                  rw [inferNodeFacts_constantBool] at inferred
                  injection inferred with equality
                  rw [← equality]
                  apply initialHolds.append_singleton
                  apply constantBoolFact_holds
                  simpa [ValueInstanceRef.ofCoreWire, FactEnvironment.ofWireEnvironment]
                    using outputLookup
              | zeroMatrix matrixType matrixParams typeEvaluates outputTypes =>
                  have localSound := zeroMatrixNode_local_sound runChild samplers params inputs
                    currentState matrixType matrixParams 1 typeEvaluates valuesMember
                  have outputLookup := singletonOutputLookup tailPath
                    (fresh current 0 (by omega)) (.matrix (zeroConstantOutput matrixParams))
                    localSound.1
                  rw [inferNodeFacts_zeroMatrix] at inferred
                  cases applicationResult : applyRule initial {
                    stage
                    scope := ⟨[]⟩
                    nodeId := current
                    node := {
                      kind := .zeroMatrix matrixType
                      arguments := []
                      outputCount := 1
                      outputTypes
                    }
                    rule := .introduceExactConstant
                  } with
                  | error error =>
                      rw [applicationResult] at inferred
                      change Except.error error = Except.ok next at inferred
                      contradiction
                  | ok application =>
                      rcases application with ⟨nextResult, obligations, endpoints⟩
                      rw [applicationResult] at inferred
                      change Except.ok nextResult = Except.ok next at inferred
                      injection inferred with equality
                      rw [← equality]
                      apply applyRule_sound_zero applicationResult initialHolds typeEvaluates
                      simpa [ValueInstanceRef.ofCoreWire, FactEnvironment.ofWireEnvironment]
                        using outputLookup
              | identityMatrix matrixType matrixParams typeEvaluates modulusPositive outputTypes =>
                  have localSound := identityMatrixNode_local_sound runChild samplers params inputs
                    currentState matrixType matrixParams 1 typeEvaluates modulusPositive valuesMember
                  have outputLookup := singletonOutputLookup tailPath
                    (fresh current 0 (by omega)) (.matrix (identityConstantOutput matrixParams))
                    localSound.1
                  rw [inferNodeFacts_identityMatrix] at inferred
                  cases applicationResult : applyRule initial {
                    stage
                    scope := ⟨[]⟩
                    nodeId := current
                    node := {
                      kind := .identityMatrix matrixType
                      arguments := []
                      outputCount := 1
                      outputTypes
                    }
                    rule := .introduceExactConstant
                  } with
                  | error error =>
                      rw [applicationResult] at inferred
                      change Except.error error = Except.ok next at inferred
                      contradiction
                  | ok application =>
                      rcases application with ⟨nextResult, obligations, endpoints⟩
                      rw [applicationResult] at inferred
                      change Except.ok nextResult = Except.ok next at inferred
                      injection inferred with equality
                      rw [← equality]
                      apply applyRule_sound_identity applicationResult initialHolds
                      · simpa [ValueInstanceRef.ofCoreWire, FactEnvironment.ofWireEnvironment]
                          using outputLookup
                      · exact localSound.2
              | constantMatrix matrixType coefficients matrixParams evaluated typeEvaluates
                  coefficientsEvaluate modulusPositive bound boundEvaluates boundExact
                  outputTypes =>
                  let matrixOutput := Mxx.Matrix.withSamplerParams {
                    coefficients := evaluated.map (Mxx.reduceCoefficient matrixParams.modulus)
                  } matrixParams
                  have localSound := constantMatrixNode_local_sound runChild samplers params inputs
                    currentState matrixType coefficients matrixParams evaluated 1 typeEvaluates
                    coefficientsEvaluate modulusPositive valuesMember
                  have outputLookup := singletonOutputLookup tailPath
                    (fresh current 0 (by omega)) (.matrix matrixOutput) localSound.1
                  rw [inferNodeFacts_constantMatrix] at inferred
                  cases applicationResult : applyRule initial {
                    stage
                    scope := ⟨[]⟩
                    nodeId := current
                    node := {
                      kind := .constantMatrix matrixType coefficients
                      arguments := []
                      outputCount := 1
                      outputTypes
                    }
                    rule := .introduceExactConstant
                  } with
                  | error error =>
                      rw [applicationResult] at inferred
                      change Except.error error = Except.ok next at inferred
                      contradiction
                  | ok application =>
                      rcases application with ⟨nextResult, obligations, endpoints⟩
                      rw [applicationResult] at inferred
                      change Except.ok nextResult = Except.ok next at inferred
                      injection inferred with equality
                      rw [← equality]
                      apply applyRule_sound_constantMatrix applicationResult initialHolds
                      · simpa [matrixOutput, ValueInstanceRef.ofCoreWire,
                          FactEnvironment.ofWireEnvironment] using outputLookup
                      · exact boundEvaluates
                      · rw [boundExact]
                        exact localSound.2
              | gadgetMatrix matrixType base matrixParams evaluatedBase typeEvaluates
                  baseEvaluates modulusPositive bound boundEvaluates boundExact outputTypes =>
                  let digits := if matrixParams.rows = 0 then 0
                    else matrixParams.columns / matrixParams.rows
                  let matrixOutput := Mxx.gadgetMatrix matrixParams evaluatedBase digits
                  have localSound := gadgetMatrixNode_local_sound runChild samplers params inputs
                    currentState matrixType base matrixParams evaluatedBase 1 typeEvaluates
                    baseEvaluates modulusPositive valuesMember
                  have outputLookup := singletonOutputLookup tailPath
                    (fresh current 0 (by omega)) (.matrix matrixOutput) localSound.1
                  rw [inferNodeFacts_gadgetMatrix] at inferred
                  cases applicationResult : applyRule initial {
                    stage
                    scope := ⟨[]⟩
                    nodeId := current
                    node := {
                      kind := .gadgetMatrix matrixType base
                      arguments := []
                      outputCount := 1
                      outputTypes
                    }
                    rule := .introduceExactConstant
                  } with
                  | error error =>
                      rw [applicationResult] at inferred
                      change Except.error error = Except.ok next at inferred
                      contradiction
                  | ok application =>
                      rcases application with ⟨nextResult, obligations, endpoints⟩
                      rw [applicationResult] at inferred
                      change Except.ok nextResult = Except.ok next at inferred
                      injection inferred with equality
                      rw [← equality]
                      apply applyRule_sound_gadgetMatrix applicationResult initialHolds
                        typeEvaluates baseEvaluates
                      · simpa [digits, matrixOutput, ValueInstanceRef.ofCoreWire,
                          FactEnvironment.ofWireEnvironment] using outputLookup
                      · exact boundEvaluates
                      · rw [boundExact]
                        exact localSound.2
              | gaussianSample matrixType cutoff matrixParams typeEvaluates boundEvaluates
                  outputTypes =>
                  obtain ⟨matrixOutput, valuesExact, outputNorm⟩ :=
                    gaussianNode_local_sound runChild samplers contract params inputs currentState
                      matrixType cutoff matrixParams 1 typeEvaluates valuesMember
                  have outputLookup := singletonOutputLookup tailPath
                    (fresh current 0 (by omega)) (.matrix matrixOutput) valuesExact
                  rw [inferNodeFacts_gaussianSample] at inferred
                  cases applicationResult : applyRule initial {
                    stage
                    scope := ⟨[]⟩
                    nodeId := current
                    node := {
                      kind := .gaussianSample matrixType cutoff
                      arguments := []
                      outputCount := 1
                      outputTypes
                    }
                    rule := .introduceGaussian
                  } with
                  | error error =>
                      rw [applicationResult] at inferred
                      change Except.error error = Except.ok next at inferred
                      contradiction
                  | ok application =>
                      rcases application with ⟨nextResult, obligations, endpoints⟩
                      rw [applicationResult] at inferred
                      change Except.ok nextResult = Except.ok next at inferred
                      injection inferred with equality
                      rw [← equality]
                      apply applyRule_sound applicationResult initialHolds
                      · simpa [ValueInstanceRef.ofCoreWire, FactEnvironment.ofWireEnvironment]
                          using outputLookup
                      · exact boundEvaluates
                      · exact outputNorm
            exact induction tailClosed nextFresh accepted nextHolds
      exact compose

/-- Closed root-scope induction for the initial scalar-leaf fragment. Every derived fact is
connected to the selected executable path; no theorem caller supplies per-node soundness. -/
private theorem inferRulesFrom_sound_closedPrimitives
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {stage : StageId}
    {nodes : List Mxx.Ir.Node}
    {final : Mxx.Ir.WireEnvironment}
    {initial result : ScopedWireFactTable}
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (path : Mxx.Ir.EvaluatesNodesPath runChild samplers params inputs 0 nodes [] final)
    (closed : List.Forall (ClosedPrimitiveNode params) nodes)
    (accepted : inferRulesFrom stage ⟨[]⟩ 0 nodes initial = .ok result)
    (initialHolds : initial.Holds
      (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ final)) :
    result.Holds (FactEnvironment.ofWireEnvironment params stage ⟨[]⟩ final) := by
  apply inferRulesFrom_sound_closedPrimitives_aux contract path closed _ accepted initialHolds
  intro target port _
  rfl

end Mxx.Certificate
