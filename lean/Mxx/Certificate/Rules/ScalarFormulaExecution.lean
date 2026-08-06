import Mxx.Certificate.Rules.BggLaneFormulaExecution
import Mxx.Certificate.Semantics

namespace Mxx.Certificate

/-!
# Trace-indexed scalar and control execution

This module interprets analyzer-owned scalar expressions only against one actual executed scope.
It replaces the callback accepted by `FrozenPointwiseScalarFormula.evaluate` with direct wire
lookups and resolves arena references from the analyzer-owned immutable `ExpressionArena`.
-/

/-- A stable value identity resolves to one wire in the actual executed scope.  The stage and
scope checks are part of the evidence, so an equal-looking wire from another instance cannot be
substituted. -/
inductive ExecutedScope.ValueAt
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (stage : StageId)
    (current : ExecutedScope samplers program) : ValueInstanceRef → Mxx.Ir.Value → Prop where
  | core
      (wire : CoreWireRef)
      (value : Mxx.Ir.Value)
      (stageMatches : wire.stage = stage)
      (scopeMatches : wire.scope = current.scopeId)
      (found : Mxx.Ir.lookupWire ⟨wire.node.value, wire.port⟩
        current.execution.wires = some value) :
      ExecutedScope.ValueAt stage current (.ofCoreWire wire) value

/-- Exact execution semantics for the closed pointwise scalar view.  Every atom is read from the
actual scope environment and every operation delegates to the executable IR evaluator. -/
inductive FrozenPointwiseScalarFormula.DenotesExecuted
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program) :
    FrozenPointwiseScalarFormula → Mxx.Ir.Value → Prop where
  | atom
      (wire : Mxx.Ir.WireRef)
      (value : Mxx.Ir.Value)
      (found : Mxx.Ir.lookupWire wire current.execution.wires = some value) :
      DenotesExecuted current (.atom wire) value
  | integer (value : Int) : DenotesExecuted current (.integer value) (.integer value)
  | boolean (value : Bool) : DenotesExecuted current (.boolean value) (.boolean value)
  | boolToInt
      {input : FrozenPointwiseScalarFormula}
      {value : Bool}
      (inputDenotes : DenotesExecuted current input (.boolean value)) :
      DenotesExecuted current (.boolToInt input) (.integer (if value then 1 else 0))
  | intBinary
      {operation : Mxx.Ir.IntBinaryOp}
      {left right : FrozenPointwiseScalarFormula}
      {leftValue rightValue value : Int}
      (leftDenotes : DenotesExecuted current left (.integer leftValue))
      (rightDenotes : DenotesExecuted current right (.integer rightValue))
      (evaluates : Mxx.Ir.evaluateIntBinary operation leftValue rightValue = some value) :
      DenotesExecuted current (.intBinary operation left right) (.integer value)
  | compare
      {operation : Mxx.Ir.IntCompareOp}
      {left right : FrozenPointwiseScalarFormula}
      {leftValue rightValue : Int}
      (leftDenotes : DenotesExecuted current left (.integer leftValue))
      (rightDenotes : DenotesExecuted current right (.integer rightValue)) :
      DenotesExecuted current (.compare operation left right)
        (.boolean (Mxx.Ir.evaluateIntCompare operation leftValue rightValue))

/-- Trace-indexed semantics agrees with the legacy evaluator when its atom callback is fixed to
the actual executed wire environment. -/
theorem FrozenPointwiseScalarFormula.DenotesExecuted.evaluate_eq
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {current : ExecutedScope samplers program}
    {formula : FrozenPointwiseScalarFormula}
    {value : Mxx.Ir.Value}
    (denotes : formula.DenotesExecuted current value) :
    formula.evaluate (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) =
      some value := by
  induction denotes with
  | atom wire value found => exact found
  | integer | boolean => rfl
  | boolToInt inputDenotes induction => simp [FrozenPointwiseScalarFormula.evaluate, induction]
  | intBinary leftDenotes rightDenotes evaluates leftInduction rightInduction =>
      simp [FrozenPointwiseScalarFormula.evaluate, leftInduction, rightInduction, evaluates]
  | compare leftDenotes rightDenotes leftInduction rightInduction =>
      simp [FrozenPointwiseScalarFormula.evaluate, leftInduction, rightInduction]

/-- A scalar result computed from the actual executed wire environment. -/
structure FrozenPointwiseScalarFormula.ExecutedResult
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program)
    (formula : FrozenPointwiseScalarFormula) : Type where
  value : Mxx.Ir.Value
  denotes : formula.DenotesExecuted current value

/-- Successful evaluation against the actual wire environment yields trace-indexed denotation. -/
theorem FrozenPointwiseScalarFormula.denotesExecuted_of_evaluate
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program)
    (formula : FrozenPointwiseScalarFormula)
    (value : Mxx.Ir.Value)
    (evaluates : formula.evaluate
      (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) = some value) :
    formula.DenotesExecuted current value := by
  induction formula generalizing value with
  | atom wire => exact .atom wire value evaluates
  | integer integerValue =>
      simp [FrozenPointwiseScalarFormula.evaluate] at evaluates
      subst value
      exact .integer integerValue
  | boolean booleanValue =>
      simp [FrozenPointwiseScalarFormula.evaluate] at evaluates
      subst value
      exact .boolean booleanValue
  | boolToInt input induction =>
      cases inputEvaluates : input.evaluate
          (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) with
      | none => simp [FrozenPointwiseScalarFormula.evaluate, inputEvaluates] at evaluates
      | some inputValue =>
          cases inputValue <;>
            simp [FrozenPointwiseScalarFormula.evaluate, inputEvaluates] at evaluates
          rename_i booleanValue
          subst value
          exact .boolToInt (induction (.boolean booleanValue) inputEvaluates)
  | intBinary operation left right leftInduction rightInduction =>
      cases leftEvaluates : left.evaluate
          (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) with
      | none => simp [FrozenPointwiseScalarFormula.evaluate, leftEvaluates] at evaluates
      | some leftValue =>
          cases leftValue <;>
            simp [FrozenPointwiseScalarFormula.evaluate, leftEvaluates] at evaluates
          rename_i leftInteger
          cases rightEvaluates : right.evaluate
              (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) with
          | none => simp [rightEvaluates] at evaluates
          | some rightValue =>
              cases rightValue <;> simp [rightEvaluates] at evaluates
              rename_i rightInteger
              cases operationEvaluates : Mxx.Ir.evaluateIntBinary operation leftInteger
                  rightInteger with
              | none =>
                  simp [operationEvaluates] at evaluates
              | some result =>
                  simp [operationEvaluates] at evaluates
                  subst value
                  exact .intBinary (leftInduction (.integer leftInteger) leftEvaluates)
                    (rightInduction (.integer rightInteger) rightEvaluates) operationEvaluates
  | compare operation left right leftInduction rightInduction =>
      cases leftEvaluates : left.evaluate
          (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) with
      | none => simp [FrozenPointwiseScalarFormula.evaluate, leftEvaluates] at evaluates
      | some leftValue =>
          cases leftValue <;>
            simp [FrozenPointwiseScalarFormula.evaluate, leftEvaluates] at evaluates
          rename_i leftInteger
          cases rightEvaluates : right.evaluate
              (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) with
          | none => simp [rightEvaluates] at evaluates
          | some rightValue =>
              cases rightValue <;> simp [rightEvaluates] at evaluates
              rename_i rightInteger
              subst value
              exact .compare (leftInduction (.integer leftInteger) leftEvaluates)
                (rightInduction (.integer rightInteger) rightEvaluates)

/-- Execute the closed scalar view without accepting an atom resolver. -/
def FrozenPointwiseScalarFormula.execute?
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program)
    (formula : FrozenPointwiseScalarFormula) : Option (ExecutedResult current formula) :=
  match evaluates : formula.evaluate
      (fun wire ↦ Mxx.Ir.lookupWire wire current.execution.wires) with
  | none => none
  | some value => some {
      value
      denotes := formula.denotesExecuted_of_evaluate current value evaluates
    }

/-- Integer-specialized result of actual scalar execution. -/
structure FrozenPointwiseScalarFormula.ExecutedIntegerResult
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program)
    (formula : FrozenPointwiseScalarFormula) : Type where
  value : Int
  denotes : formula.DenotesExecuted current (.integer value)

def FrozenPointwiseScalarFormula.executeInteger?
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (current : ExecutedScope samplers program)
    (formula : FrozenPointwiseScalarFormula) : Option (ExecutedIntegerResult current formula) := do
  match executed : formula.execute? current with
  | none => none
  | some result =>
      match valueEq : result.value with
      | .integer value => some {
          value
          denotes := by simpa only [valueEq] using result.denotes
        }
      | _ => none

mutual

/-- Exact integer expression semantics indexed by the actual scope and analyzer-owned arena.
Unsupported identities have no constructor and therefore fail closed. -/
inductive RuntimeIntExpr.DenotesExecuted
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (stage : StageId)
    (arena : ExpressionArena)
    (current : ExecutedScope samplers program) : RuntimeExpr .integer → Int → Prop where
  | intWire
      {wire : ValueInstanceRef}
      {value : Int}
      (found : current.ValueAt stage wire (.integer value)) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.intWire wire) value
  | intConstant (value : Int) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.intConstant value) value
  | parameter
      {expression : IntExpr}
      {value : Int}
      (evaluates : expression.evaluate current.params = some value) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.parameter expression) value
  | boolToInt
      {expression : RuntimeExpr .boolean}
      {value : Bool}
      (input : RuntimeBoolExpr.DenotesExecuted stage arena current expression value) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.boolToInt expression)
        (if value then 1 else 0)
  | intBinary
      {operation : IntBinaryOp}
      {left right : RuntimeExpr .integer}
      {leftValue rightValue value : Int}
      (leftDenotes : RuntimeIntExpr.DenotesExecuted stage arena current left leftValue)
      (rightDenotes : RuntimeIntExpr.DenotesExecuted stage arena current right rightValue)
      (evaluates : Mxx.Ir.evaluateIntBinary operation leftValue rightValue = some value) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.intBinary operation left right) value
  | extractCoefficient
      {matrixRef : MatrixExprRef}
      {matrixExpr : MatrixExpr}
      {matrixIdentity : MatrixInstanceRef}
      {matrix : Mxx.Matrix}
      {position : IntExpr}
      {positionValue value : Int}
      (arenaLookup : arena.lookupMatrix matrixRef = some matrixExpr)
      (matrixWire : matrixExpr = .wire matrixIdentity)
      (matrixFound : current.ValueAt stage matrixIdentity.value (.matrix matrix))
      (positionEvaluates : position.evaluate current.params = some positionValue)
      (valueEq : value = Mxx.reduceCoefficient matrix.modulus
        (matrix.coefficients.getD positionValue.toNat 0)) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.extractCoefficient matrixRef position)
        value
  | select
      {index : RuntimeExpr .integer}
      {branches : List (RuntimeExprRef .integer)}
      {indexValue : Int}
      {branchRef : RuntimeExprRef .integer}
      {branch : RuntimeExpr .integer}
      {value : Int}
      (indexDenotes : RuntimeIntExpr.DenotesExecuted stage arena current index indexValue)
      (nonnegative : 0 ≤ indexValue)
      (selected : branches[indexValue.toNat]? = some branchRef)
      (arenaLookup : arena.lookupInteger branchRef = some branch)
      (branchDenotes : RuntimeIntExpr.DenotesExecuted stage arena current branch value) :
      RuntimeIntExpr.DenotesExecuted stage arena current (.select .integer index branches) value

/-- Exact Boolean expression semantics indexed by the same actual scope and arena. -/
inductive RuntimeBoolExpr.DenotesExecuted
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    (stage : StageId)
    (arena : ExpressionArena)
    (current : ExecutedScope samplers program) : RuntimeExpr .boolean → Bool → Prop where
  | boolWire
      {wire : ValueInstanceRef}
      {value : Bool}
      (found : current.ValueAt stage wire (.boolean value)) :
      RuntimeBoolExpr.DenotesExecuted stage arena current (.boolWire wire) value
  | boolConstant (value : Bool) :
      RuntimeBoolExpr.DenotesExecuted stage arena current (.boolConstant value) value
  | compare
      {operation : IntCompareOp}
      {left right : RuntimeExpr .integer}
      {leftValue rightValue : Int}
      (leftDenotes : RuntimeIntExpr.DenotesExecuted stage arena current left leftValue)
      (rightDenotes : RuntimeIntExpr.DenotesExecuted stage arena current right rightValue) :
      RuntimeBoolExpr.DenotesExecuted stage arena current (.compare operation left right)
        (Mxx.Ir.evaluateIntCompare operation leftValue rightValue)
  | bitExtract
      {expression : RuntimeExpr .integer}
      {value : Int}
      {position : IntExpr}
      {positionValue : Int}
      (input : RuntimeIntExpr.DenotesExecuted stage arena current expression value)
      (positionEvaluates : position.evaluate current.params = some positionValue)
      (nonnegative : 0 ≤ positionValue) :
      RuntimeBoolExpr.DenotesExecuted stage arena current (.bitExtract expression position)
        (((value / (2 ^ positionValue.toNat)) % 2) ≠ 0)
  | thresholdDecodeBool
      {matrix : ValueInstanceRef}
      {matrixValue : Mxx.Matrix}
      {ciphertextModulus plaintextModulus position : IntExpr}
      {q p index coefficient : Int}
      (matrixFound : current.ValueAt stage matrix (.matrix matrixValue))
      (qEvaluates : ciphertextModulus.evaluate current.params = some q)
      (pEvaluates : plaintextModulus.evaluate current.params = some p)
      (positionEvaluates : position.evaluate current.params = some index)
      (nonnegative : 0 ≤ index)
      (coefficientFound : matrixValue.coefficients[index.toNat]? = some coefficient) :
      RuntimeBoolExpr.DenotesExecuted stage arena current
        (.thresholdDecodeBool matrix ciphertextModulus plaintextModulus position)
        (Mxx.Ir.thresholdDecodeBool q p coefficient)
  | select
      {index : RuntimeExpr .integer}
      {branches : List (RuntimeExprRef .boolean)}
      {indexValue : Int}
      {branchRef : RuntimeExprRef .boolean}
      {branch : RuntimeExpr .boolean}
      {value : Bool}
      (indexDenotes : RuntimeIntExpr.DenotesExecuted stage arena current index indexValue)
      (nonnegative : 0 ≤ indexValue)
      (selected : branches[indexValue.toNat]? = some branchRef)
      (arenaLookup : arena.lookupBoolean branchRef = some branch)
      (branchDenotes : RuntimeBoolExpr.DenotesExecuted stage arena current branch value) :
      RuntimeBoolExpr.DenotesExecuted stage arena current (.select .boolean index branches) value

end

/-- The selector wire's integer is obtained directly from the actual lane child environment. -/
theorem CheckedRecurrenceLaneOutput.CandidateFrame.gateSelectorValue
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    (candidateFrame : lane.CandidateFrame execution position)
    (selectorValue : Int)
    (found : Mxx.Ir.lookupWire lane.gateSelection.gateSelector
      candidateFrame.edge.child.execution.wires = some (.integer selectorValue)) :
    FrozenPointwiseScalarFormula.DenotesExecuted candidateFrame.edge.child
      (.atom lane.gateSelection.gateSelector) (.integer selectorValue) :=
  .atom lane.gateSelection.gateSelector (.integer selectorValue) found

/-- Compute the actual gate-selector integer directly from the selected child execution.  Failure
means the executable selector wire is absent or has the wrong runtime type. -/
def CheckedRecurrenceLaneOutput.CandidateFrame.gateSelectorExecution?
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    (candidateFrame : lane.CandidateFrame execution position) :
    Option (FrozenPointwiseScalarFormula.ExecutedIntegerResult candidateFrame.edge.child
      (.atom lane.gateSelection.gateSelector)) :=
  FrozenPointwiseScalarFormula.executeInteger? candidateFrame.edge.child
    (.atom lane.gateSelection.gateSelector)

def bggBooleanGates : List BggBooleanGate :=
  [.zero, .one, .copyLeft, .notLeft, .and, .xor]

/-- The exact gate and retained matrix candidate selected by the actual executable selector. -/
structure CheckedRecurrenceLaneOutput.CandidateFrame.ExecutedGateSelection
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    (candidateFrame : lane.CandidateFrame execution position) : Type where
  selector : FrozenPointwiseScalarFormula.ExecutedIntegerResult candidateFrame.edge.child
    (.atom lane.gateSelection.gateSelector)
  gate : BggBooleanGate
  gateFound : bggBooleanGates[selector.value.toNat]? = some gate
  candidate : FrozenPointwiseMatrixProgramFormula
  candidateFound : lane.gateCandidateProgramFormulas[selector.value.toNat]? = some candidate

/-- Resolve gate selection entirely by executing the real selector wire and indexing the two
checker-fixed six-way lists at that same value. -/
def CheckedRecurrenceLaneOutput.CandidateFrame.executedGateSelection?
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    {execution : lane.Execution scopeExecution}
    {position : Nat}
    (candidateFrame : lane.CandidateFrame execution position) :
    Option candidateFrame.ExecutedGateSelection :=
  match candidateFrame.gateSelectorExecution? with
  | none => none
  | some selector =>
      match gateFound : bggBooleanGates[selector.value.toNat]? with
      | none => none
      | some gate =>
          match candidateFound : lane.gateCandidateProgramFormulas[selector.value.toNat]? with
          | none => none
          | some candidate => some {
              selector
              gate
              gateFound
              candidate
              candidateFound
            }

end Mxx.Certificate
