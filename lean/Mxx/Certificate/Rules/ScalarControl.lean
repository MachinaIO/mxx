import Mxx.Certificate.Facts
import Mxx.Ir.ExecutionFacts

namespace Mxx.Certificate

/-- Closed scalar/control subset required by the current Diamond workflow. `IntCompare.less` is
deliberately absent and therefore cannot acquire semantics through this rule module. -/
inductive ScalarControlRule where
  | intAdd
  | intSubtract
  | intMultiply
  | intDivide
  | intRemainder
  | intEqual
  | intLessEqual
  | bitExtract
  | extractCoefficient
  deriving BEq, DecidableEq, Repr

inductive ScalarControlRuleError where
  | unsupportedNodeKind (kind : Mxx.Ir.NodeKind)

def inferScalarControlRule : Mxx.Ir.NodeKind → Except ScalarControlRuleError ScalarControlRule
  | .intBinary .add => .ok .intAdd
  | .intBinary .subtract => .ok .intSubtract
  | .intBinary .multiply => .ok .intMultiply
  | .intBinary .divide => .ok .intDivide
  | .intBinary .remainder => .ok .intRemainder
  | .intCompare .equal => .ok .intEqual
  | .intCompare .lessEqual => .ok .intLessEqual
  | .bitExtract _ => .ok .bitExtract
  | .extractCoefficient _ => .ok .extractCoefficient
  | kind => .error (.unsupportedNodeKind kind)

def ScalarControlRule.intBinaryOperation : ScalarControlRule → Option Mxx.Ir.IntBinaryOp
  | .intAdd => some .add
  | .intSubtract => some .subtract
  | .intMultiply => some .multiply
  | .intDivide => some .divide
  | .intRemainder => some .remainder
  | _ => none

def ScalarControlRule.compareOperation : ScalarControlRule → Option Mxx.Ir.IntCompareOp
  | .intEqual => some .equal
  | .intLessEqual => some .lessEqual
  | _ => none

/-- Range evidence that must be discharged before the analyzer may install an `IntegerFact`.

Keeping the requirement in the typed output prevents a future caller from treating the convenient
nonnegative formulas for multiplication, division, remainder, or coefficient extraction as
unconditional interval bounds. -/
inductive ScalarRangeRequirement where
  | multiplicationInterval (left right : IntegerFact)
  | divisionInterval (dividend divisor : IntegerFact)
  | remainderInterval (dividend divisor : IntegerFact)

/-- Typed scalar result using the accepted `IntBoundExpr` foundation. Add/subtract are
unconditional; multiply requires ordered nonnegative inputs; divide/remainder additionally require
a positive divisor. No caller-provided bound is accepted. -/
inductive ScalarControlOutput where
  | integer
      (expression : RuntimeExpr .integer)
      (lower upper : IntBoundExpr)
  | integerPending
      (expression : RuntimeExpr .integer)
      (requirement : ScalarRangeRequirement)
  | boolean (expression : RuntimeExpr .boolean)

def deriveIntBinaryOutput
    (rule : ScalarControlRule)
    (left right : IntegerFact) : Except ScalarControlRuleError ScalarControlOutput := do
  let operation ← match rule.intBinaryOperation with
    | some operation => pure operation
    | none => throw (.unsupportedNodeKind (.intBinary .add))
  let expression := RuntimeExpr.intBinary operation left.expression right.expression
  match rule with
  | .intAdd => return .integer expression (.add left.lower right.lower) (.add left.upper right.upper)
  | .intSubtract =>
      return .integer expression (.subtract left.lower right.upper) (.subtract left.upper right.lower)
  | .intMultiply =>
      return .integer expression (.multiply left.lower right.lower)
        (.multiply left.upper right.upper)
  | .intDivide =>
      return .integer expression (.divide left.lower right.upper)
        (.divide left.upper right.lower)
  | .intRemainder =>
      return .integer expression (.integer (.constant 0))
        (.subtract right.upper (.integer (.constant 1)))
  | _ => throw (.unsupportedNodeKind (.intBinary operation))

def deriveCompareOutput
    (rule : ScalarControlRule)
    (left right : IntegerFact) : Except ScalarControlRuleError ScalarControlOutput := do
  let operation ← match rule.compareOperation with
    | some operation => pure operation
    | none => throw (.unsupportedNodeKind (.intCompare .equal))
  return .boolean (.compare operation left.expression right.expression)

def deriveBitExtractOutput
    (input : IntegerFact) (position : IntExpr) : ScalarControlOutput :=
  .boolean (.bitExtract input.expression position)

def deriveExtractCoefficientOutput
    (matrix : MatrixExprRef) (position : IntExpr)
    (modulus : IntExpr) : ScalarControlOutput :=
  .integer
    (.extractCoefficient matrix position)
    (.integer (.constant 0))
    (.subtract (.integer modulus) (.integer (.constant 1)))

/-- Executable constant-integer semantics.  Keeping this alongside the other scalar leaves lets
the workflow soundness induction discharge the frozen node directly, without a per-node callback. -/
theorem constantIntNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (value : Int)
    (outputCount : Nat)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .constantInt value
      arguments := []
      outputCount
    }) :
    values = [.integer value] := by
  simpa [Mxx.Ir.evaluateNode] using member

/-- Executable parameter-expression semantics. -/
theorem evaluateIntNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (expression : IntExpr)
    (value : Int)
    (outputCount : Nat)
    (evaluates : expression.evaluate params = some value)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .evaluateInt expression
      arguments := []
      outputCount
    }) :
    values = [.integer value] := by
  simpa [Mxx.Ir.evaluateNode, evaluates] using member

/-- Executable constant-Boolean semantics. -/
theorem constantBoolNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (value : Bool)
    (outputCount : Nat)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .constantBool value
      arguments := []
      outputCount
    }) :
    values = [.boolean value] := by
  simpa [Mxx.Ir.evaluateNode] using member

/-- Executable Boolean-to-integer conversion semantics. -/
theorem boolToIntNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (value : Bool)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.boolean value])
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .boolToInt
      arguments := [inputRef]
      outputCount
    }) :
    values = [.integer (if value then 1 else 0)] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member

/-- Executable select semantics shared by integer, Boolean, and matrix selections.  The analyzer
still performs the type-specific fact derivation; this theorem only fixes the selected runtime
branch from the real node support. -/
theorem selectNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (indexRef : Mxx.Ir.WireRef)
    (branchRefs : List Mxx.Ir.WireRef)
    (index : Int)
    (branches : List Mxx.Ir.Value)
    (outputCount : Nat)
    (argumentsEvaluate : (indexRef :: branchRefs).mapM
      (fun wire => Mxx.Ir.lookupWire wire wires) = some (.integer index :: branches))
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .select
      arguments := indexRef :: branchRefs
      outputCount
    }) :
    values = [branches[index.toNat]?.getD (.invalid "Select index out of range")] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member

theorem intBinaryNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (operation : Mxx.Ir.IntBinaryOp)
    (left right result : Int)
    (outputCount : Nat)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
        some [.integer left, .integer right])
    (operationEvaluate : Mxx.Ir.evaluateIntBinary operation left right = some result)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .intBinary operation
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.integer result] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, operationEvaluate] using member

theorem intCompareNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (operation : Mxx.Ir.IntCompareOp)
    (left right : Int)
    (outputCount : Nat)
    (argumentsEvaluate :
      [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
        some [.integer left, .integer right])
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .intCompare operation
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.boolean (Mxx.Ir.evaluateIntCompare operation left right)] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member

theorem bitExtractNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (position : IntExpr)
    (value evaluatedPosition : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.integer value])
    (positionEvaluate : position.evaluate params = some evaluatedPosition)
    (nonnegative : 0 ≤ evaluatedPosition)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .bitExtract position
      arguments := [inputRef]
      outputCount
    }) :
    values = [.boolean (((value / (2 ^ evaluatedPosition.toNat)) % 2) ≠ 0)] := by
  simp only [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, positionEvaluate,
    if_neg (not_lt.mpr nonnegative)] at member
  simpa only [List.mem_singleton] using member

theorem extractCoefficientNode_local_sound
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (position : IntExpr)
    (matrix : Mxx.Matrix)
    (evaluatedPosition : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix matrix])
    (positionEvaluate : position.evaluate params = some evaluatedPosition)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .extractCoefficient position
      arguments := [inputRef]
      outputCount
    }) :
    values = [.integer (Mxx.reduceCoefficient matrix.modulus
      (matrix.coefficients.getD evaluatedPosition.toNat 0))] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, positionEvaluate] using member

theorem intAdd_interval_sound
    {left right leftLower leftUpper rightLower rightUpper : Int}
    (leftInRange : leftLower ≤ left ∧ left ≤ leftUpper)
    (rightInRange : rightLower ≤ right ∧ right ≤ rightUpper) :
    leftLower + rightLower ≤ left + right ∧ left + right ≤ leftUpper + rightUpper := by
  omega

theorem intSubtract_interval_sound
    {left right leftLower leftUpper rightLower rightUpper : Int}
    (leftInRange : leftLower ≤ left ∧ left ≤ leftUpper)
    (rightInRange : rightLower ≤ right ∧ right ≤ rightUpper) :
    leftLower - rightUpper ≤ left - right ∧ left - right ≤ leftUpper - rightLower := by
  omega

/-- Accepted v4 nonnegative multiplication interval. General signed multiplication remains
unsupported rather than guessing endpoint signs. -/
theorem intMultiply_nonnegative_interval_sound
    {left right leftLower leftUpper rightLower rightUpper : Int}
    (leftInRange : leftLower ≤ left ∧ left ≤ leftUpper)
    (rightInRange : rightLower ≤ right ∧ right ≤ rightUpper)
    (leftNonnegative : 0 ≤ leftLower)
    (rightNonnegative : 0 ≤ rightLower) :
    leftLower * rightLower ≤ left * right ∧ left * right ≤ leftUpper * rightUpper := by
  constructor
  · have first := mul_nonneg (sub_nonneg.mpr leftInRange.1)
      (le_trans rightNonnegative rightInRange.1)
    have second := mul_nonneg leftNonnegative (sub_nonneg.mpr rightInRange.1)
    nlinarith
  · have first := mul_nonneg (sub_nonneg.mpr leftInRange.2)
      (le_trans (le_trans rightNonnegative rightInRange.1) rightInRange.2)
    have second := mul_nonneg (le_trans leftNonnegative leftInRange.1)
      (sub_nonneg.mpr rightInRange.2)
    nlinarith

/-- Euclidean integer division is monotone on the closed nonnegative/positive fragment used by
Diamond's parameter arithmetic. -/
private theorem ediv_antitone_positive_denominator
    {value smaller larger : Int}
    (valueNonnegative : 0 ≤ value)
    (smallerPositive : 0 < smaller)
    (ordered : smaller ≤ larger) :
    value / larger ≤ value / smaller := by
  have largerPositive : 0 < larger := lt_of_lt_of_le smallerPositive ordered
  apply Int.le_ediv_of_mul_le smallerPositive
  calc
    value / larger * smaller ≤ value / larger * larger :=
      mul_le_mul_of_nonneg_left ordered
        (Int.ediv_nonneg valueNonnegative largerPositive.le)
    _ ≤ value := Int.ediv_mul_le value (Int.ne_of_gt largerPositive)

theorem intDivide_nonnegative_interval_sound
    {dividend divisor dividendLower dividendUpper divisorLower divisorUpper : Int}
    (dividendInRange : dividendLower ≤ dividend ∧ dividend ≤ dividendUpper)
    (divisorInRange : divisorLower ≤ divisor ∧ divisor ≤ divisorUpper)
    (dividendNonnegative : 0 ≤ dividendLower)
    (divisorPositive : 0 < divisorLower) :
    dividendLower / divisorUpper ≤ dividend / divisor ∧
      dividend / divisor ≤ dividendUpper / divisorLower := by
  have divisorPositive' : 0 < divisor := lt_of_lt_of_le divisorPositive divisorInRange.1
  have divisorUpperPositive : 0 < divisorUpper :=
    lt_of_lt_of_le divisorPositive (le_trans divisorInRange.1 divisorInRange.2)
  have dividendNonnegative' : 0 ≤ dividend :=
    le_trans dividendNonnegative dividendInRange.1
  have dividendUpperNonnegative : 0 ≤ dividendUpper :=
    le_trans dividendNonnegative' dividendInRange.2
  constructor
  · exact le_trans
      (Int.ediv_le_ediv divisorUpperPositive dividendInRange.1)
      (ediv_antitone_positive_denominator dividendNonnegative' divisorPositive'
        divisorInRange.2)
  · exact le_trans
      (Int.ediv_le_ediv divisorPositive' dividendInRange.2)
      (ediv_antitone_positive_denominator dividendUpperNonnegative divisorPositive
        divisorInRange.1)

/-- A nonnegative dividend modulo a positive divisor lies in the canonical remainder interval. -/
theorem intRemainder_nonnegative_interval_sound
    {dividend divisor divisorUpper : Int}
    (_dividendNonnegative : 0 ≤ dividend)
    (divisorPositive : 0 < divisor)
    (divisorUpperBound : divisor ≤ divisorUpper) :
    0 ≤ dividend % divisor ∧ dividend % divisor ≤ divisorUpper - 1 := by
  constructor
  · exact Int.emod_nonneg _ (Int.ne_of_gt divisorPositive)
  · have remainderLt := Int.emod_lt_of_pos dividend divisorPositive
    omega

example : inferScalarControlRule (.intCompare .less) =
    .error (.unsupportedNodeKind (.intCompare .less)) := rfl

private def fixtureInteger (wireName : String) (lower upper : Int) : IntegerFact := {
  expression := .intWire (.protocolInput ⟨wireName⟩)
  lower := .integer (.constant lower)
  upper := .integer (.constant upper)
}

example : deriveIntBinaryOutput .intAdd
    (fixtureInteger "left" 1 3) (fixtureInteger "right" 4 8) =
    .ok (.integer
      (.intBinary .add
        (.intWire (.protocolInput ⟨"left"⟩))
        (.intWire (.protocolInput ⟨"right"⟩)))
      (.add (.integer (.constant 1)) (.integer (.constant 4)))
      (.add (.integer (.constant 3)) (.integer (.constant 8)))) := rfl

/-- Closed ordered nonnegative intervals are discharged during Phase A. -/
example : deriveIntBinaryOutput .intMultiply
    (fixtureInteger "left" 1 3) (fixtureInteger "right" 4 8) =
    .ok (.integer
      (.intBinary .multiply
        (.intWire (.protocolInput ⟨"left"⟩))
        (.intWire (.protocolInput ⟨"right"⟩)))
      (.multiply (.integer (.constant 1)) (.integer (.constant 4)))
      (.multiply (.integer (.constant 3)) (.integer (.constant 8)))) := rfl

/-- Phase A constructs the unique candidate interval; Phase B rejects this fixture because its
derived nonnegativity obligation is false. -/
example : deriveIntBinaryOutput .intMultiply
    (fixtureInteger "left" (-1) 3) (fixtureInteger "right" 4 8) =
    .ok (.integer
      (.intBinary .multiply
        (.intWire (.protocolInput ⟨"left"⟩))
        (.intWire (.protocolInput ⟨"right"⟩)))
      (.multiply (.integer (.constant (-1))) (.integer (.constant 4)))
      (.multiply (.integer (.constant 3)) (.integer (.constant 8)))) := rfl

example : deriveExtractCoefficientOutput ⟨7⟩ (.constant 0)
    (.constant 17) = .integer
      (.extractCoefficient ⟨7⟩ (.constant 0))
      (.integer (.constant 0))
      (.subtract (.integer (.constant 17)) (.integer (.constant 1))) := rfl

end Mxx.Certificate
