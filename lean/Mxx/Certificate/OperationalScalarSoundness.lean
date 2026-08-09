import Mxx.Ir.ExecutionFacts

namespace Mxx.Certificate

/-! Exact executable scalar/control facts used by the operational proof path. -/

theorem constantIntNode_execution
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

theorem evaluateIntNode_execution
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (expression : Mxx.Ir.IntExpr)
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

theorem constantBoolNode_execution
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

theorem boolToIntNode_execution
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

theorem selectNode_execution
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

theorem intBinaryNode_execution
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (operation : Mxx.Ir.IntBinaryOp)
    (left right result : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
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

theorem intCompareNode_execution
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (operation : Mxx.Ir.IntCompareOp)
    (left right : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [leftRef, rightRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.integer left, .integer right])
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .intCompare operation
      arguments := [leftRef, rightRef]
      outputCount
    }) :
    values = [.boolean (Mxx.Ir.evaluateIntCompare operation left right)] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate] using member

theorem bitExtractNode_execution
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (position : Mxx.Ir.IntExpr)
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

theorem extractCoefficientNode_execution
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (position : Mxx.Ir.IntExpr)
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

end Mxx.Certificate
