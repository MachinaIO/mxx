import Mxx.Certificate.LocalSoundness

namespace Mxx.Certificate

/-!
# Nested sequential-loop trace extraction

This module provides the execution-only part of nested recurrence evidence.  A child execution
path is recovered from the exact support member stored by its parent trace, and a sequential-loop
trace is recovered from one exact node member on that path.  Neither interface accepts a
replacement child runner, execution trace, or invariant-argument list.
-/

/-- An execution path through the frozen definition selected by a parent child-support member.
The child runner used by the path is definitionally the predecessor of the runner containing that
member. -/
structure ChildScopeExecutionPath
    (samplers : Mxx.MxxSamplerFamily)
    (program : Mxx.Ir.Prog)
    (fuel : Nat)
    (definition : String)
    (params : Mxx.Ir.ParamEnvironment)
    (arguments values : List Mxx.Ir.Value) where
  scope : Mxx.Ir.Scope
  definitionFound : Mxx.Ir.lookupDefinition definition program.definitions = some scope
  wires : Mxx.Ir.WireEnvironment
  path : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scope.inputNames.zip arguments) 0 scope.nodes [] wires
  outputEq : values = (Mxx.Ir.collectOutputs scope.outputs wires).map Prod.snd

/-- Recover the exact child-scope path selected by a parent trace.  In particular, `fuel`,
`definition`, `params`, and `arguments` occur in the support member itself and cannot be replaced
by fields of the returned evidence. -/
theorem ChildScopeExecutionPath.nonempty_of_childMember
    (samplers : Mxx.MxxSamplerFamily)
    (program : Mxx.Ir.Prog)
    (fuel : Nat)
    (definition : String)
    (scope : Mxx.Ir.Scope)
    (params : Mxx.Ir.ParamEnvironment)
    (arguments values : List Mxx.Ir.Value)
    (definitionFound : Mxx.Ir.lookupDefinition definition program.definitions = some scope)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers program (fuel + 1)
      definition params arguments) :
    Nonempty (ChildScopeExecutionPath samplers program fuel definition params arguments values) := by
  obtain ⟨wires, path, outputEq⟩ :=
    (Mxx.Ir.mem_childRunnerWithFuel_succ_iff_path samplers program fuel definition scope params
      arguments values definitionFound).mp childMember
  exact ⟨{ scope, definitionFound, wires, path, outputEq }⟩

/-- The sequential-loop fields of one exact node.  `nodeEq` ties every field, including its
argument wires, to the frozen node selected from the child execution path. -/
structure SequentialLoopNodeView (node : Mxx.Ir.Node) where
  definition : String
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  carriedCount : Nat
  argumentRefs : List Mxx.Ir.WireRef
  outputCount : Nat
  outputTypes : List Mxx.Ir.WireTypeExpr
  nodeEq : node = {
    kind := .sequentialLoop definition count indexSlot bindings carriedCount
    arguments := argumentRefs
    outputCount
    outputTypes
  }

/-- The deterministic projection of one node member from an execution path.  This packages the
existential returned by `EvaluatesNodesPath.atNodeIndex`, so all later premises refer to the same
selected pre-node environment and support member. -/
structure SelectedNodeExecution
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length) where
  before : Mxx.Ir.WireEnvironment
  nodeValues : List Mxx.Ir.Value
  prefixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) 0
    (scopeExecution.scope.nodes.take nodeIndex) [] before
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) before scopeExecution.scope.nodes[nodeIndex]
  suffixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) (nodeIndex + 1)
    (scopeExecution.scope.nodes.drop (nodeIndex + 1))
    (before ++ Mxx.Ir.bindOutputs nodeIndex nodeValues) scopeExecution.wires

/-- Mechanically select one node from the stored execution path. -/
noncomputable def ChildScopeExecutionPath.nodeExecutionAt
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length) :
    SelectedNodeExecution scopeExecution nodeIndex nodeInBounds := by
  classical
  let selected := scopeExecution.path.atNodeIndex nodeIndex nodeInBounds
  let before := Classical.choose selected
  let afterBefore := Classical.choose_spec selected
  let nodeValues := Classical.choose afterBefore
  let facts := Classical.choose_spec afterBefore
  exact {
    before
    nodeValues
    prefixPath := facts.1
    nodeMember := facts.2.1
    suffixPath := by
      dsimp [before, nodeValues]
      simpa only [Nat.zero_add] using facts.2.2
  }

/-- Every output of a mechanically selected node is the value of its exact SSA wire in the
completed child-scope environment.  Freshness is derived from the prefix path; callers do not
supply it. -/
theorem SelectedNodeExecution.outputLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    {scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues}
    {nodeIndex : Nat}
    {nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length}
    (execution : SelectedNodeExecution scopeExecution nodeIndex nodeInBounds)
    (port : Nat)
    (portInBounds : port < execution.nodeValues.length) :
    Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires =
      some execution.nodeValues[port] := by
  have prefixLength : (scopeExecution.scope.nodes.take nodeIndex).length = nodeIndex := by
    simp [List.length_take, nodeInBounds.le]
  have missing : Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ execution.before = none := by
    apply execution.prefixPath.lookupWire_after_end nodeIndex port
    · simp [prefixLength]
    · rfl
  apply execution.suffixPath.lookupWire_preserved
  exact Mxx.Ir.lookupWire_append_bindOutputs missing portInBounds

/-- A value of an earlier SSA wire in the completed child scope is already present immediately
before the selected node.  The ordering premise is supplied by the closed formula validator. -/
theorem SelectedNodeExecution.argumentLookup_of_final
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    {scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues}
    {nodeIndex : Nat}
    {nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length}
    (execution : SelectedNodeExecution scopeExecution nodeIndex nodeInBounds)
    (wire : Mxx.Ir.WireRef)
    (value : Mxx.Ir.Value)
    (earlier : wire.node < nodeIndex)
    (found : Mxx.Ir.lookupWire wire scopeExecution.wires = some value) :
    Mxx.Ir.lookupWire wire execution.before = some value := by
  have throughSuffix := execution.suffixPath.lookupWire_before_start wire.node wire.port (by
    omega)
  rw [found] at throughSuffix
  cases beforeFound : Mxx.Ir.lookupWire wire execution.before with
  | some existing =>
      have appended : Mxx.Ir.lookupWire wire
          (execution.before ++ Mxx.Ir.bindOutputs nodeIndex execution.nodeValues) =
          some existing := Mxx.Ir.lookupWire_append_of_eq_some beforeFound
      rw [appended] at throughSuffix
      have valuesEqual : value = existing := Option.some.inj throughSuffix
      rw [valuesEqual]
  | none =>
      have appended : Mxx.Ir.lookupWire wire
          (execution.before ++ Mxx.Ir.bindOutputs nodeIndex execution.nodeValues) = none := by
        rw [Mxx.Ir.lookupWire_append_of_eq_none beforeFound]
        exact Mxx.Ir.lookupWire_bindOutputs_of_node_ne nodeIndex wire.node wire.port
          execution.nodeValues (by omega)
      rw [appended] at throughSuffix
      contradiction

/-- Conversely, any completed-environment value at the selected node comes from an in-bounds
port of the exact support member chosen for that node. -/
theorem SelectedNodeExecution.outputValue_of_lookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    {scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues}
    {nodeIndex : Nat}
    {nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length}
    (execution : SelectedNodeExecution scopeExecution nodeIndex nodeInBounds)
    (port : Nat)
    (value : Mxx.Ir.Value)
    (found : Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires = some value) :
    ∃ portInBounds : port < execution.nodeValues.length,
      execution.nodeValues[port] = value := by
  have throughSuffix := execution.suffixPath.lookupWire_before_start nodeIndex port (by omega)
  rw [found] at throughSuffix
  have missing : Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ execution.before = none := by
    apply execution.prefixPath.lookupWire_after_end nodeIndex port
    · simp [List.length_take, nodeInBounds.le]
    · rfl
  rw [Mxx.Ir.lookupWire_append_of_eq_none missing] at throughSuffix
  by_cases portInBounds : port < execution.nodeValues.length
  · refine ⟨portInBounds, ?_⟩
    rw [Mxx.Ir.lookupWire_bindOutputs nodeIndex port execution.nodeValues portInBounds]
      at throughSuffix
    exact Option.some.inj throughSuffix.symm
  · have outputMissing : Mxx.Ir.lookupWire ⟨nodeIndex, port⟩
        (Mxx.Ir.bindOutputs nodeIndex execution.nodeValues) = none := by
      exact Mxx.Ir.lookupWire_bindOutputs_eq_none_of_not_lt nodeIndex port
        execution.nodeValues portInBounds
    rw [outputMissing] at throughSuffix
    contradiction

/-- Close a deterministic single-matrix node once its exact evaluator result has been derived
from the selected pre-node environment. -/
theorem ChildScopeExecutionPath.deterministicMatrixOutputLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (baseNode : Mxx.Ir.Node)
    (output : Mxx.Matrix)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      baseNode with outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes })
    (evaluation : Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments)
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before baseNode =
        [[.matrix output]]) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix output) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have member : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before baseNode := by
    have typedMember := selected.nodeMember
    rw [nodeEq] at typedMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before baseNode
      scopeExecution.scope.nodes[nodeIndex].outputTypes]
    exact typedMember
  rw [evaluation] at member
  have valuesEq : selected.nodeValues = [.matrix output] := by simpa using member
  have outputLookup := selected.outputLookup 0 (by simp [valuesEq])
  simpa [valuesEq] using outputLookup

/-- Local runtime equation for an exact matrix-addition node selected from a completed child
scope. -/
theorem ChildScopeExecutionPath.matrixAddLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (leftEarlier : leftRef.node < nodeIndex)
    (rightEarlier : rightRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .matrixAdd
      arguments := [leftRef, rightRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (leftFound : Mxx.Ir.lookupWire leftRef scopeExecution.wires = some (.matrix left))
    (rightFound : Mxx.Ir.lookupWire rightRef scopeExecution.wires = some (.matrix right)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.matrixAdd left right)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have leftBefore := selected.argumentLookup_of_final leftRef (.matrix left) leftEarlier leftFound
  have rightBefore := selected.argumentLookup_of_final rightRef (.matrix right) rightEarlier
    rightFound
  have argumentsEvaluate : [leftRef, rightRef].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire selected.before) =
      some [.matrix left, .matrix right] := by
    simp [leftBefore, rightBefore]
  have member : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .matrixAdd
        arguments := [leftRef, rightRef]
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } := by
    have typedMember := selected.nodeMember
    rw [nodeEq] at typedMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .matrixAdd
        arguments := [leftRef, rightRef]
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } scopeExecution.scope.nodes[nodeIndex].outputTypes]
    exact typedMember
  have valuesEq := Mxx.Ir.mem_evaluateNode_matrixAdd_of_arguments
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) selected.before leftRef rightRef left right
    scopeExecution.scope.nodes[nodeIndex].outputCount argumentsEvaluate member
  have outputLookup := selected.outputLookup 0 (by simp [valuesEq])
  simpa [valuesEq] using outputLookup

theorem ChildScopeExecutionPath.matrixSubtractLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (leftEarlier : leftRef.node < nodeIndex)
    (rightEarlier : rightRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .matrixSubtract
      arguments := [leftRef, rightRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (leftFound : Mxx.Ir.lookupWire leftRef scopeExecution.wires = some (.matrix left))
    (rightFound : Mxx.Ir.lookupWire rightRef scopeExecution.wires = some (.matrix right)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.matrixSubtract left right)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have leftBefore := selected.argumentLookup_of_final leftRef (.matrix left) leftEarlier leftFound
  have rightBefore := selected.argumentLookup_of_final rightRef (.matrix right) rightEarlier
    rightFound
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .matrixSubtract
    arguments := [leftRef, rightRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixSubtract left right) nodeEq
  change Mxx.Ir.evaluateNode _ _ _ _ selected.before _ = _
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, leftBefore, rightBefore]

theorem ChildScopeExecutionPath.matrixMultiplyLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (leftRef rightRef : Mxx.Ir.WireRef)
    (left right : Mxx.Matrix)
    (leftEarlier : leftRef.node < nodeIndex)
    (rightEarlier : rightRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .matrixMultiply
      arguments := [leftRef, rightRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (leftFound : Mxx.Ir.lookupWire leftRef scopeExecution.wires = some (.matrix left))
    (rightFound : Mxx.Ir.lookupWire rightRef scopeExecution.wires = some (.matrix right)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.matrixMultiply left right)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have leftBefore := selected.argumentLookup_of_final leftRef (.matrix left) leftEarlier leftFound
  have rightBefore := selected.argumentLookup_of_final rightRef (.matrix right) rightEarlier
    rightFound
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .matrixMultiply
    arguments := [leftRef, rightRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixMultiply left right) nodeEq
  change Mxx.Ir.evaluateNode _ _ _ _ selected.before _ = _
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, leftBefore, rightBefore]

theorem ChildScopeExecutionPath.matrixNegateLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef)
    (input : Mxx.Matrix)
    (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .matrixNegate
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.matrixNegate input)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  have inputBefore' : Mxx.Ir.lookupWire inputRef
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before = some (.matrix input) := by
    simpa only [selected] using inputBefore
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .matrixNegate
    arguments := [inputRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixNegate input) nodeEq
  change Mxx.Ir.evaluateNode _ _ _ _ selected.before _ = _
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, inputBefore]

theorem ChildScopeExecutionPath.matrixScaleLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef)
    (input : Mxx.Matrix)
    (scalar : Mxx.Ir.IntExpr)
    (scalarValue : Int)
    (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .matrixScale scalar
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input))
    (scalarEvaluates : scalar.evaluate params = some scalarValue) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.matrixScale scalarValue input)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  have inputBefore' : Mxx.Ir.lookupWire inputRef
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before = some (.matrix input) := by
    simpa only [selected] using inputBefore
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .matrixScale scalar
    arguments := [inputRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixScale scalarValue input) nodeEq
  change Mxx.Ir.evaluateNode _ _ _ _ selected.before _ = _
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, inputBefore, scalarEvaluates]

theorem ChildScopeExecutionPath.matrixReshapeLookup
    {samplers : Mxx.MxxSamplerFamily} {program : Mxx.Ir.Prog} {fuel : Nat}
    {parentDefinition : String} {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat) (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef) (input : Mxx.Matrix) (rows columns : Mxx.Ir.IntExpr)
    (rowValue columnValue : Int) (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .reshape rows columns
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input))
    (rowsEvaluate : rows.evaluate params = some rowValue)
    (columnsEvaluate : columns.evaluate params = some columnValue)
    (rowsNonnegative : 0 ≤ rowValue) (columnsNonnegative : 0 ≤ columnValue) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix
      (Mxx.matrixReshape input rowValue.toNat columnValue.toNat)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  have inputBefore' : Mxx.Ir.lookupWire inputRef
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before = some (.matrix input) := by
    simpa only [selected] using inputBefore
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .reshape rows columns
    arguments := [inputRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixReshape input rowValue.toNat columnValue.toNat) nodeEq
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, inputBefore', rowsEvaluate, columnsEvaluate,
    not_lt.mpr rowsNonnegative, not_lt.mpr columnsNonnegative]

theorem ChildScopeExecutionPath.matrixSliceLookup
    {samplers : Mxx.MxxSamplerFamily} {program : Mxx.Ir.Prog} {fuel : Nat}
    {parentDefinition : String} {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat) (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef) (input : Mxx.Matrix)
    (rowStart rowEnd columnStart columnEnd : Mxx.Ir.IntExpr)
    (rowStartValue rowEndValue columnStartValue columnEndValue : Int)
    (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .slice (some (rowStart, rowEnd)) (some (columnStart, columnEnd))
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input))
    (rowStartEvaluate : rowStart.evaluate params = some rowStartValue)
    (rowEndEvaluate : rowEnd.evaluate params = some rowEndValue)
    (columnStartEvaluate : columnStart.evaluate params = some columnStartValue)
    (columnEndEvaluate : columnEnd.evaluate params = some columnEndValue)
    (rowStartNonnegative : 0 ≤ rowStartValue) (rowOrdered : rowStartValue ≤ rowEndValue)
    (columnStartNonnegative : 0 ≤ columnStartValue)
    (columnOrdered : columnStartValue ≤ columnEndValue) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix
      (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat columnStartValue.toNat
        columnEndValue.toNat)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  have inputBefore' : Mxx.Ir.lookupWire inputRef
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before = some (.matrix input) := by
    simpa only [selected] using inputBefore
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .slice (some (rowStart, rowEnd)) (some (columnStart, columnEnd))
    arguments := [inputRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat columnStartValue.toNat
    columnEndValue.toNat) nodeEq
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, inputBefore', rowStartEvaluate, rowEndEvaluate,
    columnStartEvaluate, columnEndEvaluate, not_lt.mpr rowStartNonnegative,
    not_lt.mpr rowOrdered, not_lt.mpr columnStartNonnegative, not_lt.mpr columnOrdered]

theorem ChildScopeExecutionPath.matrixSliceRowsLookup
    {samplers : Mxx.MxxSamplerFamily} {program : Mxx.Ir.Prog} {fuel : Nat}
    {parentDefinition : String} {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat) (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef) (input : Mxx.Matrix) (rowStart rowEnd : Mxx.Ir.IntExpr)
    (rowStartValue rowEndValue : Int) (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .slice (some (rowStart, rowEnd)) none
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input))
    (rowStartEvaluate : rowStart.evaluate params = some rowStartValue)
    (rowEndEvaluate : rowEnd.evaluate params = some rowEndValue)
    (rowStartNonnegative : 0 ≤ rowStartValue) (rowOrdered : rowStartValue ≤ rowEndValue) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix
      (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat 0 input.columns)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .slice (some (rowStart, rowEnd)) none
    arguments := [inputRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixSlice input rowStartValue.toNat rowEndValue.toNat 0 input.columns) nodeEq
  change Mxx.Ir.evaluateNode _ _ _ _ selected.before _ = _
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, inputBefore, rowStartEvaluate, rowEndEvaluate,
    not_lt.mpr rowStartNonnegative, not_lt.mpr rowOrdered]

theorem ChildScopeExecutionPath.matrixSliceColumnsLookup
    {samplers : Mxx.MxxSamplerFamily} {program : Mxx.Ir.Prog} {fuel : Nat}
    {parentDefinition : String} {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat) (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef) (input : Mxx.Matrix) (columnStart columnEnd : Mxx.Ir.IntExpr)
    (columnStartValue columnEndValue : Int) (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .slice none (some (columnStart, columnEnd))
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input))
    (columnStartEvaluate : columnStart.evaluate params = some columnStartValue)
    (columnEndEvaluate : columnEnd.evaluate params = some columnEndValue)
    (columnStartNonnegative : 0 ≤ columnStartValue)
    (columnOrdered : columnStartValue ≤ columnEndValue) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix
      (Mxx.matrixSlice input 0 input.rows columnStartValue.toNat columnEndValue.toNat)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .slice none (some (columnStart, columnEnd))
    arguments := [inputRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixSlice input 0 input.rows columnStartValue.toNat columnEndValue.toNat) nodeEq
  change Mxx.Ir.evaluateNode _ _ _ _ selected.before _ = _
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, inputBefore, columnStartEvaluate,
    columnEndEvaluate, not_lt.mpr columnStartNonnegative, not_lt.mpr columnOrdered]

theorem ChildScopeExecutionPath.matrixConcatRowsTwoLookup
    {samplers : Mxx.MxxSamplerFamily} {program : Mxx.Ir.Prog} {fuel : Nat}
    {parentDefinition : String} {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat) (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (leftRef rightRef : Mxx.Ir.WireRef) (left right : Mxx.Matrix)
    (leftEarlier : leftRef.node < nodeIndex) (rightEarlier : rightRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .concat .rows
      arguments := [leftRef, rightRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes })
    (leftFound : Mxx.Ir.lookupWire leftRef scopeExecution.wires = some (.matrix left))
    (rightFound : Mxx.Ir.lookupWire rightRef scopeExecution.wires = some (.matrix right)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.matrixConcatRows [left, right])) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have leftBefore := selected.argumentLookup_of_final leftRef (.matrix left) leftEarlier leftFound
  have rightBefore := selected.argumentLookup_of_final rightRef (.matrix right) rightEarlier
    rightFound
  have leftBefore' : Mxx.Ir.lookupWire leftRef
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before = some (.matrix left) := by
    simpa only [selected] using leftBefore
  have rightBefore' : Mxx.Ir.lookupWire rightRef
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before = some (.matrix right) := by
    simpa only [selected] using rightBefore
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .concat .rows
    arguments := [leftRef, rightRef]
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.matrixConcatRows [left, right]) nodeEq
  simp [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, leftBefore', rightBefore']

private theorem SelectedNodeExecution.argumentsLookup_of_final
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel nodeIndex : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    {scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues}
    {nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length}
    (selected : SelectedNodeExecution scopeExecution nodeIndex nodeInBounds)
    (refs : List Mxx.Ir.WireRef)
    (values : List Mxx.Ir.Value)
    (earlier : ∀ ref ∈ refs, ref.node < nodeIndex)
    (found : refs.mapM (fun wire => Mxx.Ir.lookupWire wire scopeExecution.wires) = some values) :
    refs.mapM (fun wire => Mxx.Ir.lookupWire wire selected.before) = some values := by
  induction refs generalizing values with
  | nil => simpa using found
  | cons ref refs induction =>
      cases refFound : Mxx.Ir.lookupWire ref scopeExecution.wires <;>
        simp [refFound] at found
      rename_i value
      cases tailFound : refs.mapM
          (fun wire => Mxx.Ir.lookupWire wire scopeExecution.wires) <;>
        simp [tailFound] at found
      rename_i tail
      subst values
      have refBefore := selected.argumentLookup_of_final ref value (earlier ref (by simp)) refFound
      have tailBefore := induction tail (fun nested nestedMember =>
        earlier nested (by simp [nestedMember])) tailFound
      simp [refBefore, tailBefore]

private theorem lookup_of_mapM_getElem
    (refs : List Mxx.Ir.WireRef)
    (values : List Mxx.Ir.Value)
    (environment : Mxx.Ir.WireEnvironment)
    (found : refs.mapM (fun wire => Mxx.Ir.lookupWire wire environment) = some values)
    (index : Nat)
    (ref : Mxx.Ir.WireRef)
    (refFound : refs[index]? = some ref) :
    ∃ value, values[index]? = some value ∧
      Mxx.Ir.lookupWire ref environment = some value := by
  induction refs generalizing values index with
  | nil => simp at refFound
  | cons head tail induction =>
      cases headFound : Mxx.Ir.lookupWire head environment <;> simp [headFound] at found
      rename_i headValue
      cases tailFound : tail.mapM (fun wire => Mxx.Ir.lookupWire wire environment) <;>
        simp [tailFound] at found
      rename_i tailValues
      subst values
      cases index with
      | zero =>
          simp at refFound
          subst ref
          exact ⟨headValue, by simp, headFound⟩
      | succ index =>
          exact induction tailValues tailFound index (by simpa using refFound)

/-- Exact matrix selected by one runtime integer wire. -/
theorem ChildScopeExecutionPath.matrixSelectLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (indexRef selectedRef : Mxx.Ir.WireRef)
    (branchRefs : List Mxx.Ir.WireRef)
    (branchValues : List Mxx.Ir.Value)
    (index : Int)
    (selectedMatrix : Mxx.Matrix)
    (indexEarlier : indexRef.node < nodeIndex)
    (branchesEarlier : ∀ ref ∈ branchRefs, ref.node < nodeIndex)
    (selectedBranch : branchRefs[index.toNat]? = some selectedRef)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .select
      arguments := indexRef :: branchRefs
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (indexFound : Mxx.Ir.lookupWire indexRef scopeExecution.wires = some (.integer index))
    (branchesFound : branchRefs.mapM
      (fun wire => Mxx.Ir.lookupWire wire scopeExecution.wires) = some branchValues)
    (selectedFound : Mxx.Ir.lookupWire selectedRef scopeExecution.wires =
      some (.matrix selectedMatrix)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix selectedMatrix) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have indexBefore := selected.argumentLookup_of_final indexRef (.integer index) indexEarlier
    indexFound
  have branchesBefore := selected.argumentsLookup_of_final branchRefs branchValues
    branchesEarlier branchesFound
  have member := selected.nodeMember
  rw [nodeEq] at member
  have argumentsEvaluate : (indexRef :: branchRefs).mapM
      (fun wire => Mxx.Ir.lookupWire wire selected.before) =
      some (.integer index :: branchValues) := by
    simp [indexBefore, branchesBefore]
  have valuesEq := Mxx.Ir.mem_evaluateNode_select_of_arguments
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) selected.before (indexRef :: branchRefs)
    index branchValues scopeExecution.scope.nodes[nodeIndex].outputCount argumentsEvaluate member
  have selectedValue : branchValues[index.toNat]? = some (.matrix selectedMatrix) := by
    obtain ⟨selectedValue, selectedValueFound, selectedLookup⟩ :=
      lookup_of_mapM_getElem branchRefs branchValues scopeExecution.wires branchesFound
        index.toNat selectedRef selectedBranch
    rw [selectedFound] at selectedLookup
    cases Option.some.inj selectedLookup
    exact selectedValueFound
  rw [selectedValue] at valuesEq
  have outputLookup := selected.outputLookup 0 (by simp [valuesEq])
  simpa [valuesEq] using outputLookup

theorem ChildScopeExecutionPath.gadgetDecomposeLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (inputRef : Mxx.Ir.WireRef)
    (input : Mxx.Matrix)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (base digitCount : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (baseValue digitCountValue : Int)
    (inputEarlier : inputRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .gadgetDecompose matrixType base digitCount
      arguments := [inputRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (inputFound : Mxx.Ir.lookupWire inputRef scopeExecution.wires = some (.matrix input))
    (typeEvaluates : matrixType.evaluate params (.constant 0) = some matrixParams)
    (baseEvaluates : base.evaluate params = some baseValue)
    (digitCountEvaluates : digitCount.evaluate params = some digitCountValue) :
    ∃ output ∈ samplers.gadgetDecompose matrixParams baseValue digitCountValue.toNat input,
      Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
        some (.matrix (output.withSamplerParams matrixParams)) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have inputBefore := selected.argumentLookup_of_final inputRef (.matrix input) inputEarlier
    inputFound
  have argumentsEvaluate : [inputRef].mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire selected.before) = some [.matrix input] := by
    simp [inputBefore]
  have member : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .gadgetDecompose matrixType base digitCount
        arguments := [inputRef]
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } := by
    have typedMember := selected.nodeMember
    rw [nodeEq] at typedMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .gadgetDecompose matrixType base digitCount
        arguments := [inputRef]
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } scopeExecution.scope.nodes[nodeIndex].outputTypes]
    exact typedMember
  obtain ⟨output, outputMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_gadgetDecompose_of_arguments
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before inputRef input matrixType
      base digitCount matrixParams baseValue digitCountValue
      scopeExecution.scope.nodes[nodeIndex].outputCount argumentsEvaluate typeEvaluates
      baseEvaluates digitCountEvaluates member
  refine ⟨output, outputMember, ?_⟩
  have outputLookup := selected.outputLookup 0 (by simp [valuesEq])
  simpa [valuesEq] using outputLookup

/-- An actual preimage-sampling node returns a sampler-supported matrix together with the
sampler contract's quotient-ring relation.  In particular this does not strengthen `B * K = P`
to an equality of integer representatives. -/
theorem ChildScopeExecutionPath.preimageSampleLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (publicRef trapdoorRef targetRef : Mxx.Ir.WireRef)
    (publicMatrix target : Mxx.Matrix)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (cutoff : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (publicEarlier : publicRef.node < nodeIndex)
    (trapdoorEarlier : trapdoorRef.node < nodeIndex)
    (targetEarlier : targetRef.node < nodeIndex)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .preimageSample matrixType cutoff
      arguments := [publicRef, trapdoorRef, targetRef]
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (publicFound : Mxx.Ir.lookupWire publicRef scopeExecution.wires =
      some (.matrix publicMatrix))
    (trapdoorFound : Mxx.Ir.lookupWire trapdoorRef scopeExecution.wires =
      some (.trapdoor publicMatrix))
    (targetFound : Mxx.Ir.lookupWire targetRef scopeExecution.wires = some (.matrix target))
    (typeEvaluates : matrixType.evaluate params cutoff = some matrixParams) :
    ∃ sample ∈ samplers.samplePreimage matrixParams publicMatrix target,
      Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
        some (.matrix (sample.withSamplerParams matrixParams)) ∧
      Mxx.MatrixModEq (Mxx.matrixMul publicMatrix
        (sample.withSamplerParams matrixParams)) target ∧
      Mxx.maxCenteredCoefficientNorm (sample.withSamplerParams matrixParams) ≤
        matrixParams.maxCoefficientBound := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have publicBefore := selected.argumentLookup_of_final publicRef (.matrix publicMatrix)
    publicEarlier publicFound
  have trapdoorBefore := selected.argumentLookup_of_final trapdoorRef (.trapdoor publicMatrix)
    trapdoorEarlier trapdoorFound
  have targetBefore := selected.argumentLookup_of_final targetRef (.matrix target) targetEarlier
    targetFound
  have argumentsEvaluate : [publicRef, trapdoorRef, targetRef].mapM
      (fun wire => Mxx.Ir.lookupWire wire selected.before) =
      some [.matrix publicMatrix, .trapdoor publicMatrix, .matrix target] := by
    simp [publicBefore, trapdoorBefore, targetBefore]
  have member : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .preimageSample matrixType cutoff
        arguments := [publicRef, trapdoorRef, targetRef]
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } := by
    have typedMember := selected.nodeMember
    rw [nodeEq] at typedMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .preimageSample matrixType cutoff
        arguments := [publicRef, trapdoorRef, targetRef]
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } scopeExecution.scope.nodes[nodeIndex].outputTypes]
    exact typedMember
  obtain ⟨sample, sampleMember, valuesEq⟩ :=
    Mxx.Ir.mem_evaluateNode_preimageSample_of_arguments
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before publicRef trapdoorRef targetRef
      publicMatrix target matrixType cutoff matrixParams
      scopeExecution.scope.nodes[nodeIndex].outputCount argumentsEvaluate typeEvaluates member
  obtain ⟨relation, bound⟩ := contract.preimageContract matrixParams publicMatrix target sample
    sampleMember
  have outputLookup := selected.outputLookup 0 (by simp [valuesEq])
  exact ⟨sample, sampleMember, by simpa [valuesEq] using outputLookup, relation, bound⟩

theorem ChildScopeExecutionPath.zeroMatrixLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .zeroMatrix matrixType
      arguments := []
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (typeEvaluates : matrixType.evaluate params = some matrixParams) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (zeroConstantOutput matrixParams)) := by
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .zeroMatrix matrixType
    arguments := []
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (zeroConstantOutput matrixParams) nodeEq
  simp [Mxx.Ir.evaluateNode, zeroConstantOutput, typeEvaluates]

theorem ChildScopeExecutionPath.identityMatrixLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (matrixParams : Mxx.SamplerParams)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .identityMatrix matrixType
      arguments := []
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (typeEvaluates : matrixType.evaluate params = some matrixParams) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (identityConstantOutput matrixParams)) := by
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .identityMatrix matrixType
    arguments := []
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (identityConstantOutput matrixParams) nodeEq
  simp [Mxx.Ir.evaluateNode, identityConstantOutput, typeEvaluates]

theorem ChildScopeExecutionPath.constantMatrixLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (coefficients : List Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (values : List Int)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .constantMatrix matrixType coefficients
      arguments := []
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (typeEvaluates : matrixType.evaluate params = some matrixParams)
    (coefficientsEvaluate : coefficients.mapM (Mxx.Ir.IntExpr.evaluate params) = some values) :
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires = some (.matrix
      (Mxx.Matrix.withSamplerParams {
        coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus)
      } matrixParams)) := by
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .constantMatrix matrixType coefficients
    arguments := []
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.Matrix.withSamplerParams {
    coefficients := values.map (Mxx.reduceCoefficient matrixParams.modulus)
  } matrixParams) nodeEq
  simp [Mxx.Ir.evaluateNode, typeEvaluates, coefficientsEvaluate]

theorem ChildScopeExecutionPath.gadgetMatrixLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (matrixType : Mxx.Ir.MatrixTypeExpr)
    (base : Mxx.Ir.IntExpr)
    (matrixParams : Mxx.SamplerParams)
    (baseValue : Int)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .gadgetMatrix matrixType base
      arguments := []
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (typeEvaluates : matrixType.evaluate params = some matrixParams)
    (baseEvaluates : base.evaluate params = some baseValue) :
    let digitCount := if matrixParams.rows = 0 then 0 else
      matrixParams.columns / matrixParams.rows
    Mxx.Ir.lookupWire ⟨nodeIndex, 0⟩ scopeExecution.wires =
      some (.matrix (Mxx.gadgetMatrix matrixParams baseValue digitCount)) := by
  dsimp
  apply scopeExecution.deterministicMatrixOutputLookup nodeIndex nodeInBounds {
    kind := .gadgetMatrix matrixType base
    arguments := []
    outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
  } (Mxx.gadgetMatrix matrixParams baseValue
    (if matrixParams.rows = 0 then 0 else matrixParams.columns / matrixParams.rows)) nodeEq
  simp [Mxx.Ir.evaluateNode, typeEvaluates, baseEvaluates]

/-- Recover the exact nested execution selected by a subgraph-call node.  Arguments and bindings
are evaluations of the selected frozen node in its real pre-node environment. -/
theorem ChildScopeExecutionPath.subgraphChildExecutionExists
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program (fuel + 1) parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (definition : String)
    (bindings : List (String × Mxx.Ir.IntExpr))
    (argumentRefs : List Mxx.Ir.WireRef)
    (argumentValues : List Mxx.Ir.Value)
    (evaluatedBindings : Mxx.Ir.ParamEnvironment)
    (body : Mxx.Ir.Scope)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .subgraphCall definition bindings
      arguments := argumentRefs
      outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (argumentsEvaluate : argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before) = some argumentValues)
    (bindingsEvaluate : Mxx.Ir.evaluateBindings params bindings = some evaluatedBindings)
    (bodyFound : Mxx.Ir.lookupDefinition definition program.definitions = some body) :
    Nonempty (ChildScopeExecutionPath samplers program fuel definition
      (evaluatedBindings ++ params) argumentValues
      (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).nodeValues) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have argumentsEvaluate' : argumentRefs.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire selected.before) = some argumentValues := by
    simpa only [selected] using argumentsEvaluate
  have member : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program (fuel + 1)) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .subgraphCall definition bindings
        arguments := argumentRefs
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } := by
    have typedMember := selected.nodeMember
    rw [nodeEq] at typedMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program (fuel + 1)) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .subgraphCall definition bindings
        arguments := argumentRefs
        outputCount := scopeExecution.scope.nodes[nodeIndex].outputCount
      } scopeExecution.scope.nodes[nodeIndex].outputTypes]
    exact typedMember
  have childMember : selected.nodeValues ∈ Mxx.Ir.childRunnerWithFuel samplers program (fuel + 1)
      definition (evaluatedBindings ++ params) argumentValues := by
    simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate', bindingsEvaluate] using member
  exact ChildScopeExecutionPath.nonempty_of_childMember samplers program fuel definition body
    (evaluatedBindings ++ params) argumentValues selected.nodeValues bodyFound childMember

/-- The output wire of a formal matrix-input node carries the value supplied by the exact child
invocation.  `formalInputFound` is a lookup in the child's real `inputNames.zip arguments`
environment, rather than a caller-provided replacement binding. -/
theorem ChildScopeExecutionPath.formalInputMatrixLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (name : String)
    (outputCount : Nat)
    (port : Nat)
    (value : Mxx.Matrix)
    (nodeEq : scopeExecution.scope.nodes[nodeIndex] = {
      kind := .input name
      arguments := []
      outputCount
      outputTypes := scopeExecution.scope.nodes[nodeIndex].outputTypes
    })
    (formalInputFound : Mxx.Ir.lookupEnvironment name
      (scopeExecution.scope.inputNames.zip arguments) = some (.matrix value))
    (portInBounds : port < outputCount) :
    Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires = some (.matrix value) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have member : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .input name
        arguments := []
        outputCount
      } := by
    have typedMember := selected.nodeMember
    rw [nodeEq] at typedMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .input name
        arguments := []
        outputCount
      } scopeExecution.scope.nodes[nodeIndex].outputTypes]
    exact typedMember
  have valuesEq : selected.nodeValues = List.replicate outputCount (.matrix value) := by
    have evaluated := Mxx.Ir.mem_evaluateNode_input
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before name outputCount member
    simpa [formalInputFound] using evaluated
  have outputLookup := selected.outputLookup port (by simpa [valuesEq] using portInBounds)
  simpa [valuesEq, portInBounds] using outputLookup

/-- Transport one selected child output back to the exact parent subgraph-call output wire.  The
child path's returned value list is definitionally the selected call node's `nodeValues`, so no
independent runner or output environment can be substituted. -/
theorem ChildScopeExecutionPath.subgraphCallMatrixOutputLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program (fuel + 1) parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    {definition : String}
    {childParams : Mxx.Ir.ParamEnvironment}
    {childArguments : List Mxx.Ir.Value}
    (childExecution : ChildScopeExecutionPath samplers program fuel definition childParams
      childArguments (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).nodeValues)
    (outputPort : Nat)
    (outputName : String)
    (outputWire : Mxx.Ir.WireRef)
    (value : Mxx.Matrix)
    (outputEntry : childExecution.scope.outputs[outputPort]? = some (outputName, outputWire))
    (childOutputFound : Mxx.Ir.lookupWire outputWire childExecution.wires =
      some (.matrix value)) :
    Mxx.Ir.lookupWire ⟨nodeIndex, outputPort⟩ scopeExecution.wires =
      some (.matrix value) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have outputPortInBounds : outputPort < childExecution.scope.outputs.length :=
    List.getElem?_eq_some_iff.mp outputEntry |>.1
  have returnedValue : selected.nodeValues[outputPort]? = some (.matrix value) := by
    rw [childExecution.outputEq]
    simp [Mxx.Ir.collectOutputs, List.getElem?_map, outputEntry, childOutputFound]
  have selectedPortInBounds : outputPort < selected.nodeValues.length :=
    List.getElem?_eq_some_iff.mp returnedValue |>.1
  have outputLookup := selected.outputLookup outputPort selectedPortInBounds
  have returnedValue' : selected.nodeValues[outputPort] = .matrix value := by
    exact Option.some.inj ((List.getElem?_eq_getElem selectedPortInBounds).symm.trans returnedValue)
  simpa [returnedValue'] using outputLookup

/-- The actual trace of one sequential node selected from a child execution path.  Its invariant
arguments are definitionally `argumentValues.drop view.carriedCount`; there is no independent
field through which a caller can replace them. -/
structure NestedSequentialTrace
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (view : SequentialLoopNodeView scopeExecution.scope.nodes[nodeIndex]) where
  before : Mxx.Ir.WireEnvironment
  prefixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) 0
    (scopeExecution.scope.nodes.take nodeIndex) [] before
  argumentValues : List Mxx.Ir.Value
  argumentsEvaluate : view.argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire before) =
    some argumentValues
  evaluatedCount : Int
  countEvaluate : view.count.evaluate params = some evaluatedCount
  nodeValues : List Mxx.Ir.Value
  nodeValuesEq : nodeValues =
    (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).nodeValues
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) before scopeExecution.scope.nodes[nodeIndex]
  suffixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) (nodeIndex + 1)
    (scopeExecution.scope.nodes.drop (nodeIndex + 1))
    (before ++ Mxx.Ir.bindOutputs nodeIndex nodeValues) scopeExecution.wires
  executionTrace : Mxx.Ir.SequentialIterationsTrace
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) view.definition params view.indexSlot
    view.bindings (argumentValues.drop view.carriedCount) (List.range evaluatedCount.toNat)
    (argumentValues.take view.carriedCount) nodeValues

/-- Invert a selected sequential node on the actual child path.  The two evaluation equalities do
not supply semantic values: they certify the unique results of evaluating the selected frozen
node's actual argument wires and count expression in its actual pre-node environment. -/
theorem NestedSequentialTrace.nonempty_atNode
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (view : SequentialLoopNodeView scopeExecution.scope.nodes[nodeIndex])
    (argumentValues : List Mxx.Ir.Value)
    (argumentsEvaluate : view.argumentRefs.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire
        (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before) =
      some argumentValues)
    (evaluatedCount : Int)
    (countEvaluate : view.count.evaluate params = some evaluatedCount) :
    Nonempty (NestedSequentialTrace scopeExecution nodeIndex nodeInBounds view) := by
  classical
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have baseMember : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
          view.carriedCount
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } := by
    have typedMember : selected.nodeValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
        (scopeExecution.scope.inputNames.zip arguments) selected.before {
          kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
            view.carriedCount
          arguments := view.argumentRefs
          outputCount := view.outputCount
          outputTypes := view.outputTypes
        } := by
      simpa [view.nodeEq] using selected.nodeMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .sequentialLoop view.definition view.count view.indexSlot view.bindings
          view.carriedCount
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } view.outputTypes]
    exact typedMember
  have executionTrace :=
    (Mxx.Ir.mem_evaluateNode_sequentialLoop_iff_trace
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before view.definition view.count
      view.indexSlot view.bindings view.carriedCount view.argumentRefs view.outputCount
      argumentValues evaluatedCount argumentsEvaluate countEvaluate selected.nodeValues).mp baseMember
  exact ⟨{
    before := selected.before
    prefixPath := selected.prefixPath
    argumentValues
    argumentsEvaluate
    evaluatedCount
    countEvaluate
    nodeValues := selected.nodeValues
    nodeValuesEq := rfl
    nodeMember := selected.nodeMember
    suffixPath := selected.suffixPath
    executionTrace
  }⟩

/-- The parallel-loop fields of one exact node.  Like `SequentialLoopNodeView`, this view is
only a typed equality for a node already selected from an executable child scope. -/
structure ParallelLoopNodeView (node : Mxx.Ir.Node) where
  definition : String
  count : Mxx.Ir.IntExpr
  indexSlot : Nat
  bindings : List (String × Mxx.Ir.IntExpr)
  modes : List Mxx.Ir.LoopInputMode
  argumentRefs : List Mxx.Ir.WireRef
  outputCount : Nat
  outputTypes : List Mxx.Ir.WireTypeExpr
  nodeEq : node = {
    kind := .parallelLoop definition count indexSlot bindings modes
    arguments := argumentRefs
    outputCount
    outputTypes
  }

/-- Exact execution trace of one parallel-loop node selected from a child execution path. -/
structure NestedParallelTrace
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (view : ParallelLoopNodeView scopeExecution.scope.nodes[nodeIndex]) where
  before : Mxx.Ir.WireEnvironment
  prefixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) 0
    (scopeExecution.scope.nodes.take nodeIndex) [] before
  argumentValues : List Mxx.Ir.Value
  argumentsEvaluate : view.argumentRefs.mapM (fun wire ↦ Mxx.Ir.lookupWire wire before) =
    some argumentValues
  evaluatedCount : Int
  countEvaluate : view.count.evaluate params = some evaluatedCount
  nodeValues : List Mxx.Ir.Value
  nodeValuesEq : nodeValues =
    (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).nodeValues
  nodeMember : nodeValues ∈ Mxx.Ir.evaluateNode
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) before scopeExecution.scope.nodes[nodeIndex]
  suffixPath : Mxx.Ir.EvaluatesNodesPath
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
    (scopeExecution.scope.inputNames.zip arguments) (nodeIndex + 1)
    (scopeExecution.scope.nodes.drop (nodeIndex + 1))
    (before ++ Mxx.Ir.bindOutputs nodeIndex nodeValues) scopeExecution.wires
  final : List (List Mxx.Ir.Value)
  executionTrace : Mxx.Ir.ParallelIterationsTrace
    (Mxx.Ir.childRunnerWithFuel samplers program fuel) view.definition params view.indexSlot
    view.bindings view.modes argumentValues (List.range evaluatedCount.toNat)
    (List.replicate view.outputCount []) final
  finalEq : nodeValues = final.map Mxx.Ir.Value.family

/-- Invert an actual parallel-loop member after recording the deterministic argument and count
evaluations of the selected frozen node.  The caller cannot supply a runner or execution trace;
the two equality premises merely rule out the executable evaluator's explicit `invalid` branch,
as in `NestedSequentialTrace.nonempty_atNode`. -/
theorem NestedParallelTrace.nonempty_atNode
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    (scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues)
    (nodeIndex : Nat)
    (nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length)
    (view : ParallelLoopNodeView scopeExecution.scope.nodes[nodeIndex])
    (argumentValues : List Mxx.Ir.Value)
    (argumentsEvaluate : view.argumentRefs.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire
        (scopeExecution.nodeExecutionAt nodeIndex nodeInBounds).before) = some argumentValues)
    (evaluatedCount : Int)
    (countEvaluate : view.count.evaluate params = some evaluatedCount) :
    Nonempty (NestedParallelTrace scopeExecution nodeIndex nodeInBounds view) := by
  classical
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have baseMember : selected.nodeValues ∈ Mxx.Ir.evaluateNode
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .parallelLoop view.definition view.count view.indexSlot view.bindings view.modes
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } := by
    have typedMember : selected.nodeValues ∈ Mxx.Ir.evaluateNode
        (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
        (scopeExecution.scope.inputNames.zip arguments) selected.before {
          kind := .parallelLoop view.definition view.count view.indexSlot view.bindings view.modes
          arguments := view.argumentRefs
          outputCount := view.outputCount
          outputTypes := view.outputTypes
        } := by
      simpa [view.nodeEq] using selected.nodeMember
    rw [← evaluateNode_outputTypes_irrelevant
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before {
        kind := .parallelLoop view.definition view.count view.indexSlot view.bindings view.modes
        arguments := view.argumentRefs
        outputCount := view.outputCount
      } view.outputTypes]
    exact typedMember
  obtain ⟨final, executionTrace, finalEq⟩ :=
    (Mxx.Ir.mem_evaluateNode_parallelLoop_iff_trace
      (Mxx.Ir.childRunnerWithFuel samplers program fuel) samplers params
      (scopeExecution.scope.inputNames.zip arguments) selected.before view.definition
      view.count view.indexSlot view.bindings view.modes view.argumentRefs
      view.outputCount argumentValues evaluatedCount argumentsEvaluate countEvaluate
      selected.nodeValues).mp baseMember
  exact ⟨{
    before := selected.before
    prefixPath := selected.prefixPath
    argumentValues
    argumentsEvaluate
    evaluatedCount
    countEvaluate
    nodeValues := selected.nodeValues
    nodeValuesEq := rfl
    nodeMember := selected.nodeMember
    suffixPath := selected.suffixPath
    final
    executionTrace
    finalEq
  }⟩

/-- Resolve one output port of the exact nested parallel trace into its ordered gathered values
and the actual child support member that produced every coordinate. -/
theorem NestedParallelTrace.gatheredPort
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    {scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues}
    {nodeIndex : Nat}
    {nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length}
    {view : ParallelLoopNodeView scopeExecution.scope.nodes[nodeIndex]}
    (trace : NestedParallelTrace scopeExecution nodeIndex nodeInBounds view)
    (port : Nat)
    (portInBounds : port < view.outputCount)
    (childArity : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings ((.loopIndex view.indexSlot, .integer index) :: params)
          view.bindings = some evaluatedBindings →
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
        (evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer index) :: params))
        ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
          Mxx.Ir.loopArgument mode index value) →
      childValues.length = view.outputCount) :
    ∃ gathered,
      trace.final[port]? = some gathered ∧
      gathered.length = trace.evaluatedCount.toNat ∧
      ∀ position : Nat, position < trace.evaluatedCount.toNat →
        ∃ evaluatedBindings childValues,
          Mxx.Ir.evaluateBindings
              ((.loopIndex view.indexSlot, .integer position) :: params) view.bindings =
            some evaluatedBindings ∧
          childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
            (evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer position) :: params))
            ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
              Mxx.Ir.loopArgument mode position value) ∧
          childValues[port]? = gathered[position]? := by
  have initialLength :
      (List.replicate view.outputCount ([] : List Mxx.Ir.Value)).length = view.outputCount := by
    simp
  have initialPort :
      (List.replicate view.outputCount ([] : List Mxx.Ir.Value))[port]? = some [] := by
    have replicateBound : port <
        (List.replicate view.outputCount ([] : List Mxx.Ir.Value)).length := by
      simpa using portInBounds
    rw [List.getElem?_eq_getElem replicateBound]
    simp
  obtain ⟨gathered, finalPort, gatheredLength, coordinates⟩ :=
    trace.executionTrace.portValues view.outputCount port initialLength portInBounds childArity
      initialPort
  refine ⟨gathered, ?_, by simpa using gatheredLength, ?_⟩
  · simpa using finalPort
  · intro position positionInBounds
    have rangeBound : position < (List.range trace.evaluatedCount.toNat).length := by
      rw [List.length_range]
      exact positionInBounds
    have indexEq : (List.range trace.evaluatedCount.toNat)[position]'rangeBound = position := by
      simp
    simpa only [indexEq] using coordinates position rangeBound

/-- The parent SSA output of a nested parallel node is the exact family gathered by its actual
iteration trace. -/
theorem NestedParallelTrace.parentFamilyLookup
    {samplers : Mxx.MxxSamplerFamily}
    {program : Mxx.Ir.Prog}
    {fuel : Nat}
    {parentDefinition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments parentValues : List Mxx.Ir.Value}
    {scopeExecution : ChildScopeExecutionPath samplers program fuel parentDefinition params
      arguments parentValues}
    {nodeIndex : Nat}
    {nodeInBounds : nodeIndex < scopeExecution.scope.nodes.length}
    {view : ParallelLoopNodeView scopeExecution.scope.nodes[nodeIndex]}
    (trace : NestedParallelTrace scopeExecution nodeIndex nodeInBounds view)
    (port : Nat)
    (_portInBounds : port < view.outputCount)
    (gathered : List Mxx.Ir.Value)
    (finalPort : trace.final[port]? = some gathered) :
    Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires =
      some (.family gathered) := by
  let selected := scopeExecution.nodeExecutionAt nodeIndex nodeInBounds
  have nodePort : trace.nodeValues[port]? = some (.family gathered) := by
    rw [trace.finalEq]
    simp [List.getElem?_map, finalPort]
  have selectedNodePort : selected.nodeValues[port]? = some (.family gathered) := by
    rw [← trace.nodeValuesEq]
    exact nodePort
  have selectedPortInBounds : port < selected.nodeValues.length :=
    List.getElem?_eq_some_iff.mp selectedNodePort |>.1
  have outputLookup := selected.outputLookup port selectedPortInBounds
  have nodePortValue : selected.nodeValues[port] = .family gathered :=
    Option.some.inj ((List.getElem?_eq_getElem selectedPortInBounds).symm.trans selectedNodePort)
  simpa [nodePortValue] using outputLookup

end Mxx.Certificate
