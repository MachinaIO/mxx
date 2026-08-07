import Mxx.Certificate.Rules.BggThreeTrace

namespace Mxx.Certificate

/-!
# Execution-indexed BGG three-trace steps

This module connects one iteration selected by an actual nested parallel trace to the matching
element of the family stored at the parent parallel-loop output wire.  It is deliberately phrased
only in terms of executable support members.  Formula semantics can consume the recovered child
scope and matrix value without accepting a caller-supplied lane output.
-/

/-- One selected port of a nested parallel execution contains one value per iteration.  The
selected child support member and the parent family wire are recovered from the same executable
trace. -/
theorem NestedParallelTrace.gatheredOutputAt
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
    (port position : Nat)
    (portInBounds : port < view.outputCount)
    (positionInBounds : position < trace.evaluatedCount.toNat)
    (childArity : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings
          ((.loopIndex view.indexSlot, .integer index) :: params) view.bindings =
        some evaluatedBindings →
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
        (evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer index) :: params))
        ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
          Mxx.Ir.loopArgument mode index value) →
      childValues.length = view.outputCount) :
    ∃ gathered value evaluatedBindings childValues,
      trace.final[port]? = some gathered ∧
      gathered[position]? = some value ∧
      Mxx.Ir.evaluateBindings
          ((.loopIndex view.indexSlot, .integer position) :: params)
          view.bindings = some evaluatedBindings ∧
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
        (evaluatedBindings ++
          ((.loopIndex view.indexSlot, .integer position) :: params))
        ((view.modes.zip trace.argumentValues).map fun (mode, argument) ↦
          Mxx.Ir.loopArgument mode position argument) ∧
      childValues[port]? = some value ∧
      Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires =
        some (.family gathered) := by
  have rangePositionInBounds :
      position < (List.range trace.evaluatedCount.toNat).length := by
    simpa using positionInBounds
  have initialPort :
      (List.replicate view.outputCount ([] : List Mxx.Ir.Value))[port]? = some [] := by
    rw [List.getElem?_eq_getElem (by simpa using portInBounds)]
    simp
  obtain ⟨gathered, finalPort, gatheredLength, childEvidence⟩ :=
    trace.executionTrace.portValues view.outputCount port (by simp) portInBounds childArity
      initialPort
  obtain ⟨evaluatedBindings, childValues, bindingsEvaluate, childMember, childPort⟩ :=
    childEvidence position rangePositionInBounds
  have rangeAt : (List.range trace.evaluatedCount.toNat)[position] = position := by
    simp
  rw [rangeAt] at bindingsEvaluate childMember
  have gatheredPositionInBounds : position < gathered.length := by
    simpa [gatheredLength] using positionInBounds
  have gatheredAt : gathered[position]? = some gathered[position] :=
    List.getElem?_eq_getElem gatheredPositionInBounds
  have childAt : childValues[port]? = some gathered[position] := by
    rw [childPort, gatheredAt]
  have nodePortInBounds : port < trace.nodeValues.length := by
    rw [trace.finalEq]
    simpa using (List.getElem?_eq_some_iff.mp finalPort).1
  have missing : Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ trace.before = none := by
    apply trace.prefixPath.lookupWire_after_end nodeIndex port
    · simp [List.length_take, nodeInBounds.le]
    · rfl
  have parentWire : Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires =
      some trace.nodeValues[port] := by
    apply trace.suffixPath.lookupWire_preserved
    exact Mxx.Ir.lookupWire_append_bindOutputs missing nodePortInBounds
  have finalPort' : trace.final[port]? = some gathered := by simpa using finalPort
  have nodeAt : trace.nodeValues[port]? = some (.family gathered) := by
    rw [trace.finalEq, List.getElem?_map, finalPort']
    rfl
  have nodeValueEq : trace.nodeValues[port] = .family gathered := by
    exact Option.some.inj ((List.getElem?_eq_getElem nodePortInBounds).symm.trans nodeAt)
  refine ⟨gathered, gathered[position], evaluatedBindings, childValues, finalPort', gatheredAt,
    bindingsEvaluate, childMember, childAt, ?_⟩
  simpa [nodeValueEq] using parentWire

/-- Matrix-specialized view of `gatheredOutputAt`.  The matrix premise is about the exact child
support member returned by that theorem; the parent family value cannot be substituted
independently. -/
theorem NestedParallelTrace.gatheredMatrixAt
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
    (port position : Nat)
    (portInBounds : port < view.outputCount)
    (positionInBounds : position < trace.evaluatedCount.toNat)
    (childArity : ∀ (index : Nat) evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings
          ((.loopIndex view.indexSlot, .integer index) :: params) view.bindings =
        some evaluatedBindings →
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
        (evaluatedBindings ++ ((.loopIndex view.indexSlot, .integer index) :: params))
        ((view.modes.zip trace.argumentValues).map fun (mode, value) ↦
          Mxx.Ir.loopArgument mode index value) →
      childValues.length = view.outputCount)
    (matrixOutput : ∀ evaluatedBindings childValues,
      Mxx.Ir.evaluateBindings
          ((.loopIndex view.indexSlot, .integer position) :: params)
          view.bindings = some evaluatedBindings →
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
        (evaluatedBindings ++
          ((.loopIndex view.indexSlot, .integer position) :: params))
        ((view.modes.zip trace.argumentValues).map fun (mode, argument) ↦
          Mxx.Ir.loopArgument mode position argument) →
      ∃ matrix, childValues[port]? = some (.matrix matrix)) :
    ∃ gathered matrix evaluatedBindings childValues,
      trace.final[port]? = some gathered ∧
      gathered[position]? = some (.matrix matrix) ∧
      childValues[port]? = some (.matrix matrix) ∧
      childValues ∈ Mxx.Ir.childRunnerWithFuel samplers program fuel view.definition
        (evaluatedBindings ++
          ((.loopIndex view.indexSlot, .integer position) :: params))
        ((view.modes.zip trace.argumentValues).map fun (mode, argument) ↦
          Mxx.Ir.loopArgument mode position argument) ∧
      Mxx.Ir.lookupWire ⟨nodeIndex, port⟩ scopeExecution.wires =
        some (.family gathered) := by
  obtain ⟨gathered, value, evaluatedBindings, childValues, finalPort, gatheredAt,
    bindingsEvaluate, childMember, childAt, parentWire⟩ :=
    trace.gatheredOutputAt port position portInBounds positionInBounds childArity
  obtain ⟨matrix, matrixAt⟩ := matrixOutput evaluatedBindings childValues bindingsEvaluate
    childMember
  have valueEq : value = .matrix matrix := by
    rw [childAt] at matrixAt
    exact Option.some.inj matrixAt
  subst value
  exact ⟨gathered, matrix, evaluatedBindings, childValues, finalPort, gatheredAt, matrixAt,
    childMember, parentWire⟩

end Mxx.Certificate
