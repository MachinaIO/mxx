import Mxx.Certificate.Rules.BggThreeTraceStep
import Mxx.Certificate.Rules.PointwiseFormulaBoundaries
import Mxx.Certificate.Rules.PointwiseFormulaElaboration

namespace Mxx.Certificate

/-!
# Exact BGG lane formula execution

This module binds one matcher-retained BGG gate candidate to one coordinate of the actual nested
parallel execution.  The child frame is reconstructed from the executable trace, including its
static scope identity; neither a node number nor a runtime-value callback is accepted.
-/

/-- The exact recurrence-body scope in which the matcher-selected parallel lane executes. -/
def CheckedRecurrenceLaneOutput.Execution.parentExecutedScope
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    (_execution : lane.Execution scopeExecution) : ExecutedScope samplers interface.program := {
  scopeId := ⟨interface.transfer.source.loop.site.scope.path ++ [interface.definition]⟩
  fuel
  definition := interface.definition
  params
  arguments
  outputs := values
  execution := scopeExecution
}

/-- One actual coordinate of the matcher-selected BGG lane, with the precise child scope and
frame used to interpret its retained program formulas. -/
structure CheckedRecurrenceLaneOutput.CandidateFrame
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    (execution : lane.Execution scopeExecution)
    (position : Nat) where
  parent : FormulaExecutionFrame samplers interface.program
    (execution.parentExecutedScope) := .root _
  edge : ExactParallelLaneExecutionEdge execution.parentExecutedScope
  traceMatches : HEq edge.trace execution.trace
  positionMatches : edge.position = position
  childScopeMatches : edge.child.scopeId =
    ⟨interface.transfer.source.loop.site.scope.path ++
      [interface.definition, lane.definition]⟩
  childFrameValid : LocalFormulaFrameValid edge.child

/-- Reconstruct one exact child frame from one concrete coordinate of the actual parallel trace.
The only numeric premise is membership in that trace's evaluated coordinate range. -/
theorem CheckedRecurrenceLaneOutput.CandidateFrame.nonempty
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    (execution : lane.Execution scopeExecution)
    (fuelMatches : fuel = interface.program.definitions.length - 1)
    (position : Nat)
    (positionInBounds : position < execution.trace.evaluatedCount.toNat) :
    Nonempty (lane.CandidateFrame execution position) := by
  let parent := execution.parentExecutedScope
  have rangeMember : position ∈ List.range execution.trace.evaluatedCount.toNat := by
    simpa using positionInBounds
  obtain ⟨childParams, childInputs, childOutputs, childMember,
      ⟨evaluatedBindings, bindingsEvaluate, paramsEq⟩, inputsEq⟩ :=
    execution.trace.executionTrace.everyChild
      (fun index childParams childInputs _ ↦
        (∃ evaluatedBindings,
          Mxx.Ir.evaluateBindings
              ((.loopIndex execution.view.indexSlot, .integer index) :: params)
                execution.view.bindings = some evaluatedBindings ∧
          childParams = evaluatedBindings ++
            ((.loopIndex execution.view.indexSlot, .integer index) :: params)) ∧
        childInputs = ((execution.view.modes.zip execution.trace.argumentValues).map
          fun (mode, value) ↦ Mxx.Ir.loopArgument mode index value))
      (by
        intro index evaluatedBindings childValues evaluated childMember
        exact ⟨⟨evaluatedBindings, evaluated, rfl⟩, rfl⟩)
      position rangeMember
  subst childParams
  subst childInputs
  have nestedFuelEq : fuel = interface.program.definitions.length - 2 + 1 := by
    rw [fuelMatches]
    have positive := lane.nestedFuelPositive
    omega
  have childMemberAtLane : childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers
      interface.program fuel lane.definition
      (evaluatedBindings ++
        ((.loopIndex execution.view.indexSlot, .integer position) :: params))
      ((execution.view.modes.zip execution.trace.argumentValues).map fun (mode, value) ↦
        Mxx.Ir.loopArgument mode position value) := by
    simpa only [execution.definitionMatches] using childMember
  have childMemberAtFuel : childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers
      interface.program (interface.program.definitions.length - 2 + 1) lane.definition
      (evaluatedBindings ++
        ((.loopIndex execution.view.indexSlot, .integer position) :: params))
      ((execution.view.modes.zip execution.trace.argumentValues).map fun (mode, value) ↦
        Mxx.Ir.loopArgument mode position value) := by
    simpa only [← nestedFuelEq] using childMemberAtLane
  obtain ⟨childExecution⟩ := ChildScopeExecutionPath.nonempty_of_childMember samplers
    interface.program (interface.program.definitions.length - 2) lane.definition lane.body
    (evaluatedBindings ++
      ((.loopIndex execution.view.indexSlot, .integer position) :: params))
    ((execution.view.modes.zip execution.trace.argumentValues).map fun (mode, value) ↦
      Mxx.Ir.loopArgument mode position value)
    childOutputs lane.bodyFound childMemberAtFuel
  let child : ExecutedScope samplers interface.program := {
    scopeId := ⟨interface.transfer.source.loop.site.scope.path ++
      [interface.definition, lane.definition]⟩
    fuel := interface.program.definitions.length - 2
    definition := lane.definition
    params := evaluatedBindings ++
      ((.loopIndex execution.view.indexSlot, .integer position) :: params)
    arguments := (execution.view.modes.zip execution.trace.argumentValues).map
      fun (mode, value) ↦ Mxx.Ir.loopArgument mode position value
    outputs := childOutputs
    execution := childExecution
  }
  let edge : ExactParallelLaneExecutionEdge parent := {
    nodeIndex := lane.output.node
    nodeInBounds := execution.nodeInBounds
    view := execution.view
    trace := execution.trace
    position
    positionInBounds
    evaluatedBindings
    childValues := childOutputs
    bindingsEvaluate
    childMember := by
      simpa [parent, CheckedRecurrenceLaneOutput.Execution.parentExecutedScope] using childMember
    child
    fuelEq := by simpa [parent, CheckedRecurrenceLaneOutput.Execution.parentExecutedScope]
      using nestedFuelEq
    definitionEq := by simpa [child] using execution.definitionMatches.symm
    paramsEq := rfl
    argumentsEq := rfl
    outputsEq := rfl
  }
  have childFrameValid : LocalFormulaFrameValid child := by
    constructor
    simpa [child, scopeAtStaticPath?] using childExecution.definitionFound
  exact ⟨{
    parent := .root parent
    edge
    traceMatches := HEq.rfl
    positionMatches := rfl
    childScopeMatches := rfl
    childFrameValid
  }⟩

/-- Local arithmetic elaboration for one exact matcher-retained candidate at one actual lane
coordinate.  Parameter evaluations and matrix layouts remain explicit typed premises; the
runtime matrix and every arithmetic equation are recovered by `LocalElaborationInputs.elaborate`
from the frozen candidate and this trace-derived frame. -/
theorem CheckedRecurrenceLaneOutput.CandidateFrame.elaborateCandidate
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
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    {formula : FrozenPointwiseMatrixProgramFormula}
    (candidateMember : formula ∈ lane.gateCandidateProgramFormulas)
    {q ringDimension rows columns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    (inputs : formula.LocalElaborationInputs
      (.parallelLane candidateFrame.parent candidateFrame.edge)
      q ringDimension rows columns) :
    Nonempty (formula.LocalElaborationResult
      (.parallelLane candidateFrame.parent candidateFrame.edge)
      q ringDimension rows columns) := by
  exact inputs.elaborate contract candidateFrame.childFrameValid
    (lane.programFormulaValid formula candidateMember)

end Mxx.Certificate
