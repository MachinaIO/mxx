import Mxx.Certificate.Rules.ParallelFamilyAnalysis
import Mxx.Certificate.Rules.PointwiseFormulaElaboration
import Mxx.Certificate.Rules.TraceBoundRecurrence

namespace Mxx.Certificate

/-!
# Root parallel-family lane execution

This bridge reconstructs one child scope directly from the actual root-stage parallel trace.
It is intentionally independent of `FormulaExecutionFrame`'s parent-edge constructors: the
returned child frame is rooted at the exact child execution selected by the trace.  No lane
value, runner, or child execution can be supplied separately.
-/

/-- One exact element of an analyzer-owned root parallel family. -/
structure RootParallelFamilyLaneFrame
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {stage : StageExecution samplers}
    {joint : JointFamilyId}
    {family : JointFamilyFact}
    (owned : ParallelFamilyAnalysisEvidence analysis joint family)
    (execution : ParallelLoopSemanticResult analysis stage.rootChildRunner samplers stage.params
      stage.inputs joint)
    (executionMatches : owned.MatchesExecution execution)
    (position : Nat) where
  positionInBounds : position < execution.evaluatedCount.toNat
  childValues : List Mxx.Ir.Value
  childMember : childValues ∈ stage.rootChildRunner execution.definition
    ((.loopIndex execution.indexSlot, .integer position) :: stage.params)
    ((execution.modes.zip execution.argumentValues).map fun (mode, value) =>
      Mxx.Ir.loopArgument mode position value)
  child : ExecutedScope samplers stage.stage.program
  definitionEq : child.definition = execution.definition
  paramsEq : child.params =
    ((.loopIndex execution.indexSlot, .integer position) :: stage.params)
  argumentsEq : child.arguments =
    ((execution.modes.zip execution.argumentValues).map fun (mode, value) =>
      Mxx.Ir.loopArgument mode position value)
  outputsEq : child.outputs = childValues
  frame : FormulaExecutionFrame samplers stage.stage.program child := .root child
  childFrameValid : LocalFormulaFrameValid child

/-- Recover one lane solely by inverting the actual `ParallelIterationsTrace` retained by the
root node.  The initial version supports the no-binding loops used by Diamond's family/gather
pipeline; bound root loops fail closed until an actual protocol requires them. -/
theorem RootParallelFamilyLaneFrame.nonempty_of_noBindings
    {analysis : AnalysisResult}
    {samplers : Mxx.MxxSamplerFamily}
    {stage : StageExecution samplers}
    {joint : JointFamilyId}
    {family : JointFamilyFact}
    (owned : ParallelFamilyAnalysisEvidence analysis joint family)
    (execution : ParallelLoopSemanticResult analysis stage.rootChildRunner samplers stage.params
      stage.inputs joint)
    (executionMatches : owned.MatchesExecution execution)
    (bindingsEmpty : execution.bindings = [])
    (body : Mxx.Ir.Scope)
    (bodyFound : Mxx.Ir.lookupDefinition execution.definition
      stage.stage.program.definitions = some body)
    (childScopeFound : scopeAtStaticPath? stage.stage.program owned.source.childScope = some body)
    (fuelPositive : 0 < stage.stage.program.definitions.length)
    (position : Nat)
    (positionInBounds : position < execution.evaluatedCount.toNat) :
    Nonempty (RootParallelFamilyLaneFrame owned execution executionMatches position) := by
  have rangeMember : position ∈ List.range execution.evaluatedCount.toNat := by
    simpa using positionInBounds
  obtain ⟨childParams, childInputs, childOutputs, childMember, paramsEq, inputsEq⟩ :=
    execution.executionTrace.everyChild
      (fun index childParams childInputs _ =>
        childParams = ((.loopIndex execution.indexSlot, .integer index) :: stage.params) ∧
        childInputs = ((execution.modes.zip execution.argumentValues).map fun (mode, value) =>
          Mxx.Ir.loopArgument mode index value))
      (by
        intro index evaluatedBindings childValues evaluated childMember
        rw [bindingsEmpty] at evaluated
        simp only [Mxx.Ir.evaluateBindings] at evaluated
        cases Option.some.inj evaluated
        exact ⟨rfl, rfl⟩)
      position rangeMember
  subst childParams
  subst childInputs
  have childMemberAtFuel : childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers
      stage.stage.program stage.stage.program.definitions.length execution.definition
      ((.loopIndex execution.indexSlot, .integer position) :: stage.params)
      ((execution.modes.zip execution.argumentValues).map fun (mode, value) =>
        Mxx.Ir.loopArgument mode position value) := by
    simpa [StageExecution.rootChildRunner] using childMember
  have fuelEq : stage.stage.program.definitions.length =
      (stage.stage.program.definitions.length - 1) + 1 := by omega
  rw [fuelEq] at childMemberAtFuel
  obtain ⟨childExecution⟩ := ChildScopeExecutionPath.nonempty_of_childMember samplers
    stage.stage.program (stage.stage.program.definitions.length - 1) execution.definition body
    ((.loopIndex execution.indexSlot, .integer position) :: stage.params)
    ((execution.modes.zip execution.argumentValues).map fun (mode, value) =>
      Mxx.Ir.loopArgument mode position value)
    childOutputs bodyFound childMemberAtFuel
  have executionScopeEq : childExecution.scope = body :=
    Option.some.inj (childExecution.definitionFound.symm.trans bodyFound)
  let child : ExecutedScope samplers stage.stage.program := {
    scopeId := owned.source.childScope
    fuel := stage.stage.program.definitions.length - 1
    definition := execution.definition
    params := ((.loopIndex execution.indexSlot, .integer position) :: stage.params)
    arguments := ((execution.modes.zip execution.argumentValues).map fun (mode, value) =>
      Mxx.Ir.loopArgument mode position value)
    outputs := childOutputs
    execution := childExecution
  }
  exact ⟨{
    positionInBounds
    childValues := childOutputs
    childMember := by simpa [StageExecution.rootChildRunner] using childMember
    child
    definitionEq := rfl
    paramsEq := rfl
    argumentsEq := rfl
    outputsEq := rfl
    frame := .root child
    childFrameValid := by
      constructor
      simpa [child, executionScopeEq] using childScopeFound
  }⟩

end Mxx.Certificate
