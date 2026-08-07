import Mxx.Certificate.SymbolicEvaluationSoundness
import Mxx.Certificate.Rules.AggregateExecution

namespace Mxx.Certificate

/-!
# Analyzer-owned parallel-family provenance

This module exposes the retained derivation source of a parallel family only through the unique
source proved by `AnalysisHolds`.  It does not accept element facts, sampler relations, or a body
soundness callback.
-/

/-- The unique analyzer-produced derivation source for one family table entry. -/
structure ParallelFamilyAnalysisEvidence
    (analysis : AnalysisResult)
    (joint : JointFamilyId)
    (family : JointFamilyFact) : Type where
  familyMember : (joint, family) ∈ analysis.families
  source : ParallelFamilyDerivationSource
  uniqueSource :
    analysis.parallelFamilyDerivations.filter (fun candidate => candidate.family = joint) =
      [source]
  sourceMatches : source.MatchesFamily family
  indexExpressionExact : source.indexExpression = .loopIndex { site := source.loopSite }
  indexExpressionOwned :
    analysis.expressionArena.lookupInteger source.indexReference = some source.indexExpression
  outputsMatchBody : source.OutputFactsMatchBody
  outputTemplatesDerived :
    source.outputFacts.mapM ScopedWireFact.toTemplate = some source.elementTemplates

/-- Recover family provenance from the strengthened analysis soundness judgment. -/
theorem AnalysisHolds.parallelFamilyAnalysisEvidence
    {environment : FactEnvironment}
    {analysis : AnalysisResult}
    (holds : AnalysisHolds environment analysis)
    {joint : JointFamilyId}
    {family : JointFamilyFact}
    (member : (joint, family) ∈ analysis.families) :
    Nonempty (ParallelFamilyAnalysisEvidence analysis joint family) := by
  obtain ⟨source, unique, sourceMatchesProof, indexExact, indexOwned, outputsMatch, outputs⟩ :=
    holds.2.2 joint family member
  exact ⟨{
    familyMember := member
    source
    uniqueSource := unique
    sourceMatches := sourceMatchesProof
    indexExpressionExact := indexExact
    indexExpressionOwned := indexOwned
    outputsMatchBody := outputsMatch
    outputTemplatesDerived := outputs
  }⟩

/-- The retained source and the executable parallel-loop view describe the same frozen node.
All fields are equalities against analyzer-produced data; there is no replacement runner or
per-lane semantic premise. -/
structure ParallelFamilyAnalysisEvidence.MatchesExecution
    {analysis : AnalysisResult}
    {joint : JointFamilyId}
    {family : JointFamilyFact}
    (evidence : ParallelFamilyAnalysisEvidence analysis joint family)
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (execution : ParallelLoopSemanticResult analysis runChild samplers params inputs joint) :
    Prop where
  familyEq : execution.family = family
  definitionEq : execution.definition = evidence.source.definition
  countEq : execution.countExpression = evidence.source.count
  indexSlotEq : execution.indexSlot = evidence.source.indexSlot
  bindingsEq : execution.bindings = evidence.source.bindings
  modesEq : execution.modes = evidence.source.modes
  argumentsEq : execution.argumentRefs = evidence.source.argumentRefs
  outputCountEq : execution.outputCount = evidence.source.outputCount
  outputTypesEq : execution.outputTypes = evidence.source.outputTypes

/-- One actual lane selected from the executable parallel trace, paired with the exact
analyzer-produced family source for that coordinate.  Template instantiation is intentionally
deferred to the new scope-semantic layer, where it can thread the exact arena and prove every
rewritten expression reference. -/
structure ParallelFamilyLaneInstance
    {analysis : AnalysisResult}
    {joint : JointFamilyId}
    {family : JointFamilyFact}
    (evidence : ParallelFamilyAnalysisEvidence analysis joint family)
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (execution : ParallelLoopSemanticResult analysis runChild samplers params inputs joint)
    (position : Nat) : Type where
  executionMatches : evidence.MatchesExecution execution
  positionInBounds : position < execution.evaluatedCount.toNat
  evaluatedBindings : Mxx.Ir.ParamEnvironment
  childValues : List Mxx.Ir.Value
  bindingsEvaluate : Mxx.Ir.evaluateBindings
      ((.loopIndex execution.indexSlot, .integer position) :: params) execution.bindings =
    some evaluatedBindings
  childMember : childValues ∈ runChild execution.definition
    (evaluatedBindings ++ ((.loopIndex execution.indexSlot, .integer position) :: params))
    ((execution.modes.zip execution.argumentValues).map fun (mode, value) ↦
      Mxx.Ir.loopArgument mode position value)

/-- Actual trace induction selects every lane and mechanically instantiates the retained output
templates.  There is no caller-provided fact table, lane relation, or body-soundness callback. -/
theorem ParallelFamilyAnalysisEvidence.laneInstance
    {analysis : AnalysisResult}
    {joint : JointFamilyId}
    {family : JointFamilyFact}
    (evidence : ParallelFamilyAnalysisEvidence analysis joint family)
    {runChild : Mxx.Ir.ChildRunner}
    {samplers : Mxx.MxxSamplerFamily}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (execution : ParallelLoopSemanticResult analysis runChild samplers params inputs joint)
    (executionMatches : evidence.MatchesExecution execution)
    (position : Nat)
    (positionInBounds : position < execution.evaluatedCount.toNat) :
    Nonempty (ParallelFamilyLaneInstance evidence execution position) := by
  have member : position ∈ List.range execution.evaluatedCount.toNat := by
    simpa using positionInBounds
  obtain ⟨childParams, childInputs, childValues, _childMember,
      evaluatedBindings, bindingsEvaluate, paramsEq, inputsEq, exactMember⟩ :=
    execution.executionTrace.everyChild
      (fun index childParams childInputs childValues ↦
        ∃ evaluatedBindings,
          Mxx.Ir.evaluateBindings
              ((.loopIndex execution.indexSlot, .integer index) :: params) execution.bindings =
                some evaluatedBindings ∧
          childParams = evaluatedBindings ++
            ((.loopIndex execution.indexSlot, .integer index) :: params) ∧
          childInputs = ((execution.modes.zip execution.argumentValues).map fun (mode, value) ↦
            Mxx.Ir.loopArgument mode index value) ∧
          childValues ∈ runChild execution.definition childParams childInputs)
      (by
        intro index evaluatedBindings childValues evaluated member
        exact ⟨evaluatedBindings, evaluated, rfl, rfl, member⟩)
      position member
  subst childParams
  subst childInputs
  exact ⟨{
    executionMatches
    positionInBounds
    evaluatedBindings
    childValues
    bindingsEvaluate
    childMember := exactMember
  }⟩

end Mxx.Certificate
