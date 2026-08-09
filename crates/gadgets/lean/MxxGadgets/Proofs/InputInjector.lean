import MxxGadgets.InputInjector
import Mxx.Certificate.SymbolicEvaluationSoundness
import Mxx.Certificate.Rules.TraceBoundRecurrence

namespace MxxGadgets.InputInjector

noncomputable section

/-- The exact affine rewrite performed by one online transition. -/
theorem transition_affine
    {q n signalColumns stateColumns : Nat}
    (state : RingMatrix q n 1 stateColumns)
    (signal : RingMatrix q n 1 signalColumns)
    (base : RingMatrix q n signalColumns stateColumns)
    (stateNoise : RingMatrix q n 1 stateColumns)
    (transition : RingMatrix q n stateColumns stateColumns)
    (selector : RingMatrix q n signalColumns signalColumns)
    (nextBase transitionNoise : RingMatrix q n signalColumns stateColumns)
    (stateEquation : state = signal * base + stateNoise)
    (transitionEquation : base * transition = selector * nextBase + transitionNoise) :
    state * transition =
      (signal * selector) * nextBase +
        (signal * transitionNoise + stateNoise * transition) := by
  rw [stateEquation, _root_.Matrix.add_mul, _root_.Matrix.mul_assoc,
    transitionEquation, _root_.Matrix.mul_add]
  rw [← _root_.Matrix.mul_assoc]
  abel

/-- Projecting an injected state preserves its signal and right-multiplies only its noise. -/
theorem projection_affine
    {q n signalColumns stateColumns outputColumns : Nat}
    (state : RingMatrix q n 1 stateColumns)
    (signal : RingMatrix q n 1 signalColumns)
    (base : RingMatrix q n signalColumns stateColumns)
    (stateNoise : RingMatrix q n 1 stateColumns)
    (preimage : RingMatrix q n stateColumns outputColumns)
    (target : RingMatrix q n signalColumns outputColumns)
    (stateEquation : state = signal * base + stateNoise)
    (preimageEquation : base * preimage = target) :
    state * preimage = signal * target + stateNoise * preimage := by
  rw [stateEquation, _root_.Matrix.add_mul, _root_.Matrix.mul_assoc, preimageEquation]

/-- Artifact transport uses the value selected by the executable workflow semantics. -/
theorem ValidatedInputInjectorFacts.transitionsArtifact_source_value
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow params inputs}
    {request : ProjectionRequest}
    (validated : ValidatedInputInjectorFacts analysis execution request) :
    validated.transitionsArtifact.val.value =
      Mxx.Ir.resolveStageInput validated.transitionsArtifact.val.protocolInputs
        validated.transitionsArtifact.val.sourceStages
        (.artifact validated.transitionsArtifact.val.sourceStage
          validated.transitionsArtifact.val.sourceOutput) :=
  validated.transitionsArtifact.val.valueEq

/-- Gaussian and preimage randomness contributes only the reviewed hard-support contract. -/
theorem sampled_transition_contract
    (samplers : Mxx.MxxSamplerFamily)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (errorParams preimageParams : Mxx.SamplerParams)
    (publicMatrix target error preimage : Mxx.Matrix)
    (errorMember : error ∈ samplers.gaussianSample errorParams)
    (preimageMember : preimage ∈
      samplers.samplePreimage preimageParams publicMatrix target) :
    Mxx.maxCenteredCoefficientNorm (error.withSamplerParams errorParams) ≤
        errorParams.maxCoefficientBound ∧
      Mxx.MatrixModEq
        (Mxx.matrixMul publicMatrix (preimage.withSamplerParams preimageParams)) target ∧
      Mxx.maxCenteredCoefficientNorm (preimage.withSamplerParams preimageParams) ≤
        preimageParams.maxCoefficientBound := by
  exact ⟨contract.gaussianHardSupport errorParams error errorMember,
    contract.preimageContract preimageParams publicMatrix target preimage preimageMember⟩

/-- A successful projector result contains only facts found in the analyzer and execution. -/
theorem projectInputInjectorFacts_validated
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow params inputs}
    {request : ProjectionRequest}
    {validated : ValidatedInputInjectorFacts analysis execution request}
    (_projected : projectInputInjectorFacts analysis execution request = .ok validated) :
      validated.preprocessing.val ∈ analysis.facts ∧
      (request.transitionsFamily, validated.transitions.val) ∈ analysis.families ∧
      validated.outputStates.val ∈ analysis.symbolicRecurrences ∧
      validated.initialArtifact.val ∈ execution.artifactBindings ∧
      validated.transitionsArtifact.val ∈ execution.artifactBindings := by
  exact ⟨validated.preprocessing.property.1, validated.transitions.property.1,
    validated.outputStates.property.1, validated.initialArtifact.property.1,
    validated.transitionsArtifact.property.1⟩

/-- Proof result for one trace-bound input-injector execution. -/
structure ValidatedInputInjectorSoundnessResult
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {workflowParams : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow workflowParams inputs}
    {request : ProjectionRequest}
    (validated : ValidatedInputInjectorFacts analysis execution request)
    (environment : Mxx.Certificate.FactEnvironment)
    {stageExecution : Mxx.Certificate.StageExecution samplers}
    {recurrenceInstance : Mxx.Certificate.SequentialRecurrenceInstanceRef}
    (recurrenceEvidence : Mxx.Certificate.TraceBoundSequentialRecurrence analysis stageExecution
      recurrenceInstance) where
  preprocessingHolds : validated.preprocessing.val.Holds environment
  transitionsInputHolds : validated.transitionsInput.val.Holds environment
  inputDigitsHold : validated.inputDigits.val.Holds environment
  stageInExecution : stageExecution ∈ execution.stageExecutions
  recurrenceTransfer : recurrenceEvidence.transfer = validated.outputStates.val
  finalProjection :
    Mxx.Certificate.TraceBoundSequentialRecurrence.FinalProjectedCarriedFacts recurrenceEvidence
      environment
  initialArtifactValue : validated.initialArtifact.val.value =
    Mxx.Ir.resolveStageInput validated.initialArtifact.val.protocolInputs
      validated.initialArtifact.val.sourceStages
      (.artifact validated.initialArtifact.val.sourceStage
        validated.initialArtifact.val.sourceOutput)
  transitionsArtifactValue : validated.transitionsArtifact.val.value =
    Mxx.Ir.resolveStageInput validated.transitionsArtifact.val.protocolInputs
      validated.transitionsArtifact.val.sourceStages
      (.artifact validated.transitionsArtifact.val.sourceStage
        validated.transitionsArtifact.val.sourceOutput)

/-- End-to-end reusable input-injector theorem over the trace-bound symbolic recurrence.

The recurrence derivation is indexed by the exact trace stored in `recurrenceEvidence`; callers
cannot provide a runner, definition, invariant arguments, state predicate, or preservation
callback. The selected stage is also required to be a member of the workflow execution trace.
-/
def deriveValidatedInputInjectorSoundness
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {workflowParams : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow workflowParams inputs}
    {request : ProjectionRequest}
    (validated : ValidatedInputInjectorFacts analysis execution request)
    (environment : Mxx.Certificate.FactEnvironment)
    (analysisHolds : Mxx.Certificate.AnalysisHolds environment analysis)
    {stageExecution : Mxx.Certificate.StageExecution samplers}
    (stageInExecution : stageExecution ∈ execution.stageExecutions)
    {recurrenceInstance : Mxx.Certificate.SequentialRecurrenceInstanceRef}
    (recurrenceEvidence : Mxx.Certificate.TraceBoundSequentialRecurrence analysis stageExecution
      recurrenceInstance)
    (recurrenceTransfer : recurrenceEvidence.transfer = validated.outputStates.val)
    (contract : Mxx.MxxBoundedSamplerContract samplers)
    (initialFacts : Mxx.Certificate.CarriedState.Holds stageExecution.params
      recurrenceEvidence.transfer.carriedSchemas
      (recurrenceEvidence.argumentValues.take recurrenceEvidence.view.carriedCount))
    (derivation : Mxx.Certificate.TraceBoundSequentialRecurrence.Derivation recurrenceEvidence
      contract recurrenceEvidence.executionTrace initialFacts) :
    ValidatedInputInjectorSoundnessResult validated environment recurrenceEvidence := {
  preprocessingHolds := analysisHolds.1.2.2.2 _ validated.preprocessing.property.1
  transitionsInputHolds := analysisHolds.1.2.2.2 _ validated.transitionsInput.property.1
  inputDigitsHold := analysisHolds.1.2.2.2 _ validated.inputDigits.property.1
  stageInExecution
  recurrenceTransfer
  finalProjection :=
    Mxx.Certificate.TraceBoundSequentialRecurrence.Derivation.finalProjection
      recurrenceEvidence contract environment analysisHolds.1 derivation
  initialArtifactValue := validated.initialArtifact.val.valueEq
  transitionsArtifactValue := validated.transitionsArtifact.val.valueEq
}

end

end MxxGadgets.InputInjector
