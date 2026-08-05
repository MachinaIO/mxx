import MxxGadgets.InputInjector
import Mxx.Certificate.Semantics
import Mathlib.Tactic

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

/-- Input-injector loop soundness is an instance of the core recurrence theorem. -/
theorem sequential_scan_preserves
    {runChild : Mxx.Ir.ChildRunner}
    {definition : String}
    {params : Mxx.Ir.ParamEnvironment}
    {indexSlot count carriedArity : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariantArguments : List Mxx.Ir.Value}
    {statePredicate : List Mxx.Ir.Value → Prop}
    (bodyAnalysisSound : ∀ index state next,
      index < count →
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate state →
      Mxx.Certificate.ExecutesLoopBody runChild definition params indexSlot bindings
        invariantArguments index state next →
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate next) :
    ∀ {indices initial final},
      Mxx.Ir.SequentialIterationsTrace runChild definition params indexSlot bindings
        invariantArguments indices initial final →
      (∀ index ∈ indices, index < count) →
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate initial →
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate final :=
  Mxx.Certificate.sequentialIterationsTrace_recurrence bodyAnalysisSound

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
      (request.outputStates, validated.outputStates.val) ∈ analysis.recurrences ∧
      validated.initialArtifact.val ∈ execution.artifactBindings ∧
      validated.transitionsArtifact.val ∈ execution.artifactBindings := by
  exact ⟨validated.preprocessing.property.1, validated.transitions.property.1,
    validated.outputStates.property, validated.initialArtifact.property.1,
    validated.transitionsArtifact.property.1⟩

/-- End-to-end reusable input-injector theorem.

The projector fixes the actual preprocessing, family, recurrence, and artifact origins. The
semantic premise is only the analyzer's global soundness result plus the local-rule composition
for the executable sequential-loop body. No state equation, preimage equation, or bound can be
supplied independently at this layer.
-/
theorem validated_input_injector_end_to_end
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
    {runChild : Mxx.Ir.ChildRunner}
    {definition : String}
    {loopParams : Mxx.Ir.ParamEnvironment}
    {indexSlot count carriedArity : Nat}
    {bindings : List (String × Mxx.Ir.IntExpr)}
    {invariantArguments : List Mxx.Ir.Value}
    {statePredicate : List Mxx.Ir.Value → Prop}
    (bodyAnalysisSound : ∀ index state next,
      index < count →
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate state →
      Mxx.Certificate.ExecutesLoopBody runChild definition loopParams indexSlot bindings
        invariantArguments index state next →
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate next)
    {indices initial final}
    (trace : Mxx.Ir.SequentialIterationsTrace runChild definition loopParams indexSlot bindings
      invariantArguments indices initial final)
    (indicesInRange : ∀ index ∈ indices, index < count)
    (initialHolds : Mxx.Certificate.FactTupleHolds carriedArity statePredicate initial) :
    validated.preprocessing.val.Holds environment ∧
      validated.transitionsInput.val.Holds environment ∧
      validated.inputDigits.val.Holds environment ∧
      Mxx.Certificate.FactTupleHolds carriedArity statePredicate final ∧
      validated.initialArtifact.val.value =
        Mxx.Ir.resolveStageInput validated.initialArtifact.val.protocolInputs
          validated.initialArtifact.val.sourceStages
          (.artifact validated.initialArtifact.val.sourceStage
            validated.initialArtifact.val.sourceOutput) ∧
      validated.transitionsArtifact.val.value =
        Mxx.Ir.resolveStageInput validated.transitionsArtifact.val.protocolInputs
          validated.transitionsArtifact.val.sourceStages
          (.artifact validated.transitionsArtifact.val.sourceStage
            validated.transitionsArtifact.val.sourceOutput) := by
  refine ⟨analysisHolds.2.2.2 _ validated.preprocessing.property.1,
    analysisHolds.2.2.2 _ validated.transitionsInput.property.1,
    analysisHolds.2.2.2 _ validated.inputDigits.property.1, ?_,
    validated.initialArtifact.val.valueEq, validated.transitionsArtifact.val.valueEq⟩
  exact sequential_scan_preserves bodyAnalysisSound trace indicesInRange initialHolds

end

end MxxGadgets.InputInjector
