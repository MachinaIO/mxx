import MxxWe.DiamondWE.Operational
import MxxGadgets.SelectorMagnitude

namespace Mxx.We.DiamondWE

open Mxx.IR
open Mxx.Primitives

/- Every field below is recovered from the same stored program lookup.  This helper prevents a
   generated sampler reference from pairing the payload of one node with the output of another. -/
theorem StoredNodeRef.storedNodeWitness {program : Program}
    (site : StoredNodeRef program) :
    ∃ stage scope node,
      program.data.stages[site.reference.2.stage]? = some stage ∧
      scopeAt stage site.reference.2.wire.scope = some scope ∧
      nodeAt scope site.reference.2.wire.node = some node ∧
      node.payload = site.payload ∧ node.arguments = site.arguments ∧
      node.outputs = site.outputs := by
  have payloadFact := site.payload_stored
  have argumentsFact := site.arguments_stored
  have outputsFact := site.outputs_stored
  unfold nodePayloadAt nodeAtReference at payloadFact
  unfold nodeArgumentsAt nodeAtReference at argumentsFact
  unfold nodeOutputsAt nodeAtReference at outputsFact
  simp only [Option.map_eq_some_iff] at payloadFact argumentsFact outputsFact
  cases stageStored : program.data.stages[site.reference.2.stage]? with
  | none => simp [stageStored] at payloadFact
  | some stage =>
      cases scopeStored : scopeAt stage site.reference.2.wire.scope with
      | none => simp [stageStored, scopeStored] at payloadFact
      | some scope =>
          cases nodeStored : nodeAt scope site.reference.2.wire.node with
          | none => simp [stageStored, scopeStored, nodeStored] at payloadFact
          | some node =>
              simp [stageStored, scopeStored, nodeStored] at payloadFact argumentsFact outputsFact
              exact ⟨stage, scope, node, rfl, scopeStored, nodeStored, payloadFact,
                argumentsFact, outputsFact⟩

/- The occurrence path is dynamic, but sampler identity is entirely determined by the stored
   node and output wire.  `GoodSamples` can therefore be queried for a nested Gaussian or family
   preimage occurrence without a provenance string or raw node-number assumption. -/
noncomputable def StoredNodeRef.sampleRefAt {program : Program}
    (site : StoredNodeRef program) (path : OccurrencePath)
    (isSampler : samplerPayload site.payload = true) : SampleRef program.data := by
  exact {
    occurrence := occurrenceOf site.reference.2.stage path site.reference.2.wire
    payload := site.payload
    outputType := site.reference.1
    programValid := program.valid
    occurrenceValid := by
      rcases site.storedNodeWitness with
        ⟨stage, scope, node, stageStored, scopeStored, nodeStored, _payloadStored,
          _argumentsStored, outputsStored⟩
      have typeCorrect := site.reference.2.type_correct
      simp [wireTypeAt, Stage.wireType?, Mxx.IR.wireType?, stageStored, scopeStored,
        nodeStored, outputsStored] at typeCorrect
      refine ⟨stage, scope, node, stageStored, scopeStored, nodeStored, ?_⟩
      dsimp [occurrenceOf]
      rw [outputsStored]
      exact (Array.getElem?_eq_some_iff.mp typeCorrect.2).1
    storedPayload := by
      rcases site.storedNodeWitness with
        ⟨stage, scope, node, stageStored, scopeStored, nodeStored, payloadStored, _⟩
      exact ⟨stage, scope, node, stageStored, scopeStored, nodeStored, payloadStored⟩
    storedOutput := by
      rcases site.storedNodeWitness with
        ⟨stage, scope, node, stageStored, scopeStored, nodeStored, _payloadStored,
          _argumentsStored, outputsStored⟩
      have typeCorrect := site.reference.2.type_correct
      simp [wireTypeAt, Stage.wireType?, Mxx.IR.wireType?, stageStored, scopeStored,
        nodeStored, outputsStored] at typeCorrect
      refine ⟨stage, scope, node, stageStored, scopeStored, nodeStored, ?_⟩
      dsimp [occurrenceOf]
      rw [outputsStored]
      exact typeCorrect.2
    isSampler := isSampler
  }

/- A reached primitive at the same typed site supplies liveness for the generated sampler
   reference.  Both sides use the identical path and wire occurrence, so no external occurrence
   equality is required. -/
theorem StoredNodeRef.sampleRefAt_reached_of_run
    {backend : SemanticBackend} {program : Program} {trace : Trace backend}
    (site : StoredNodeRef program) (path : OccurrencePath)
    (isSampler : samplerPayload site.payload = true) (structural : StructuralEnv)
    (storedNode : Node)
    (run : ReachedPrimitiveRun trace structural site.reference.2.stage
      site.reference.2.wire.scope site.reference.2.wire.node path site.payload storedNode
      site.reference.2.wire.port) :
    (site.sampleRefAt path isSampler).Reached trace := by
  exact ⟨run.output, by simpa [StoredNodeRef.sampleRefAt] using run.outputTraced⟩

/- A sampler used by a reached consumer is reached at the same dynamic path.  This is how the
   target-add node establishes liveness of its Gaussian operand: argument resolution and trace
   coverage identify the exact stored sampler wire, without treating the sampler as a primitive
   operation or accepting a caller-provided occurrence equality. -/
theorem StoredNodeRef.sampleRefAt_reached_of_consumerArgument
    {backend : SemanticBackend} {program : Program} {trace : Trace backend}
    (samplerSite consumerSite : StoredNodeRef program) (path : OccurrencePath)
    (isSampler : samplerPayload samplerSite.payload = true)
    (sameStage : samplerSite.stage = consumerSite.stage)
    (argumentIndex : Nat)
    (argumentEdge : consumerSite.arguments[argumentIndex]? =
      some samplerSite.reference.2.wire)
    (structural : StructuralEnv) (storedConsumer : Node)
    (argumentsStored : storedConsumer.arguments = consumerSite.arguments)
    (run : ReachedPrimitiveRun trace structural consumerSite.reference.2.stage
      consumerSite.reference.2.wire.scope consumerSite.reference.2.wire.node path
      consumerSite.payload storedConsumer consumerSite.reference.2.wire.port) :
    (samplerSite.sampleRefAt path isSampler).Reached trace := by
  have wireBound : argumentIndex < storedConsumer.arguments.size := by
    rw [argumentsStored]
    exact (Array.getElem?_eq_some_iff.mp argumentEdge).1
  obtain ⟨argumentBound, argumentTraced⟩ :=
    resolvedArgument_trace_at_path run.argumentsResolved run.valuesTraced argumentIndex wireBound
  refine ⟨run.arguments[argumentIndex], ?_⟩
  change traceValueAt trace
      (occurrenceOf samplerSite.reference.2.stage path samplerSite.reference.2.wire) =
    some run.arguments[argumentIndex]
  have referenceStage : samplerSite.reference.2.stage = consumerSite.reference.2.stage := by
    exact samplerSite.ownership_stored.1.trans
      (sameStage.trans consumerSite.ownership_stored.1.symm)
  rw [referenceStage]
  calc
    traceValueAt trace
        (occurrenceOf consumerSite.reference.2.stage path samplerSite.reference.2.wire) =
        traceValueAt trace
          (occurrenceOf consumerSite.reference.2.stage path
            storedConsumer.arguments[argumentIndex]) := by
      have wireEq : storedConsumer.arguments[argumentIndex] =
          samplerSite.reference.2.wire := by
        have storedEdge : storedConsumer.arguments[argumentIndex]? =
            some samplerSite.reference.2.wire := by
          simpa [argumentsStored] using argumentEdge
        exact (Array.getElem?_eq_some_iff.mp storedEdge).2
      rw [wireEq]
    _ = some run.arguments[argumentIndex] := argumentTraced

/- A successful evaluator trace determines the family-preimage sampler fact.  The stored
   `SampleRef.factKind` selects the branch of `GoodSamples`; callers cannot substitute a relation
   for another sampler occurrence. -/
theorem GoodSamples.familyPreimageOccurrence
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {env : RuntimeEvalEnv oracle candidate.program.data} {trace : RuntimeTrace oracle}
    (good : GoodSamples oracle candidate env trace)
    (sample : SampleRef candidate.program.data) (reached : sample.Reached trace)
    (preimageType : MatrixType) (preimageBound : Nat)
    (factKind : sample.factKind = .familyPreimage preimageType preimageBound) :
    Nonempty (FamilyPreimageOccurrenceFact oracle candidate env trace sample
      preimageType preimageBound) := by
  have fact := good.occurrence sample reached
  simp only [SamplerOccurrenceFact, factKind] at fact
  exact fact

/- The same occurrence-indexed lookup recovers the bounded Gaussian matrix used by the target
   addition.  In particular, the cutoff is read from the sampler payload rather than supplied as
   an unrelated noise premise. -/
theorem GoodSamples.cutoffOccurrence
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {env : RuntimeEvalEnv oracle candidate.program.data} {trace : RuntimeTrace oracle}
    (good : GoodSamples oracle candidate env trace)
    (sample : SampleRef candidate.program.data) (reached : sample.Reached trace)
    (matrixType : MatrixType) (bound : Nat)
    (factKind : sample.factKind = .cutoff matrixType bound) :
    Nonempty (CutoffSampleOccurrenceFact oracle candidate env trace sample matrixType bound) := by
  have fact := good.occurrence sample reached
  simp only [SamplerOccurrenceFact, factKind] at fact
  exact fact

/- A cutoff sampler contributes only error: its exact ideal is the zero matrix and its sampled
   matrix is the reduction of the bounded integer lift supplied by `GoodSamples`. -/
noncomputable def CutoffSampleOccurrenceFact.approxZero
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {env : RuntimeEvalEnv oracle candidate.program.data} {trace : RuntimeTrace oracle}
    {sample : SampleRef candidate.program.data} {matrixType : MatrixType} {bound : Nat}
    (fact : CutoffSampleOccurrenceFact oracle candidate env trace sample matrixType bound) :
    ApproxWithin fact.actual 0 bound := {
  toApprox := {
    error := fact.bounded.witness
    equation := by simp [fact.bounded.reduce_eq]
  }
  norm_le := fact.bounded.norm_le
}

/- The same bounded lift is the selector's magnitude witness. For the retained ternary sampler
   the payload cutoff is one, so this fact starts the entire selector scan at magnitude one. -/
noncomputable def CutoffSampleOccurrenceFact.magnitude
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {env : RuntimeEvalEnv oracle candidate.program.data} {trace : RuntimeTrace oracle}
    {sample : SampleRef candidate.program.data} {matrixType : MatrixType} {bound : Nat}
    (fact : CutoffSampleOccurrenceFact oracle candidate env trace sample matrixType bound) :
    MagnitudeFact fact.actual :=
  Mxx.Gadgets.SelectorMagnitude.ofBoundedLift fact.bounded

/- The stored interval `[-1,1]` deterministically classifies as cutoff one. This equation connects
   the typed selector trace to `GoodSamples` without a caller-provided magnitude premise. -/
theorem StoredNodeRef.ternarySampleRef_factKind
    {program : Program} (site : StoredNodeRef program) (path : OccurrencePath)
    (matrixType : MatrixType)
    (payloadStored : site.payload = .uniformIntervalSample matrixType {
      start := .literal (-1), stop := .literal 1 })
    (isSampler : samplerPayload site.payload = true) :
    (site.sampleRefAt path isSampler).factKind = .cutoff matrixType 1 := by
  have cutoff : closedIntervalCutoff? {
      start := .literal (-1), stop := .literal 1 } = some 1 := by
    rfl
  simp [StoredNodeRef.sampleRefAt, SampleRef.factKind, payloadStored, cutoff]

/- The finite seven-site topology identifies the Gaussian sampler consumed by the target add.
   Its payload carries the configured cutoff, its operand edge is exact, and both operations are
   in the same linked stage.  All three facts are consequences of generated program data. -/
theorem InjectorTargetTraceSites.gaussianAddEdge
    {program : Program} {inputCount batchBits stateCount digitBase errorBound : Nat}
    (sites : InjectorTargetTraceSites program inputCount batchBits stateCount digitBase errorBound) :
    ∃ error target matrixType sigma,
      sites.entries[2]? = some error ∧ sites.entries[3]? = some target ∧
      error.site.payload = .gaussianSample matrixType sigma (.literal errorBound) ∧
      target.site.arguments[1]? = some error.site.reference.2.wire ∧
      error.site.stage = target.site.stage := by
  rcases sites.targetAddEdges with
    ⟨product, error, target, productStored, errorStored, targetStored, targetOperands⟩
  obtain ⟨matched, matchedStored, matchedRole, payloadMatch⟩ :=
    sites.traceComplete.2 2 (by decide)
  have matchedEq : matched = error := by
    exact Option.some.inj (matchedStored.symm.trans errorStored)
  subst matched
  have expectedRole : injectorTargetTraceRoles[2]! = .gaussianError := by decide
  rw [expectedRole] at matchedRole
  cases payloadStored : error.site.payload with
  | gaussianSample matrixType sigma bound =>
      cases bound with
      | literal bound =>
        simp [matchedRole, injectorTargetRolePayloadMatches, payloadStored] at payloadMatch
        subst bound
        have targetArgument : target.site.arguments[1]? =
            some error.site.reference.2.wire := by
          rw [target.arguments_eq, targetOperands]
          rfl
        have errorMember : error ∈ sites.entries := by
          have stored := Array.getElem?_eq_some_iff.mp errorStored
          exact Array.mem_iff_getElem.mpr ⟨2, stored.1, stored.2⟩
        have targetMember : target ∈ sites.entries := by
          have stored := Array.getElem?_eq_some_iff.mp targetStored
          exact Array.mem_iff_getElem.mpr ⟨3, stored.1, stored.2⟩
        have errorStage := sites.entryStages error errorMember
        have targetStage := sites.entryStages target targetMember
        exact ⟨error, target, matrixType, sigma, errorStored, targetStored, payloadStored,
          targetArgument, errorStage.trans targetStage.symm⟩
      | _ => simp [matchedRole, injectorTargetRolePayloadMatches, payloadStored] at payloadMatch
  | _ => simp [matchedRole, injectorTargetRolePayloadMatches, payloadStored] at payloadMatch

/- Once the generated payload equation identifies the retained Gaussian node, sampler
   classification reduces to the literal configured cutoff. -/
theorem StoredNodeRef.gaussianSampleRef_factKind
    {program : Program} (site : StoredNodeRef program) (path : OccurrencePath)
    (matrixType : MatrixType) (sigma : RealExpr) (bound : Nat)
    (payloadStored : site.payload = .gaussianSample matrixType sigma (.literal bound))
    (isSampler : samplerPayload site.payload = true) :
    (site.sampleRefAt path isSampler).factKind = .cutoff matrixType bound := by
  have nonnegative : (0 : Int) ≤ (bound : Int) := by omega
  simp only [StoredNodeRef.sampleRefAt, SampleRef.factKind, payloadStored, closedNatural?,
    StructuralIntExpr.eval]
  change (if (0 : Int) ≤ (bound : Int) then
      SamplerFactKind.cutoff matrixType (bound : Int).toNat else SamplerFactKind.invalid) =
    SamplerFactKind.cutoff matrixType bound
  simp [nonnegative]

/- Selecting one group and branch from a family sampler preserves both pieces needed by the
   injector transition: `source * preimage = target` and a bounded integer lift of `preimage`.
   The target may still contain an exact public source term; this projection does not erase it. -/
theorem FamilyPreimageOccurrenceFact.selectedRelationAndLift
    {oracle : Mxx.Runtime.RuntimeGadgetOracle} {candidate : Candidate}
    {env : RuntimeEvalEnv oracle candidate.program.data} {trace : RuntimeTrace oracle}
    {sample : SampleRef candidate.program.data} {preimageType : MatrixType}
    {preimageBound : Nat}
    (fact : FamilyPreimageOccurrenceFact oracle candidate env trace sample
      preimageType preimageBound)
    (group : FamilyIndex fact.groupShape) (branch : FamilyIndex [fact.branchExtent]) :
    RightPreimage (fact.source group)
        (fact.preimage (FamilyIndex.append group branch)).exactMatrix
        (fact.target (FamilyIndex.append group branch)) ∧
      Nonempty (BoundedLift
        (fact.preimage (FamilyIndex.append group branch)).exactMatrix preimageBound) :=
  ⟨fact.relation group branch, fact.bounded group branch⟩

end Mxx.We.DiamondWE
