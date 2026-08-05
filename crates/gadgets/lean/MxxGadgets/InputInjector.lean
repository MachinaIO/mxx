import Mxx.Certificate.Execution
import Mxx.Certificate.Rules.LoopRecurrence
import Mxx.Toolkit.Negacyclic

open Mxx.Toolkit

namespace MxxGadgets.InputInjector

noncomputable section

/-! # Diamond input-injector interface

This module owns the reusable algebra and the typed analysis interface of the Diamond input
injector. It does not describe a standalone protocol, choose an ideal program, or assert semantic
facts. Every exposed identity is obtained from an actual analyzer result or workflow execution.
-/

abbrev RingMatrix (q n rows columns : Nat) :=
  _root_.Matrix (Fin rows) (Fin columns) (Negacyclic q n)

/-- One expected artifact route obtained from generated workflow handles. -/
structure ArtifactRoute where
  sourceStage : String
  sourceOutput : String
  destinationStage : String
  destinationInput : String
  deriving DecidableEq, Repr

def ArtifactRoute.Matches
    (route : ArtifactRoute)
    (binding : Mxx.Certificate.ArtifactBindingExecution) : Prop :=
  binding.sourceStage = route.sourceStage ∧
    binding.sourceOutput = route.sourceOutput ∧
    binding.destinationStage = route.destinationStage ∧
    binding.destinationInput = route.destinationInput

instance (route : ArtifactRoute) (binding : Mxx.Certificate.ArtifactBindingExecution) :
    Decidable (route.Matches binding) := by
  unfold ArtifactRoute.Matches
  infer_instance

/-- Direct handles retained by the DSL workflow builder for the input-injector scopes. -/
structure ProjectionRequest where
  preprocessingProducer : Mxx.Certificate.CoreWireRef
  preprocessingConsumer : Mxx.Certificate.CoreWireRef
  transitionsFamily : Mxx.Certificate.JointFamilyId
  transitionsOutput : Mxx.Certificate.CoreWireRef
  transitionsConsumer : Mxx.Certificate.CoreWireRef
  finalTrapdoorsFamily : Mxx.Certificate.JointFamilyId
  finalTrapdoorsOutput : Mxx.Certificate.CoreWireRef
  inputDigits : Mxx.Certificate.CoreWireRef
  initialStatesFamily : Mxx.Certificate.JointFamilyId
  initialStatesOutput : Mxx.Certificate.CoreWireRef
  outputStates : Mxx.Certificate.FactRecurrenceRef
  initialArtifact : ArtifactRoute
  transitionsArtifact : ArtifactRoute

inductive ProjectionError where
  | missingFact (wire : Mxx.Certificate.CoreWireRef)
  | expectedMatrix (wire : Mxx.Certificate.CoreWireRef)
  | expectedFamily (wire : Mxx.Certificate.CoreWireRef)
  | matrixOriginMismatch (wire origin : Mxx.Certificate.CoreWireRef)
  | missingFamily (family : Mxx.Certificate.JointFamilyId)
  | familyIdentityMismatch (family : Mxx.Certificate.JointFamilyId)
  | familyOutputMismatch (family : Mxx.Certificate.JointFamilyId)
  | missingRecurrence (recurrence : Mxx.Certificate.FactRecurrenceRef)
  | recurrenceInputMismatch (recurrence : Mxx.Certificate.FactRecurrenceRef)
  | missingArtifact (route : ArtifactRoute)
  deriving Repr

private def findScopedFact
    (analysis : Mxx.Certificate.AnalysisResult)
    (wire : Mxx.Certificate.CoreWireRef) :
    Except ProjectionError { fact // fact ∈ analysis.facts ∧ fact.wire = wire } :=
  match found : analysis.facts.find?
      (fun fact : Mxx.Certificate.ScopedWireFact => decide (fact.wire = wire)) with
  | none => .error (.missingFact wire)
  | some fact => .ok ⟨fact, List.mem_of_find?_eq_some found,
      of_decide_eq_true (List.find?_some (p := fun candidate :
        Mxx.Certificate.ScopedWireFact => decide (candidate.wire = wire)) found)⟩

private def findFamily
    (analysis : Mxx.Certificate.AnalysisResult)
    (family : Mxx.Certificate.JointFamilyId) :
    Except ProjectionError { fact // (family, fact) ∈ analysis.families ∧ fact.id = family } :=
  match found : analysis.families.find? (fun entry :
      Mxx.Certificate.JointFamilyId × Mxx.Certificate.JointFamilyFact =>
        decide (entry.1 = family)) with
  | none => .error (.missingFamily family)
  | some entry =>
      have key : entry.1 = family := of_decide_eq_true (List.find?_some
        (p := fun candidate :
          Mxx.Certificate.JointFamilyId × Mxx.Certificate.JointFamilyFact =>
            decide (candidate.1 = family)) found)
      if identity : entry.2.id = family then
        .ok ⟨entry.2, by
          rw [← key]
          exact List.mem_of_find?_eq_some found, identity⟩
      else .error (.familyIdentityMismatch family)

private def findRecurrence
    (analysis : Mxx.Certificate.AnalysisResult)
    (reference : Mxx.Certificate.FactRecurrenceRef) :
    Except ProjectionError { fact // (reference, fact) ∈ analysis.recurrences } :=
  match found : analysis.recurrences.find? (fun entry :
      Mxx.Certificate.FactRecurrenceRef × Mxx.Certificate.FactRecurrence =>
        decide (entry.1 = reference)) with
  | none => .error (.missingRecurrence reference)
  | some entry =>
      have key : entry.1 = reference := of_decide_eq_true (List.find?_some
        (p := fun candidate :
          Mxx.Certificate.FactRecurrenceRef × Mxx.Certificate.FactRecurrence =>
            decide (candidate.1 = reference)) found)
      .ok ⟨entry.2, by
        rw [← key]
        exact List.mem_of_find?_eq_some found⟩

private def findArtifact
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow params inputs)
    (route : ArtifactRoute) :
    Except ProjectionError { binding //
      binding ∈ execution.artifactBindings ∧ route.Matches binding } :=
  match found : execution.artifactBindings.find? (fun binding :
      Mxx.Certificate.ArtifactBindingExecution => decide (route.Matches binding)) with
  | none => .error (.missingArtifact route)
  | some binding => .ok ⟨binding, List.mem_of_find?_eq_some found,
      of_decide_eq_true (List.find?_some
        (p := fun candidate : Mxx.Certificate.ArtifactBindingExecution =>
          decide (route.Matches candidate)) found)⟩

private def containsFamilyFact
    (aggregate : Mxx.Certificate.FamilyAggregateRef) :
    List Mxx.Certificate.ValueFact → Bool
  | [] => false
  | .family fact :: tail =>
      (fact.aggregate == aggregate) || containsFamilyFact aggregate tail
  | _ :: tail => containsFamilyFact aggregate tail

private def rootFamilyAggregate
    (joint : Mxx.Certificate.JointFamilyId)
    (outputSlot : Nat) : Mxx.Certificate.FamilyAggregateRef :=
  .joint joint outputSlot []

private structure Evidence (statement : Prop) : Type where
  proof : statement
  token : Unit

private def requireMatrixFact
    {analysis : Mxx.Certificate.AnalysisResult}
    {wire : Mxx.Certificate.CoreWireRef}
    (foundFact : { fact // fact ∈ analysis.facts ∧ fact.wire = wire }) :
    Except ProjectionError { fact // foundFact.val.fact = .matrix fact } :=
  match _equation : foundFact.val.fact with
  | .matrix fact => .ok ⟨fact, rfl⟩
  | _ => .error (.expectedMatrix wire)

private def requireFamilyFact
    {analysis : Mxx.Certificate.AnalysisResult}
    {wire : Mxx.Certificate.CoreWireRef}
    (foundFact : { fact // fact ∈ analysis.facts ∧ fact.wire = wire }) :
    Except ProjectionError { fact // foundFact.val.fact = .family fact } :=
  match _equation : foundFact.val.fact with
  | .family fact => .ok ⟨fact, rfl⟩
  | _ => .error (.expectedFamily wire)

private def requireFamilyOutput
    (family : Mxx.Certificate.JointFamilyId)
    (outputs : List Mxx.Certificate.CoreWireRef)
    (wire : Mxx.Certificate.CoreWireRef) :
    Except ProjectionError (Evidence (wire ∈ outputs)) :=
  match found : outputs.find? (fun candidate : Mxx.Certificate.CoreWireRef =>
      decide (candidate = wire)) with
  | none => .error (.familyOutputMismatch family)
  | some candidate =>
      have key : candidate = wire := of_decide_eq_true (List.find?_some
        (p := fun output : Mxx.Certificate.CoreWireRef => decide (output = wire)) found)
      .ok ⟨by
        rw [← key]
        exact List.mem_of_find?_eq_some found, ()⟩

/-- Analyzer- and execution-validated facts for the reusable input-injector scopes. -/
structure ValidatedInputInjectorFacts
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (analysis : Mxx.Certificate.AnalysisResult)
    (execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow params inputs)
    (request : ProjectionRequest) where
  preprocessing : { fact // fact ∈ analysis.facts ∧
    fact.wire = request.preprocessingConsumer }
  preprocessingMatrix : Mxx.Certificate.MatrixFact
  preprocessingIsMatrix : preprocessing.val.fact = .matrix preprocessingMatrix
  preprocessingOrigin : preprocessingMatrix.subject =
    .ofCoreWire request.preprocessingProducer
  transitions : { fact // (request.transitionsFamily, fact) ∈ analysis.families ∧
    fact.id = request.transitionsFamily }
  transitionsOutput : request.transitionsOutput ∈ transitions.val.outputFamilies
  transitionsInput : { fact // fact ∈ analysis.facts ∧
    fact.wire = request.transitionsConsumer }
  transitionsInputFact : Mxx.Certificate.FamilyFact
  transitionsInputIsFamily : transitionsInput.val.fact = .family transitionsInputFact
  transitionsInputIdentity : transitionsInputFact.aggregate =
    rootFamilyAggregate transitions.val.id 0
  finalTrapdoors : { fact // (request.finalTrapdoorsFamily, fact) ∈ analysis.families ∧
    fact.id = request.finalTrapdoorsFamily }
  finalTrapdoorsOutput : request.finalTrapdoorsOutput ∈ finalTrapdoors.val.outputFamilies
  inputDigits : { fact // fact ∈ analysis.facts ∧ fact.wire = request.inputDigits }
  inputDigitsFact : Mxx.Certificate.FamilyFact
  inputDigitsIsFamily : inputDigits.val.fact = .family inputDigitsFact
  initialStates : { fact // (request.initialStatesFamily, fact) ∈ analysis.families ∧
    fact.id = request.initialStatesFamily }
  initialStatesOutput : request.initialStatesOutput ∈ initialStates.val.outputFamilies
  outputStates : { fact // (request.outputStates, fact) ∈ analysis.recurrences }
  recurrenceInitial : containsFamilyFact (rootFamilyAggregate initialStates.val.id 0)
    outputStates.val.initial.toList = true
  recurrenceDigits : containsFamilyFact inputDigitsFact.aggregate
    outputStates.val.invariantInputs = true
  recurrenceTransitions : containsFamilyFact (rootFamilyAggregate transitions.val.id 0)
    outputStates.val.invariantInputs = true
  initialArtifact : { binding // binding ∈ execution.artifactBindings ∧
    request.initialArtifact.Matches binding }
  transitionsArtifact : { binding // binding ∈ execution.artifactBindings ∧
    request.transitionsArtifact.Matches binding }

/-- Fail-closed projection from generic analyzer output to the input-injector interface. -/
def projectInputInjectorFacts
    {samplers : Mxx.MxxSamplerFamily}
    {workflow : Mxx.Ir.Workflow}
    {params : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (analysis : Mxx.Certificate.AnalysisResult)
    (execution : Mxx.Certificate.WorkflowExecutionTrace samplers workflow params inputs)
    (request : ProjectionRequest) :
    Except ProjectionError (ValidatedInputInjectorFacts analysis execution request) := do
  let preprocessing ← findScopedFact analysis request.preprocessingConsumer
  let preprocessingMatrix ← requireMatrixFact preprocessing
  let origin ← if origin : preprocessingMatrix.val.subject =
      Mxx.Certificate.ValueInstanceRef.ofCoreWire request.preprocessingProducer then
    pure (⟨origin, ()⟩ : Evidence (preprocessingMatrix.val.subject =
      Mxx.Certificate.ValueInstanceRef.ofCoreWire request.preprocessingProducer))
  else throw (.matrixOriginMismatch request.preprocessingConsumer request.preprocessingProducer)
  let transitions ← findFamily analysis request.transitionsFamily
  let output ← requireFamilyOutput request.transitionsFamily transitions.val.outputFamilies
    request.transitionsOutput
  let transitionsInput ← findScopedFact analysis request.transitionsConsumer
  let transitionsInputFact ← requireFamilyFact transitionsInput
  let transitionIdentity ← if transitionIdentity :
      transitionsInputFact.val.aggregate = rootFamilyAggregate transitions.val.id 0 then
    pure (⟨transitionIdentity, ()⟩ : Evidence
      (transitionsInputFact.val.aggregate = rootFamilyAggregate transitions.val.id 0))
  else throw (.familyIdentityMismatch request.transitionsFamily)
  let finalTrapdoors ← findFamily analysis request.finalTrapdoorsFamily
  let finalOutput ← requireFamilyOutput request.finalTrapdoorsFamily
    finalTrapdoors.val.outputFamilies request.finalTrapdoorsOutput
  let inputDigits ← findScopedFact analysis request.inputDigits
  let inputDigitsFact ← requireFamilyFact inputDigits
  let initialStates ← findFamily analysis request.initialStatesFamily
  let initialOutput ← requireFamilyOutput request.initialStatesFamily
    initialStates.val.outputFamilies request.initialStatesOutput
  let outputStates ← findRecurrence analysis request.outputStates
  let recurrenceInitialProof ← if recurrenceInitial :
      containsFamilyFact (rootFamilyAggregate initialStates.val.id 0)
        outputStates.val.initial.toList = true then
    pure (⟨recurrenceInitial, ()⟩ : Evidence
      (containsFamilyFact (rootFamilyAggregate initialStates.val.id 0)
        outputStates.val.initial.toList = true))
  else throw (ProjectionError.recurrenceInputMismatch request.outputStates)
  let recurrenceDigitsProof ← if recurrenceDigits :
      containsFamilyFact inputDigitsFact.val.aggregate outputStates.val.invariantInputs = true then
    pure (⟨recurrenceDigits, ()⟩ : Evidence
      (containsFamilyFact inputDigitsFact.val.aggregate
        outputStates.val.invariantInputs = true))
  else throw (ProjectionError.recurrenceInputMismatch request.outputStates)
  let recurrenceTransitionsProof ← if recurrenceTransitions :
      containsFamilyFact (rootFamilyAggregate transitions.val.id 0)
        outputStates.val.invariantInputs = true then
    pure (⟨recurrenceTransitions, ()⟩ : Evidence
      (containsFamilyFact (rootFamilyAggregate transitions.val.id 0)
        outputStates.val.invariantInputs = true))
  else throw (ProjectionError.recurrenceInputMismatch request.outputStates)
  let initialArtifact ← findArtifact execution request.initialArtifact
  let transitionsArtifact ← findArtifact execution request.transitionsArtifact
  return {
    preprocessing
    preprocessingMatrix := preprocessingMatrix.val
    preprocessingIsMatrix := preprocessingMatrix.property
    preprocessingOrigin := origin.proof
    transitions
    transitionsOutput := output.proof
    transitionsInput
    transitionsInputFact := transitionsInputFact.val
    transitionsInputIsFamily := transitionsInputFact.property
    transitionsInputIdentity := transitionIdentity.proof
    finalTrapdoors
    finalTrapdoorsOutput := finalOutput.proof
    inputDigits
    inputDigitsFact := inputDigitsFact.val
    inputDigitsIsFamily := inputDigitsFact.property
    initialStates
    initialStatesOutput := initialOutput.proof
    outputStates
    recurrenceInitial := recurrenceInitialProof.proof
    recurrenceDigits := recurrenceDigitsProof.proof
    recurrenceTransitions := recurrenceTransitionsProof.proof
    initialArtifact
    transitionsArtifact
  }

end

end MxxGadgets.InputInjector
