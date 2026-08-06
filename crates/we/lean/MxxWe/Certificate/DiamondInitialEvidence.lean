import Mxx.Certificate.Rules.BggThreeTraceInitial
import MxxGadgets.Proofs.InputInjector

namespace MxxWe.Certificate

noncomputable section

/-! # Diamond initial BGG evidence

The shared three-trace theorem starts at the initial carried values of the depth recurrences.
Diamond obtains its initial encoding vectors by first running the reusable input injector and
then applying preimages, packing, and instance/witness selectors.  This module fixes that
application proof to the same closed-protocol execution and records the one fail-closed semantic
boundary that remains: closed normalization of those later operations into one modular BGG lane
relation per actual family element.

This is proof-only evidence.  It is not accepted from the Rust certificate, protocol declaration,
or endpoint metadata.
-/

/-- Selecting one of two complete BGG lanes with one shared Boolean selector preserves the BGG
relation.  This is the algebraic rule needed by Diamond's vector, public-key, and plaintext
initialization selects.  Runtime use of this theorem must still prove from the actual three
`select` nodes that they use the same selector and corresponding branch positions. -/
def selectBggEncodingLane
    (selector : Bool)
    {secret gadget : Mxx.Matrix}
    {leftEncryptionPublicKey leftVector leftEncodingPublicKey leftPlaintext : Mxx.Matrix}
    {rightEncryptionPublicKey rightVector rightEncodingPublicKey rightPlaintext : Mxx.Matrix}
    (left : Mxx.Certificate.BggEncodingLaneRelation secret gadget leftEncryptionPublicKey
      leftVector leftEncodingPublicKey leftPlaintext)
    (right : Mxx.Certificate.BggEncodingLaneRelation secret gadget rightEncryptionPublicKey
      rightVector rightEncodingPublicKey rightPlaintext) :
    Mxx.Certificate.BggEncodingLaneRelation secret gadget
      (if selector then rightEncryptionPublicKey else leftEncryptionPublicKey)
      (if selector then rightVector else leftVector)
      (if selector then rightEncodingPublicKey else leftEncodingPublicKey)
      (if selector then rightPlaintext else leftPlaintext) := by
  cases selector
  · exact left
  · exact right

/-- Exact Diamond-owned prerequisite for constructing the initial three-trace invariant.

`injectorSoundness` is indexed by the real workflow execution and the real input-injector
recurrence trace.  `injectorStageMatches` prevents a proof about another workflow stage from being
combined with the BGG recurrence.  Finally, `lanes` is indexed by all five families extracted
from the actual initial carried values; consequently it cannot describe a reordered or truncated
family.

The remaining generic gadget theorem must construct `lanes` from `injectorSoundness.finalProjection`
plus the actual preimage, packing, and select execution paths.  Until that theorem exists, this
structure is an explicit prerequisite and no initial BGG invariant is inferred.
-/
structure DiamondInitialEvidencePrerequisite
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : Mxx.Certificate.ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {trace : Mxx.Certificate.ClosedProtocolExecutionTrace samplers protocol.bundle parameters
      inputs}
    {interface : Mxx.Certificate.ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (threeTrace : Mxx.Certificate.TraceBoundBggThreeTrace analysis trace interface)
    {request : MxxGadgets.InputInjector.ProjectionRequest}
    (validated : MxxGadgets.InputInjector.ValidatedInputInjectorFacts analysis trace.workflow
      request)
    (environment : Mxx.Certificate.FactEnvironment)
    {injectorStage : Mxx.Certificate.StageExecution samplers}
    {injectorRecurrence : Mxx.Certificate.SequentialRecurrenceInstanceRef}
    (injectorEvidence : Mxx.Certificate.TraceBoundSequentialRecurrence analysis injectorStage
      injectorRecurrence)
    (injectorSoundness : MxxGadgets.InputInjector.ValidatedInputInjectorSoundnessResult validated
      environment injectorEvidence)
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (secret gadget : Mxx.Matrix) : Type where
  injectorStageMatches : injectorStage = threeTrace.decryptionStageAt.execution
  view : Mxx.Certificate.BggThreeTraceRuntimeView interface.checked.candidate.slots q
    ringDimension outputRows secretColumns publicColumns threeTrace.EncryptionInitialState
    threeTrace.DecryptionInitialState threeTrace.BooleanInitialState
  secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension outputRows secretColumns
  gadgetLayout : Mxx.Toolkit.MatrixLayout gadget q ringDimension secretColumns publicColumns
  lanes : Mxx.Certificate.BggThreeTraceInitialLaneEvidence q ringDimension outputRows
    secretColumns publicColumns secret gadget view.encryptionPublicKeys view.encodingVectors
    view.encodingPublicKeys view.plaintextMatrices view.booleanValues

/-- Forget only the Diamond/input-injector ownership witness after all runtime families and lane
relations have been tied to the exact three traces. -/
def DiamondInitialEvidencePrerequisite.toSharedInitialEvidence
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : Mxx.Certificate.ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {trace : Mxx.Certificate.ClosedProtocolExecutionTrace samplers protocol.bundle parameters
      inputs}
    {interface : Mxx.Certificate.ResolvedBggThreeTraceInterface analysis protocol.bundle}
    {threeTrace : Mxx.Certificate.TraceBoundBggThreeTrace analysis trace interface}
    {request : MxxGadgets.InputInjector.ProjectionRequest}
    {validated : MxxGadgets.InputInjector.ValidatedInputInjectorFacts analysis trace.workflow
      request}
    {environment : Mxx.Certificate.FactEnvironment}
    {injectorStage : Mxx.Certificate.StageExecution samplers}
    {injectorRecurrence : Mxx.Certificate.SequentialRecurrenceInstanceRef}
    {injectorEvidence : Mxx.Certificate.TraceBoundSequentialRecurrence analysis injectorStage
      injectorRecurrence}
    {injectorSoundness : MxxGadgets.InputInjector.ValidatedInputInjectorSoundnessResult validated
      environment injectorEvidence}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret gadget : Mxx.Matrix}
    (prerequisite : DiamondInitialEvidencePrerequisite threeTrace validated environment
      injectorEvidence injectorSoundness q ringDimension outputRows secretColumns publicColumns
      secret gadget) :
    threeTrace.InitialEvidence q ringDimension outputRows secretColumns publicColumns secret
      gadget := {
  view := prerequisite.view
  secretLayout := prerequisite.secretLayout
  gadgetLayout := prerequisite.gadgetLayout
  lanes := prerequisite.lanes
}

/-- Construct the initial quotient invariant only after the complete trace-indexed Diamond
prerequisite has been proved. -/
noncomputable def DiamondInitialEvidencePrerequisite.toQuotientState
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : Mxx.Certificate.ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : Mxx.Certificate.AnalysisResult}
    {trace : Mxx.Certificate.ClosedProtocolExecutionTrace samplers protocol.bundle parameters
      inputs}
    {interface : Mxx.Certificate.ResolvedBggThreeTraceInterface analysis protocol.bundle}
    {threeTrace : Mxx.Certificate.TraceBoundBggThreeTrace analysis trace interface}
    {request : MxxGadgets.InputInjector.ProjectionRequest}
    {validated : MxxGadgets.InputInjector.ValidatedInputInjectorFacts analysis trace.workflow
      request}
    {environment : Mxx.Certificate.FactEnvironment}
    {injectorStage : Mxx.Certificate.StageExecution samplers}
    {injectorRecurrence : Mxx.Certificate.SequentialRecurrenceInstanceRef}
    {injectorEvidence : Mxx.Certificate.TraceBoundSequentialRecurrence analysis injectorStage
      injectorRecurrence}
    {injectorSoundness : MxxGadgets.InputInjector.ValidatedInputInjectorSoundnessResult validated
      environment injectorEvidence}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret gadget : Mxx.Matrix}
    (prerequisite : DiamondInitialEvidencePrerequisite threeTrace validated environment
      injectorEvidence injectorSoundness q ringDimension outputRows secretColumns publicColumns
      secret gadget) :
    Mxx.Certificate.BggThreeTraceQuotientState interface.checked.candidate.slots q ringDimension
      outputRows secretColumns publicColumns
      (Mxx.Toolkit.matrixValue q ringDimension outputRows secretColumns secret)
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns gadget)
      threeTrace.EncryptionInitialState threeTrace.DecryptionInitialState
      threeTrace.BooleanInitialState :=
  prerequisite.toSharedInitialEvidence.toQuotientState

end


end MxxWe.Certificate
