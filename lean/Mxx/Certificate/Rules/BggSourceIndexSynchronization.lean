import Mxx.Certificate.Rules.BggLayerExecution
import Mxx.Certificate.Rules.PointwiseFormulaExecution
import Mxx.Certificate.Rules.RecurrenceCoupling

namespace Mxx.Certificate

/-!
# Closed BGG source-index synchronization

This module assigns the left/right source roles by inspecting the actual `familyGetDynamic`
producers retained by the closed lane matcher.  It then resolves their common protocol-input
origins against one `ClosedProtocolExecutionTrace`.  No source index or source equality is an
argument to either checker.
-/

/-- Resolve the direct loop control which feeds the index input of one matrix-family lookup.
The formula must be an atom produced by an actual `familyGetDynamic`; wrappers or unrelated
integer controls fail closed. -/
def checkedMatrixSourceControl?
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    (controls : List (CheckedDirectLoopControl bundle interface destinationFor))
    (lane : CheckedRecurrenceLaneOutput interface)
    (formula : FrozenPointwiseMatrixFormula) :
    Option (CheckedDirectLoopControl bundle interface destinationFor) := do
  let .atom scope wire := formula | none
  let body ← scopeAtStaticPath? interface.program scope
  let node ← body.nodes[wire.node]?
  guard (wire.port = 0)
  let .familyGetDynamic := node.kind | none
  let [_, indexWire] := node.arguments | none
  let site : CoreNodeRef := {
    stage := interface.transfer.source.loop.site.stage
    scope
    node := ⟨wire.node⟩
  }
  let [slot] ← lane.outputSlice.projectInputToOuterScopeAny? interface.program site indexWire
    | none
  controls.find? fun control => control.argumentIndex.val = slot

/-- Boolean-family counterpart of `checkedMatrixSourceControl?`. -/
def checkedBooleanSourceControl?
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    {destinationFor : String → ProtocolInputDestination}
    (controls : List (CheckedDirectLoopControl bundle interface destinationFor))
    (lane : CheckedRecurrenceLaneOutput interface)
    (formula : FrozenPointwiseScalarFormula) :
    Option (CheckedDirectLoopControl bundle interface destinationFor) := do
  let .atom wire := formula | none
  let scope := lane.gateSelection.site.scope
  let body ← scopeAtStaticPath? interface.program scope
  let node ← body.nodes[wire.node]?
  guard (wire.port = 0)
  let .familyGetDynamic := node.kind | none
  let [_, indexWire] := node.arguments | none
  let site : CoreNodeRef := {
    stage := interface.transfer.source.loop.site.stage
    scope
    node := ⟨wire.node⟩
  }
  let [slot] ← lane.outputSlice.projectInputToOuterScopeAny? interface.program site indexWire
    | none
  controls.find? fun control => control.argumentIndex.val = slot

/-- Closed role assignment for every executable lane participating in one BGG step. -/
structure CheckedBggSourceIndexRoles
    {bundle : ClosedProtocolBundle}
    (checked : CheckedBggThreeTraceInterface bundle) where
  encryptionLeft : CheckedDirectLoopControl bundle checked.candidate.prefilter.encryption
    checked.encryptionDestination
  encryptionRight : CheckedDirectLoopControl bundle checked.candidate.prefilter.encryption
    checked.encryptionDestination
  vectorLeft : CheckedDirectLoopControl bundle checked.candidate.prefilter.decryption
    checked.decryptionDestination
  vectorRight : CheckedDirectLoopControl bundle checked.candidate.prefilter.decryption
    checked.decryptionDestination
  publicKeyLeft : CheckedDirectLoopControl bundle checked.candidate.prefilter.decryption
    checked.decryptionDestination
  publicKeyRight : CheckedDirectLoopControl bundle checked.candidate.prefilter.decryption
    checked.decryptionDestination
  plaintextLeft : CheckedDirectLoopControl bundle checked.candidate.prefilter.decryption
    checked.decryptionDestination
  plaintextRight : CheckedDirectLoopControl bundle checked.candidate.prefilter.decryption
    checked.decryptionDestination
  booleanLeft : CheckedDirectLoopControl bundle
    checked.candidate.prefilter.booleanInterpreter checked.booleanDestination
  booleanRight : CheckedDirectLoopControl bundle
    checked.candidate.prefilter.booleanInterpreter checked.booleanDestination
  leftOrigin : ProtocolInputId
  rightOrigin : ProtocolInputId
  leftOrigins : [encryptionLeft.protocolInput, vectorLeft.protocolInput,
      publicKeyLeft.protocolInput, plaintextLeft.protocolInput, booleanLeft.protocolInput].all
    (· = leftOrigin) = true
  rightOrigins : [encryptionRight.protocolInput, vectorRight.protocolInput,
      publicKeyRight.protocolInput, plaintextRight.protocolInput, booleanRight.protocolInput].all
    (· = rightOrigin) = true

/-- Strengthen the static three-trace match by discovering left/right roles from the frozen
lookup DAG.  Every role is recomputed; none is selected by a caller. -/
def checkBggSourceIndexRoles
    {bundle : ClosedProtocolBundle}
    (checked : CheckedBggThreeTraceInterface bundle) :
    Option (CheckedBggSourceIndexRoles checked) := do
  let coupling := checked.gateFormulaCoupling
  let encryptionLeft ← checkedMatrixSourceControl? checked.encryptionControls
    checked.encryptionLaneControl.lane coupling.encryptionPublicKey.skeleton.leftFormula
  let encryptionRight ← checkedMatrixSourceControl? checked.encryptionControls
    checked.encryptionLaneControl.lane coupling.encryptionPublicKey.skeleton.rightFormula
  let vectorLeft ← checkedMatrixSourceControl? checked.decryptionControls
    checked.encodingVectorLane.binding.lane coupling.encodingVector.leftFormula
  let vectorRight ← checkedMatrixSourceControl? checked.decryptionControls
    checked.encodingVectorLane.binding.lane coupling.encodingVector.rightFormula
  let publicKeyLeft ← checkedMatrixSourceControl? checked.decryptionControls
    checked.decryptionPublicKeyLane.binding.lane
    coupling.decryptionPublicKey.skeleton.leftFormula
  let publicKeyRight ← checkedMatrixSourceControl? checked.decryptionControls
    checked.decryptionPublicKeyLane.binding.lane
    coupling.decryptionPublicKey.skeleton.rightFormula
  let plaintextLeft ← checkedMatrixSourceControl? checked.decryptionControls
    checked.plaintextLane.binding.lane coupling.plaintext.skeleton.leftFormula
  let plaintextRight ← checkedMatrixSourceControl? checked.decryptionControls
    checked.plaintextLane.binding.lane coupling.plaintext.skeleton.rightFormula
  let booleanLeft ← checkedBooleanSourceControl? checked.booleanControls
    checked.booleanLaneControl.lane checked.booleanGateSkeleton.leftFormula
  let booleanRight ← checkedBooleanSourceControl? checked.booleanControls
    checked.booleanLaneControl.lane checked.booleanGateSkeleton.rightFormula
  let leftOrigin := encryptionLeft.protocolInput
  let rightOrigin := encryptionRight.protocolInput
  if leftOrigins : [encryptionLeft.protocolInput, vectorLeft.protocolInput,
      publicKeyLeft.protocolInput, plaintextLeft.protocolInput, booleanLeft.protocolInput].all
        (· = leftOrigin) = true then
    if rightOrigins : [encryptionRight.protocolInput, vectorRight.protocolInput,
        publicKeyRight.protocolInput, plaintextRight.protocolInput,
        booleanRight.protocolInput].all (· = rightOrigin) = true then
      some {
        encryptionLeft, encryptionRight, vectorLeft, vectorRight, publicKeyLeft, publicKeyRight,
        plaintextLeft, plaintextRight, booleanLeft, booleanRight, leftOrigin, rightOrigin,
        leftOrigins, rightOrigins
      }
    else none
  else none

end Mxx.Certificate
