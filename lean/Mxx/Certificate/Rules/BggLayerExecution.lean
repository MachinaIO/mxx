import Mxx.Certificate.Rules.BggThreeTrace

namespace Mxx.Certificate

/-!
# Executed BGG layer assembly

The pointwise trace rules establish one gate result at a time.  This module is the unique
list-level assembly step: it consumes evidence indexed by the actual three output families and
constructs the existing `QuotientBggLayerTransfer`.  It contains no recurrence invariant and no
alternative Boolean or BGG evaluator.
-/

/-- One actual output lane identified with the result of the generic BGG gate theorem.  The
output matrices and Boolean are indices of the type, so a proof for one lane cannot be reused at
another output coordinate. -/
structure QuotientBggExecutedGateLane
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {inputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {inputVectors : List (_root_.Matrix outputRows publicColumns R)}
    {inputBooleans : List Bool}
    (inputs : QuotientBggFamilyRelation secret gadget inputPublicKeys inputVectors inputBooleans)
    {onePublicKey : _root_.Matrix secretColumns publicColumns R}
    {oneVector : _root_.Matrix outputRows publicColumns R}
    (one : QuotientBggLane secret gadget onePublicKey oneVector (1 : R))
    (outputPublicKey : _root_.Matrix secretColumns publicColumns R)
    (outputVector : _root_.Matrix outputRows publicColumns R)
    (outputBoolean : Bool) : Type where
  leftIndex : Nat
  rightIndex : Nat
  left : inputs.LaneAt leftIndex
  right : inputs.LaneAt rightIndex
  gate : BggBooleanGate
  rightDecomposition : _root_.Matrix publicColumns publicColumns R
  decomposition : gadget * rightDecomposition = right.publicKey
  publicKeyEq :
    (QuotientBggGateResult.evaluate gate rightDecomposition decomposition
      one left.lane right.lane).publicKey = outputPublicKey
  vectorEq :
    (QuotientBggGateResult.evaluate gate rightDecomposition decomposition
      one left.lane right.lane).vector = outputVector
  booleanEq : gate.evaluate left.booleanValue right.booleanValue = outputBoolean

/-- Simultaneous evidence for every actual output coordinate.  The constructors force the public
key, vector, and Boolean lists to advance in lockstep. -/
inductive QuotientBggExecutedLayer
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {inputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {inputVectors : List (_root_.Matrix outputRows publicColumns R)}
    {inputBooleans : List Bool}
    (inputs : QuotientBggFamilyRelation secret gadget inputPublicKeys inputVectors inputBooleans)
    {onePublicKey : _root_.Matrix secretColumns publicColumns R}
    {oneVector : _root_.Matrix outputRows publicColumns R}
    (one : QuotientBggLane secret gadget onePublicKey oneVector (1 : R)) :
    List (_root_.Matrix secretColumns publicColumns R) →
      List (_root_.Matrix outputRows publicColumns R) → List Bool → Type where
  | nil : QuotientBggExecutedLayer inputs one [] [] []
  | cons
      {outputPublicKey : _root_.Matrix secretColumns publicColumns R}
      {outputVector : _root_.Matrix outputRows publicColumns R}
      {outputBoolean : Bool}
      {outputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
      {outputVectors : List (_root_.Matrix outputRows publicColumns R)}
      {outputBooleans : List Bool}
      (head : QuotientBggExecutedGateLane inputs one outputPublicKey outputVector outputBoolean)
      (tail : QuotientBggExecutedLayer inputs one outputPublicKeys outputVectors outputBooleans) :
      QuotientBggExecutedLayer inputs one (outputPublicKey :: outputPublicKeys)
        (outputVector :: outputVectors) (outputBoolean :: outputBooleans)

/-- Assemble the existing closed layer-transfer object.  The only transports performed here are
the three equalities proved for the corresponding actual lane. -/
noncomputable def QuotientBggExecutedLayer.toTransfer
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {inputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {inputVectors : List (_root_.Matrix outputRows publicColumns R)}
    {inputBooleans : List Bool}
    {inputs : QuotientBggFamilyRelation secret gadget inputPublicKeys inputVectors inputBooleans}
    {onePublicKey : _root_.Matrix secretColumns publicColumns R}
    {oneVector : _root_.Matrix outputRows publicColumns R}
    {one : QuotientBggLane secret gadget onePublicKey oneVector (1 : R)}
    {outputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {outputVectors : List (_root_.Matrix outputRows publicColumns R)}
    {outputBooleans : List Bool}
    (execution : QuotientBggExecutedLayer inputs one outputPublicKeys outputVectors
      outputBooleans) :
    QuotientBggLayerTransfer inputs one outputPublicKeys outputVectors outputBooleans := by
  induction execution with
  | nil => exact .nil
  | @cons outputPublicKey outputVector outputBoolean outputPublicKeys outputVectors
      outputBooleans head tail induction =>
      rcases head with
        ⟨leftIndex, rightIndex, left, right, gate, rightDecomposition, decomposition,
          publicKeyEq, vectorEq, booleanEq⟩
      subst outputPublicKey
      subst outputVector
      subst outputBoolean
      exact .cons left right gate rightDecomposition decomposition induction

/-- The assembled output relation is therefore the one computed by the generic gate theorem for
every actual coordinate. -/
noncomputable def QuotientBggExecutedLayer.outputRelation
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {inputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {inputVectors : List (_root_.Matrix outputRows publicColumns R)}
    {inputBooleans : List Bool}
    {inputs : QuotientBggFamilyRelation secret gadget inputPublicKeys inputVectors inputBooleans}
    {onePublicKey : _root_.Matrix secretColumns publicColumns R}
    {oneVector : _root_.Matrix outputRows publicColumns R}
    {one : QuotientBggLane secret gadget onePublicKey oneVector (1 : R)}
    {outputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {outputVectors : List (_root_.Matrix outputRows publicColumns R)}
    {outputBooleans : List Bool}
    (execution : QuotientBggExecutedLayer inputs one outputPublicKeys outputVectors
      outputBooleans) :
    QuotientBggFamilyRelation secret gadget outputPublicKeys outputVectors outputBooleans :=
  execution.toTransfer.outputRelation

/-- Pointwise equality of two matrix families, with their common ordering enforced by the list
constructors. -/
inductive QuotientMatrixFamilyEquality
    {rows columns R : Type} :
    List (_root_.Matrix rows columns R) → List (_root_.Matrix rows columns R) → Prop where
  | nil : QuotientMatrixFamilyEquality [] []
  | cons
      {leftHead rightHead : _root_.Matrix rows columns R}
      {leftTail rightTail : List (_root_.Matrix rows columns R)}
      (head : leftHead = rightHead)
      (tail : QuotientMatrixFamilyEquality leftTail rightTail) :
      QuotientMatrixFamilyEquality (leftHead :: leftTail) (rightHead :: rightTail)

theorem QuotientMatrixFamilyEquality.eq
    {rows columns R : Type}
    {left right : List (_root_.Matrix rows columns R)}
    (equality : QuotientMatrixFamilyEquality left right) : left = right := by
  induction equality with
  | nil => rfl
  | cons head tail induction => simp [head, induction]

/-- All pointwise evidence needed for one actual three-trace successor state.  Every family in
the structure is extracted from `nextView`; no field can redirect the proof to another output
wire or reorder one role independently. -/
structure BggThreeTraceExecutedLayer
    {slots : CheckedBggEncodingSlots}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {encryptionState decryptionState booleanState : List Mxx.Ir.Value}
    (current : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionState decryptionState booleanState)
    (encryptionNext decryptionNext booleanNext : List Mxx.Ir.Value) : Type where
  nextView : BggThreeTraceRuntimeView slots q ringDimension outputRows secretColumns publicColumns
    encryptionNext decryptionNext booleanNext
  onePublicKey : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
    (Mxx.Toolkit.Negacyclic q ringDimension)
  oneVector : _root_.Matrix (Fin outputRows) (Fin publicColumns)
    (Mxx.Toolkit.Negacyclic q ringDimension)
  one : QuotientBggLane secret gadget onePublicKey oneVector (1 :
    Mxx.Toolkit.Negacyclic q ringDimension)
  layer : QuotientBggExecutedLayer current.relation one
    (runtimeMatrixValues q ringDimension secretColumns publicColumns
      nextView.encodingPublicKeys)
    (runtimeMatrixValues q ringDimension outputRows publicColumns nextView.encodingVectors)
    nextView.booleanValues
  publicKeys : QuotientMatrixFamilyEquality
    (runtimeMatrixValues q ringDimension secretColumns publicColumns
      nextView.encryptionPublicKeys)
    (runtimeMatrixValues q ringDimension secretColumns publicColumns
      nextView.encodingPublicKeys)
  plaintext : QuotientBooleanMatrixFamilyRelation
    (runtimeMatrixValues q ringDimension 1 1 nextView.plaintextMatrices)
    nextView.booleanValues

/-- Forget only the trace-construction details after they have produced the exact existing step
object consumed by `BggThreeTraceQuotientDerivation`. -/
noncomputable def BggThreeTraceExecutedLayer.toStep
    {slots : CheckedBggEncodingSlots}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {encryptionState decryptionState booleanState encryptionNext decryptionNext booleanNext :
      List Mxx.Ir.Value}
    {current : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionState decryptionState booleanState}
    (execution : BggThreeTraceExecutedLayer current encryptionNext decryptionNext booleanNext) :
    BggThreeTraceQuotientStep slots q ringDimension outputRows secretColumns publicColumns secret
      gadget current encryptionNext decryptionNext booleanNext := {
  nextView := execution.nextView
  onePublicKey := execution.onePublicKey
  oneVector := execution.oneVector
  one := execution.one
  transfer := execution.layer.toTransfer
  publicKeysEqual := execution.publicKeys.eq
  plaintextRelation := execution.plaintext
}

end Mxx.Certificate
