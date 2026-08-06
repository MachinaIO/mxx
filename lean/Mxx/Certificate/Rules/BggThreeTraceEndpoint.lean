import Mxx.Certificate.Rules.BggThreeTrace

namespace Mxx.Certificate

open scoped Matrix

/-!
# Generic endpoint projection for the BGG three-trace relation

This module projects one accepted final Boolean lane from the quotient-ring invariant.  The
caller supplies only an index into the actual final Boolean family and a proof that the value at
that index is `true`.  The synchronized BGG and plaintext relations determine every matrix at
that index; no protocol-specific node number, artifact name, callback, or claimed equation is
accepted.
-/

/-- A Boolean-family lookup determines the corresponding synchronized BGG lane. -/
theorem QuotientBggFamilyRelation.laneAtOfBooleanFound
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {publicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {vectors : List (_root_.Matrix outputRows publicColumns R)}
    {booleanValues : List Bool}
    (relation : QuotientBggFamilyRelation secret gadget publicKeys vectors booleanValues)
    (index : Nat) (booleanValue : Bool)
    (booleanFound : booleanValues[index]? = some booleanValue) :
    Nonempty (relation.LaneAt index) := by
  induction relation generalizing index with
  | nil => simp at booleanFound
  | @cons publicKey vector headBoolean publicKeys vectors booleanValues head tail induction =>
      cases index with
      | zero =>
          simp at booleanFound
          subst headBoolean
          exact ⟨{
            publicKey
            vector
            booleanValue
            publicKeyFound := rfl
            vectorFound := rfl
            booleanFound := rfl
            lane := head
          }⟩
      | succ index =>
          obtain ⟨lane⟩ := induction index booleanFound
          exact ⟨{
            lane with
            publicKeyFound := lane.publicKeyFound
            vectorFound := lane.vectorFound
            booleanFound := lane.booleanFound
          }⟩

/-- A Boolean-family lookup determines the corresponding actual `1 × 1` plaintext matrix and
its exact quotient-ring entry. -/
theorem QuotientBooleanMatrixFamilyRelation.matrixAtOfBooleanFound
    {R : Type} [CommRing R]
    {matrices : List (_root_.Matrix (Fin 1) (Fin 1) R)}
    {booleans : List Bool}
    (relation : QuotientBooleanMatrixFamilyRelation matrices booleans)
    (index : Nat) (booleanValue : Bool)
    (booleanFound : booleans[index]? = some booleanValue) :
    ∃ matrix, matrices[index]? = some matrix ∧ matrix 0 0 = booleanRingValue booleanValue := by
  induction relation generalizing index with
  | nil => simp at booleanFound
  | @cons matrix headBoolean matrices booleans head tail induction =>
      cases index with
      | zero =>
          simp at booleanFound
          subst headBoolean
          exact ⟨matrix, rfl, head⟩
      | succ index => exact induction index booleanFound

/-- The generic final-lane endpoint obtained from an actual synchronized quotient state.

The lane's plaintext is definitionally `1`, so `lane.equation` is the required BGG encoding
equation with one message-carrier contribution in `R_q`.  `plaintextEntry` independently ties the
actual runtime plaintext matrix at the same family index to `1`.
-/
structure BggThreeTraceAcceptedLaneAt
    {slots : CheckedBggEncodingSlots}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {encryptionState decryptionState booleanState : List Mxx.Ir.Value}
    (state : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionState decryptionState booleanState)
    (index : Nat) : Type where
  publicKey : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
    (Mxx.Toolkit.Negacyclic q ringDimension)
  vector : _root_.Matrix (Fin outputRows) (Fin publicColumns)
    (Mxx.Toolkit.Negacyclic q ringDimension)
  plaintextMatrix : _root_.Matrix (Fin 1) (Fin 1)
    (Mxx.Toolkit.Negacyclic q ringDimension)
  publicKeyFound :
    (runtimeMatrixValues q ringDimension secretColumns publicColumns
      state.view.encodingPublicKeys)[index]? = some publicKey
  vectorFound :
    (runtimeMatrixValues q ringDimension outputRows publicColumns
      state.view.encodingVectors)[index]? = some vector
  booleanFound : state.view.booleanValues[index]? = some true
  plaintextFound :
    (runtimeMatrixValues q ringDimension 1 1 state.view.plaintextMatrices)[index]? =
      some plaintextMatrix
  lane : QuotientBggLane secret gadget publicKey vector (1 :
    Mxx.Toolkit.Negacyclic q ringDimension)
  plaintextEntry : plaintextMatrix 0 0 = (1 : Mxx.Toolkit.Negacyclic q ringDimension)

/-- Project an accepted lane from the actual final state's Boolean family. -/
noncomputable def BggThreeTraceQuotientState.acceptedLaneAt
    {slots : CheckedBggEncodingSlots}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {encryptionState decryptionState booleanState : List Mxx.Ir.Value}
    (state : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionState decryptionState booleanState)
    (index : Nat) (booleanFound : state.view.booleanValues[index]? = some true) :
    BggThreeTraceAcceptedLaneAt state index := by
  let selectedLane := Classical.choice
    (state.relation.laneAtOfBooleanFound index true booleanFound)
  have plaintextExists :=
    state.plaintextRelation.matrixAtOfBooleanFound index true booleanFound
  let plaintextMatrix := Classical.choose plaintextExists
  have plaintextProperties := Classical.choose_spec plaintextExists
  have selectedBoolean : selectedLane.booleanValue = true :=
    Option.some.inj (selectedLane.booleanFound.symm.trans booleanFound)
  exact {
    publicKey := selectedLane.publicKey
    vector := selectedLane.vector
    plaintextMatrix
    publicKeyFound := selectedLane.publicKeyFound
    vectorFound := selectedLane.vectorFound
    booleanFound
    plaintextFound := plaintextProperties.1
    lane := by simpa [booleanRingValue, selectedBoolean] using selectedLane.lane
    plaintextEntry := by
      dsimp [plaintextMatrix]
      simpa [booleanRingValue] using plaintextProperties.2
  }

/-- The accepted lane's equation, exposed in the one-carrier normal form used by endpoint
reasoning. -/
theorem BggThreeTraceAcceptedLaneAt.encodingEquation
    {slots : CheckedBggEncodingSlots}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {encryptionState decryptionState booleanState : List Mxx.Ir.Value}
    {state : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionState decryptionState booleanState}
    {index : Nat} (accepted : BggThreeTraceAcceptedLaneAt state index) :
    accepted.vector = secret * accepted.publicKey - secret * gadget + accepted.lane.error := by
  simpa using accepted.lane.equation

/-- Project an accepted output directly from the final state forced by an actual three-trace
derivation.  This is the endpoint-facing form: it follows `finalState` rather than accepting a
separately supplied quotient state. -/
noncomputable def BggThreeTraceQuotientDerivation.acceptedFinalLaneAt
    {slots : CheckedBggEncodingSlots}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension)}
    {encryptionRunner decryptionRunner booleanRunner : Mxx.Ir.ChildRunner}
    {encryptionDefinition decryptionDefinition booleanDefinition : String}
    {encryptionParams decryptionParams booleanParams : Mxx.Ir.ParamEnvironment}
    {encryptionIndexSlot decryptionIndexSlot booleanIndexSlot : Nat}
    {encryptionBindings decryptionBindings booleanBindings :
      List (String × Mxx.Ir.IntExpr)}
    {encryptionInvariants decryptionInvariants booleanInvariants : List Mxx.Ir.Value}
    {indices : List Nat}
    {encryptionInitial encryptionFinal decryptionInitial decryptionFinal
      booleanInitial booleanFinal : List Mxx.Ir.Value}
    {trace : BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
      encryptionDefinition decryptionDefinition booleanDefinition
      encryptionParams decryptionParams booleanParams
      encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
      encryptionBindings decryptionBindings booleanBindings
      encryptionInvariants decryptionInvariants booleanInvariants indices
      encryptionInitial encryptionFinal decryptionInitial decryptionFinal
      booleanInitial booleanFinal}
    {initialState : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionInitial decryptionInitial booleanInitial}
    (derivation : BggThreeTraceQuotientDerivation trace initialState)
    (index : Nat)
    (booleanFound : derivation.finalState.view.booleanValues[index]? = some true) :
    BggThreeTraceAcceptedLaneAt derivation.finalState index :=
  derivation.finalState.acceptedLaneAt index booleanFound

end Mxx.Certificate
