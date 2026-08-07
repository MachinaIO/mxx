import Mxx.Certificate.Rules.BggThreeTrace

namespace Mxx.Certificate

noncomputable section

/-! # Trace-indexed initial evidence for the BGG three-trace recurrence

This module is the semantic boundary between closed straight-line normalization of the three
initial carried states and the quotient-ring recurrence theorem.  It deliberately does not
discover an input injector, a recurrence, or a family lane.  All five runtime families are
extracted by `BggThreeTraceRuntimeView` from the initial values of one actual
`TraceBoundBggThreeTrace`; the remaining lane equations are the precise results that closed
normalization must derive from those values.
-/

/-- Initial carried values selected definitionally from the three actual recurrence executions. -/
abbrev TraceBoundBggThreeTrace.EncryptionInitialState
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :=
  evidence.encryption.argumentValues.take evidence.encryption.view.carriedCount

abbrev TraceBoundBggThreeTrace.DecryptionInitialState
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :=
  evidence.decryption.argumentValues.take evidence.decryption.view.carriedCount

abbrev TraceBoundBggThreeTrace.BooleanInitialState
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :=
  evidence.boolean.trace.argumentValues.take evidence.boolean.view.carriedCount

/-- Pointwise modular facts still required after the actual initial runtime families have been
selected.  Each constructor consumes one lane from all five families simultaneously, so a fact
for a reordered, truncated, or unrelated family cannot inhabit this type.

For Diamond, `lane` is where closed normalization connects the input-injector output, its exact
preimage projections, packing, and instance/witness selection.  This shared type accepts only the
resulting executable `MatrixModEq` relation; it does not accept an input-injector callback or a
protocol-authored invariant.
-/
inductive BggThreeTraceInitialLaneEvidence
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (secret : Mxx.Matrix)
    (gadget : Mxx.Matrix) :
    List Mxx.Matrix → List Mxx.Matrix → List Mxx.Matrix → List Mxx.Matrix →
      List Bool → Type where
  | nil : BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns publicColumns
      secret gadget [] [] [] [] []
  | cons
      {encryptionPublicKey encodingVector encodingPublicKey plaintext : Mxx.Matrix}
      {booleanValue : Bool}
      {encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices :
        List Mxx.Matrix}
      {booleanValues : List Bool}
      (lane : BggEncodingLaneRelation secret gadget encryptionPublicKey encodingVector
        encodingPublicKey plaintext)
      (errorLayout : Mxx.Toolkit.MatrixLayout lane.error q ringDimension outputRows publicColumns)
      (plaintextValue :
        (Mxx.Toolkit.matrixValue q ringDimension 1 1 plaintext) 0 0 =
          booleanRingValue booleanValue)
      (tail : BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns
        publicColumns secret gadget encryptionPublicKeys encodingVectors encodingPublicKeys
        plaintextMatrices booleanValues) :
      BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns publicColumns
        secret gadget (encryptionPublicKey :: encryptionPublicKeys)
        (encodingVector :: encodingVectors) (encodingPublicKey :: encodingPublicKeys)
        (plaintext :: plaintextMatrices) (booleanValue :: booleanValues)

/-- Initial evidence tied to the exact carried inputs of one trace-bound three-recurrence
execution.  `view` performs all slot/family extraction and layout validation; `lanes` can only
describe the lists extracted by that exact view. -/
structure TraceBoundBggThreeTrace.InitialEvidence
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface)
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (secret : Mxx.Matrix)
    (gadget : Mxx.Matrix) : Type where
  view : BggThreeTraceRuntimeView interface.checked.candidate.slots q ringDimension outputRows
    secretColumns publicColumns evidence.EncryptionInitialState evidence.DecryptionInitialState
    evidence.BooleanInitialState
  secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension outputRows secretColumns
  gadgetLayout : Mxx.Toolkit.MatrixLayout gadget q ringDimension secretColumns publicColumns
  lanes : BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns publicColumns
    secret gadget view.encryptionPublicKeys view.encodingVectors view.encodingPublicKeys
    view.plaintextMatrices view.booleanValues

namespace BggThreeTraceInitialLaneEvidence

/-- The pointwise modular public-key relations imply equality of the quotient-valued families. -/
theorem publicKeysEqual
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret gadget : Mxx.Matrix}
    {encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices : List Mxx.Matrix}
    {booleanValues : List Bool}
    (evidence : BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionPublicKeys encodingVectors encodingPublicKeys
      plaintextMatrices booleanValues)
    (encryptionLayouts : RuntimeMatrixFamilyLayouts q ringDimension secretColumns publicColumns
      encryptionPublicKeys)
    (encodingLayouts : RuntimeMatrixFamilyLayouts q ringDimension secretColumns publicColumns
      encodingPublicKeys) :
    runtimeMatrixValues q ringDimension secretColumns publicColumns encryptionPublicKeys =
      runtimeMatrixValues q ringDimension secretColumns publicColumns encodingPublicKeys := by
  induction evidence with
  | nil => rfl
  | @cons encryptionPublicKey encodingVector encodingPublicKey plaintext booleanValue
      encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices booleanValues
      lane errorLayout plaintextValue tail induction =>
      cases encryptionLayouts with
      | cons encryptionLayout encryptionLayouts =>
          cases encodingLayouts with
          | cons encodingLayout encodingLayouts =>
              simp only [runtimeMatrixValues, List.map_cons, List.cons.injEq]
              exact ⟨Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension secretColumns
                publicColumns encryptionPublicKey encodingPublicKey encryptionLayout
                encodingLayout lane.publicKeyEquation.symm,
                induction encryptionLayouts encodingLayouts⟩

/-- Closed lane normalization yields the quotient BGG relation, initially indexed by the
encryption public-key family. -/
noncomputable def quotientRelation
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret gadget : Mxx.Matrix}
    {encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices : List Mxx.Matrix}
    {booleanValues : List Bool}
    (evidence : BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionPublicKeys encodingVectors encodingPublicKeys
      plaintextMatrices booleanValues)
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension outputRows secretColumns)
    (gadgetLayout : Mxx.Toolkit.MatrixLayout gadget q ringDimension secretColumns publicColumns)
    (encryptionLayouts : RuntimeMatrixFamilyLayouts q ringDimension secretColumns publicColumns
      encryptionPublicKeys)
    (vectorLayouts : RuntimeMatrixFamilyLayouts q ringDimension outputRows publicColumns
      encodingVectors)
    (encodingLayouts : RuntimeMatrixFamilyLayouts q ringDimension secretColumns publicColumns
      encodingPublicKeys)
    (plaintextLayouts : RuntimeMatrixFamilyLayouts q ringDimension 1 1 plaintextMatrices) :
    QuotientBggFamilyRelation
      (Mxx.Toolkit.matrixValue q ringDimension outputRows secretColumns secret)
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns gadget)
      (runtimeMatrixValues q ringDimension secretColumns publicColumns encryptionPublicKeys)
      (runtimeMatrixValues q ringDimension outputRows publicColumns encodingVectors)
      booleanValues := by
  induction evidence with
  | nil => exact .nil
  | @cons encryptionPublicKey encodingVector encodingPublicKey plaintext booleanValue
      encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices booleanValues
      lane errorLayout plaintextValue tail induction =>
      cases encryptionLayouts with
      | cons encryptionLayout encryptionLayouts =>
          cases vectorLayouts with
          | cons vectorLayout vectorLayouts =>
              cases encodingLayouts with
              | cons encodingLayout encodingLayouts =>
                  cases plaintextLayouts with
                  | cons plaintextLayout plaintextLayouts =>
                      have quotientLane := lane.toQuotient q ringDimension outputRows
                        secretColumns publicColumns secret gadget encryptionPublicKey
                        encodingVector encodingPublicKey plaintext secretLayout gadgetLayout
                        encryptionLayout vectorLayout encodingLayout plaintextLayout errorLayout
                      rw [plaintextValue] at quotientLane
                      exact .cons
                        quotientLane
                        (induction encryptionLayouts vectorLayouts encodingLayouts
                          plaintextLayouts)

/-- The same exact lane traversal proves the Boolean interpretation of the plaintext matrices. -/
theorem plaintextRelation
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret gadget : Mxx.Matrix}
    {encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices : List Mxx.Matrix}
    {booleanValues : List Bool}
    (evidence : BggThreeTraceInitialLaneEvidence q ringDimension outputRows secretColumns
      publicColumns secret gadget encryptionPublicKeys encodingVectors encodingPublicKeys
      plaintextMatrices booleanValues) :
    QuotientBooleanMatrixFamilyRelation
      (runtimeMatrixValues q ringDimension 1 1 plaintextMatrices) booleanValues := by
  induction evidence with
  | nil => exact .nil
  | cons lane errorLayout plaintextValue tail induction =>
      exact .cons plaintextValue induction

end BggThreeTraceInitialLaneEvidence

/-- Assemble the recurrence's initial quotient invariant from the exact trace-carried families
and the modular facts established by closed straight-line normalization. -/
noncomputable def TraceBoundBggThreeTrace.InitialEvidence.toQuotientState
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    {evidence : TraceBoundBggThreeTrace analysis trace interface}
    {q ringDimension outputRows secretColumns publicColumns : Nat}
    [Fact (1 < q)] [NeZero ringDimension]
    {secret gadget : Mxx.Matrix}
    (initial : evidence.InitialEvidence q ringDimension outputRows secretColumns publicColumns
      secret gadget) :
    BggThreeTraceQuotientState interface.checked.candidate.slots q ringDimension outputRows
      secretColumns publicColumns
      (Mxx.Toolkit.matrixValue q ringDimension outputRows secretColumns secret)
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns gadget)
      evidence.EncryptionInitialState evidence.DecryptionInitialState
      evidence.BooleanInitialState := by
  let keysEqual := initial.lanes.publicKeysEqual initial.view.encryptionPublicKeyLayouts
    initial.view.encodingPublicKeyLayouts
  let relation := initial.lanes.quotientRelation initial.secretLayout initial.gadgetLayout
    initial.view.encryptionPublicKeyLayouts initial.view.encodingVectorLayouts
    initial.view.encodingPublicKeyLayouts initial.view.plaintextLayouts
  have relation' : QuotientBggFamilyRelation
      (Mxx.Toolkit.matrixValue q ringDimension outputRows secretColumns secret)
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns gadget)
      (runtimeMatrixValues q ringDimension secretColumns publicColumns
        initial.view.encodingPublicKeys)
      (runtimeMatrixValues q ringDimension outputRows publicColumns initial.view.encodingVectors)
      initial.view.booleanValues := by
    rw [← keysEqual]
    exact relation
  exact {
    view := initial.view
    publicKeysEqual := keysEqual
    relation := relation'
    plaintextRelation := initial.lanes.plaintextRelation
  }

end

end Mxx.Certificate
