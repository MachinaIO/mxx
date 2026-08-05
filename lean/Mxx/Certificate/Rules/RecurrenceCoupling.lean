import Mxx.Certificate.Typing
import Mxx.Certificate.Workflow

namespace Mxx.Certificate

/-! # Closed recurrence-coupling matcher foundation

This file contains analyzer-owned syntax and structural checks for relating two distinct
sequential recurrences.  None of these types is reachable from `SparseCertificate` or from the
serialized protocol declaration.  Semantic initial/step evidence is intentionally not modeled
until the symbolic-form transfer exists; structural acceptance alone is not a soundness proof.
-/

/-- The closed relation universe for pairs of recurrence results. -/
inductive RecurrenceRelationKind where
  | quotientEqual
  | bggEncodingOf
  deriving BEq, DecidableEq, Repr

/-- A coarse typed path to a matrix-valued element of a carried family.  Unlike
`MatrixFactPath`, it cannot address an affine term. `nestedFamilyDepth = 0` denotes a matrix
element of the outer family, one denotes a matrix element of a nested family, and so on. -/
structure FamilyMatrixPath where
  carriedSlot : Nat
  nestedFamilyDepth : Nat := 0
  deriving BEq, DecidableEq, Repr

private def familyMatrixTypeAtDepth : ValueFactSchema → Nat → Option MatrixTypeExpr
  | .family _ (.matrix matrixType ..), 0 => some matrixType
  | .family _ element, depth + 1 => familyMatrixTypeAtDepth element depth
  | _, _ => none

private def familyMatrixTypeOfFact (fact : ValueFact) (depth : Nat) : Option MatrixTypeExpr :=
  match fact with
  | .family family => familyMatrixTypeAtDepth (.family family.count family.elementSchema) depth
  | _ => none

/-- Resolve a family-matrix role only when the initial value and one-step output template expose
the same coarse matrix type.  Exact/affine form and term count are deliberately irrelevant. -/
def FactRecurrence.resolveFamilyMatrixType
    (recurrence : FactRecurrence)
    (path : FamilyMatrixPath) : Option MatrixTypeExpr := do
  let initial ← recurrence.initial.toList[path.carriedSlot]?
  let output ← recurrence.bodyOutputs.toList[path.carriedSlot]?
  let initialType ← familyMatrixTypeOfFact initial path.nestedFamilyDepth
  let outputType ← familyMatrixTypeAtDepth output.schema path.nestedFamilyDepth
  if initialType == outputType then some initialType else none

/-- Exact analyzer evidence that two count expressions have the same frozen expression origin.
For current IR integer expressions, parameter names and loop slots are stable origin identities;
the matcher does not compare evaluated integers. -/
structure CheckedIntExprOriginEquality (left right : IntExpr) where
  origin : IntExpr
  leftEq : left = origin
  rightEq : right = origin

private def intExprHasStableGlobalOrigin : IntExpr → Bool
  | .constant _ | .parameter _ => true
  | .loopIndex _ => false
  | .add left right | .subtract left right | .multiply left right |
      .divide left right | .roundDivide left right =>
      intExprHasStableGlobalOrigin left && intExprHasStableGlobalOrigin right
  | .log2Ceil value => intExprHasStableGlobalOrigin value

def checkIntExprOriginEquality
    (left right : IntExpr) : Option (CheckedIntExprOriginEquality left right) :=
  if intExprHasStableGlobalOrigin left then
    if equal : left = right then
      some { origin := left, leftEq := rfl, rightEq := equal.symm }
    else none
  else none

/-- Exact frozen-value origin equality.  Numerically equal values with different producer
identities are rejected. -/
structure CheckedValueOriginEquality where
  origin : ValueInstanceRef
  left : ValueInstanceRef
  right : ValueInstanceRef
  leftEq : left = origin
  rightEq : right = origin

def checkValueOriginEquality
    (left right : ValueInstanceRef) : Option CheckedValueOriginEquality :=
  if equal : left = right then
    some { origin := left, left, right, leftEq := rfl, rightEq := equal.symm }
  else none

/-- Structural proof that two independently produced gadget expressions are the same canonical
constant.  Equality of the complete `MatrixTypeExpr` checks modulus, ring dimension, and shape;
equality of `base` fixes the digit decomposition convention used by the executable node. -/
structure CheckedCanonicalGadgetExpr where
  left : MatrixExpr
  right : MatrixExpr
  matrixType : MatrixTypeExpr
  base : IntExpr
  leftCanonical : left = .gadget matrixType base
  rightCanonical : right = .gadget matrixType base

inductive CanonicalGadgetMatchError where
  | leftNotCanonical
  | rightNotCanonical
  | matrixTypeMismatch
  | baseMismatch
  deriving BEq, DecidableEq, Repr

def matchCanonicalGadgetExpr
    (left right : MatrixExpr) : Except CanonicalGadgetMatchError CheckedCanonicalGadgetExpr :=
  match left with
  | .gadget leftType leftBase =>
      match right with
      | .gadget rightType rightBase =>
          if typeEqual : leftType = rightType then
            if baseEqual : leftBase = rightBase then
              .ok {
                left := .gadget leftType leftBase
                right := .gadget rightType rightBase
                matrixType := leftType
                base := leftBase
                leftCanonical := rfl
                rightCanonical := by rw [← typeEqual, ← baseEqual]
              }
            else .error .baseMismatch
          else .error .matrixTypeMismatch
      | _ => .error .rightNotCanonical
  | _ => .error .leftNotCanonical

/-- The four BGG family roles and deterministic gadget constant found by the frozen interface
matcher.  This is analyzer output, not protocol or sparse-certificate input. -/
structure CheckedBggEncodingSlots where
  encryptionPublicKeys : FamilyMatrixPath
  encodingVectors : FamilyMatrixPath
  encodingPublicKeys : FamilyMatrixPath
  plaintextMatrices : FamilyMatrixPath
  gadgetMatrix : CheckedCanonicalGadgetExpr
  encryptionPublicKeyType : MatrixTypeExpr
  encodingVectorType : MatrixTypeExpr
  plaintextMatrixType : MatrixTypeExpr

inductive BggEncodingSlotMatchError where
  | invalidEncryptionPublicKeys
  | invalidEncodingVectors
  | invalidEncodingPublicKeys
  | invalidPlaintextMatrices
  | duplicateDecryptionRole
  | publicKeyTypeMismatch
  | gadgetTypeMismatch
  | plaintextNotOneByOne
  | plaintextRingMismatch
  | plaintextGadgetProductTyping
  | plaintextGadgetProductMismatch
  | encodingVectorRingMismatch
  deriving BEq, DecidableEq, Repr

/-- Validate roles already located by the frozen BGG interface matcher.  Kept private so no
protocol-facing API can choose a role.  Future interface discovery in this module will be the
only caller outside the regression fixtures below. -/
private def checkBggEncodingSlots
    (encryption decryption : FactRecurrence)
    (encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices :
      FamilyMatrixPath)
    (gadgetMatrix : CheckedCanonicalGadgetExpr) :
    Except BggEncodingSlotMatchError CheckedBggEncodingSlots := do
  let encryptionPublicKeyType ← match encryption.resolveFamilyMatrixType encryptionPublicKeys with
    | some matrixType => pure matrixType
    | none => throw .invalidEncryptionPublicKeys
  let encodingVectorType ← match decryption.resolveFamilyMatrixType encodingVectors with
    | some matrixType => pure matrixType
    | none => throw .invalidEncodingVectors
  let encodingPublicKeyType ← match decryption.resolveFamilyMatrixType encodingPublicKeys with
    | some matrixType => pure matrixType
    | none => throw .invalidEncodingPublicKeys
  let plaintextMatrixType ← match decryption.resolveFamilyMatrixType plaintextMatrices with
    | some matrixType => pure matrixType
    | none => throw .invalidPlaintextMatrices
  if encodingVectors = encodingPublicKeys || encodingVectors = plaintextMatrices ||
      encodingPublicKeys = plaintextMatrices then
    throw .duplicateDecryptionRole
  unless encryptionPublicKeyType = encodingPublicKeyType do
    throw .publicKeyTypeMismatch
  unless gadgetMatrix.matrixType = encodingPublicKeyType do
    throw .gadgetTypeMismatch
  unless plaintextMatrixType.rows = .constant 1 &&
      plaintextMatrixType.columns = .constant 1 do
    throw .plaintextNotOneByOne
  unless plaintextMatrixType.modulus = encryptionPublicKeyType.modulus &&
      plaintextMatrixType.ringDimension = encryptionPublicKeyType.ringDimension do
    throw .plaintextRingMismatch
  let plaintextGadgetProduct ← inferMatrixProductType plaintextMatrixType gadgetMatrix.matrixType
    |>.mapError fun _ => .plaintextGadgetProductTyping
  unless plaintextGadgetProduct.output = encryptionPublicKeyType do
    throw .plaintextGadgetProductMismatch
  unless encodingVectorType.modulus = encryptionPublicKeyType.modulus &&
      encodingVectorType.ringDimension = encryptionPublicKeyType.ringDimension do
    throw .encodingVectorRingMismatch
  return {
    encryptionPublicKeys
    encodingVectors
    encodingPublicKeys
    plaintextMatrices
    gadgetMatrix
    encryptionPublicKeyType
    encodingVectorType
    plaintextMatrixType
  }

/-- Unique frozen-table resolution for a recurrence occurrence. -/
structure CheckedRecurrenceResolution
    (analysis : AnalysisResult)
    (reference : FactRecurrenceInstanceRef) where
  recurrence : FactRecurrence
  unique : analysis.recurrences.filter (fun entry => entry.1 = reference) =
    [(reference, recurrence)]

def resolveUniqueRecurrence
    (analysis : AnalysisResult)
    (reference : FactRecurrenceInstanceRef) :
    Option (CheckedRecurrenceResolution analysis reference) :=
  match resolved : analysis.recurrences.filter (fun entry => entry.1 = reference) with
  | [(candidate, recurrence)] =>
      if equal : candidate = reference then
        some {
          recurrence
          unique := by simpa [equal] using resolved
        }
      else none
  | _ => none

/-- Closed structural coupling data.  The payload invariant makes an absent BGG role table legal
only for quotient equality.  Initial and step semantic evidence will be added by the symbolic
transfer theorem; this structure alone deliberately has no soundness eliminator. -/
structure CheckedRecurrenceCoupling (analysis : AnalysisResult) where
  kind : RecurrenceRelationKind
  left : FactRecurrenceInstanceRef
  right : FactRecurrenceInstanceRef
  leftResolution : CheckedRecurrenceResolution analysis left
  rightResolution : CheckedRecurrenceResolution analysis right
  bggSlots : Option CheckedBggEncodingSlots
  payloadMatches : match kind with
    | .quotientEqual => bggSlots = none
    | .bggEncodingOf => bggSlots.isSome = true
  countIdentity : CheckedIntExprOriginEquality leftResolution.recurrence.count
    rightResolution.recurrence.count
  sharedControls : List CheckedValueOriginEquality

inductive RecurrenceCouplingMatchError where
  | missingOrAmbiguousLeft
  | missingOrAmbiguousRight
  | countOriginMismatch
  | invalidBggSlots (error : BggEncodingSlotMatchError)
  deriving BEq, DecidableEq, Repr

/-- Closed quotient-equality foundation.  Candidate discovery supplies only frozen recurrence
identities and already matched shared-control origins. -/
def deriveQuotientEqualCoupling
    (analysis : AnalysisResult)
    (left right : FactRecurrenceInstanceRef)
    (sharedControls : List CheckedValueOriginEquality) :
    Except RecurrenceCouplingMatchError (CheckedRecurrenceCoupling analysis) := do
  let leftResolution ← resolveUniqueRecurrence analysis left
    |>.elim (throw .missingOrAmbiguousLeft) pure
  let rightResolution ← resolveUniqueRecurrence analysis right
    |>.elim (throw .missingOrAmbiguousRight) pure
  let countIdentity ← checkIntExprOriginEquality leftResolution.recurrence.count
    rightResolution.recurrence.count |>.elim (throw .countOriginMismatch) pure
  return {
    kind := .quotientEqual
    left
    right
    leftResolution
    rightResolution
    bggSlots := none
    payloadMatches := rfl
    countIdentity
    sharedControls
  }

/-- Assemble the BGG coupling foundation after the frozen interface matcher has produced the
checked role table.  This function does not accept raw slot numbers, expressions, or protocol
labels. -/
private def deriveBggEncodingCouplingFromCheckedSlots
    (analysis : AnalysisResult)
    (left right : FactRecurrenceInstanceRef)
    (slots : CheckedBggEncodingSlots)
    (sharedControls : List CheckedValueOriginEquality) :
    Except RecurrenceCouplingMatchError (CheckedRecurrenceCoupling analysis) := do
  let leftResolution ← resolveUniqueRecurrence analysis left
    |>.elim (throw .missingOrAmbiguousLeft) pure
  let rightResolution ← resolveUniqueRecurrence analysis right
    |>.elim (throw .missingOrAmbiguousRight) pure
  let countIdentity ← checkIntExprOriginEquality leftResolution.recurrence.count
    rightResolution.recurrence.count |>.elim (throw .countOriginMismatch) pure
  return {
    kind := .bggEncodingOf
    left
    right
    leftResolution
    rightResolution
    bggSlots := some slots
    payloadMatches := rfl
    countIdentity
    sharedControls
  }

private def fixtureMatrixType (rows columns : Int) : MatrixTypeExpr := {
  modulus := .parameter "q"
  ringDimension := .parameter "n"
  rows := .constant rows
  columns := .constant columns
}

private def fixtureMatrixSchema (matrixType : MatrixTypeExpr) : ValueFactSchema :=
  .matrix matrixType .exact [] .unknown

private def fixtureFamilyFact (matrixType : MatrixTypeExpr) : ValueFact :=
  .family {
    aggregate := .carriedInput 0
    count := .parameter "width"
    elementSchema := fixtureMatrixSchema matrixType
  }

private def fixtureFamilyFactAt (slot : Nat) (matrixType : MatrixTypeExpr) : ValueFact :=
  .family {
    aggregate := .carriedInput slot
    count := .parameter "width"
    elementSchema := fixtureMatrixSchema matrixType
  }

private def fixtureFamilyFactWithSchemaAt
    (slot : Nat) (schema : ValueFactSchema) : ValueFact :=
  .family {
    aggregate := .carriedInput slot
    count := .parameter "width"
    elementSchema := schema
  }

private def fixtureRecurrence (matrixType : MatrixTypeExpr) : FactRecurrence where
  loop := { site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨0⟩ } }
  count := .parameter "depth"
  carriedArity := 1
  initial := ⟨#[fixtureFamilyFact matrixType], rfl⟩
  bodyInputs := ⟨#[{
    definition := { stage := ⟨"fixture"⟩, name := "body" }
    bodyScope := ⟨[]⟩
    node := ⟨0⟩
    port := 0
  }], rfl⟩
  bodyOutputs := ⟨#[{
    fact := fixtureFamilyFact matrixType
    schema := .family (.parameter "width") (fixtureMatrixSchema matrixType)
  }], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩

private def fixturePublicKeyType := fixtureMatrixType 2 4
private def fixtureOtherPublicKeyType := fixtureMatrixType 2 5
private def fixtureEncodingVectorType := fixtureMatrixType 1 1
private def fixturePlaintextType := fixtureMatrixType 1 1
private def fixtureCheckedGadget : CheckedCanonicalGadgetExpr := {
  left := .gadget fixturePublicKeyType (.constant 2)
  right := .gadget fixturePublicKeyType (.constant 2)
  matrixType := fixturePublicKeyType
  base := .constant 2
  leftCanonical := rfl
  rightCanonical := rfl
}

private def fixtureBggDecryptionRecurrenceWithPlaintext
    (plaintextFact : ValueFact)
    (plaintextSchema : ValueFactSchema) : FactRecurrence where
  loop := { site := { stage := ⟨"fixture"⟩, scope := ⟨[]⟩, node := ⟨1⟩ } }
  count := .parameter "depth"
  carriedArity := 3
  initial := ⟨#[
    fixtureFamilyFactAt 0 fixtureEncodingVectorType,
    fixtureFamilyFactAt 1 fixturePublicKeyType,
    plaintextFact
  ], rfl⟩
  bodyInputs := ⟨#[
    { definition := { stage := ⟨"fixture"⟩, name := "body" }, bodyScope := ⟨[]⟩,
      node := ⟨0⟩, port := 0 },
    { definition := { stage := ⟨"fixture"⟩, name := "body" }, bodyScope := ⟨[]⟩,
      node := ⟨1⟩, port := 0 },
    { definition := { stage := ⟨"fixture"⟩, name := "body" }, bodyScope := ⟨[]⟩,
      node := ⟨2⟩, port := 0 }
  ], rfl⟩
  bodyOutputs := ⟨#[
    { fact := fixtureFamilyFactAt 0 fixtureEncodingVectorType,
      schema := .family (.parameter "width") (fixtureMatrixSchema fixtureEncodingVectorType) },
    { fact := fixtureFamilyFactAt 1 fixturePublicKeyType,
      schema := .family (.parameter "width") (fixtureMatrixSchema fixturePublicKeyType) },
    { fact := plaintextFact,
      schema := .family (.parameter "width") plaintextSchema }
  ], rfl⟩
  invariantInputs := []
  iterationVariable := ⟨0⟩

private def fixtureBggDecryptionRecurrence : FactRecurrence :=
  fixtureBggDecryptionRecurrenceWithPlaintext
    (fixtureFamilyFactAt 2 fixturePlaintextType)
    (fixtureMatrixSchema fixturePlaintextType)

private def fixtureNonMatrixPlaintextRecurrence : FactRecurrence :=
  fixtureBggDecryptionRecurrenceWithPlaintext
    (fixtureFamilyFactWithSchemaAt 2 .boolean) .boolean

private def fixtureNonScalarPlaintextRecurrence : FactRecurrence :=
  fixtureBggDecryptionRecurrenceWithPlaintext
    (fixtureFamilyFactAt 2 fixtureOtherPublicKeyType)
    (fixtureMatrixSchema fixtureOtherPublicKeyType)

private def fixtureLeftRecurrenceRef : FactRecurrenceInstanceRef := {
  recurrence := ⟨"fixture-encryption"⟩
  path := []
}

private def fixtureRightRecurrenceRef : FactRecurrenceInstanceRef := {
  recurrence := ⟨"fixture-decryption"⟩
  path := []
}

private def fixtureCouplingAnalysis : AnalysisResult where
  facts := []
  families := []
  recurrences := [
    (fixtureLeftRecurrenceRef, fixtureRecurrence fixturePublicKeyType),
    (fixtureRightRecurrenceRef, fixtureBggDecryptionRecurrence)
  ]
  staticObligations := []
  inputObligations := []
  semanticObligations := []
  endpointFacts := []
  usedRules := []

private def fixtureCheckedBggSlots : CheckedBggEncodingSlots := {
  encryptionPublicKeys := ⟨0, 0⟩
  encodingVectors := ⟨0, 0⟩
  encodingPublicKeys := ⟨1, 0⟩
  plaintextMatrices := ⟨2, 0⟩
  gadgetMatrix := fixtureCheckedGadget
  encryptionPublicKeyType := fixturePublicKeyType
  encodingVectorType := fixtureEncodingVectorType
  plaintextMatrixType := fixturePlaintextType
}

example : (fixtureRecurrence fixturePublicKeyType).resolveFamilyMatrixType ⟨0, 0⟩ =
    some fixturePublicKeyType := rfl

example : (fixtureRecurrence fixturePublicKeyType).resolveFamilyMatrixType ⟨1, 0⟩ = none := rfl

example : (matchCanonicalGadgetExpr
    (.gadget fixturePublicKeyType (.constant 2))
    (.gadget fixturePublicKeyType (.constant 2))).isOk = true := rfl

example : matchCanonicalGadgetExpr
    (.gadget fixturePublicKeyType (.constant 2))
    (.gadget fixtureOtherPublicKeyType (.constant 2)) =
    .error .matrixTypeMismatch := rfl

example : matchCanonicalGadgetExpr
    (.gadget fixturePublicKeyType (.constant 2))
    (.gadget fixturePublicKeyType (.constant 4)) =
    .error .baseMismatch := rfl

example : matchCanonicalGadgetExpr
    (.identity fixturePublicKeyType)
    (.gadget fixturePublicKeyType (.constant 2)) =
    .error .leftNotCanonical := rfl

example : matchCanonicalGadgetExpr
    (.gadget fixturePublicKeyType (.constant 2))
    (.identity fixturePublicKeyType) =
    .error .rightNotCanonical := rfl

example : checkIntExprOriginEquality (.parameter "depth") (.parameter "other") = none := rfl

example : checkIntExprOriginEquality (.loopIndex 0) (.loopIndex 0) = none := rfl

example : (checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureBggDecryptionRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨1, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget).isOk = true := rfl

example : checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureBggDecryptionRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨0, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget =
    .error .duplicateDecryptionRole := by
  rfl

example : checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureNonMatrixPlaintextRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨1, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget =
    .error .invalidPlaintextMatrices := by
  rfl

example : checkBggEncodingSlots (fixtureRecurrence fixturePublicKeyType)
    fixtureNonScalarPlaintextRecurrence ⟨0, 0⟩ ⟨0, 0⟩ ⟨1, 0⟩ ⟨2, 0⟩
    fixtureCheckedGadget =
    .error .plaintextNotOneByOne := by
  rfl

example : (deriveBggEncodingCouplingFromCheckedSlots fixtureCouplingAnalysis
    fixtureLeftRecurrenceRef fixtureRightRecurrenceRef fixtureCheckedBggSlots []).isOk = true := by
  rfl

end Mxx.Certificate
