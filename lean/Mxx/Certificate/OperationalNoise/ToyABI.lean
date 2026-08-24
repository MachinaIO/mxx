import Mxx.Certificate.OperationalNoise.RelationReplay
import Mxx.Certificate.OperationalNoise.SchemaV1

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyABI

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1

/-! A fixed, hand-authored ABI for the singleton-preimage plus Gaussian G2 fixture. It is not a
    decoder or a general certificate language. Every proof reference is a chronological `Nat`. -/

structure ToyRows where
  root : ExpressionRef
  publicExpression : ExpressionRef
  preimageExpression : ExpressionRef
  targetExpression : ExpressionRef
  noiseExpression : ExpressionRef
  preimageEvent : EventRef
  noiseEvent : EventRef
deriving DecidableEq, Repr

structure ToyOwner where
  scope : StatementScope
  expression : ExpressionRef
deriving DecidableEq, Repr

structure ToyMonomial where
  ordered : List ToyOwner
deriving DecidableEq, Repr

structure ToyTerm where
  coefficient : Int
  monomial : ToyMonomial
deriving DecidableEq, Repr

structure ToyValue where
  coefficient : Int
  terms : List ToyTerm
  bound : Nat
deriving DecidableEq, Repr

structure ToyTermRef where
  valueEvent : Nat
  termOrdinal : Nat
deriving DecidableEq, Repr

inductive ToyBoundRule where
  | authorityRelationPreimageSource (source : EventRef)
  | authorityNoiseOperator (source : EventRef)
  | sum (inputs : List Nat)
  | monomialProduct (factors : List Nat)
  | product (left right : Nat)
deriving DecidableEq, Repr

inductive ToyCoefficientMerge where
  | operator (left right : ToyTermRef) (contribution : ToyTerm)
  | relation (appliedEvent ordinal : Nat) (contribution : ToyTerm)
deriving DecidableEq, Repr

inductive ToyEvent where
  | invocationStart (owner : ToyOwner)
  | predecessor (owner : ToyOwner) (inputPosition : Nat) (expression : ExpressionRef)
      (resultEvent : Nat)
  | result (owner : ToyOwner) (value : ToyValue)
  | invocationEnd (owner : ToyOwner) (result : ToyValue)
  | specializationComputed (owner : ToyOwner) (sourceStart sourceEnd selector : Nat)
      (rhsResult rhsNestedEnd : Nat)
  | appliedUniversal (owner : ToyOwner) (specialization : Nat) (source : EventRef)
      (sourceMonomial : ToyMonomial) (outerCoefficient : Int)
      (orderedStart orderedEndExclusive : Nat) (left right : ToyTerm)
      (rhsResult rhsTermOrdinal : Nat)
  | boundTransfer (owner : ToyOwner) (rule : ToyBoundRule) (bound : Nat)
  | coefficientMerge (owner : ToyOwner) (merge : ToyCoefficientMerge)
  | preFold (owner : ToyOwner) (accumulator : List ToyTerm) (summaryBound : Nat)
  | survivorFold (coefficient : Int) (transferEvent : Nat)
deriving DecidableEq, Repr

/-- Statement rows are separate from the proof-row map and chronological proof events. -/
structure ToyCertificate where
  expressions : List ExpressionRow
  statementEvents : List SchemaV1.EventRow
deriving DecidableEq, Repr

def listAt? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | value :: _, 0 => some value
  | _ :: values, index + 1 => listAt? values index

def matrixType : ValueType := .matrix "257" 1 1 1

def finiteRawContract (maximum : String) : RawValueContract where
  signedRange := none
  coefficientClass := some (.finite maximum)
  canonicalCoefficientExclusiveUpper := none
  polynomialSupportUpper := some 1

def toyWire (node : Nat) : ObservedWire where
  stage := "toy"
  definition := .root
  path := 0
  node := node
  port := 0

def toyRows : ToyRows where
  root := ⟨4⟩
  publicExpression := ⟨0⟩
  preimageExpression := ⟨1⟩
  targetExpression := ⟨2⟩
  noiseExpression := ⟨3⟩
  preimageEvent := ⟨0⟩
  noiseEvent := ⟨1⟩

def publicIdentity : SourceIdentity where
  definition := "toy-public"
  sampleEvent := none
  outputRole := "public"
  artifact := none
  valueType := matrixType
  coordinates := []
  matrixConstant := none

def publicRow : ExpressionRow :=
  { descriptor := .operation (.stable (.source publicIdentity)) matrixType
    inputs := []
    program := none }

def preimageRow (rows : ToyRows) : ExpressionRow :=
  { descriptor := .event (.sampler rows.preimageEvent)
    inputs := []
    program := none }

def targetRow (rows : ToyRows) : ExpressionRow :=
  { descriptor := .operation (.stable (.matrix .multiply)) matrixType
    inputs := [rows.publicExpression, rows.preimageExpression]
    program := none }

def noiseRow (rows : ToyRows) : ExpressionRow :=
  { descriptor := .event (.sampler rows.noiseEvent)
    inputs := []
    program := none }

def rootRow (rows : ToyRows) : ExpressionRow :=
  { descriptor := .operation (.stable (.matrix .add)) matrixType
    inputs := [rows.targetExpression, rows.noiseExpression]
    program := none }

def preimageStatementRow (rows : ToyRows) : SchemaV1.EventRow :=
  .sampler (toyWire rows.preimageExpression.row) (.preimage matrixType "8")
    (some (finiteRawContract "8"))

def noiseStatementRow (rows : ToyRows) : SchemaV1.EventRow :=
  .sampler (toyWire rows.noiseExpression.row) (.gaussian matrixType "1" "1")
    (some (finiteRawContract "1"))

def owner (rows : ToyRows) (expression : ExpressionRef) : ToyOwner :=
  ⟨.closed rows.root, expression⟩

def monomial (rows : ToyRows) (ordered : List ExpressionRef) : ToyMonomial :=
  ⟨ordered.map (owner rows)⟩

def term (rows : ToyRows) (coefficient : Int) (ordered : List ExpressionRef) : ToyTerm :=
  ⟨coefficient, monomial rows ordered⟩

def publicTerm (rows : ToyRows) : ToyTerm := term rows 1 [rows.publicExpression]
def preimageTerm (rows : ToyRows) : ToyTerm := term rows 1 [rows.preimageExpression]
def relationLeftTerm (rows : ToyRows) : ToyTerm :=
  term rows 1 [rows.publicExpression, rows.preimageExpression]
def targetTerm (rows : ToyRows) : ToyTerm := term rows 1 [rows.targetExpression]
def targetCancellation (rows : ToyRows) : ToyTerm := term rows (-1) [rows.targetExpression]
def noiseTerm (rows : ToyRows) : ToyTerm := term rows 1 [rows.noiseExpression]

def publicValue (rows : ToyRows) : ToyValue := ⟨1, [publicTerm rows], 1⟩
def preimageValue (rows : ToyRows) : ToyValue := ⟨1, [preimageTerm rows], 8⟩
def targetValue (rows : ToyRows) : ToyValue := ⟨1, [targetTerm rows], 8⟩
def noiseValue (rows : ToyRows) : ToyValue := ⟨1, [noiseTerm rows], 1⟩
def rootValue (rows : ToyRows) : ToyValue := ⟨1, [noiseTerm rows], 1⟩

/-- The only ordinal in the fixed relation replacement. A later generator must reproduce it. -/
def relationContributionOrdinal : Nat := 0

def rowsValid (certificate : ToyCertificate) (rows : ToyRows) : Prop :=
  rows = toyRows ∧
    listAt? certificate.expressions rows.publicExpression.row = some publicRow ∧
    listAt? certificate.expressions rows.preimageExpression.row = some (preimageRow rows) ∧
    listAt? certificate.expressions rows.targetExpression.row = some (targetRow rows) ∧
    listAt? certificate.expressions rows.noiseExpression.row = some (noiseRow rows) ∧
    listAt? certificate.expressions rows.root.row = some (rootRow rows)

def samplerRowsValid (certificate : ToyCertificate) (rows : ToyRows) : Prop :=
  listAt? certificate.statementEvents rows.preimageEvent.row =
      some (preimageStatementRow rows) ∧
    listAt? certificate.statementEvents rows.noiseEvent.row = some (noiseStatementRow rows)

def frameEventsValid (rows : ToyRows) (events : List ToyEvent) : Prop :=
  listAt? events 0 = some (.invocationStart (owner rows rows.root)) ∧
    listAt? events 1 = some (.invocationStart (owner rows rows.publicExpression)) ∧
    listAt? events 2 = some (.result (owner rows rows.publicExpression) (publicValue rows)) ∧
    listAt? events 3 =
      some (.invocationEnd (owner rows rows.publicExpression) (publicValue rows)) ∧
    listAt? events 4 = some (.invocationStart (owner rows rows.preimageExpression)) ∧
    listAt? events 5 =
      some (.result (owner rows rows.preimageExpression) (preimageValue rows)) ∧
    listAt? events 6 =
      some (.invocationEnd (owner rows rows.preimageExpression) (preimageValue rows)) ∧
    listAt? events 7 = some (.invocationStart (owner rows rows.targetExpression)) ∧
    listAt? events 8 = some (.predecessor (owner rows rows.targetExpression) 0
      rows.publicExpression 2) ∧
    listAt? events 9 = some (.predecessor (owner rows rows.targetExpression) 1
      rows.preimageExpression 5) ∧
    listAt? events 10 = some (.result (owner rows rows.targetExpression) (targetValue rows)) ∧
    listAt? events 11 =
      some (.invocationEnd (owner rows rows.targetExpression) (targetValue rows)) ∧
    listAt? events 12 = some (.invocationStart (owner rows rows.noiseExpression)) ∧
    listAt? events 13 = some (.result (owner rows rows.noiseExpression) (noiseValue rows)) ∧
    listAt? events 14 =
      some (.invocationEnd (owner rows rows.noiseExpression) (noiseValue rows)) ∧
    listAt? events 15 = some (.predecessor (owner rows rows.root) 0 rows.targetExpression 10) ∧
    listAt? events 16 = some (.predecessor (owner rows rows.root) 1 rows.noiseExpression 13)

def relationEventsValid (rows : ToyRows) (events : List ToyEvent) : Prop :=
  listAt? events 17 =
      some (.specializationComputed (owner rows rows.root) 7 12 0 10 11) ∧
    listAt? events 18 = some (.appliedUniversal (owner rows rows.root) 17
      rows.preimageEvent (relationLeftTerm rows).monomial (-1) 0 2
      (relationLeftTerm rows) (targetTerm rows) 10 0) ∧
    listAt? events 19 = some (.boundTransfer (owner rows rows.root)
      (.authorityRelationPreimageSource rows.preimageEvent) 8) ∧
    listAt? events 20 = some (.boundTransfer (owner rows rows.root)
      (.authorityNoiseOperator rows.noiseEvent) 1) ∧
    listAt? events 21 = some (.boundTransfer (owner rows rows.root)
      (.monomialProduct [19]) 8) ∧
    listAt? events 22 =
      some (.boundTransfer (owner rows rows.root) (.product 19 20) 8) ∧
    listAt? events 23 = some (.boundTransfer (owner rows rows.root) (.sum [19, 20]) 9) ∧
    listAt? events 24 = some (.coefficientMerge (owner rows rows.root)
      (.relation 18 relationContributionOrdinal (targetCancellation rows))) ∧
    listAt? events 25 = some (.coefficientMerge (owner rows rows.root)
      (.operator ⟨10, 0⟩ ⟨13, 0⟩ (noiseTerm rows))) ∧
    7 < 11 ∧ 11 < 12 ∧ 12 ≤ 17 ∧ rows.targetExpression ≠ rows.root ∧
    18 < 24 ∧ 10 < 25 ∧ 13 < 25

def foldEventsValid (rows : ToyRows) (events : List ToyEvent) : Prop :=
  listAt? events 26 = some (.preFold (owner rows rows.root)
      [targetTerm rows, targetCancellation rows] 0) ∧
    listAt? events 27 = some (.survivorFold 1 20) ∧
    listAt? events 28 = some (.result (owner rows rows.root) (rootValue rows)) ∧
    listAt? events 29 = some (.invocationEnd (owner rows rows.root) (rootValue rows)) ∧
    (rootValue rows).bound = 0 + (noiseValue rows).bound ∧ 20 < 27

/-- Structural validation checks the external proof event list at each fixed ABI position. -/
def ToyValid (certificate : ToyCertificate) (rows : ToyRows) (events : List ToyEvent) : Prop :=
  events.length = 30 ∧ rowsValid certificate rows ∧ samplerRowsValid certificate rows ∧
    frameEventsValid rows events ∧ relationEventsValid rows events ∧
    foldEventsValid rows events

instance (certificate : ToyCertificate) (rows : ToyRows) (events : List ToyEvent) :
    Decidable (ToyValid certificate rows events) := by
  letI : Decidable (rowsValid certificate rows) := by
    unfold rowsValid
    infer_instance
  letI : Decidable (samplerRowsValid certificate rows) := by
    unfold samplerRowsValid
    infer_instance
  letI : Decidable (frameEventsValid rows events) := by
    unfold frameEventsValid
    infer_instance
  letI : Decidable (relationEventsValid rows events) := by
    unfold relationEventsValid
    infer_instance
  letI : Decidable (foldEventsValid rows events) := by
    unfold foldEventsValid
    infer_instance
  unfold ToyValid
  infer_instance

/-- Constructive decimal decoder for the only two strings emitted by the fixed toy ABI. -/
def toyDecimalCutoff? (value : String) : Option Nat :=
  match value.toByteArray.data.toList with
  | [49] => some 1
  | [56] => some 8
  | _ => none

def noiseCutoff? (certificate : ToyCertificate) (rows : ToyRows) : Option Nat :=
  match listAt? certificate.statementEvents rows.noiseEvent.row with
  | some (.sampler eventOwner (.gaussian output sigma cutoff) (some contract)) =>
      if eventOwner = toyWire rows.noiseExpression.row ∧ output = matrixType ∧ sigma = "1" ∧
          contract = finiteRawContract cutoff then toyDecimalCutoff? cutoff
      else none
  | _ => none

def preimageCutoff? (certificate : ToyCertificate) (rows : ToyRows) : Option Nat :=
  match listAt? certificate.statementEvents rows.preimageEvent.row with
  | some (.sampler eventOwner (.preimage output cutoff) (some contract)) =>
      if eventOwner = toyWire rows.preimageExpression.row ∧ output = matrixType ∧
          contract = finiteRawContract cutoff then toyDecimalCutoff? cutoff
      else none
  | _ => none

def ToySamplerContract (certificate : ToyCertificate) (rows : ToyRows)
    (events : List ToyEvent) (actual : Int) : Prop :=
  ∃ cutoff, noiseCutoff? certificate rows = some cutoff ∧
    listAt? events 13 = some (.result (owner rows rows.noiseExpression) (noiseValue rows)) ∧
    actual = (noiseValue rows).coefficient ∧ actual.natAbs ≤ cutoff

def ToyPreimageContract (certificate : ToyCertificate) (rows : ToyRows)
    (events : List ToyEvent) (actual : Int) : Prop :=
  ∃ cutoff, preimageCutoff? certificate rows = some cutoff ∧
    listAt? events 5 =
      some (.result (owner rows rows.preimageExpression) (preimageValue rows)) ∧
    actual = (preimageValue rows).coefficient ∧ actual.natAbs ≤ cutoff

theorem ToyValid.noiseCutoff {certificate : ToyCertificate} {rows : ToyRows}
    {events : List ToyEvent} (valid : ToyValid certificate rows events) :
    noiseCutoff? certificate rows = some 1 := by
  rcases valid with ⟨_, rowsValidProof, samplerValid, _⟩
  rcases rowsValidProof with ⟨rfl, _⟩
  simp [noiseCutoff?, samplerValid.2, noiseStatementRow, finiteRawContract, matrixType,
    toyWire, toyDecimalCutoff?]
  rfl

theorem ToyValid.preimageCutoff {certificate : ToyCertificate} {rows : ToyRows}
    {events : List ToyEvent} (valid : ToyValid certificate rows events) :
    preimageCutoff? certificate rows = some 8 := by
  rcases valid with ⟨_, rowsValidProof, samplerValid, _⟩
  rcases rowsValidProof with ⟨rfl, _⟩
  simp [preimageCutoff?, samplerValid.1, preimageStatementRow, finiteRawContract, matrixType,
    toyWire, toyDecimalCutoff?]
  rfl

theorem ToySamplerContract.sound {certificate : ToyCertificate} {rows : ToyRows}
    {events : List ToyEvent} {actual : Int} (valid : ToyValid certificate rows events)
    (contract : ToySamplerContract certificate rows events actual) :
    (recordedFiniteContract 1).Interprets actual.natAbs := by
  rcases contract with ⟨cutoff, cutoffRow, _, _, actualBound⟩
  rw [valid.noiseCutoff] at cutoffRow
  cases cutoffRow
  exact gaussianCutoff_sound actualBound

theorem ToyPreimageContract.sound {certificate : ToyCertificate} {rows : ToyRows}
    {events : List ToyEvent} {actual : Int} (valid : ToyValid certificate rows events)
    (contract : ToyPreimageContract certificate rows events actual) :
    (recordedFiniteContract 8).Interprets actual.natAbs := by
  rcases contract with ⟨cutoff, cutoffRow, _, _, actualBound⟩
  rw [valid.preimageCutoff] at cutoffRow
  cases cutoffRow
  exact preimageCutoff_sound actualBound

/-- Numeric equality is attached to the exact B/K source, target result, source authority, and
    specialization event; it cannot be reused with another row map or proof event list. -/
def ToyUniversalRelation (certificate : ToyCertificate) (rows : ToyRows)
    (events : List ToyEvent) (left right : Int) : Prop :=
  rowsValid certificate rows ∧ relationEventsValid rows events ∧
    listAt? events 2 = some (.result (owner rows rows.publicExpression) (publicValue rows)) ∧
    listAt? events 5 = some (.result (owner rows rows.preimageExpression) (preimageValue rows)) ∧
    listAt? events 10 = some (.result (owner rows rows.targetExpression) (targetValue rows)) ∧
    left = (publicValue rows).coefficient * (preimageValue rows).coefficient ∧
    right = (targetValue rows).coefficient ∧
    (left - right) % (257 : Int) = 0

theorem ToyValid.universalRelation {certificate : ToyCertificate} {rows : ToyRows}
    {events : List ToyEvent} (valid : ToyValid certificate rows events) :
    ToyUniversalRelation certificate rows events 1 1 := by
  rcases valid with ⟨_, rowProof, _, frameProof, relationProof, _⟩
  rcases rowProof with ⟨rfl, rowChecks⟩
  exact ⟨⟨rfl, rowChecks⟩, relationProof, frameProof.2.2.1,
    frameProof.2.2.2.2.2.1, frameProof.2.2.2.2.2.2.2.2.2.2.1,
    by decide, by decide, by decide⟩

/-- The six scalar values needed to connect the fixed 1-by-1 replay to its recorded results. -/
structure ToyExecutionValues where
  publicCoefficient : Int
  preimageCoefficient : Int
  relationLeft : Int
  relationRight : Int
  error : Int
  finalCoefficient : Int
deriving DecidableEq, Repr

/-- Fixed execution-value association. It reuses structural validators and adds only the numeric
    equations joining result events, merge contributions, the survivor, and the final result. -/
def ToyExecutionValues.Valid (rows : ToyRows) (events : List ToyEvent)
    (values : ToyExecutionValues) : Prop :=
  frameEventsValid rows events ∧ relationEventsValid rows events ∧
    foldEventsValid rows events ∧
    values.publicCoefficient = (publicValue rows).coefficient ∧
    values.preimageCoefficient = (preimageValue rows).coefficient ∧
    values.relationLeft = values.publicCoefficient * values.preimageCoefficient ∧
    values.relationRight = (targetValue rows).coefficient ∧
    values.error = (noiseValue rows).coefficient ∧
    (targetCancellation rows).coefficient = -values.relationRight ∧
    (noiseTerm rows).coefficient = values.error ∧
    (targetTerm rows).coefficient + (targetCancellation rows).coefficient = 0 ∧
    values.finalCoefficient = values.relationLeft - values.relationRight + values.error ∧
    values.finalCoefficient = (rootValue rows).coefficient ∧
    (rootValue rows).bound = 0 + (noiseValue rows).bound

/-- A modularly exact relation may be removed before centering the remaining error. -/
theorem centeredCoefficient_add_relation {modulus : Nat} {left right error : Int}
    (modulusPositive : 0 < modulus)
    (relationExact : (left - right) % Int.ofNat modulus = 0) :
    centeredCoefficient modulus (left - right + error) = centeredCoefficient modulus error := by
  have modulusNonzero : modulus ≠ 0 := Nat.ne_of_gt modulusPositive
  have remainderEquality :
      (left - right + error) % Int.ofNat modulus = error % Int.ofNat modulus := by
    rw [Int.add_emod, relationExact]
    simp
  simp only [centeredCoefficient, modulusNonzero, ↓reduceIte]
  rw [remainderEquality]

def liftCoefficient (coefficient : Int) : Matrix where
  shape := ⟨257, 1, 1, 1⟩
  coefficients := [coefficient]

@[simp]
theorem liftCoefficient_norm (coefficient : Int) :
    (liftCoefficient coefficient).maxCenteredCoefficientNorm 257 =
      centeredNorm 257 coefficient := by
  simp [liftCoefficient, Matrix.maxCenteredCoefficientNorm, maxNatList]

def ToyOperationalClaim (certificate : ToyCertificate) (rows : ToyRows)
    (events : List ToyEvent) (values : ToyExecutionValues) : Prop :=
  ToyValid certificate rows events ∧ values.Valid rows events ∧
    centeredCoefficient 257
        (values.relationLeft - values.relationRight + values.error) =
      centeredCoefficient 257 values.finalCoefficient ∧
    2 * 2 * centeredNorm 257 values.finalCoefficient < 257

theorem operationalProof {certificate : ToyCertificate} {rows : ToyRows}
    {events : List ToyEvent} {values : ToyExecutionValues}
    (valid : ToyValid certificate rows events)
    (execution : values.Valid rows events)
    (sampler : ToySamplerContract certificate rows events values.error)
    (relation : ToyUniversalRelation certificate rows events values.relationLeft
      values.relationRight) : ToyOperationalClaim certificate rows events values := by
  have errorBound : values.error.natAbs ≤ 1 := sampler.sound valid
  have errorCases : values.error = -1 ∨ values.error = 0 ∨ values.error = 1 := by omega
  have centeredErrorBound : centeredNorm 257 values.error ≤ 1 := by
    rcases errorCases with h | h | h <;> rw [h] <;> decide
  have centeredRelation := centeredCoefficient_add_relation (modulus := 257)
    (left := values.relationLeft) (right := values.relationRight) (error := values.error)
    (by decide) relation.2.2.2.2.2.2.2
  have executionProof := execution
  rcases execution with ⟨_, _, _, _, _, _, _, _, _, _, _, residualIsFinal, _, _⟩
  have centeredNormEquality : centeredNorm 257 values.finalCoefficient =
      centeredNorm 257 values.error := by
    unfold centeredNorm
    rw [residualIsFinal, centeredRelation]
  refine ⟨valid, executionProof, ?_, ?_⟩
  · rw [residualIsFinal]
  rw [centeredNormEquality]
  omega

end Mxx.Certificate.OperationalNoise.ToyABI
