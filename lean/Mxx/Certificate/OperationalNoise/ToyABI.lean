import Mxx.Certificate.OperationalNoise.RelationReplay
import Mxx.Certificate.OperationalNoise.SchemaV1

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.ToyABI

open Mxx.Certificate.OperationalNoise
open EventReplay
open SchemaV1

/-! A fixed, hand-authored ABI for the singleton-preimage plus Gaussian G2 fixture. It is not a
    decoder or a general certificate language. The exact event list below deliberately makes every
    proof reference a chronological `Nat` index. -/

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
  terms : List ToyTerm
  bound : CoeffClass
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
  | specializationComputed (owner : ToyOwner) (minimum maximumExclusive selector : Nat)
      (rhsNestedEnd : Nat)
  | appliedUniversal (owner : ToyOwner) (specialization : Nat) (source : EventRef)
      (sourceMonomial : ToyMonomial) (outerCoefficient : Int)
      (orderedStart orderedEndExclusive : Nat) (left right : ToyTerm)
  | boundTransfer (owner : ToyOwner) (rule : ToyBoundRule) (bound : Nat)
  | coefficientMerge (owner : ToyOwner) (merge : ToyCoefficientMerge)
  | preFold (owner : ToyOwner) (accumulator : List ToyTerm) (summaryBound : Nat)
  | survivorFold (coefficient : Int) (transferEvent : Nat)
deriving DecidableEq, Repr

structure ToyCertificate where
  rows : ToyRows
  expressions : List ExpressionRow
  statementEvents : List SchemaV1.EventRow
  proofEvents : List ToyEvent
deriving DecidableEq, Repr

def matrixType : ValueType := .matrix "257" 1 1 1

def finiteOneRawContract : RawValueContract where
  signedRange := none
  coefficientClass := some (.finite "1")
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

def expectedExpressions : List ExpressionRow :=
  [ { descriptor := .operation (.stable (.source publicIdentity)) matrixType
      inputs := []
      program := none },
    { descriptor := .event (.sampler toyRows.preimageEvent)
      inputs := []
      program := none },
    { descriptor := .operation (.stable (.matrix .multiply)) matrixType
      inputs := [toyRows.publicExpression, toyRows.preimageExpression]
      program := none },
    { descriptor := .event (.sampler toyRows.noiseEvent)
      inputs := []
      program := none },
    { descriptor := .operation (.stable (.matrix .add)) matrixType
      inputs := [toyRows.targetExpression, toyRows.noiseExpression]
      program := none } ]

def expectedStatementEvents : List SchemaV1.EventRow :=
  [ SchemaV1.EventRow.sampler (toyWire toyRows.preimageExpression.row)
      (.preimage matrixType "1")
      (some finiteOneRawContract),
    SchemaV1.EventRow.sampler (toyWire toyRows.noiseExpression.row)
      (.gaussian matrixType "1" "1")
      (some finiteOneRawContract) ]

def owner (expression : ExpressionRef) : ToyOwner := ⟨.closed toyRows.root, expression⟩

def monomial (ordered : List ExpressionRef) : ToyMonomial :=
  ⟨ordered.map owner⟩

def term (coefficient : Int) (ordered : List ExpressionRef) : ToyTerm :=
  ⟨coefficient, monomial ordered⟩

def finiteOne : CoeffClass := .finite ⟨1, by decide⟩

def publicTerm : ToyTerm := term 1 [toyRows.publicExpression]
def preimageTerm : ToyTerm := term 1 [toyRows.preimageExpression]
def relationLeftTerm : ToyTerm :=
  term 1 [toyRows.publicExpression, toyRows.preimageExpression]
def targetTerm : ToyTerm := term 1 [toyRows.targetExpression]
def targetCancellation : ToyTerm := term (-1) [toyRows.targetExpression]
def noiseTerm : ToyTerm := term 1 [toyRows.noiseExpression]

def publicValue : ToyValue := ⟨[publicTerm], finiteOne⟩
def preimageValue : ToyValue := ⟨[preimageTerm], finiteOne⟩
def targetValue : ToyValue := ⟨[targetTerm], finiteOne⟩
def noiseValue : ToyValue := ⟨[noiseTerm], finiteOne⟩
def rootValue : ToyValue := ⟨[noiseTerm], finiteOne⟩

/-- The only ordinal in the fixed relation replacement. A later generator must reproduce it. -/
def relationContributionOrdinal : Nat := 0

def expectedProofEvents : List ToyEvent :=
  [ .invocationStart (owner toyRows.root),
    .invocationStart (owner toyRows.publicExpression),
    .result (owner toyRows.publicExpression) publicValue,
    .invocationEnd (owner toyRows.publicExpression) publicValue,
    .invocationStart (owner toyRows.preimageExpression),
    .result (owner toyRows.preimageExpression) preimageValue,
    .invocationEnd (owner toyRows.preimageExpression) preimageValue,
    .invocationStart (owner toyRows.targetExpression),
    .predecessor (owner toyRows.targetExpression) 0 toyRows.publicExpression 2,
    .predecessor (owner toyRows.targetExpression) 1 toyRows.preimageExpression 5,
    .result (owner toyRows.targetExpression) targetValue,
    .invocationEnd (owner toyRows.targetExpression) targetValue,
    .invocationStart (owner toyRows.noiseExpression),
    .result (owner toyRows.noiseExpression) noiseValue,
    .invocationEnd (owner toyRows.noiseExpression) noiseValue,
    .predecessor (owner toyRows.root) 0 toyRows.targetExpression 10,
    .predecessor (owner toyRows.root) 1 toyRows.noiseExpression 13,
    .specializationComputed (owner toyRows.root) 0 1 0 11,
    .appliedUniversal (owner toyRows.root) 17 toyRows.preimageEvent
      relationLeftTerm.monomial (-1) 0 2 relationLeftTerm targetTerm,
    .boundTransfer (owner toyRows.root)
      (.authorityRelationPreimageSource toyRows.preimageEvent) 1,
    .boundTransfer (owner toyRows.root) (.authorityNoiseOperator toyRows.noiseEvent) 1,
    .boundTransfer (owner toyRows.root) (.monomialProduct [19]) 1,
    .boundTransfer (owner toyRows.root) (.product 19 20) 1,
    .boundTransfer (owner toyRows.root) (.sum [19, 20]) 2,
    .coefficientMerge (owner toyRows.root)
      (.relation 18 relationContributionOrdinal targetCancellation),
    .coefficientMerge (owner toyRows.root) (.operator ⟨10, 0⟩ ⟨13, 0⟩ noiseTerm),
    .preFold (owner toyRows.root) [targetTerm, targetCancellation] 0,
    .survivorFold 1 20,
    .result (owner toyRows.root) rootValue,
    .invocationEnd (owner toyRows.root) rootValue ]

def listAt? {α : Type} : List α → Nat → Option α
  | [], _ => none
  | value :: _, 0 => some value
  | _ :: values, index + 1 => listAt? values index

def rowsValid (certificate : ToyCertificate) : Prop :=
  certificate.rows = toyRows ∧
    listAt? certificate.expressions certificate.rows.publicExpression.row =
      listAt? expectedExpressions toyRows.publicExpression.row ∧
    listAt? certificate.expressions certificate.rows.preimageExpression.row =
      listAt? expectedExpressions toyRows.preimageExpression.row ∧
    listAt? certificate.expressions certificate.rows.targetExpression.row =
      listAt? expectedExpressions toyRows.targetExpression.row ∧
    listAt? certificate.expressions certificate.rows.noiseExpression.row =
      listAt? expectedExpressions toyRows.noiseExpression.row ∧
    listAt? certificate.expressions certificate.rows.root.row =
      listAt? expectedExpressions toyRows.root.row

def samplerRowsValid (certificate : ToyCertificate) : Prop :=
  listAt? certificate.statementEvents certificate.rows.preimageEvent.row =
      listAt? expectedStatementEvents toyRows.preimageEvent.row ∧
    listAt? certificate.statementEvents certificate.rows.noiseEvent.row =
      listAt? expectedStatementEvents toyRows.noiseEvent.row

/-- Exact list equality fixes chronological references, owner scopes, balanced nested frames,
    specialization range and nested end, relation and operator merge inputs, and the fold chain. -/
def replayValid (certificate : ToyCertificate) : Prop :=
  certificate.proofEvents = expectedProofEvents

def ToyValid (certificate : ToyCertificate) : Prop :=
  rowsValid certificate ∧ samplerRowsValid certificate ∧ replayValid certificate

instance (certificate : ToyCertificate) : Decidable (ToyValid certificate) := by
  unfold ToyValid rowsValid samplerRowsValid replayValid
  infer_instance

/-- The Gaussian contract is read from the exact statement event rather than supplied by an
    authority tag. -/
def ToySamplerContract (certificate : ToyCertificate) (actual : Int) : Prop :=
  listAt? certificate.statementEvents certificate.rows.noiseEvent.row =
      some (SchemaV1.EventRow.sampler (toyWire toyRows.noiseExpression.row)
        (.gaussian matrixType "1" "1")
        (some finiteOneRawContract)) ∧
    actual.natAbs ≤ 1

theorem ToySamplerContract.sound {certificate : ToyCertificate} {actual : Int}
    (contract : ToySamplerContract certificate actual) :
    (recordedFiniteContract 1).Interprets actual.natAbs :=
  gaussianCutoff_sound contract.2

/-- The relation-preimage authority likewise consumes the exact preimage event and cutoff. -/
def ToyPreimageContract (certificate : ToyCertificate) (actual : Int) : Prop :=
  listAt? certificate.statementEvents certificate.rows.preimageEvent.row =
      some (SchemaV1.EventRow.sampler (toyWire toyRows.preimageExpression.row)
        (.preimage matrixType "1") (some finiteOneRawContract)) ∧
    actual.natAbs ≤ 1

theorem ToyPreimageContract.sound {certificate : ToyCertificate} {actual : Int}
    (contract : ToyPreimageContract certificate actual) :
    (recordedFiniteContract 1).Interprets actual.natAbs :=
  preimageCutoff_sound contract.2

/-- The universal relation is the exact typed statement-row and replay-event association. -/
def ToyUniversalRelation (certificate : ToyCertificate) : Prop :=
  rowsValid certificate ∧
    listAt? certificate.proofEvents 18 =
      some (ToyEvent.appliedUniversal (owner toyRows.root) 17 toyRows.preimageEvent
        relationLeftTerm.monomial (-1) 0 2 relationLeftTerm targetTerm)

theorem ToyValid.universalRelation {certificate : ToyCertificate}
    (valid : ToyValid certificate) : ToyUniversalRelation certificate := by
  refine ⟨valid.1, ?_⟩
  rw [valid.2.2]
  decide

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

/-- Fixed scalar lift for the 1-by-1, ring-dimension-one toy residual. -/
def liftCoefficient (coefficient : Int) : Matrix where
  shape := ⟨257, 1, 1, 1⟩
  coefficients := [coefficient]

@[simp]
theorem liftCoefficient_norm (coefficient : Int) :
    (liftCoefficient coefficient).maxCenteredCoefficientNorm 257 =
      centeredNorm 257 coefficient := by
  simp [liftCoefficient, Matrix.maxCenteredCoefficientNorm, maxNatList]

def ToyOperationalClaim (residual : Int) : Prop :=
  2 * 2 * centeredNorm 257 residual < 257

end Mxx.Certificate.OperationalNoise.ToyABI
