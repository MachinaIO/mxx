import Mxx.Certificate.Rules.RecurrenceCoupling
import Mxx.Certificate.Rules.RequirementExecution
import Mxx.Certificate.Rules.TraceBoundRecurrence
import Mxx.Certificate.Rules.NestedSequentialTrace

namespace Mxx.Certificate

open scoped Matrix

/-!
# Trace-bound BGG three-recurrence evidence

The static matcher in `RecurrenceCoupling` identifies the encryption, decryption, and Boolean
interpreter loops.  This module binds those three occurrences to executions selected from one
`ClosedProtocolExecutionTrace`.  Consequently, this evidence cannot be constructed from a
structural match alone and exposes no caller-supplied runner, loop trace, or invariant.
-/

/-- Generic BGG multiplication identity over any commutative coefficient ring.

This is the algebraic core used by every Boolean `AND` lane.  It requires one shared secret and
only the exact decomposition relation `G * D = A_R`; it does not know about Diamond, a node
number, or a gate position.  Instantiating `R` with the negacyclic quotient makes the statement
precisely a mod-`R_q` theorem. -/
theorem bggEncodingMultiply_algebra
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget leftPublicKey rightPublicKey : _root_.Matrix secretColumns publicColumns R)
    (leftVector rightVector leftError rightError : _root_.Matrix outputRows publicColumns R)
    (rightDecomposition : _root_.Matrix publicColumns publicColumns R)
    (leftPlaintext rightPlaintext : R)
    (leftEncoding : leftVector =
      secret * leftPublicKey - leftPlaintext • (secret * gadget) + leftError)
    (rightEncoding : rightVector =
      secret * rightPublicKey - rightPlaintext • (secret * gadget) + rightError)
    (decomposition : gadget * rightDecomposition = rightPublicKey) :
    leftVector * rightDecomposition + leftPlaintext • rightVector =
      secret * (leftPublicKey * rightDecomposition) -
        (leftPlaintext * rightPlaintext) • (secret * gadget) +
          (leftError * rightDecomposition + leftPlaintext • rightError) := by
  rw [leftEncoding, rightEncoding]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.mul_assoc,
    smul_add, smul_sub, smul_smul, Matrix.smul_mul]
  rw [decomposition]
  module

/-- Generic BGG addition rule. -/
theorem bggEncodingAdd_algebra
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget leftPublicKey rightPublicKey : _root_.Matrix secretColumns publicColumns R)
    (leftVector rightVector leftError rightError : _root_.Matrix outputRows publicColumns R)
    (leftPlaintext rightPlaintext : R)
    (leftEncoding : leftVector =
      secret * leftPublicKey - leftPlaintext • (secret * gadget) + leftError)
    (rightEncoding : rightVector =
      secret * rightPublicKey - rightPlaintext • (secret * gadget) + rightError) :
    leftVector + rightVector =
      secret * (leftPublicKey + rightPublicKey) -
        (leftPlaintext + rightPlaintext) • (secret * gadget) +
          (leftError + rightError) := by
  rw [leftEncoding, rightEncoding]
  simp only [Matrix.mul_add, add_smul]
  module

/-- Generic BGG subtraction rule. -/
theorem bggEncodingSubtract_algebra
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget leftPublicKey rightPublicKey : _root_.Matrix secretColumns publicColumns R)
    (leftVector rightVector leftError rightError : _root_.Matrix outputRows publicColumns R)
    (leftPlaintext rightPlaintext : R)
    (leftEncoding : leftVector =
      secret * leftPublicKey - leftPlaintext • (secret * gadget) + leftError)
    (rightEncoding : rightVector =
      secret * rightPublicKey - rightPlaintext • (secret * gadget) + rightError) :
    leftVector - rightVector =
      secret * (leftPublicKey - rightPublicKey) -
        (leftPlaintext - rightPlaintext) • (secret * gadget) +
          (leftError - rightError) := by
  rw [leftEncoding, rightEncoding]
  simp only [Matrix.mul_sub, sub_smul]
  module

/-- Generic BGG multiplication by a public polynomial scalar. -/
theorem bggEncodingScale_algebra
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget publicKey : _root_.Matrix secretColumns publicColumns R)
    (vector error : _root_.Matrix outputRows publicColumns R)
    (plaintext scalar : R)
    (encoding : vector =
      secret * publicKey - plaintext • (secret * gadget) + error) :
    scalar • vector =
      secret * (scalar • publicKey) -
        (scalar * plaintext) • (secret * gadget) + scalar • error := by
  rw [encoding]
  simp only [smul_add, smul_sub, smul_smul, Matrix.mul_smul]

/-- One BGG encoding relation interpreted directly in a commutative quotient ring.  This compact
relation is the induction payload; executable `MatrixModEq` facts are converted to it through the
proved negacyclic-value bridge, never through integer equality. -/
structure QuotientBggLane
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget publicKey : _root_.Matrix secretColumns publicColumns R)
    (vector : _root_.Matrix outputRows publicColumns R)
    (plaintext : R) : Type where
  error : _root_.Matrix outputRows publicColumns R
  equation : vector = secret * publicKey - plaintext • (secret * gadget) + error

def QuotientBggLane.add
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget leftPublicKey rightPublicKey : _root_.Matrix secretColumns publicColumns R}
    {leftVector rightVector : _root_.Matrix outputRows publicColumns R}
    {leftPlaintext rightPlaintext : R}
    (left : QuotientBggLane secret gadget leftPublicKey leftVector leftPlaintext)
    (right : QuotientBggLane secret gadget rightPublicKey rightVector rightPlaintext) :
    QuotientBggLane secret gadget (leftPublicKey + rightPublicKey)
      (leftVector + rightVector) (leftPlaintext + rightPlaintext) := {
  error := left.error + right.error
  equation := bggEncodingAdd_algebra secret gadget leftPublicKey rightPublicKey leftVector
    rightVector left.error right.error leftPlaintext rightPlaintext left.equation right.equation
}

def QuotientBggLane.subtract
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget leftPublicKey rightPublicKey : _root_.Matrix secretColumns publicColumns R}
    {leftVector rightVector : _root_.Matrix outputRows publicColumns R}
    {leftPlaintext rightPlaintext : R}
    (left : QuotientBggLane secret gadget leftPublicKey leftVector leftPlaintext)
    (right : QuotientBggLane secret gadget rightPublicKey rightVector rightPlaintext) :
    QuotientBggLane secret gadget (leftPublicKey - rightPublicKey)
      (leftVector - rightVector) (leftPlaintext - rightPlaintext) := {
  error := left.error - right.error
  equation := bggEncodingSubtract_algebra secret gadget leftPublicKey rightPublicKey leftVector
    rightVector left.error right.error leftPlaintext rightPlaintext left.equation right.equation
}

def QuotientBggLane.scale
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget publicKey : _root_.Matrix secretColumns publicColumns R}
    {vector : _root_.Matrix outputRows publicColumns R}
    {plaintext scalar : R}
    (input : QuotientBggLane secret gadget publicKey vector plaintext) :
    QuotientBggLane secret gadget (scalar • publicKey) (scalar • vector)
      (scalar * plaintext) := {
  error := scalar • input.error
  equation := bggEncodingScale_algebra secret gadget publicKey vector input.error plaintext scalar
    input.equation
}

def QuotientBggLane.multiply
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget leftPublicKey rightPublicKey : _root_.Matrix secretColumns publicColumns R}
    {leftVector rightVector : _root_.Matrix outputRows publicColumns R}
    {leftPlaintext rightPlaintext : R}
    (rightDecomposition : _root_.Matrix publicColumns publicColumns R)
    (decomposition : gadget * rightDecomposition = rightPublicKey)
    (left : QuotientBggLane secret gadget leftPublicKey leftVector leftPlaintext)
    (right : QuotientBggLane secret gadget rightPublicKey rightVector rightPlaintext) :
    QuotientBggLane secret gadget (leftPublicKey * rightDecomposition)
      (leftVector * rightDecomposition + leftPlaintext • rightVector)
      (leftPlaintext * rightPlaintext) := {
  error := left.error * rightDecomposition + leftPlaintext • right.error
  equation := bggEncodingMultiply_algebra secret gadget leftPublicKey rightPublicKey leftVector
    rightVector left.error right.error rightDecomposition leftPlaintext rightPlaintext
    left.equation right.equation decomposition
}

/-- Closed Boolean gate universe used by the BGG and plain Boolean interpreters. -/
inductive BggBooleanGate where
  | zero
  | one
  | copyLeft
  | notLeft
  | and
  | xor
  deriving BEq, DecidableEq, Repr

def BggBooleanGate.evaluate (gate : BggBooleanGate) (left right : Bool) : Bool :=
  match gate with
  | .zero => false
  | .one => true
  | .copyLeft => left
  | .notLeft => !left
  | .and => left && right
  | .xor => left != right

/-- The frozen scalar skeleton computes exactly the closed Boolean gate universe.  This theorem
uses the same integer-operation and comparison evaluators as executable IR through
`FrozenPointwiseScalarFormula.evaluate`. -/
theorem CheckedSixWayBooleanSkeleton.evaluateFormulas
    (skeleton : CheckedSixWayBooleanSkeleton)
    (atoms : Mxx.Ir.WireRef → Option Mxx.Ir.Value)
    (left right : Bool)
    (leftEvaluates : skeleton.leftFormula.evaluate atoms = some (.boolean left))
    (rightEvaluates : skeleton.rightFormula.evaluate atoms = some (.boolean right)) :
    skeleton.formulas.mapM (·.evaluate atoms) = some [
      .boolean (BggBooleanGate.zero.evaluate left right),
      .boolean (BggBooleanGate.one.evaluate left right),
      .boolean (BggBooleanGate.copyLeft.evaluate left right),
      .boolean (BggBooleanGate.notLeft.evaluate left right),
      .boolean (BggBooleanGate.and.evaluate left right),
      .boolean (BggBooleanGate.xor.evaluate left right)
    ] := by
  rw [skeleton.formulasMatch]
  cases left <;> cases right <;>
    simp [FrozenPointwiseScalarFormula.evaluate, leftEvaluates, rightEvaluates,
      BggBooleanGate.evaluate, Mxx.Ir.evaluateIntBinary, Mxx.Ir.evaluateIntCompare]

def booleanRingValue {R : Type} [CommRing R] (value : Bool) : R :=
  if value then 1 else 0

/-- One gate candidate together with its mechanically derived BGG relation and exact Boolean
plaintext.  The public key and vector are data because their formulas differ by gate; the
plaintext is forced to the zero/one interpretation of `booleanValue`. -/
structure QuotientBggGateResult
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget : _root_.Matrix secretColumns publicColumns R)
    (booleanValue : Bool) : Type where
  publicKey : _root_.Matrix secretColumns publicColumns R
  vector : _root_.Matrix outputRows publicColumns R
  lane : QuotientBggLane secret gadget publicKey vector (booleanRingValue booleanValue)

private theorem booleanRingValue_add_sub_twice_mul
    {R : Type} [CommRing R]
    (left right : Bool) :
    booleanRingValue left + booleanRingValue right -
        (2 : R) * (booleanRingValue left * booleanRingValue right) =
      booleanRingValue (left != right) := by
  cases left <;> cases right
  all_goals simp [booleanRingValue]
  all_goals ring

private theorem booleanRingValue_and
    {R : Type} [CommRing R]
    (left right : Bool) :
    booleanRingValue (left && right) =
      (booleanRingValue left : R) * booleanRingValue right := by
  cases left <;> cases right <;> simp [booleanRingValue]

/-- Generic gate preservation theorem.  The six branches are exactly the public-key/BGG/Boolean
candidate ordering used by the DSL bodies.  Once the frozen body matcher identifies a branch,
this function constructs its relation without a protocol-specific algebra script. -/
def QuotientBggGateResult.evaluate
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type}
    [Fintype secretColumns] [Fintype publicColumns]
    [DecidableEq secretColumns] [DecidableEq publicColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget onePublicKey leftPublicKey rightPublicKey :
      _root_.Matrix secretColumns publicColumns R}
    {oneVector leftVector rightVector : _root_.Matrix outputRows publicColumns R}
    {leftBoolean rightBoolean : Bool}
    (gate : BggBooleanGate)
    (rightDecomposition : _root_.Matrix publicColumns publicColumns R)
    (decomposition : gadget * rightDecomposition = rightPublicKey)
    (one : QuotientBggLane secret gadget onePublicKey oneVector (1 : R))
    (left : QuotientBggLane secret gadget leftPublicKey leftVector
      (booleanRingValue leftBoolean))
    (right : QuotientBggLane secret gadget rightPublicKey rightVector
      (booleanRingValue rightBoolean)) :
    QuotientBggGateResult secret gadget (gate.evaluate leftBoolean rightBoolean) := by
  let zero := one.subtract one
  let not := one.subtract left
  let product := left.multiply rightDecomposition decomposition right
  let xor := (left.add right).subtract (product.scale (scalar := (2 : R)))
  cases gate with
  | zero =>
      exact {
        publicKey := onePublicKey - onePublicKey
        vector := oneVector - oneVector
        lane := by simpa [BggBooleanGate.evaluate, booleanRingValue, zero] using zero
      }
  | one =>
      exact {
        publicKey := onePublicKey
        vector := oneVector
        lane := by simpa [BggBooleanGate.evaluate, booleanRingValue] using one
      }
  | copyLeft =>
      exact {
        publicKey := leftPublicKey
        vector := leftVector
        lane := by simpa [BggBooleanGate.evaluate] using left
      }
  | notLeft =>
      exact {
        publicKey := onePublicKey - leftPublicKey
        vector := oneVector - leftVector
        lane := by
          cases leftBoolean <;>
            simpa [BggBooleanGate.evaluate, booleanRingValue, not] using not
      }
  | and =>
      exact {
        publicKey := leftPublicKey * rightDecomposition
        vector := leftVector * rightDecomposition +
          booleanRingValue leftBoolean • rightVector
        lane := by
          rw [BggBooleanGate.evaluate, booleanRingValue_and]
          exact product
      }
  | xor =>
      exact {
        publicKey := (leftPublicKey + rightPublicKey) -
          (2 : R) • (leftPublicKey * rightDecomposition)
        vector := (leftVector + rightVector) -
          (2 : R) • (leftVector * rightDecomposition +
            booleanRingValue leftBoolean • rightVector)
        lane := by
          simp only [BggBooleanGate.evaluate]
          rw [← booleanRingValue_add_sub_twice_mul leftBoolean rightBoolean]
          exact xor
      }

/-- Pointwise quotient-ring relation for one whole circuit layer.  Equal lengths are enforced by
the constructors, so gate-source lookup cannot silently pair a public key, vector, and Boolean
from different lanes. -/
inductive QuotientBggFamilyRelation
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    (secret : _root_.Matrix outputRows secretColumns R)
    (gadget : _root_.Matrix secretColumns publicColumns R) :
    List (_root_.Matrix secretColumns publicColumns R) →
      List (_root_.Matrix outputRows publicColumns R) → List Bool → Type where
  | nil : QuotientBggFamilyRelation secret gadget [] [] []
  | cons
      {publicKey : _root_.Matrix secretColumns publicColumns R}
      {vector : _root_.Matrix outputRows publicColumns R}
      {booleanValue : Bool}
      {publicKeys : List (_root_.Matrix secretColumns publicColumns R)}
      {vectors : List (_root_.Matrix outputRows publicColumns R)}
      {booleanValues : List Bool}
      (head : QuotientBggLane secret gadget publicKey vector
        (booleanRingValue booleanValue))
      (tail : QuotientBggFamilyRelation secret gadget publicKeys vectors booleanValues) :
      QuotientBggFamilyRelation secret gadget (publicKey :: publicKeys)
        (vector :: vectors) (booleanValue :: booleanValues)

/-- A typed lane lookup from one pointwise family relation. -/
structure QuotientBggFamilyRelation.LaneAt
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {publicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {vectors : List (_root_.Matrix outputRows publicColumns R)}
    {booleanValues : List Bool}
    (_relation : QuotientBggFamilyRelation secret gadget publicKeys vectors booleanValues)
    (index : Nat) : Type where
  publicKey : _root_.Matrix secretColumns publicColumns R
  vector : _root_.Matrix outputRows publicColumns R
  booleanValue : Bool
  publicKeyFound : publicKeys[index]? = some publicKey
  vectorFound : vectors[index]? = some vector
  booleanFound : booleanValues[index]? = some booleanValue
  lane : QuotientBggLane secret gadget publicKey vector (booleanRingValue booleanValue)

noncomputable def QuotientBggFamilyRelation.laneAt?
    {R : Type} [CommRing R]
    {outputRows secretColumns publicColumns : Type} [Fintype secretColumns]
    {secret : _root_.Matrix outputRows secretColumns R}
    {gadget : _root_.Matrix secretColumns publicColumns R}
    {publicKeys : List (_root_.Matrix secretColumns publicColumns R)}
    {vectors : List (_root_.Matrix outputRows publicColumns R)}
    {booleanValues : List Bool}
    (relation : QuotientBggFamilyRelation secret gadget publicKeys vectors booleanValues)
    (index : Nat) : Option (relation.LaneAt index) := by
  induction relation generalizing index with
  | nil => exact none
  | @cons publicKey vector booleanValue publicKeys vectors booleanValues head tail induction =>
      cases index with
      | zero =>
          exact some {
            publicKey
            vector
            booleanValue
            publicKeyFound := rfl
            vectorFound := rfl
            booleanFound := rfl
            lane := head
          }
      | succ index =>
          exact (induction index).map fun lane => {
            lane with
            publicKeyFound := lane.publicKeyFound
            vectorFound := lane.vectorFound
            booleanFound := lane.booleanFound
          }

/-- One complete circuit layer obtained solely by typed source lookups and the generic gate
theorem.  `cons` stores no arbitrary output relation: its output lane is definitionally the
result of `QuotientBggGateResult.evaluate`. -/
inductive QuotientBggLayerTransfer
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
  | nil : QuotientBggLayerTransfer inputs one [] [] []
  | cons
      {leftIndex rightIndex : Nat}
      (left : inputs.LaneAt leftIndex)
      (right : inputs.LaneAt rightIndex)
      (gate : BggBooleanGate)
      (rightDecomposition : _root_.Matrix publicColumns publicColumns R)
      (decomposition : gadget * rightDecomposition = right.publicKey)
      {outputPublicKeys : List (_root_.Matrix secretColumns publicColumns R)}
      {outputVectors : List (_root_.Matrix outputRows publicColumns R)}
      {outputBooleans : List Bool}
      (tail : QuotientBggLayerTransfer inputs one outputPublicKeys outputVectors outputBooleans) :
      QuotientBggLayerTransfer inputs one
        ((QuotientBggGateResult.evaluate gate rightDecomposition decomposition
          one left.lane right.lane).publicKey :: outputPublicKeys)
        ((QuotientBggGateResult.evaluate gate rightDecomposition decomposition
          one left.lane right.lane).vector :: outputVectors)
        (gate.evaluate left.booleanValue right.booleanValue :: outputBooleans)

/-- Every mechanically constructed layer transfer carries the pointwise BGG/Boolean relation
for its outputs. -/
noncomputable def QuotientBggLayerTransfer.outputRelation
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
    (transfer : QuotientBggLayerTransfer inputs one outputPublicKeys outputVectors
      outputBooleans) :
    QuotientBggFamilyRelation secret gadget outputPublicKeys outputVectors outputBooleans := by
  induction transfer with
  | nil => exact .nil
  | cons left right gate rightDecomposition decomposition tail induction =>
      exact .cons
        (QuotientBggGateResult.evaluate gate rightDecomposition decomposition
          one left.lane right.lane).lane
        induction

/-- Complete layouts for a runtime matrix family.  The shape is uniform and the constructor
forces exact length agreement with the matrix list. -/
private def matrixFamilyValuesAtDepth :
    Nat → Mxx.Ir.Value → Option (List Mxx.Matrix)
  | 0, .family values => values.mapM fun value => match value with
      | .matrix matrix => some matrix
      | _ => none
  | depth + 1, .family values => do
      let nested ← values.mapM (matrixFamilyValuesAtDepth depth)
      some nested.flatten
  | _, _ => none

/-- Runtime matrix family selected only through a matcher-derived typed carried path. -/
def matrixFamilyAt
    (state : List Mxx.Ir.Value)
    (path : FamilyMatrixPath) : Option (List Mxx.Matrix) := do
  let value ← state[path.carriedSlot]?
  matrixFamilyValuesAtDepth path.nestedFamilyDepth value

private def booleanFamilyValuesAtDepth :
    Nat → Mxx.Ir.Value → Option (List Bool)
  | 0, .family values => values.mapM fun value => match value with
      | .boolean boolean => some boolean
      | _ => none
  | depth + 1, .family values => do
      let nested ← values.mapM (booleanFamilyValuesAtDepth depth)
      some nested.flatten
  | _, _ => none

/-- The current closed Boolean interpreter has one outer carried Boolean family.  The function
still validates the runtime constructor rather than silently treating any slot as Boolean. -/
def booleanFamilyAt
    (state : List Mxx.Ir.Value)
    (slot : Nat)
    (nestedFamilyDepth : Nat := 0) : Option (List Bool) := do
  let value ← state[slot]?
  booleanFamilyValuesAtDepth nestedFamilyDepth value

inductive RuntimeMatrixFamilyLayouts
    (q ringDimension rows columns : Nat) : List Mxx.Matrix → Type where
  | nil : RuntimeMatrixFamilyLayouts q ringDimension rows columns []
  | cons {matrix : Mxx.Matrix} {matrices : List Mxx.Matrix}
      (head : Mxx.Toolkit.MatrixLayout matrix q ringDimension rows columns)
      (tail : RuntimeMatrixFamilyLayouts q ringDimension rows columns matrices) :
      RuntimeMatrixFamilyLayouts q ringDimension rows columns (matrix :: matrices)

theorem RuntimeMatrixFamilyLayouts.layoutAt
    {q ringDimension rows columns : Nat}
    {matrices : List Mxx.Matrix}
    (layouts : RuntimeMatrixFamilyLayouts q ringDimension rows columns matrices)
    (index : Nat)
    (matrix : Mxx.Matrix)
    (found : matrices[index]? = some matrix) :
    Mxx.Toolkit.MatrixLayout matrix q ringDimension rows columns := by
  induction layouts generalizing index matrix with
  | nil => simp at found
  | cons head tail induction =>
      cases index with
      | zero =>
          simp at found
          subst matrix
          exact head
      | succ index =>
          exact induction index matrix (by simpa using found)

noncomputable def runtimeMatrixValues
    (q ringDimension rows columns : Nat)
    (matrices : List Mxx.Matrix) :
    List (_root_.Matrix (Fin rows) (Fin columns)
      (Mxx.Toolkit.Negacyclic q ringDimension)) :=
  matrices.map (Mxx.Toolkit.matrixValue q ringDimension rows columns)

/-- Exact quotient-ring interpretation of a family of executable `1 × 1` plaintext
matrices as Boolean zero/one values. -/
inductive QuotientBooleanMatrixFamilyRelation
    {R : Type} [CommRing R] :
    List (_root_.Matrix (Fin 1) (Fin 1) R) → List Bool → Prop where
  | nil : QuotientBooleanMatrixFamilyRelation [] []
  | cons
      {matrix : _root_.Matrix (Fin 1) (Fin 1) R}
      {boolean : Bool}
      {matrices : List (_root_.Matrix (Fin 1) (Fin 1) R)}
      {booleans : List Bool}
      (head : matrix 0 0 = booleanRingValue boolean)
      (tail : QuotientBooleanMatrixFamilyRelation matrices booleans) :
      QuotientBooleanMatrixFamilyRelation (matrix :: matrices) (boolean :: booleans)

/-- Actual three carried states viewed through the unique matcher-selected roles. -/
structure BggThreeTraceRuntimeView
    (slots : CheckedBggEncodingSlots)
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    (encryptionState decryptionState booleanState : List Mxx.Ir.Value) : Type where
  encryptionPublicKeys : List Mxx.Matrix
  encodingVectors : List Mxx.Matrix
  encodingPublicKeys : List Mxx.Matrix
  plaintextMatrices : List Mxx.Matrix
  booleanValues : List Bool
  encryptionPublicKeysFound :
    matrixFamilyAt encryptionState slots.encryptionPublicKeys = some encryptionPublicKeys
  encodingVectorsFound :
    matrixFamilyAt decryptionState slots.encodingVectors = some encodingVectors
  encodingPublicKeysFound :
    matrixFamilyAt decryptionState slots.encodingPublicKeys = some encodingPublicKeys
  plaintextMatricesFound :
    matrixFamilyAt decryptionState slots.plaintextMatrices = some plaintextMatrices
  booleanValuesFound : booleanFamilyAt booleanState 0 = some booleanValues
  encryptionPublicKeyLayouts : RuntimeMatrixFamilyLayouts q ringDimension secretColumns
    publicColumns encryptionPublicKeys
  encodingVectorLayouts : RuntimeMatrixFamilyLayouts q ringDimension outputRows publicColumns
    encodingVectors
  encodingPublicKeyLayouts : RuntimeMatrixFamilyLayouts q ringDimension secretColumns
    publicColumns encodingPublicKeys
  plaintextLayouts : RuntimeMatrixFamilyLayouts q ringDimension 1 1 plaintextMatrices

/-- Quotient-ring invariant synchronized across the encryption public-key, decryption BGG, and
plain Boolean states.  Public-key equality is explicit; the BGG relation is carried against the
decryption keys and therefore cannot be obtained by merely reusing the encryption list. -/
structure BggThreeTraceQuotientState
    (slots : CheckedBggEncodingSlots)
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension))
    (gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension))
    (encryptionState decryptionState booleanState : List Mxx.Ir.Value) : Type where
  view : BggThreeTraceRuntimeView slots q ringDimension outputRows secretColumns publicColumns
    encryptionState decryptionState booleanState
  publicKeysEqual :
    runtimeMatrixValues q ringDimension secretColumns publicColumns
        view.encryptionPublicKeys =
      runtimeMatrixValues q ringDimension secretColumns publicColumns view.encodingPublicKeys
  relation : QuotientBggFamilyRelation secret gadget
    (runtimeMatrixValues q ringDimension secretColumns publicColumns view.encodingPublicKeys)
    (runtimeMatrixValues q ringDimension outputRows publicColumns view.encodingVectors)
    view.booleanValues
  plaintextRelation : QuotientBooleanMatrixFamilyRelation
    (runtimeMatrixValues q ringDimension 1 1 view.plaintextMatrices) view.booleanValues

/-- Pointwise executable public keys in the encryption and evaluation recurrences are equal in
`R_q`.  This is the exact premise needed by canonical gadget decomposition; raw coefficient-list
equality is intentionally unnecessary. -/
theorem BggThreeTraceQuotientState.publicKeyModEqAt
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
    (index : Nat)
    (encryptionPublicKey encodingPublicKey : Mxx.Matrix)
    (encryptionFound : state.view.encryptionPublicKeys[index]? = some encryptionPublicKey)
    (encodingFound : state.view.encodingPublicKeys[index]? = some encodingPublicKey) :
    Mxx.MatrixModEq encryptionPublicKey encodingPublicKey := by
  have encryptionLayout := state.view.encryptionPublicKeyLayouts.layoutAt index
    encryptionPublicKey encryptionFound
  have encodingLayout := state.view.encodingPublicKeyLayouts.layoutAt index encodingPublicKey
    encodingFound
  have valuesAt := congrArg (fun values => values[index]?) state.publicKeysEqual
  have encryptionValueFound :
      (runtimeMatrixValues q ringDimension secretColumns publicColumns
        state.view.encryptionPublicKeys)[index]? =
        some (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns
          encryptionPublicKey) := by
    simp [runtimeMatrixValues, encryptionFound]
  have encodingValueFound :
      (runtimeMatrixValues q ringDimension secretColumns publicColumns
        state.view.encodingPublicKeys)[index]? =
        some (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns
          encodingPublicKey) := by
    simp [runtimeMatrixValues, encodingFound]
  rw [encryptionValueFound, encodingValueFound] at valuesAt
  have valueEq := Option.some.inj valuesAt
  exact Mxx.Toolkit.modEq_of_matrixValue_eq q ringDimension secretColumns publicColumns
    encryptionPublicKey encodingPublicKey encryptionLayout encodingLayout valueEq

/-- One quotient-ring semantic transition between actual carried states.  The output lists in
`transfer` are definitionally the values extracted from `nextView`; consequently a valid step
cannot prove a relation for a different or reordered family.  The enclosing trace derivation
additionally indexes this object by the three actual child-support members for this iteration. -/
structure BggThreeTraceQuotientStep
    (slots : CheckedBggEncodingSlots)
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (secret : _root_.Matrix (Fin outputRows) (Fin secretColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension))
    (gadget : _root_.Matrix (Fin secretColumns) (Fin publicColumns)
      (Mxx.Toolkit.Negacyclic q ringDimension))
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
  transfer : QuotientBggLayerTransfer current.relation one
    (runtimeMatrixValues q ringDimension secretColumns publicColumns
      nextView.encodingPublicKeys)
    (runtimeMatrixValues q ringDimension outputRows publicColumns nextView.encodingVectors)
    nextView.booleanValues
  publicKeysEqual :
    runtimeMatrixValues q ringDimension secretColumns publicColumns
        nextView.encryptionPublicKeys =
      runtimeMatrixValues q ringDimension secretColumns publicColumns
        nextView.encodingPublicKeys
  plaintextRelation : QuotientBooleanMatrixFamilyRelation
    (runtimeMatrixValues q ringDimension 1 1 nextView.plaintextMatrices)
    nextView.booleanValues

noncomputable def BggThreeTraceQuotientStep.nextState
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
    (step : BggThreeTraceQuotientStep slots q ringDimension outputRows secretColumns
      publicColumns secret gadget current encryptionNext decryptionNext booleanNext) :
    BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns publicColumns
      secret gadget encryptionNext decryptionNext booleanNext := {
  view := step.nextView
  publicKeysEqual := step.publicKeysEqual
  relation := step.transfer.outputRelation
  plaintextRelation := step.plaintextRelation
}

/-- Unique analyzer resolution whose executable recurrence interface agrees on every field used
by the three-trace matcher.  The dependent body/bound payload remains owned by `resolution` and
is never reconstructed from the structural interface. -/
structure ResolvedFrozenRecurrence
    (analysis : AnalysisResult)
    (interface : FrozenSequentialRecurrenceInterface) where
  resolution : CheckedRecurrenceResolution analysis interface.transfer.identity
  countMatches : resolution.transfer.source.count = interface.transfer.source.count
  arityMatches : resolution.transfer.source.carriedArity =
    interface.transfer.source.carriedArity
  schemasMatch : resolution.transfer.carriedSchemas = interface.transfer.carriedSchemas

/-- Recover the body path selected by one actual child-support member of a frozen recurrence.
The scope equality is retained by `FrozenSequentialRecurrenceInterface.bodyFound`; no definition
lookup or body value is supplied by the theorem caller. -/
theorem FrozenSequentialRecurrenceInterface.childExecutionExists
    (interface : FrozenSequentialRecurrenceInterface)
    (samplers : Mxx.MxxSamplerFamily)
    (fuel : Nat)
    (params : Mxx.Ir.ParamEnvironment)
    (arguments values : List Mxx.Ir.Value)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers interface.program (fuel + 1)
      interface.definition params arguments) :
    Nonempty (ChildScopeExecutionPath samplers interface.program fuel interface.definition params
      arguments values) :=
  ChildScopeExecutionPath.nonempty_of_childMember samplers interface.program fuel
    interface.definition interface.body params arguments values interface.bodyFound childMember

/-- Full-program child runners use `definitions.length` fuel.  A successful frozen body lookup
proves that this length is positive, so the exact predecessor runner needed by the child path is
derived rather than guessed. -/
theorem FrozenSequentialRecurrenceInterface.childExecutionExistsAtFullFuel
    (interface : FrozenSequentialRecurrenceInterface)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (arguments values : List Mxx.Ir.Value)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers interface.program
      interface.program.definitions.length interface.definition params arguments) :
    Nonempty (ChildScopeExecutionPath samplers interface.program
      (interface.program.definitions.length - 1) interface.definition params arguments values) := by
  have positive : 0 < interface.program.definitions.length := by
    apply List.length_pos_of_ne_nil
    intro definitionsEmpty
    have bodyFound := interface.bodyFound
    rw [definitionsEmpty] at bodyFound
    simp [Mxx.Ir.lookupDefinition] at bodyFound
  have fuelEq : interface.program.definitions.length - 1 + 1 =
      interface.program.definitions.length := by omega
  rw [← fuelEq] at childMember
  exact interface.childExecutionExists samplers (interface.program.definitions.length - 1)
    params arguments values childMember

/-- Transport a full-fuel child member from the actual executable program and definition to the
frozen interface.  Equality elimination is isolated here, avoiding dependent rewrites through a
larger trace-evidence structure. -/
theorem FrozenSequentialRecurrenceInterface.childExecutionExistsOfMatches
    (interface : FrozenSequentialRecurrenceInterface)
    (samplers : Mxx.MxxSamplerFamily)
    (actualProgram : Mxx.Ir.Prog)
    (programMatches : actualProgram = interface.program)
    (actualDefinition : String)
    (definitionMatches : actualDefinition = interface.definition)
    (params : Mxx.Ir.ParamEnvironment)
    (arguments values : List Mxx.Ir.Value)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers actualProgram
      actualProgram.definitions.length actualDefinition params arguments) :
    Nonempty (ChildScopeExecutionPath samplers interface.program
      (interface.program.definitions.length - 1) interface.definition params arguments values) := by
  subst actualProgram
  subst actualDefinition
  exact interface.childExecutionExistsAtFullFuel samplers params arguments values childMember

/-- A child path recovered through a frozen recurrence definition executes exactly the body scope
stored by that interface. -/
theorem FrozenSequentialRecurrenceInterface.childScopeMatches
    (interface : FrozenSequentialRecurrenceInterface)
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    (execution : ChildScopeExecutionPath samplers interface.program fuel interface.definition
      params arguments values) :
    execution.scope = interface.body := by
  exact Option.some.inj (execution.definitionFound.symm.trans interface.bodyFound)

private theorem node_eq_parallelLoop_of_kind
    (node : Mxx.Ir.Node)
    (definition : String)
    (count : Mxx.Ir.IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × Mxx.Ir.IntExpr))
    (modes : List Mxx.Ir.LoopInputMode)
    (kindMatches : node.kind = .parallelLoop definition count indexSlot bindings modes) :
    node = {
      kind := .parallelLoop definition count indexSlot bindings modes
      arguments := node.arguments
      outputCount := node.outputCount
      outputTypes := node.outputTypes
    } := by
  cases node with
  | mk kind arguments outputCount outputTypes =>
      cases kindMatches
      rfl

/-- Construct the exact parallel-loop view certified by a node-kind equality. -/
def parallelLoopViewOfKind
    (node : Mxx.Ir.Node)
    (definition : String)
    (count : Mxx.Ir.IntExpr)
    (indexSlot : Nat)
    (bindings : List (String × Mxx.Ir.IntExpr))
    (modes : List Mxx.Ir.LoopInputMode)
    (kindMatches : node.kind = .parallelLoop definition count indexSlot bindings modes) :
    ParallelLoopNodeView node := {
  definition
  count
  indexSlot
  bindings
  modes
  argumentRefs := node.arguments
  outputCount := node.outputCount
  outputTypes := node.outputTypes
  nodeEq := node_eq_parallelLoop_of_kind node definition count indexSlot bindings modes kindMatches
}

/-- The exact parallel-loop view already certified by the static lane matcher. -/
def CheckedRecurrenceLaneOutput.parallelView
    {interface : FrozenSequentialRecurrenceInterface}
    (lane : CheckedRecurrenceLaneOutput interface) : ParallelLoopNodeView lane.node :=
  parallelLoopViewOfKind lane.node lane.definition lane.count lane.indexSlot lane.bindings
    lane.inputModes lane.kindMatches

/-- Every retained candidate provenance tree passed the closed frozen-program validator. -/
theorem CheckedRecurrenceLaneOutput.programFormulaValid
    {interface : FrozenSequentialRecurrenceInterface}
    (lane : CheckedRecurrenceLaneOutput interface)
    (formula : FrozenPointwiseMatrixProgramFormula)
    (member : formula ∈ lane.gateCandidateProgramFormulas) :
    formula.validIn interface.program = true := by
  have valid := lane.gateCandidateProgramFormulasValid
  simp only [List.all_eq_true] at valid
  exact valid formula member

/-- Execution of the matcher-selected outer lane node on one actual recurrence-body path.  The
node index comes from the body's output wire and is not a certificate or protocol label. -/
structure CheckedRecurrenceLaneOutput.Execution
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    (lane : CheckedRecurrenceLaneOutput interface)
    (scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values) where
  scopeMatches : scopeExecution.scope = interface.body
  nodeFound : scopeExecution.scope.nodes[lane.output.node]? = some lane.node
  nodeInBounds : lane.output.node < scopeExecution.scope.nodes.length
  actualNodeEq : scopeExecution.scope.nodes[lane.output.node] = lane.node
  view : ParallelLoopNodeView scopeExecution.scope.nodes[lane.output.node]
  definitionMatches : view.definition = lane.definition
  trace : NestedParallelTrace scopeExecution lane.output.node nodeInBounds view

/-- Bind the static lane selection to an actual parallel execution.  As with every executable
loop inversion, the two equality premises only certify evaluation of the frozen node's own
arguments and count; the returned trace is derived from its node member. -/
theorem CheckedRecurrenceLaneOutput.Execution.nonempty
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    (lane : CheckedRecurrenceLaneOutput interface)
    (scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values)
    (argumentValues : List Mxx.Ir.Value)
    (evaluatedCount : Int)
    (argumentsEvaluate : lane.node.arguments.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire
        (scopeExecution.nodeExecutionAt lane.output.node
          ((List.getElem?_eq_some_iff.mp (by
            simpa [interface.childScopeMatches scopeExecution] using lane.nodeFound)).1)).before) =
      some argumentValues)
    (countEvaluate : lane.count.evaluate params = some evaluatedCount) :
    Nonempty (lane.Execution scopeExecution) := by
  have scopeMatches := interface.childScopeMatches scopeExecution
  have nodeFound : scopeExecution.scope.nodes[lane.output.node]? = some lane.node := by
    simpa [scopeMatches] using lane.nodeFound
  have nodeInBounds := (List.getElem?_eq_some_iff.mp nodeFound).1
  have actualNodeEq := (List.getElem?_eq_some_iff.mp nodeFound).2
  have actualKindMatches : scopeExecution.scope.nodes[lane.output.node].kind =
      .parallelLoop lane.definition lane.count lane.indexSlot lane.bindings lane.inputModes := by
    rw [actualNodeEq]
    exact lane.kindMatches
  let view : ParallelLoopNodeView scopeExecution.scope.nodes[lane.output.node] :=
    parallelLoopViewOfKind scopeExecution.scope.nodes[lane.output.node] lane.definition lane.count
      lane.indexSlot lane.bindings lane.inputModes actualKindMatches
  have viewArguments : view.argumentRefs = lane.node.arguments := by
    change scopeExecution.scope.nodes[lane.output.node].arguments = lane.node.arguments
    exact congrArg Mxx.Ir.Node.arguments actualNodeEq
  have viewCount : view.count = lane.count := by
    rfl
  have definitionMatches : view.definition = lane.definition := by
    rfl
  have argumentsEvaluate' : view.argumentRefs.mapM
      (fun wire ↦ Mxx.Ir.lookupWire wire
        (scopeExecution.nodeExecutionAt lane.output.node nodeInBounds).before) =
      some argumentValues := by
    simpa [viewArguments] using argumentsEvaluate
  have countEvaluate' : view.count.evaluate params = some evaluatedCount := by
    simpa [viewCount] using countEvaluate
  obtain ⟨trace⟩ := NestedParallelTrace.nonempty_atNode scopeExecution lane.output.node
    nodeInBounds view argumentValues argumentsEvaluate' evaluatedCount countEvaluate'
  exact ⟨{
    scopeMatches
    nodeFound
    nodeInBounds
    actualNodeEq
    view
    definitionMatches
    trace
  }⟩

/-- Recover the exact nested lane-body path selected by one iteration of an outer parallel loop.
The predecessor fuel is derived from the frozen program length; the matcher has already proved
that this nested call is reachable with positive fuel. -/
theorem CheckedRecurrenceLaneOutput.childScopeExists
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    (lane : CheckedRecurrenceLaneOutput interface)
    (fuelMatches : fuel = interface.program.definitions.length - 1)
    (childParams : Mxx.Ir.ParamEnvironment)
    (childInputs childOutputs : List Mxx.Ir.Value)
    (childMember : childOutputs ∈ Mxx.Ir.childRunnerWithFuel samplers interface.program fuel
      lane.definition childParams childInputs) :
    Nonempty (ChildScopeExecutionPath samplers interface.program
      (interface.program.definitions.length - 2) lane.definition childParams childInputs
      childOutputs) := by
  have fuelSuccessor : interface.program.definitions.length - 2 + 1 =
      interface.program.definitions.length - 1 := by
    have positive := lane.nestedFuelPositive
    omega
  rw [fuelMatches, ← fuelSuccessor] at childMember
  exact ChildScopeExecutionPath.nonempty_of_childMember samplers interface.program
    (interface.program.definitions.length - 2) lane.definition lane.body childParams childInputs
    childOutputs lane.bodyFound childMember

/-- Every lane coordinate in the actual outer parallel trace selects an actual execution path of
the matcher-frozen lane body.  The result follows from the executable trace rather than from an
analyzer-provided lane formula. -/
theorem CheckedRecurrenceLaneOutput.Execution.everyChildScope
    {interface : FrozenSequentialRecurrenceInterface}
    {samplers : Mxx.MxxSamplerFamily}
    {fuel : Nat}
    {params : Mxx.Ir.ParamEnvironment}
    {arguments values : List Mxx.Ir.Value}
    {lane : CheckedRecurrenceLaneOutput interface}
    {scopeExecution : ChildScopeExecutionPath samplers interface.program fuel
      interface.definition params arguments values}
    (execution : lane.Execution scopeExecution)
    (fuelMatches : fuel = interface.program.definitions.length - 1) :
    ∀ index ∈ List.range execution.trace.evaluatedCount.toNat,
      ∃ childParams childInputs childOutputs,
        Nonempty (ChildScopeExecutionPath samplers interface.program
          (interface.program.definitions.length - 2) lane.definition childParams childInputs
          childOutputs) := by
  intro index indexMember
  obtain ⟨childParams, childInputs, childOutputs, childMember, _⟩ :=
    execution.trace.executionTrace.everyChild (fun _ _ _ _ ↦ True)
      (fun _ _ _ _ _ ↦ trivial) index indexMember
  rw [execution.definitionMatches] at childMember
  exact ⟨childParams, childInputs, childOutputs,
    lane.childScopeExists fuelMatches childParams childInputs childOutputs childMember⟩

theorem CheckedRecurrenceResolution.eq_of_same_reference
    {analysis : AnalysisResult}
    {reference : SequentialRecurrenceInstanceRef}
    (left right : CheckedRecurrenceResolution analysis reference) :
    left.transfer = right.transfer := by
  have singletonEq : [left.transfer] = [right.transfer] := left.unique.symm.trans right.unique
  exact List.cons.inj singletonEq |>.1

private def resolveFrozenRecurrence
    (analysis : AnalysisResult)
    (interface : FrozenSequentialRecurrenceInterface) :
    Option (ResolvedFrozenRecurrence analysis interface) := do
  let resolution ← resolveUniqueRecurrence analysis interface.transfer.identity
  if countMatches : resolution.transfer.source.count = interface.transfer.source.count then
    if arityMatches : resolution.transfer.source.carriedArity =
        interface.transfer.source.carriedArity then
      if schemasMatch : resolution.transfer.carriedSchemas = interface.transfer.carriedSchemas then
        return { resolution, countMatches, arityMatches, schemasMatch }
      else none
    else none
  else none

/-- Static three-trace matching reattached to the unique recurrences in the final analyzer
result. -/
structure ResolvedBggThreeTraceInterface
    (analysis : AnalysisResult)
    (bundle : ClosedProtocolBundle) where
  checked : CheckedBggThreeTraceInterface bundle
  encryption : ResolvedFrozenRecurrence analysis checked.candidate.prefilter.encryption
  decryption : ResolvedFrozenRecurrence analysis checked.candidate.prefilter.decryption
  booleanInterpreter : ResolvedFrozenRecurrence analysis
    checked.candidate.prefilter.booleanInterpreter

def resolveBggThreeTraceInterface
    (analysis : AnalysisResult)
    {bundle : ClosedProtocolBundle}
    (checked : CheckedBggThreeTraceInterface bundle) :
    Option (ResolvedBggThreeTraceInterface analysis bundle) := do
  let encryption ← resolveFrozenRecurrence analysis checked.candidate.prefilter.encryption
  let decryption ← resolveFrozenRecurrence analysis checked.candidate.prefilter.decryption
  let booleanInterpreter ← resolveFrozenRecurrence analysis
    checked.candidate.prefilter.booleanInterpreter
  return { checked, encryption, decryption, booleanInterpreter }

inductive ResolveBggThreeTraceError where
  | analyzer (error : VerifyError)
  | prefilter (error : BggRecurrencePrefilterError)
  | roles (error : BggCarriedRoleInferenceError)
  | interface (error : BggThreeTraceInterfaceError)
  | recurrenceResolution

/-- Reconstruct the complete static and analyzer-resolution package from the final analysis.
This is deterministic analyzer-owned work; no protocol certificate chooses a recurrence or role.
-/
def resolveBggThreeTraceFromAnalysis
    (bundle : ClosedProtocolBundle)
    (analysis : AnalysisResult) :
    Except ResolveBggThreeTraceError (ResolvedBggThreeTraceInterface analysis bundle) := do
  let interfaces ← constructFrozenRecurrenceInterfaces bundle analysis.symbolicRecurrences
    |>.mapError .analyzer
  let mut acceptances := []
  for requirement in bundle.requirements, index in [0:bundle.requirements.length] do
    match bundle.preconditionSpec.requirementOutputs[index]? with
    | none => pure ()
    | some outputName =>
        match checkRequirementAcceptance analysis.facts index requirement outputName with
        | none => pure ()
        | some acceptance => acceptances := acceptances ++ [acceptance]
  let prefilter ← checkBggRecurrencePrefilter interfaces acceptances |>.mapError .prefilter
  let candidate ← inferBggCarriedRoles prefilter |>.mapError .roles
  let checked ← checkBggThreeTraceInterface bundle candidate |>.mapError .interface
  resolveBggThreeTraceInterface analysis checked |>.elim
    (throw .recurrenceResolution) pure

/-- Canonical proof-only index of the bundle stage selected by the static matcher. -/
noncomputable def CheckedWorkflowRecurrenceOrigin.stageIndex
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    (origin : CheckedWorkflowRecurrenceOrigin bundle interface) : Nat :=
  Classical.choose (List.mem_iff_getElem?.mp origin.stageMember)

theorem CheckedWorkflowRecurrenceOrigin.stageAt
    {bundle : ClosedProtocolBundle}
    {interface : FrozenSequentialRecurrenceInterface}
    (origin : CheckedWorkflowRecurrenceOrigin bundle interface) :
    bundle.workflow.stages[origin.stageIndex]? = some origin.stage :=
  Classical.choose_spec (List.mem_iff_getElem?.mp origin.stageMember)

/-- Select the exact stage execution corresponding to the matcher-owned bundle origin. -/
theorem CheckedWorkflowRecurrenceOrigin.executionExists
    {samplers : Mxx.MxxSamplerFamily}
    {bundle : ClosedProtocolBundle}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (trace : WorkflowExecutionTrace samplers bundle.workflow parameters inputs)
    {interface : FrozenSequentialRecurrenceInterface}
    (origin : CheckedWorkflowRecurrenceOrigin bundle interface) :
    Nonempty (WorkflowStageExecutionAt trace origin.stageIndex origin.stage) :=
  WorkflowStageExecutionAt.exists trace origin.stageIndex origin.stage origin.stageAt

/-- The three matched recurrences selected from one closed protocol execution.

The workflow recurrences carry membership proofs in the exact workflow trace.  The Boolean
recurrence is selected from the exact accepted requirement execution stored in the same closed
trace.  Transfer equalities prevent a recurrence with a coincidentally equal identity from being
substituted for the frozen transfer checked by `CheckedBggThreeTraceInterface`.

This structure only binds executions.  It deliberately does not assert the BGG preimage
relation, Boolean/BGG coherence, or the final endpoint equation; those require the simultaneous
three-trace preservation theorem built on top of this evidence.
-/
structure TraceBoundBggThreeTrace
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    (analysis : AnalysisResult)
    (trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs)
    (interface : ResolvedBggThreeTraceInterface analysis protocol.bundle) where
  encryptionStageAt : WorkflowStageExecutionAt trace.workflow
    interface.checked.encryptionProgramOrigin.stageIndex
    interface.checked.encryptionProgramOrigin.stage
  encryption : TraceBoundSequentialRecurrence analysis encryptionStageAt.execution
    interface.checked.candidate.prefilter.encryption.transfer.identity
  decryptionStageAt : WorkflowStageExecutionAt trace.workflow
    interface.checked.decryptionProgramOrigin.stageIndex
    interface.checked.decryptionProgramOrigin.stage
  decryption : TraceBoundSequentialRecurrence analysis decryptionStageAt.execution
    interface.checked.candidate.prefilter.decryption.transfer.identity
  requirement : ClosedRequirementAcceptedExecution trace
    interface.checked.candidate.prefilter.requirementAcceptance.requirementIndex
    interface.checked.candidate.prefilter.booleanInterpreter.program
    interface.checked.candidate.prefilter.requirementAcceptance.outputName
  boolean : TraceBoundPureSequentialRecurrence analysis
    interface.checked.candidate.prefilter.booleanInterpreter.transfer.source.loop.site.stage
    requirement.execution
    interface.checked.candidate.prefilter.booleanInterpreter.transfer.identity
  encryptionDefinitionMatches : encryption.view.definition =
    interface.checked.candidate.prefilter.encryption.definition
  decryptionDefinitionMatches : decryption.view.definition =
    interface.checked.candidate.prefilter.decryption.definition
  booleanDefinitionMatches : boolean.view.definition =
    interface.checked.candidate.prefilter.booleanInterpreter.definition

/-- All three recurrences use the parameters of their common closed protocol execution. -/
theorem TraceBoundBggThreeTrace.workflowParameters
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.encryptionStageAt.execution.params = parameters ∧
      evidence.decryptionStageAt.execution.params = parameters ∧
      evidence.requirement.execution.params = parameters := by
  constructor
  · exact evidence.encryptionStageAt.paramsMatch
  constructor
  · exact evidence.decryptionStageAt.paramsMatch
  · exact evidence.requirement.parametersMatch

theorem TraceBoundBggThreeTrace.encryptionProgramMatches
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.encryptionStageAt.execution.stage.program =
      interface.checked.candidate.prefilter.encryption.program := by
  rw [evidence.encryptionStageAt.stageMatches]
  exact interface.checked.encryptionProgramOrigin.programMatches

theorem TraceBoundBggThreeTrace.decryptionProgramMatches
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.decryptionStageAt.execution.stage.program =
      interface.checked.candidate.prefilter.decryption.program := by
  rw [evidence.decryptionStageAt.stageMatches]
  exact interface.checked.decryptionProgramOrigin.programMatches

theorem TraceBoundBggThreeTrace.booleanProgramMatches
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.requirement.execution.program =
      interface.checked.candidate.prefilter.booleanInterpreter.program := by
  exact evidence.requirement.programMatches

/-- Body paths for one synchronized iteration, recovered from the three child-support members
stored by the actual traces.  The program, definition, runner fuel, parameters, and arguments are
all fixed by `evidence`; only the support members selected by that iteration are premises. -/
theorem TraceBoundBggThreeTrace.childExecutionsExist
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface)
    {index : Nat}
    {encryptionState encryptionNext decryptionState decryptionNext booleanState booleanNext :
      List Mxx.Ir.Value}
    {encryptionEvaluatedBindings decryptionEvaluatedBindings booleanEvaluatedBindings :
      Mxx.Ir.ParamEnvironment}
    (encryptionChildMember : encryptionNext ∈
      evidence.encryptionStageAt.execution.rootChildRunner evidence.encryption.view.definition
        (encryptionEvaluatedBindings ++
          ((.loopIndex evidence.encryption.view.indexSlot, .integer index) ::
            evidence.encryptionStageAt.execution.params))
        (encryptionState ++ evidence.encryption.invariantValues))
    (decryptionChildMember : decryptionNext ∈
      evidence.decryptionStageAt.execution.rootChildRunner evidence.decryption.view.definition
        (decryptionEvaluatedBindings ++
          ((.loopIndex evidence.decryption.view.indexSlot, .integer index) ::
            evidence.decryptionStageAt.execution.params))
        (decryptionState ++ evidence.decryption.invariantValues))
    (booleanChildMember : booleanNext ∈
      Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily
        evidence.requirement.execution.program
        evidence.requirement.execution.program.definitions.length
        evidence.boolean.view.definition
        (booleanEvaluatedBindings ++
          ((.loopIndex evidence.boolean.view.indexSlot, .integer index) ::
            evidence.requirement.execution.params))
        (booleanState ++
          evidence.boolean.trace.argumentValues.drop evidence.boolean.view.carriedCount)) :
    Nonempty (
      ChildScopeExecutionPath samplers
          interface.checked.candidate.prefilter.encryption.program
          (interface.checked.candidate.prefilter.encryption.program.definitions.length - 1)
          interface.checked.candidate.prefilter.encryption.definition
          (encryptionEvaluatedBindings ++
            ((.loopIndex evidence.encryption.view.indexSlot, .integer index) ::
              evidence.encryptionStageAt.execution.params))
          (encryptionState ++ evidence.encryption.invariantValues) encryptionNext ×
        ChildScopeExecutionPath samplers
          interface.checked.candidate.prefilter.decryption.program
          (interface.checked.candidate.prefilter.decryption.program.definitions.length - 1)
          interface.checked.candidate.prefilter.decryption.definition
          (decryptionEvaluatedBindings ++
            ((.loopIndex evidence.decryption.view.indexSlot, .integer index) ::
              evidence.decryptionStageAt.execution.params))
          (decryptionState ++ evidence.decryption.invariantValues) decryptionNext ×
        ChildScopeExecutionPath Mxx.Ir.emptySamplerFamily
          interface.checked.candidate.prefilter.booleanInterpreter.program
          (interface.checked.candidate.prefilter.booleanInterpreter.program.definitions.length - 1)
          interface.checked.candidate.prefilter.booleanInterpreter.definition
          (booleanEvaluatedBindings ++
            ((.loopIndex evidence.boolean.view.indexSlot, .integer index) ::
              evidence.requirement.execution.params))
          (booleanState ++
            evidence.boolean.trace.argumentValues.drop evidence.boolean.view.carriedCount)
          booleanNext) := by
  have encryptionMember : encryptionNext ∈ Mxx.Ir.childRunnerWithFuel samplers
      evidence.encryptionStageAt.execution.stage.program
      evidence.encryptionStageAt.execution.stage.program.definitions.length
      evidence.encryption.view.definition
      (encryptionEvaluatedBindings ++
        ((.loopIndex evidence.encryption.view.indexSlot, .integer index) ::
          evidence.encryptionStageAt.execution.params))
      (encryptionState ++ evidence.encryption.invariantValues) := by
    simpa only [StageExecution.rootChildRunner] using encryptionChildMember
  have decryptionMember : decryptionNext ∈ Mxx.Ir.childRunnerWithFuel samplers
      evidence.decryptionStageAt.execution.stage.program
      evidence.decryptionStageAt.execution.stage.program.definitions.length
      evidence.decryption.view.definition
      (decryptionEvaluatedBindings ++
        ((.loopIndex evidence.decryption.view.indexSlot, .integer index) ::
          evidence.decryptionStageAt.execution.params))
      (decryptionState ++ evidence.decryption.invariantValues) := by
    simpa only [StageExecution.rootChildRunner] using decryptionChildMember
  obtain ⟨encryptionPath⟩ :=
    interface.checked.candidate.prefilter.encryption.childExecutionExistsOfMatches samplers
      evidence.encryptionStageAt.execution.stage.program evidence.encryptionProgramMatches
      evidence.encryption.view.definition evidence.encryptionDefinitionMatches
      _ _ _ encryptionMember
  obtain ⟨decryptionPath⟩ :=
    interface.checked.candidate.prefilter.decryption.childExecutionExistsOfMatches samplers
      evidence.decryptionStageAt.execution.stage.program evidence.decryptionProgramMatches
      evidence.decryption.view.definition evidence.decryptionDefinitionMatches
      _ _ _ decryptionMember
  obtain ⟨booleanPath⟩ :=
    interface.checked.candidate.prefilter.booleanInterpreter.childExecutionExistsOfMatches
      Mxx.Ir.emptySamplerFamily evidence.requirement.execution.program
      evidence.booleanProgramMatches evidence.boolean.view.definition
      evidence.booleanDefinitionMatches _ _ _ booleanChildMember
  exact ⟨encryptionPath, decryptionPath, booleanPath⟩

theorem TraceBoundBggThreeTrace.encryptionTransferResolved
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.encryption.transfer = interface.encryption.resolution.transfer :=
  CheckedRecurrenceResolution.eq_of_same_reference
    { transfer := evidence.encryption.transfer, unique := evidence.encryption.uniqueResolution }
    interface.encryption.resolution

theorem TraceBoundBggThreeTrace.decryptionTransferResolved
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.decryption.transfer = interface.decryption.resolution.transfer :=
  CheckedRecurrenceResolution.eq_of_same_reference
    { transfer := evidence.decryption.transfer, unique := evidence.decryption.uniqueResolution }
    interface.decryption.resolution

theorem TraceBoundBggThreeTrace.booleanTransferResolved
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    evidence.boolean.transfer = interface.booleanInterpreter.resolution.transfer :=
  CheckedRecurrenceResolution.eq_of_same_reference
    { transfer := evidence.boolean.transfer, unique := evidence.boolean.uniqueResolution }
    interface.booleanInterpreter.resolution

/-- The three actual loop traces evaluated the same closed count expression under the same
parameter environment. -/
structure ThreeTraceCountsAgree
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) : Prop where
  encryptionDecryption : evidence.encryption.evaluatedCount =
    evidence.decryption.evaluatedCount
  encryptionBoolean : evidence.encryption.evaluatedCount =
    evidence.boolean.trace.evaluatedCount

/-- Pointwise semantic relation maintained by the two BGG recurrences.  Both equations are in
the quotient ring.  The secret and error matrices remain explicit so endpoint normalization can
preserve their independent hard-bound witnesses instead of hiding them in an aggregate matrix.
-/
structure BggEncodingLaneRelation
    (secret gadget encryptionPublicKey encodingVector encodingPublicKey plaintext : Mxx.Matrix) :
    Type
    where
  error : Mxx.Matrix
  secretBound : Nat
  errorBound : Nat
  publicKeyEquation : Mxx.MatrixModEq encodingPublicKey encryptionPublicKey
  encodingEquation : Mxx.MatrixModEq encodingVector
    (Mxx.matrixAdd
      (Mxx.matrixSubtract
        (Mxx.matrixMultiply secret encodingPublicKey)
        (Mxx.matrixMultiply plaintext (Mxx.matrixMultiply secret gadget)))
      error)
  secretNorm : Mxx.maxCenteredCoefficientNorm secret ≤ secretBound
  errorNorm : Mxx.maxCenteredCoefficientNorm error ≤ errorBound

/-- Interpret one executable BGG lane in the exact negacyclic quotient.  Both premises are only
`MatrixModEq`; the conversion never assumes that stored integer representatives are equal. -/
noncomputable def BggEncodingLaneRelation.toQuotient
    (q ringDimension outputRows secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (secret gadget encryptionPublicKey encodingVector encodingPublicKey plaintext : Mxx.Matrix)
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension outputRows secretColumns)
    (gadgetLayout : Mxx.Toolkit.MatrixLayout gadget q ringDimension secretColumns publicColumns)
    (encryptionPublicKeyLayout : Mxx.Toolkit.MatrixLayout encryptionPublicKey q ringDimension
      secretColumns publicColumns)
    (encodingVectorLayout : Mxx.Toolkit.MatrixLayout encodingVector q ringDimension
      outputRows publicColumns)
    (encodingPublicKeyLayout : Mxx.Toolkit.MatrixLayout encodingPublicKey q ringDimension
      secretColumns publicColumns)
    (plaintextLayout : Mxx.Toolkit.MatrixLayout plaintext q ringDimension 1 1)
    (relation : BggEncodingLaneRelation secret gadget encryptionPublicKey encodingVector
      encodingPublicKey plaintext)
    (errorLayout : Mxx.Toolkit.MatrixLayout relation.error q ringDimension
      outputRows publicColumns) :
    QuotientBggLane
      (Mxx.Toolkit.matrixValue q ringDimension outputRows secretColumns secret)
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns gadget)
      (Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns encryptionPublicKey)
      (Mxx.Toolkit.matrixValue q ringDimension outputRows publicColumns encodingVector)
      ((Mxx.Toolkit.matrixValue q ringDimension 1 1 plaintext) 0 0) := by
  let secretPublicKey := Mxx.matrixMultiply secret encodingPublicKey
  let secretGadget := Mxx.matrixMultiply secret gadget
  let plaintextSecretGadget := Mxx.matrixMultiply plaintext secretGadget
  let signal := Mxx.matrixSubtract secretPublicKey plaintextSecretGadget
  let reconstructed := Mxx.matrixAdd signal relation.error
  have secretPublicKeyLayout := Mxx.Toolkit.matrixMultiply_layout secret encodingPublicKey
    secretLayout encodingPublicKeyLayout
  have secretGadgetLayout := Mxx.Toolkit.matrixMultiply_layout secret gadget
    secretLayout gadgetLayout
  have plaintextSecretGadgetLayout := Mxx.Toolkit.matrixMultiply_leftBroadcast_layout
    plaintext secretGadget plaintextLayout secretGadgetLayout
  have signalLayout := Mxx.Toolkit.matrixSubtract_layout secretPublicKey plaintextSecretGadget
    secretPublicKeyLayout plaintextSecretGadgetLayout
  have reconstructedLayout := Mxx.Toolkit.matrixAdd_layout signal relation.error
    signalLayout errorLayout
  have encodingValue := Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension outputRows
    publicColumns encodingVector reconstructed encodingVectorLayout reconstructedLayout
    relation.encodingEquation
  have publicKeyValue := Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension secretColumns
    publicColumns encodingPublicKey encryptionPublicKey encodingPublicKeyLayout
    encryptionPublicKeyLayout relation.publicKeyEquation
  refine {
    error := Mxx.Toolkit.matrixValue q ringDimension outputRows publicColumns relation.error
    equation := ?_
  }
  rw [encodingValue]
  dsimp [reconstructed, signal, secretPublicKey, plaintextSecretGadget, secretGadget]
  rw [Mxx.Toolkit.matrixValue_add q ringDimension outputRows publicColumns _ _
      ⟨signalLayout.modulus, signalLayout.ringDimension, signalLayout.rows,
        signalLayout.columns⟩
      ⟨errorLayout.modulus, errorLayout.ringDimension, errorLayout.rows,
        errorLayout.columns⟩,
    Mxx.Toolkit.matrixValue_subtract q ringDimension outputRows publicColumns _ _
      ⟨secretPublicKeyLayout.modulus, secretPublicKeyLayout.ringDimension,
        secretPublicKeyLayout.rows, secretPublicKeyLayout.columns⟩
      ⟨plaintextSecretGadgetLayout.modulus, plaintextSecretGadgetLayout.ringDimension,
        plaintextSecretGadgetLayout.rows, plaintextSecretGadgetLayout.columns⟩,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows secretColumns
      publicColumns secret encodingPublicKey secretLayout encodingPublicKeyLayout,
    Mxx.Toolkit.matrixValue_matrixMultiply_leftBroadcast q ringDimension outputRows
      publicColumns plaintext (Mxx.matrixMultiply secret gadget) plaintextLayout
      secretGadgetLayout,
    Mxx.Toolkit.matrixValue_matrixMultiply q ringDimension outputRows secretColumns
      publicColumns secret gadget secretLayout gadgetLayout,
    publicKeyValue]
  rfl

/-- Convert the executable gadget-decomposition contract directly into the quotient equality
consumed by `QuotientBggLayerTransfer`.  No equality of integer coefficient representatives is
used. -/
theorem gadgetDecomposition_toQuotient
    (q ringDimension secretColumns publicColumns : Nat)
    [Fact (1 < q)] [NeZero ringDimension]
    (gadget decomposition publicKey : Mxx.Matrix)
    (gadgetLayout : Mxx.Toolkit.MatrixLayout gadget q ringDimension secretColumns publicColumns)
    (decompositionLayout : Mxx.Toolkit.MatrixLayout decomposition q ringDimension publicColumns
      publicColumns)
    (publicKeyLayout : Mxx.Toolkit.MatrixLayout publicKey q ringDimension secretColumns
      publicColumns)
    (relation : Mxx.MatrixModEq (Mxx.matrixMul gadget decomposition) publicKey) :
    Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns gadget *
        Mxx.Toolkit.matrixValue q ringDimension publicColumns publicColumns decomposition =
      Mxx.Toolkit.matrixValue q ringDimension secretColumns publicColumns publicKey := by
  have productLayout := Mxx.Toolkit.matrixMul_layout gadget decomposition gadgetLayout
    decompositionLayout
  have quotientEq := Mxx.Toolkit.matrixValue_eq_of_modEq q ringDimension secretColumns
    publicColumns (Mxx.matrixMul gadget decomposition) publicKey productLayout publicKeyLayout
    relation
  rw [Mxx.Toolkit.matrixValue_mul q ringDimension secretColumns publicColumns publicColumns
      gadget decomposition
      ⟨gadgetLayout.modulus, gadgetLayout.ringDimension, gadgetLayout.rows,
        gadgetLayout.columns⟩
      ⟨decompositionLayout.modulus, decompositionLayout.ringDimension,
        decompositionLayout.rows, decompositionLayout.columns⟩] at quotientEq
  exact quotientEq

/-- Boolean/BGG coherence for one lane.  The zero and one matrices are actual frozen constants
selected by the checked body, not values invented by this relation. -/
structure BggInterpreterLaneRelation
    (secret gadget zero one encryptionPublicKey encodingVector encodingPublicKey plaintext :
      Mxx.Matrix)
    (booleanValue : Bool) : Type extends
      BggEncodingLaneRelation secret gadget encryptionPublicKey encodingVector encodingPublicKey
        plaintext
    where
  plaintextEquation : Mxx.MatrixModEq plaintext (if booleanValue then one else zero)

/-- Pointwise relation over all lanes.  Equal lengths are forced by the constructors, and no
lane can be silently omitted or reordered. -/
inductive BggInterpreterFamilyRelation
    (secret gadget zero one : Mxx.Matrix) :
    List Mxx.Matrix → List Mxx.Matrix → List Mxx.Matrix → List Mxx.Matrix →
      List Bool → Prop where
  | nil : BggInterpreterFamilyRelation secret gadget zero one [] [] [] [] []
  | cons
      {encryptionPublicKey encodingVector encodingPublicKey plaintext : Mxx.Matrix}
      {booleanValue : Bool}
      {encryptionPublicKeys encodingVectors encodingPublicKeys plaintextMatrices :
        List Mxx.Matrix}
      {booleanValues : List Bool}
      (head : BggInterpreterLaneRelation secret gadget zero one encryptionPublicKey
        encodingVector encodingPublicKey plaintext booleanValue)
      (tail : BggInterpreterFamilyRelation secret gadget zero one encryptionPublicKeys
        encodingVectors encodingPublicKeys plaintextMatrices booleanValues) :
      BggInterpreterFamilyRelation secret gadget zero one
        (encryptionPublicKey :: encryptionPublicKeys)
        (encodingVector :: encodingVectors)
        (encodingPublicKey :: encodingPublicKeys)
        (plaintext :: plaintextMatrices)
        (booleanValue :: booleanValues)

/-- Semantic state relation used by the simultaneous three-trace induction.  All five families
are extracted from actual carried states through the unique roles chosen by the frozen matcher;
the theorem caller cannot substitute differently ordered families. -/
structure BggThreeTraceStateRelation
    (slots : CheckedBggEncodingSlots)
    (secret gadget zero one : Mxx.Matrix)
    (encryptionState decryptionState booleanState : List Mxx.Ir.Value) : Type where
  encryptionPublicKeys : List Mxx.Matrix
  encodingVectors : List Mxx.Matrix
  encodingPublicKeys : List Mxx.Matrix
  plaintextMatrices : List Mxx.Matrix
  booleanValues : List Bool
  encryptionPublicKeysFound :
    matrixFamilyAt encryptionState slots.encryptionPublicKeys = some encryptionPublicKeys
  encodingVectorsFound :
    matrixFamilyAt decryptionState slots.encodingVectors = some encodingVectors
  encodingPublicKeysFound :
    matrixFamilyAt decryptionState slots.encodingPublicKeys = some encodingPublicKeys
  plaintextMatricesFound :
    matrixFamilyAt decryptionState slots.plaintextMatrices = some plaintextMatrices
  booleanValuesFound : booleanFamilyAt booleanState 0 = some booleanValues
  relation : BggInterpreterFamilyRelation secret gadget zero one encryptionPublicKeys
    encodingVectors encodingPublicKeys plaintextMatrices booleanValues

theorem TraceBoundBggThreeTrace.countsAgree
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) :
    ThreeTraceCountsAgree evidence := by
  have params := evidence.workflowParameters
  have encryptionDecryptionExpr : evidence.encryption.view.count =
      evidence.decryption.view.count := by
    calc
      evidence.encryption.view.count = evidence.encryption.transfer.source.count :=
        evidence.encryption.countMatches.symm
      _ = interface.encryption.resolution.transfer.source.count :=
        congrArg (fun transfer => transfer.source.count) evidence.encryptionTransferResolved
      _ = interface.checked.candidate.prefilter.encryption.transfer.source.count :=
        interface.encryption.countMatches
      _ = interface.checked.candidate.prefilter.encryptionDecryptionCount.origin :=
        interface.checked.candidate.prefilter.encryptionDecryptionCount.leftEq
      _ = interface.checked.candidate.prefilter.decryption.transfer.source.count :=
        interface.checked.candidate.prefilter.encryptionDecryptionCount.rightEq.symm
      _ = interface.decryption.resolution.transfer.source.count :=
        interface.decryption.countMatches.symm
      _ = evidence.decryption.transfer.source.count :=
        congrArg (fun transfer => transfer.source.count) evidence.decryptionTransferResolved.symm
      _ = evidence.decryption.view.count := evidence.decryption.countMatches
  have encryptionBooleanExpr : evidence.encryption.view.count =
      evidence.boolean.view.count := by
    calc
      evidence.encryption.view.count = evidence.encryption.transfer.source.count :=
        evidence.encryption.countMatches.symm
      _ = interface.encryption.resolution.transfer.source.count :=
        congrArg (fun transfer => transfer.source.count) evidence.encryptionTransferResolved
      _ = interface.checked.candidate.prefilter.encryption.transfer.source.count :=
        interface.encryption.countMatches
      _ = interface.checked.candidate.prefilter.encryptionBooleanCount.origin :=
        interface.checked.candidate.prefilter.encryptionBooleanCount.leftEq
      _ = interface.checked.candidate.prefilter.booleanInterpreter.transfer.source.count :=
        interface.checked.candidate.prefilter.encryptionBooleanCount.rightEq.symm
      _ = interface.booleanInterpreter.resolution.transfer.source.count :=
        interface.booleanInterpreter.countMatches.symm
      _ = evidence.boolean.transfer.source.count :=
        congrArg (fun transfer => transfer.source.count) evidence.booleanTransferResolved.symm
      _ = evidence.boolean.view.count := evidence.boolean.countMatches
  constructor
  · have encryptionEvaluate := evidence.encryption.countEvaluate
    have decryptionEvaluate := evidence.decryption.countEvaluate
    rw [params.1, encryptionDecryptionExpr] at encryptionEvaluate
    rw [params.2.1] at decryptionEvaluate
    exact Option.some.inj (encryptionEvaluate.symm.trans decryptionEvaluate)
  · have encryptionEvaluate := evidence.encryption.countEvaluate
    have booleanEvaluate := evidence.boolean.trace.countEvaluate
    rw [params.1, encryptionBooleanExpr] at encryptionEvaluate
    rw [params.2.2] at booleanEvaluate
    exact Option.some.inj (encryptionEvaluate.symm.trans booleanEvaluate)

/-! ## Simultaneous execution trace

The three executable loops use different child runners and frozen definitions.  Equal evaluated
counts therefore do not make their traces definitionally equal.  The following execution-only
trace zips the three actual traces by their common index list.  Its constructors retain every
binding evaluation and child-support member from the source traces and contain no semantic
invariant or user-provided preservation callback.
-/

/-- Three sequential-loop executions aligned at exactly the same iteration indices. -/
inductive BggThreeSequentialTrace
    (encryptionRunner decryptionRunner booleanRunner : Mxx.Ir.ChildRunner)
    (encryptionDefinition decryptionDefinition booleanDefinition : String)
    (encryptionParams decryptionParams booleanParams : Mxx.Ir.ParamEnvironment)
    (encryptionIndexSlot decryptionIndexSlot booleanIndexSlot : Nat)
    (encryptionBindings decryptionBindings booleanBindings :
      List (String × Mxx.Ir.IntExpr))
    (encryptionInvariants decryptionInvariants booleanInvariants : List Mxx.Ir.Value) :
    List Nat →
      List Mxx.Ir.Value → List Mxx.Ir.Value →
      List Mxx.Ir.Value → List Mxx.Ir.Value →
      List Mxx.Ir.Value → List Mxx.Ir.Value → Prop where
  | nil (encryptionState decryptionState booleanState : List Mxx.Ir.Value) :
      BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
        encryptionDefinition decryptionDefinition booleanDefinition
        encryptionParams decryptionParams booleanParams
        encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
        encryptionBindings decryptionBindings booleanBindings
        encryptionInvariants decryptionInvariants booleanInvariants []
        encryptionState encryptionState decryptionState decryptionState booleanState booleanState
  | cons
      (index : Nat)
      (tail : List Nat)
      (encryptionState encryptionNext encryptionFinal : List Mxx.Ir.Value)
      (decryptionState decryptionNext decryptionFinal : List Mxx.Ir.Value)
      (booleanState booleanNext booleanFinal : List Mxx.Ir.Value)
      (encryptionEvaluatedBindings decryptionEvaluatedBindings booleanEvaluatedBindings :
        Mxx.Ir.ParamEnvironment)
      (encryptionBindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex encryptionIndexSlot, .integer index) :: encryptionParams)
        encryptionBindings = some encryptionEvaluatedBindings)
      (encryptionChildMember : encryptionNext ∈ encryptionRunner encryptionDefinition
        (encryptionEvaluatedBindings ++
          ((.loopIndex encryptionIndexSlot, .integer index) :: encryptionParams))
        (encryptionState ++ encryptionInvariants))
      (decryptionBindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex decryptionIndexSlot, .integer index) :: decryptionParams)
        decryptionBindings = some decryptionEvaluatedBindings)
      (decryptionChildMember : decryptionNext ∈ decryptionRunner decryptionDefinition
        (decryptionEvaluatedBindings ++
          ((.loopIndex decryptionIndexSlot, .integer index) :: decryptionParams))
        (decryptionState ++ decryptionInvariants))
      (booleanBindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex booleanIndexSlot, .integer index) :: booleanParams)
        booleanBindings = some booleanEvaluatedBindings)
      (booleanChildMember : booleanNext ∈ booleanRunner booleanDefinition
        (booleanEvaluatedBindings ++
          ((.loopIndex booleanIndexSlot, .integer index) :: booleanParams))
        (booleanState ++ booleanInvariants))
      (rest : BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
        encryptionDefinition decryptionDefinition booleanDefinition
        encryptionParams decryptionParams booleanParams
        encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
        encryptionBindings decryptionBindings booleanBindings
        encryptionInvariants decryptionInvariants booleanInvariants tail
        encryptionNext encryptionFinal decryptionNext decryptionFinal booleanNext booleanFinal) :
      BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
        encryptionDefinition decryptionDefinition booleanDefinition
        encryptionParams decryptionParams booleanParams
        encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
        encryptionBindings decryptionBindings booleanBindings
        encryptionInvariants decryptionInvariants booleanInvariants (index :: tail)
        encryptionState encryptionFinal decryptionState decryptionFinal booleanState booleanFinal

/-- Zip three actual sequential traces.  Since all three arguments are indexed by the same list,
the impossible constructor combinations are eliminated by dependent pattern matching. -/
theorem BggThreeSequentialTrace.ofTraces
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
    (encryptionTrace : Mxx.Ir.SequentialIterationsTrace encryptionRunner encryptionDefinition
      encryptionParams encryptionIndexSlot encryptionBindings encryptionInvariants indices
      encryptionInitial encryptionFinal)
    (decryptionTrace : Mxx.Ir.SequentialIterationsTrace decryptionRunner decryptionDefinition
      decryptionParams decryptionIndexSlot decryptionBindings decryptionInvariants indices
      decryptionInitial decryptionFinal)
    (booleanTrace : Mxx.Ir.SequentialIterationsTrace booleanRunner booleanDefinition
      booleanParams booleanIndexSlot booleanBindings booleanInvariants indices
      booleanInitial booleanFinal) :
    BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
      encryptionDefinition decryptionDefinition booleanDefinition
      encryptionParams decryptionParams booleanParams
      encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
      encryptionBindings decryptionBindings booleanBindings
      encryptionInvariants decryptionInvariants booleanInvariants indices
      encryptionInitial encryptionFinal decryptionInitial decryptionFinal
      booleanInitial booleanFinal := by
  induction encryptionTrace generalizing decryptionInitial decryptionFinal booleanInitial
    booleanFinal with
  | nil encryptionState =>
      cases decryptionTrace
      cases booleanTrace
      exact .nil encryptionState decryptionInitial booleanInitial
  | cons index tail encryptionState encryptionEvaluatedBindings encryptionNext encryptionFinal
      encryptionBindingsEvaluate encryptionChildMember encryptionRest induction =>
      cases decryptionTrace with
      | cons _ _ decryptionState decryptionEvaluatedBindings decryptionNext decryptionFinal
          decryptionBindingsEvaluate decryptionChildMember decryptionRest =>
          cases booleanTrace with
          | cons _ _ booleanState booleanEvaluatedBindings booleanNext booleanFinal
              booleanBindingsEvaluate booleanChildMember booleanRest =>
              exact .cons index tail encryptionState encryptionNext encryptionFinal
                decryptionInitial decryptionNext decryptionFinal booleanInitial booleanNext
                booleanFinal encryptionEvaluatedBindings decryptionEvaluatedBindings
                booleanEvaluatedBindings encryptionBindingsEvaluate encryptionChildMember
                decryptionBindingsEvaluate decryptionChildMember booleanBindingsEvaluate
                booleanChildMember (induction decryptionRest booleanRest)

/-- The simultaneous trace proposition specialized to the three matcher-selected executions. -/
def TraceBoundBggThreeTrace.AlignedExecution
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) : Prop :=
  BggThreeSequentialTrace
    evidence.encryptionStageAt.execution.rootChildRunner
    evidence.decryptionStageAt.execution.rootChildRunner
    (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily
      evidence.requirement.execution.program
      evidence.requirement.execution.program.definitions.length)
    evidence.encryption.view.definition evidence.decryption.view.definition
    evidence.boolean.view.definition
    evidence.encryptionStageAt.execution.params evidence.decryptionStageAt.execution.params
    evidence.requirement.execution.params
    evidence.encryption.view.indexSlot evidence.decryption.view.indexSlot
    evidence.boolean.view.indexSlot
    evidence.encryption.view.bindings evidence.decryption.view.bindings
    evidence.boolean.view.bindings
    evidence.encryption.invariantValues evidence.decryption.invariantValues
    (evidence.boolean.trace.argumentValues.drop evidence.boolean.view.carriedCount)
    (List.range evidence.encryption.evaluatedCount.toNat)
    (evidence.encryption.argumentValues.take evidence.encryption.view.carriedCount)
    evidence.encryption.nodeValues
    (evidence.decryption.argumentValues.take evidence.decryption.view.carriedCount)
    evidence.decryption.nodeValues
    (evidence.boolean.trace.argumentValues.take evidence.boolean.view.carriedCount)
    evidence.boolean.trace.nodeValues

/-- Zip the three actual traces selected from one closed-protocol execution.  Count equality is
derived from frozen count origins and exact evaluation; no trace is rerun or supplied separately.
-/
theorem TraceBoundBggThreeTrace.alignedExecution
    {samplers : Mxx.MxxSamplerFamily}
    {protocol : ClosedProtocolDecl}
    {parameters : Mxx.Ir.ParamEnvironment}
    {inputs : Mxx.Ir.Environment}
    {analysis : AnalysisResult}
    {trace : ClosedProtocolExecutionTrace samplers protocol.bundle parameters inputs}
    {interface : ResolvedBggThreeTraceInterface analysis protocol.bundle}
    (evidence : TraceBoundBggThreeTrace analysis trace interface) : evidence.AlignedExecution := by
  have counts := evidence.countsAgree
  have decryptionTrace : Mxx.Ir.SequentialIterationsTrace
      evidence.decryptionStageAt.execution.rootChildRunner evidence.decryption.view.definition
      evidence.decryptionStageAt.execution.params evidence.decryption.view.indexSlot
      evidence.decryption.view.bindings evidence.decryption.invariantValues
      (List.range evidence.encryption.evaluatedCount.toNat)
      (evidence.decryption.argumentValues.take evidence.decryption.view.carriedCount)
      evidence.decryption.nodeValues := by
    simpa only [TraceBoundSequentialRecurrence.invariantValues,
      counts.encryptionDecryption] using evidence.decryption.executionTrace
  have booleanTrace : Mxx.Ir.SequentialIterationsTrace
      (Mxx.Ir.childRunnerWithFuel Mxx.Ir.emptySamplerFamily
        evidence.requirement.execution.program
        evidence.requirement.execution.program.definitions.length)
      evidence.boolean.view.definition evidence.requirement.execution.params
      evidence.boolean.view.indexSlot evidence.boolean.view.bindings
      (evidence.boolean.trace.argumentValues.drop evidence.boolean.view.carriedCount)
      (List.range evidence.encryption.evaluatedCount.toNat)
      (evidence.boolean.trace.argumentValues.take evidence.boolean.view.carriedCount)
      evidence.boolean.trace.nodeValues := by
    simpa [counts.encryptionBoolean] using evidence.boolean.trace.executionTrace
  exact BggThreeSequentialTrace.ofTraces evidence.encryption.executionTrace decryptionTrace
    booleanTrace

/-- Dependent quotient-ring derivation over one simultaneous execution trace.  A successor can
only be formed with a `BggThreeTraceQuotientStep` for the exact three `next` states stored by the
trace constructor.  The next induction state is computed by `step.nextState`; there is no field
for a caller-supplied invariant or arbitrary next relation. -/
inductive BggThreeTraceQuotientDerivation
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
    {encryptionInvariants decryptionInvariants booleanInvariants : List Mxx.Ir.Value} :
    {indices : List Nat} →
      {encryptionInitial encryptionFinal decryptionInitial decryptionFinal
        booleanInitial booleanFinal : List Mxx.Ir.Value} →
      (trace : BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
        encryptionDefinition decryptionDefinition booleanDefinition
        encryptionParams decryptionParams booleanParams
        encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
        encryptionBindings decryptionBindings booleanBindings
        encryptionInvariants decryptionInvariants booleanInvariants indices
        encryptionInitial encryptionFinal decryptionInitial decryptionFinal
        booleanInitial booleanFinal) →
      BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns publicColumns
        secret gadget encryptionInitial decryptionInitial booleanInitial → Type where
  | nil
      {encryptionState decryptionState booleanState : List Mxx.Ir.Value}
      (state : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
        publicColumns secret gadget encryptionState decryptionState booleanState) :
      BggThreeTraceQuotientDerivation
        (.nil encryptionState decryptionState booleanState) state
  | cons
      {index : Nat}
      {tail : List Nat}
      {encryptionState encryptionNext encryptionFinal : List Mxx.Ir.Value}
      {decryptionState decryptionNext decryptionFinal : List Mxx.Ir.Value}
      {booleanState booleanNext booleanFinal : List Mxx.Ir.Value}
      {encryptionEvaluatedBindings decryptionEvaluatedBindings booleanEvaluatedBindings :
        Mxx.Ir.ParamEnvironment}
      {encryptionBindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex encryptionIndexSlot, .integer index) :: encryptionParams)
        encryptionBindings = some encryptionEvaluatedBindings}
      {encryptionChildMember : encryptionNext ∈ encryptionRunner encryptionDefinition
        (encryptionEvaluatedBindings ++
          ((.loopIndex encryptionIndexSlot, .integer index) :: encryptionParams))
        (encryptionState ++ encryptionInvariants)}
      {decryptionBindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex decryptionIndexSlot, .integer index) :: decryptionParams)
        decryptionBindings = some decryptionEvaluatedBindings}
      {decryptionChildMember : decryptionNext ∈ decryptionRunner decryptionDefinition
        (decryptionEvaluatedBindings ++
          ((.loopIndex decryptionIndexSlot, .integer index) :: decryptionParams))
        (decryptionState ++ decryptionInvariants)}
      {booleanBindingsEvaluate : Mxx.Ir.evaluateBindings
        ((.loopIndex booleanIndexSlot, .integer index) :: booleanParams)
        booleanBindings = some booleanEvaluatedBindings}
      {booleanChildMember : booleanNext ∈ booleanRunner booleanDefinition
        (booleanEvaluatedBindings ++
          ((.loopIndex booleanIndexSlot, .integer index) :: booleanParams))
        (booleanState ++ booleanInvariants)}
      {rest : BggThreeSequentialTrace encryptionRunner decryptionRunner booleanRunner
        encryptionDefinition decryptionDefinition booleanDefinition
        encryptionParams decryptionParams booleanParams
        encryptionIndexSlot decryptionIndexSlot booleanIndexSlot
        encryptionBindings decryptionBindings booleanBindings
        encryptionInvariants decryptionInvariants booleanInvariants tail
        encryptionNext encryptionFinal decryptionNext decryptionFinal booleanNext booleanFinal}
      (state : BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns
        publicColumns secret gadget encryptionState decryptionState booleanState)
      (step : BggThreeTraceQuotientStep slots q ringDimension outputRows secretColumns
        publicColumns secret gadget state encryptionNext decryptionNext booleanNext)
      (restDerivation : BggThreeTraceQuotientDerivation rest step.nextState) :
      BggThreeTraceQuotientDerivation
        (.cons index tail encryptionState encryptionNext encryptionFinal
          decryptionState decryptionNext decryptionFinal booleanState booleanNext booleanFinal
          encryptionEvaluatedBindings decryptionEvaluatedBindings booleanEvaluatedBindings
          encryptionBindingsEvaluate encryptionChildMember decryptionBindingsEvaluate
          decryptionChildMember booleanBindingsEvaluate booleanChildMember rest)
        state

/-- The final synchronized relation is obtained solely by following the actual simultaneous
trace and the mechanically computed `nextState` at each constructor. -/
noncomputable def BggThreeTraceQuotientDerivation.finalState
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
    (derivation : BggThreeTraceQuotientDerivation trace initialState) :
    BggThreeTraceQuotientState slots q ringDimension outputRows secretColumns publicColumns
      secret gadget encryptionFinal decryptionFinal booleanFinal := by
  induction derivation with
  | nil state => exact state
  | cons _ _ restDerivation induction => exact induction

end Mxx.Certificate
