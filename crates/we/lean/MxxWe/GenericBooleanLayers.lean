import MxxWe.GenericInputInjection

open Mxx

namespace MxxWe

namespace DynamicBoolean

/-- One fixed-allocation Boolean circuit supplied as protocol data.  The shape is external:
`depth` and `maxWidth` are protocol parameters, while every list here is a dynamic input. -/
structure CircuitData where
  activeGateCounts : List Int
  gateKinds : List Int
  leftSources : List Int
  rightSources : List Int
  outputSources : List Int

def entry (values : List Int) (maxWidth layer slot : Nat) : Int :=
  values[layer * maxWidth + slot]?.getD 0

def activeCount (circuit : CircuitData) (layer : Nat) : Nat :=
  (circuit.activeGateCounts[layer]?.getD 0).toNat

def gateKind : Int → BooleanGate
  | 0 => .constantFalse
  | 1 => .constantTrue
  | 2 => .copy
  | 3 => .not
  | 4 => .and
  | 5 => .xor
  | _ => .constantFalse

def evalGate : BooleanGate → Bool → Bool → Bool
  | .constantFalse, _, _ => false
  | .constantTrue, _, _ => true
  | .copy, left, _ => left
  | .not, left, _ => !left
  | .and, left, right => left && right
  | .xor, left, right => left != right

/-- Canonical fixed-width input state.  Only the declared instance/witness prefixes are used;
the remainder of the allocated family is zero padding. -/
def canonicalInputs (maxWidth instanceWidth witnessWidth : Nat)
    (instanceValues witnessValues : List Int) : List Bool :=
  (List.range maxWidth).map fun slot ↦
    if slot < instanceWidth then decide (instanceValues[slot]?.getD 0 = 1)
    else if slot < instanceWidth + witnessWidth then
      decide (witnessValues[slot - instanceWidth]?.getD 0 = 1)
    else false

def evaluateLayer (maxWidth layer : Nat) (circuit : CircuitData)
    (previous : List Bool) : List Bool :=
  (List.range maxWidth).map fun slot ↦
    if slot < activeCount circuit layer then
      evalGate (gateKind (entry circuit.gateKinds maxWidth layer slot))
        (previous[(entry circuit.leftSources maxWidth layer slot).toNat]?.getD false)
        (previous[(entry circuit.rightSources maxWidth layer slot).toNat]?.getD false)
    else false

def evaluateLayers (maxWidth : Nat) (circuit : CircuitData) (initial : List Bool) :
    Nat → List Bool
  | 0 => initial
  | depth + 1 => evaluateLayer maxWidth depth circuit (evaluateLayers maxWidth circuit initial depth)

def output (depth maxWidth instanceWidth witnessWidth : Nat) (circuit : CircuitData)
    (instanceValues witnessValues : List Int) : Bool :=
  let final := evaluateLayers maxWidth circuit
    (canonicalInputs maxWidth instanceWidth witnessWidth instanceValues witnessValues) depth
  final[(circuit.outputSources[0]?.getD 0).toNat]?.getD false

def outputSource (circuit : CircuitData) : Int :=
  circuit.outputSources[0]?.getD 0

/-- Implementation-independent validity of a fixed-allocation dynamic circuit. -/
structure Valid (depth maxWidth : Nat) (circuit : CircuitData) : Prop where
  depthPositive : 0 < depth
  maxWidthPositive : 0 < maxWidth
  activeCountLength : circuit.activeGateCounts.length = depth
  gateKindsLength : circuit.gateKinds.length = depth * maxWidth
  leftSourcesLength : circuit.leftSources.length = depth * maxWidth
  rightSourcesLength : circuit.rightSources.length = depth * maxWidth
  outputSourcesLength : circuit.outputSources.length = 1
  activeCountBounds : ∀ layer, layer < depth →
    0 < activeCount circuit layer ∧ activeCount circuit layer ≤ maxWidth
  activeGateKinds : ∀ layer slot, layer < depth → slot < activeCount circuit layer →
    0 ≤ entry circuit.gateKinds maxWidth layer slot ∧
      entry circuit.gateKinds maxWidth layer slot ≤ 5
  activeLeftSources : ∀ layer slot, layer < depth → slot < activeCount circuit layer →
    0 ≤ entry circuit.leftSources maxWidth layer slot ∧
      (entry circuit.leftSources maxWidth layer slot).toNat < maxWidth
  activeRightSources : ∀ layer slot, layer < depth → slot < activeCount circuit layer →
    0 ≤ entry circuit.rightSources maxWidth layer slot ∧
      (entry circuit.rightSources maxWidth layer slot).toNat < maxWidth
  inactivePadding : ∀ layer slot, layer < depth →
    activeCount circuit layer ≤ slot → slot < maxWidth →
      entry circuit.gateKinds maxWidth layer slot = 0 ∧
        entry circuit.leftSources maxWidth layer slot = 0 ∧
        entry circuit.rightSources maxWidth layer slot = 0
  outputSourceValid : 0 ≤ DynamicBoolean.outputSource circuit ∧
    (DynamicBoolean.outputSource circuit).toNat < activeCount circuit (depth - 1)

/-- Exact value-domain contract used by the executable Boolean-family predicate.  Active entries
are bits and the unused fixed-allocation suffix is canonical zero padding. -/
def CanonicalValues (maxWidth activeWidth : Nat) (values : List Int) : Prop :=
  ∀ slot, slot < maxWidth →
    if slot < activeWidth then
      values[slot]?.getD 0 = 0 ∨ values[slot]?.getD 0 = 1
    else
      values[slot]?.getD 0 = 0

/-- A satisfying assignment uses the same canonical integer encoding as the executable predicate
and makes the selected circuit output true. -/
structure Satisfied (depth maxWidth instanceWidth witnessWidth : Nat) (circuit : CircuitData)
    (instanceValues witnessValues : List Int) : Prop where
  instanceCanonical : CanonicalValues maxWidth instanceWidth instanceValues
  witnessCanonical : CanonicalValues maxWidth witnessWidth witnessValues
  outputTrue :
    output depth maxWidth instanceWidth witnessWidth circuit instanceValues witnessValues = true

@[simp] theorem canonicalInputs_length (maxWidth instanceWidth witnessWidth : Nat)
    (instanceValues witnessValues : List Int) :
    (canonicalInputs maxWidth instanceWidth witnessWidth instanceValues witnessValues).length =
      maxWidth := by
  simp [canonicalInputs]

@[simp] theorem evaluateLayer_length (maxWidth layer : Nat) (circuit : CircuitData)
    (previous : List Bool) :
    (evaluateLayer maxWidth layer circuit previous).length = maxWidth := by
  simp [evaluateLayer]

theorem evaluateLayers_length (maxWidth : Nat) (circuit : CircuitData) (initial : List Bool)
    (initialLength : initial.length = maxWidth) (depth : Nat) :
    (evaluateLayers maxWidth circuit initial depth).length = maxWidth := by
  induction depth with
  | zero => exact initialLength
  | succ depth _ => exact evaluateLayer_length maxWidth depth circuit _

end DynamicBoolean

/-! Reusable Boolean BGG layer algebra.

This file deliberately does not mention generated node identifiers or a concrete Diamond shape.
Runtime lists are validated once and then accessed through `Fin`, so active gates never use a
fallback value. -/

abbrev AlgebraMatrix (R : Type) (rows columns : Nat) :=
  _root_.Matrix (Fin rows) (Fin columns) R

noncomputable def algebraScale {R : Type} [Mul R] {rows columns : Nat}
    (scalar : AlgebraMatrix R 1 1) (matrix : AlgebraMatrix R rows columns) :
    AlgebraMatrix R rows columns :=
  fun row column ↦ scalar 0 0 * matrix row column

structure BooleanEncoding (R : Type) (columns : Nat) where
  vector : AlgebraMatrix R 1 columns
  publicKey : AlgebraMatrix R 1 columns
  plaintext : AlgebraMatrix R 1 1
  error : AlgebraMatrix R 1 columns

def BooleanEncoding.Holds {R : Type} [CommRing R] {columns : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (encoding : BooleanEncoding R columns) : Prop :=
  encoding.vector =
    algebraScale secret
      (encoding.publicKey - algebraScale encoding.plaintext gadget) + encoding.error

noncomputable def BooleanEncoding.add {R : Type} [CommRing R] {columns : Nat}
    (left right : BooleanEncoding R columns) : BooleanEncoding R columns where
  vector := left.vector + right.vector
  publicKey := left.publicKey + right.publicKey
  plaintext := left.plaintext + right.plaintext
  error := left.error + right.error

noncomputable def BooleanEncoding.sub {R : Type} [CommRing R] {columns : Nat}
    (left right : BooleanEncoding R columns) : BooleanEncoding R columns where
  vector := left.vector - right.vector
  publicKey := left.publicKey - right.publicKey
  plaintext := left.plaintext - right.plaintext
  error := left.error - right.error

noncomputable def BooleanEncoding.scale {R : Type} [CommRing R] {columns : Nat}
    (factor : AlgebraMatrix R 1 1) (encoding : BooleanEncoding R columns) :
    BooleanEncoding R columns where
  vector := algebraScale factor encoding.vector
  publicKey := algebraScale factor encoding.publicKey
  plaintext := factor * encoding.plaintext
  error := algebraScale factor encoding.error

theorem BooleanEncoding.add_holds {R : Type} [CommRing R] {columns : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (left right : BooleanEncoding R columns)
    (leftHolds : left.Holds secret gadget) (rightHolds : right.Holds secret gadget) :
    (left.add right).Holds secret gadget := by
  change left.vector = _ at leftHolds
  change right.vector = _ at rightHolds
  change left.vector + right.vector = _
  rw [leftHolds, rightHolds]
  simp only [BooleanEncoding.add]
  ext row column
  fin_cases row
  simp [algebraScale]
  ring

theorem BooleanEncoding.sub_holds {R : Type} [CommRing R] {columns : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (left right : BooleanEncoding R columns)
    (leftHolds : left.Holds secret gadget) (rightHolds : right.Holds secret gadget) :
    (left.sub right).Holds secret gadget := by
  change left.vector = _ at leftHolds
  change right.vector = _ at rightHolds
  change left.vector - right.vector = _
  rw [leftHolds, rightHolds]
  simp only [BooleanEncoding.sub]
  ext row column
  fin_cases row
  simp [algebraScale]
  ring

theorem BooleanEncoding.scale_holds {R : Type} [CommRing R] {columns : Nat}
    (factor secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (encoding : BooleanEncoding R columns) (holds : encoding.Holds secret gadget) :
    (encoding.scale factor).Holds secret gadget := by
  rw [BooleanEncoding.Holds] at holds
  rw [BooleanEncoding.Holds, BooleanEncoding.scale, holds]
  ext row column
  fin_cases row
  simp [algebraScale, _root_.Matrix.mul_apply]
  ring

noncomputable def BooleanEncoding.multiply {R : Type} [CommRing R] {columns : Nat}
    (left right : BooleanEncoding R columns)
    (rightDecomposition : AlgebraMatrix R columns columns) : BooleanEncoding R columns where
  vector := left.vector * rightDecomposition + algebraScale left.plaintext right.vector
  publicKey := left.publicKey * rightDecomposition
  plaintext := left.plaintext * right.plaintext
  error := left.error * rightDecomposition + algebraScale left.plaintext right.error

theorem BooleanEncoding.multiply_holds {R : Type} [CommRing R] {columns : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (left right : BooleanEncoding R columns)
    (rightDecomposition : AlgebraMatrix R columns columns)
    (leftHolds : left.Holds secret gadget) (rightHolds : right.Holds secret gadget)
    (decomposes : gadget * rightDecomposition = right.publicKey) :
    (left.multiply right rightDecomposition).Holds secret gadget := by
  rw [BooleanEncoding.Holds] at leftHolds rightHolds
  rw [BooleanEncoding.Holds, BooleanEncoding.multiply, leftHolds, rightHolds]
  ext row column
  fin_cases row
  simp only [_root_.Matrix.add_apply, _root_.Matrix.sub_apply, algebraScale,
    _root_.Matrix.mul_apply]
  have decomposesEntry := congrFun (congrFun decomposes 0) column
  simp only [_root_.Matrix.mul_apply] at decomposesEntry
  have plaintextProduct :
      (∑ index : Fin 1, left.plaintext 0 index * right.plaintext index 0) =
        left.plaintext 0 0 * right.plaintext 0 0 := by
    simp
  rw [plaintextProduct]
  have leftSum :
      (∑ x,
        (secret 0 0 * (left.publicKey 0 x - left.plaintext 0 0 * gadget 0 x) +
          left.error 0 x) * rightDecomposition x column) =
        secret 0 0 * ((∑ x, left.publicKey 0 x * rightDecomposition x column) -
          left.plaintext 0 0 * (∑ x, gadget 0 x * rightDecomposition x column)) +
        ∑ x, left.error 0 x * rightDecomposition x column := by
    rw [mul_sub]
    simp only [Finset.mul_sum]
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro index _
    ring
  rw [show (∑ x,
      (secret 0 0 * (left.publicKey ⟨0, by omega⟩ x - left.plaintext 0 0 *
        gadget ⟨0, by omega⟩ x) + left.error ⟨0, by omega⟩ x) *
          rightDecomposition x column) =
      secret 0 0 * ((∑ x, left.publicKey 0 x * rightDecomposition x column) -
        left.plaintext 0 0 * (∑ x, gadget 0 x * rightDecomposition x column)) +
      ∑ x, left.error 0 x * rightDecomposition x column by simpa using leftSum]
  rw [decomposesEntry]
  have finOne (index : Fin 1) : index = 0 := Subsingleton.elim _ _
  simp_rw [finOne]
  ring

/-! `rightDecomposition` is intentionally not part of an encoding or ciphertext artifact.  For
the concrete sampler semantics, `samePublicKeyDecomposition` proves that two decompositions
obtained from the same public key, sampler parameters, base, and digit count normalize to the same
matrix.  The algebra below therefore only needs the resulting decomposition equation. -/

noncomputable def BooleanEncoding.applyGate {R : Type} [CommRing R] {columns : Nat}
    (gate : BooleanGate) (one left right product : BooleanEncoding R columns) :
    BooleanEncoding R columns :=
  match gate with
  | .constantFalse => one.sub one
  | .constantTrue => one
  | .copy => left
  | .not => one.sub left
  | .and => product
  | .xor => (left.add right).sub (product.scale 2)

/-- The public-key component of one Boolean gate.  This is the common recurrence followed by the
encryption public-key evaluator and by the public-key component of the decryption evaluator. -/
noncomputable def applyBooleanPublicKeyGate {R : Type} [CommRing R] {columns : Nat}
    (gate : BooleanGate) (one left right product : AlgebraMatrix R 1 columns) :
    AlgebraMatrix R 1 columns :=
  match gate with
  | .constantFalse => one - one
  | .constantTrue => one
  | .copy => left
  | .not => one - left
  | .and => product
  | .xor => left + right - algebraScale 2 product

@[simp] theorem BooleanEncoding.applyGate_publicKey {R : Type} [CommRing R] {columns : Nat}
    (gate : BooleanGate) (one left right product : BooleanEncoding R columns) :
    (applyGate gate one left right product).publicKey =
      applyBooleanPublicKeyGate gate one.publicKey left.publicKey right.publicKey
        product.publicKey := by
  cases gate <;> rfl

theorem BooleanEncoding.applyGate_holds {R : Type} [CommRing R] {columns : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (gate : BooleanGate) (one left right product : BooleanEncoding R columns)
    (oneHolds : one.Holds secret gadget) (leftHolds : left.Holds secret gadget)
    (rightHolds : right.Holds secret gadget) (productHolds : product.Holds secret gadget) :
    (applyGate gate one left right product).Holds secret gadget := by
  cases gate
  · exact sub_holds secret gadget one one oneHolds oneHolds
  · exact oneHolds
  · exact leftHolds
  · exact sub_holds secret gadget one left oneHolds leftHolds
  · exact productHolds
  · exact sub_holds secret gadget (left.add right) (product.scale 2)
      (add_holds secret gadget left right leftHolds rightHolds)
      (scale_holds 2 secret gadget product productHolds)

structure BooleanLayerProgram where
  kinds : List BooleanGate
  leftPredecessors : List Nat
  rightPredecessors : List Nat
  deriving Repr

def BooleanLayerProgram.activeWidth (layer : BooleanLayerProgram) : Nat := layer.kinds.length

def BooleanLayerProgram.Valid (layer : BooleanLayerProgram) (precedingWidth maxWidth : Nat) :
    Prop :=
  layer.leftPredecessors.length = layer.activeWidth ∧
  layer.rightPredecessors.length = layer.activeWidth ∧
  layer.activeWidth ≤ maxWidth ∧
  (∀ predecessor ∈ layer.leftPredecessors, predecessor < precedingWidth) ∧
  ∀ predecessor ∈ layer.rightPredecessors, predecessor < precedingWidth

def BooleanLayerProgram.leftIndex (layer : BooleanLayerProgram) {precedingWidth maxWidth : Nat}
    (valid : layer.Valid precedingWidth maxWidth) (i : Fin layer.activeWidth) :
    Fin precedingWidth :=
  let position : Fin layer.leftPredecessors.length := Fin.cast valid.1.symm i
  ⟨layer.leftPredecessors.get position,
    valid.2.2.2.1 _ (List.get_mem layer.leftPredecessors position)⟩

def BooleanLayerProgram.rightIndex (layer : BooleanLayerProgram) {precedingWidth maxWidth : Nat}
    (valid : layer.Valid precedingWidth maxWidth) (i : Fin layer.activeWidth) :
    Fin precedingWidth :=
  let position : Fin layer.rightPredecessors.length := Fin.cast valid.2.1.symm i
  ⟨layer.rightPredecessors.get position,
    valid.2.2.2.2 _ (List.get_mem layer.rightPredecessors position)⟩

/-- One layer of the public-key recurrence, separated from encoding vectors and errors. -/
noncomputable def evaluateBooleanPublicKeyLayer {R : Type} [CommRing R]
    {columns maxWidth : Nat} (one : AlgebraMatrix R 1 columns)
    (previous : List (AlgebraMatrix R 1 columns)) (layer : BooleanLayerProgram)
    (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns) :
    List (AlgebraMatrix R 1 columns) :=
  List.ofFn fun i : Fin maxWidth ↦
    if active : i < layer.activeWidth then
      let gate : Fin layer.activeWidth := ⟨i, active⟩
      let left := previous.get (layer.leftIndex valid gate)
      let right := previous.get (layer.rightIndex valid gate)
      applyBooleanPublicKeyGate (layer.kinds.get gate) one left right
        (left * rightDecompositions gate)
    else one - one

noncomputable def evaluateBooleanLayer {R : Type} [CommRing R] {columns maxWidth : Nat}
    (one : BooleanEncoding R columns) (previous : List (BooleanEncoding R columns))
    (layer : BooleanLayerProgram) (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns) :
    List (BooleanEncoding R columns) :=
  List.ofFn fun i : Fin maxWidth ↦
    if active : i < layer.activeWidth then
      let gate : Fin layer.activeWidth := ⟨i, active⟩
      let left := previous.get (layer.leftIndex valid gate)
      let right := previous.get (layer.rightIndex valid gate)
      let product := left.multiply right (rightDecompositions gate)
      BooleanEncoding.applyGate (layer.kinds.get gate) one left right product
  else one.sub one

theorem evaluateBooleanLayer_publicKeys {R : Type} [CommRing R]
    {columns maxWidth : Nat} (one : BooleanEncoding R columns)
    (previous : List (BooleanEncoding R columns)) (layer : BooleanLayerProgram)
    (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns) :
    (evaluateBooleanLayer one previous layer valid rightDecompositions).map
        BooleanEncoding.publicKey =
      evaluateBooleanPublicKeyLayer one.publicKey (previous.map BooleanEncoding.publicKey) layer
        (by simpa using valid) rightDecompositions := by
  apply List.ext_get
  · simp [evaluateBooleanLayer, evaluateBooleanPublicKeyLayer]
  · intro i leftBound rightBound
    by_cases active : i < layer.activeWidth
    · simp [evaluateBooleanLayer, evaluateBooleanPublicKeyLayer, active,
        BooleanEncoding.multiply, BooleanLayerProgram.leftIndex,
        BooleanLayerProgram.rightIndex]
    · simp [evaluateBooleanLayer, evaluateBooleanPublicKeyLayer, active,
        BooleanEncoding.sub]

theorem evaluateBooleanLayer_length {R : Type} [CommRing R] {columns maxWidth : Nat}
    (one : BooleanEncoding R columns) (previous : List (BooleanEncoding R columns))
    (layer : BooleanLayerProgram) (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns) :
    (evaluateBooleanLayer one previous layer valid rightDecompositions).length =
      maxWidth := by
  simp [evaluateBooleanLayer]

theorem evaluateBooleanLayer_holds {R : Type} [CommRing R] {columns maxWidth : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (one : BooleanEncoding R columns) (previous : List (BooleanEncoding R columns))
    (layer : BooleanLayerProgram) (valid : layer.Valid previous.length maxWidth)
    (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns)
    (oneHolds : one.Holds secret gadget)
    (previousHolds : ∀ i : Fin previous.length, (previous.get i).Holds secret gadget)
    (decomposes : ∀ i : Fin layer.activeWidth,
      gadget * rightDecompositions i =
        (previous.get (layer.rightIndex valid i)).publicKey) :
    ∀ i : Fin (evaluateBooleanLayer one previous layer valid rightDecompositions).length,
      ((evaluateBooleanLayer one previous layer valid rightDecompositions).get i).Holds
        secret gadget := by
  intro i
  have indexLt : i.val < maxWidth := by
    simpa [evaluateBooleanLayer_length one previous layer valid rightDecompositions] using i.isLt
  by_cases active : i.val < layer.activeWidth
  · let gate : Fin layer.activeWidth := ⟨i, active⟩
    simpa [evaluateBooleanLayer, active, gate] using
      BooleanEncoding.applyGate_holds secret gadget (layer.kinds.get gate) one
        (previous.get (layer.leftIndex valid gate)) (previous.get (layer.rightIndex valid gate))
        ((previous.get (layer.leftIndex valid gate)).multiply
          (previous.get (layer.rightIndex valid gate)) (rightDecompositions gate)) oneHolds
        (previousHolds (layer.leftIndex valid gate))
        (previousHolds (layer.rightIndex valid gate))
        (BooleanEncoding.multiply_holds secret gadget
          (previous.get (layer.leftIndex valid gate))
          (previous.get (layer.rightIndex valid gate)) (rightDecompositions gate)
          (previousHolds (layer.leftIndex valid gate))
          (previousHolds (layer.rightIndex valid gate)) (decomposes gate))
  · simpa [evaluateBooleanLayer, active] using
      BooleanEncoding.sub_holds secret gadget one one oneHolds oneHolds

inductive BooleanLayersEvaluation {R : Type} [CommRing R] (columns maxWidth : Nat)
    (one : BooleanEncoding R columns) :
    List BooleanLayerProgram → List (BooleanEncoding R columns) →
      List (BooleanEncoding R columns) → Prop where
  | nil (state) : BooleanLayersEvaluation columns maxWidth one [] state state
  | cons (layer layers initial middle final)
      (valid : layer.Valid initial.length maxWidth)
      (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns)
      (step : middle = evaluateBooleanLayer one initial layer valid rightDecompositions)
      (rest : BooleanLayersEvaluation columns maxWidth one layers middle final) :
      BooleanLayersEvaluation columns maxWidth one (layer :: layers) initial final

/-- Public-key projection of the same arbitrary-depth Boolean recurrence.  A concrete certificate
connects its encryption and decryption executions to this relation, then proves corresponding
decompositions equal with `gadgetDecomposeUnique`. -/
inductive BooleanPublicKeyLayersEvaluation {R : Type} [CommRing R] (columns maxWidth : Nat)
    (one : AlgebraMatrix R 1 columns) :
    List BooleanLayerProgram → List (AlgebraMatrix R 1 columns) →
      List (AlgebraMatrix R 1 columns) → Prop where
  | nil (state) : BooleanPublicKeyLayersEvaluation columns maxWidth one [] state state
  | cons (layer layers initial middle final)
      (valid : layer.Valid initial.length maxWidth)
      (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns)
      (step : middle =
        evaluateBooleanPublicKeyLayer one initial layer valid rightDecompositions)
      (rest : BooleanPublicKeyLayersEvaluation columns maxWidth one layers middle final) :
      BooleanPublicKeyLayersEvaluation columns maxWidth one (layer :: layers) initial final

theorem BooleanLayersEvaluation.publicKeys {R : Type} [CommRing R]
    {columns maxWidth : Nat} (one : BooleanEncoding R columns)
    (layers : List BooleanLayerProgram) (initial final : List (BooleanEncoding R columns))
    (evaluation : BooleanLayersEvaluation columns maxWidth one layers initial final) :
    BooleanPublicKeyLayersEvaluation columns maxWidth one.publicKey layers
      (initial.map BooleanEncoding.publicKey) (final.map BooleanEncoding.publicKey) := by
  induction evaluation with
  | nil => exact .nil _
  | cons layer layers initial middle final valid rightDecompositions step rest induction =>
      subst middle
      exact .cons layer layers _ _ _ (by simpa using valid) rightDecompositions
        (evaluateBooleanLayer_publicKeys one initial layer valid rightDecompositions)
        induction

theorem BooleanLayersEvaluation.holds {R : Type} [CommRing R] {columns maxWidth : Nat}
    (secret : AlgebraMatrix R 1 1) (gadget : AlgebraMatrix R 1 columns)
    (one : BooleanEncoding R columns) (layers : List BooleanLayerProgram)
    (initial final : List (BooleanEncoding R columns))
    (evaluation : BooleanLayersEvaluation columns maxWidth one layers initial final)
    (oneHolds : one.Holds secret gadget)
    (initialHolds : ∀ i : Fin initial.length, (initial.get i).Holds secret gadget)
    (decomposes : ∀ (layer : BooleanLayerProgram)
      (state : List (BooleanEncoding R columns))
      (valid : layer.Valid state.length maxWidth)
      (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns),
      ∀ i : Fin layer.activeWidth,
        gadget * rightDecompositions i = (state.get (layer.rightIndex valid i)).publicKey) :
    ∀ i : Fin final.length, (final.get i).Holds secret gadget := by
  induction evaluation with
  | nil => exact initialHolds
  | cons layer layers initial middle final valid rightDecompositions step rest induction =>
      subst middle
      apply induction
      exact evaluateBooleanLayer_holds secret gadget one initial layer valid
        rightDecompositions oneHolds initialHolds
        (decomposes layer initial valid rightDecompositions)

theorem BooleanLayersEvaluation.output_holds {R : Type} [CommRing R]
    {columns maxWidth : Nat} (secret : AlgebraMatrix R 1 1)
    (gadget : AlgebraMatrix R 1 columns) (one : BooleanEncoding R columns)
    (layers : List BooleanLayerProgram) (initial final : List (BooleanEncoding R columns))
    (evaluation : BooleanLayersEvaluation columns maxWidth one layers initial final)
    (oneHolds : one.Holds secret gadget)
    (initialHolds : ∀ i : Fin initial.length, (initial.get i).Holds secret gadget)
    (decomposes : ∀ (layer : BooleanLayerProgram)
      (state : List (BooleanEncoding R columns))
      (valid : layer.Valid state.length maxWidth)
      (rightDecompositions : Fin layer.activeWidth → AlgebraMatrix R columns columns),
      ∀ i : Fin layer.activeWidth,
        gadget * rightDecompositions i = (state.get (layer.rightIndex valid i)).publicKey)
    (output : Fin final.length) :
    (final.get output).Holds secret gadget :=
  BooleanLayersEvaluation.holds secret gadget one layers initial final evaluation oneHolds
    initialHolds decomposes output

/-- One dynamic gate preserves the conservative worst-case layer bound. -/
theorem dynamicGateNoise_norm_le (q ringDimension publicColumns digitBound oneBound inputBound :
    Nat) [NeZero q] (gate : BooleanGate)
    (oneError leftError rightError leftPlaintext rightDecomposed : Mxx.Matrix)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (leftShape : Mxx.Toolkit.MatrixShape leftError q ringDimension 1 publicColumns)
    (rightShape : Mxx.Toolkit.MatrixShape rightError q ringDimension 1 publicColumns)
    (plaintextShape : Mxx.Toolkit.MatrixShape leftPlaintext q ringDimension 1 1)
    (decomposedShape :
      Mxx.Toolkit.MatrixShape rightDecomposed q ringDimension publicColumns publicColumns)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound)
    (leftNorm : Mxx.maxCenteredCoefficientNorm leftError ≤ inputBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm rightError ≤ inputBound)
    (plaintextNorm : Mxx.maxCenteredCoefficientNorm leftPlaintext ≤ 1)
    (decomposedNorm : Mxx.maxCenteredCoefficientNorm rightDecomposed ≤ digitBound) :
    Mxx.maxCenteredCoefficientNorm
        (booleanGateNoiseMatrix gate oneError leftError rightError leftPlaintext
          rightDecomposed) ≤
      gateStep ringDimension publicColumns digitBound oneBound inputBound := by
  apply le_trans
    (booleanGateNoiseMatrix_norm_le q ringDimension publicColumns digitBound oneBound
      inputBound inputBound gate oneError leftError rightError leftPlaintext rightDecomposed
      oneShape leftShape rightShape plaintextShape decomposedShape oneNorm leftNorm rightNorm
      plaintextNorm decomposedNorm)
  exact gateNoise_le_gateStep gate ringDimension publicColumns digitBound oneBound inputBound
    inputBound inputBound (Nat.le_refl _) (Nat.le_refl _)

def EveryNoiseBounded (matrices : List Mxx.Matrix) (bound : Nat) : Prop :=
  ∀ i : Fin matrices.length, Mxx.maxCenteredCoefficientNorm (matrices.get i) ≤ bound

theorem dynamicBooleanLayer_noise_norm_le (q ringDimension publicColumns digitBound oneBound
    inputBound maxWidth : Nat) [NeZero q] (layer : BooleanLayerProgram)
    (previousErrors : List Mxx.Matrix) (valid : layer.Valid previousErrors.length maxWidth)
    (oneError : Mxx.Matrix)
    (leftPlaintexts rightDecompositions : Fin layer.activeWidth → Mxx.Matrix)
    (oneShape : Mxx.Toolkit.MatrixShape oneError q ringDimension 1 publicColumns)
    (previousShape : ∀ i : Fin previousErrors.length,
      Mxx.Toolkit.MatrixShape (previousErrors.get i) q ringDimension 1 publicColumns)
    (plaintextShape : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixShape (leftPlaintexts i) q ringDimension 1 1)
    (decompositionShape : ∀ i : Fin layer.activeWidth,
      Mxx.Toolkit.MatrixShape (rightDecompositions i) q ringDimension publicColumns publicColumns)
    (oneNorm : Mxx.maxCenteredCoefficientNorm oneError ≤ oneBound)
    (previousNorm : ∀ i : Fin previousErrors.length,
      Mxx.maxCenteredCoefficientNorm (previousErrors.get i) ≤ inputBound)
    (plaintextNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (leftPlaintexts i) ≤ 1)
    (decompositionNorm : ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm (rightDecompositions i) ≤ digitBound) :
    ∀ i : Fin layer.activeWidth,
      Mxx.maxCenteredCoefficientNorm
        (booleanGateNoiseMatrix (layer.kinds.get i) oneError
        (previousErrors.get (layer.leftIndex valid i))
        (previousErrors.get (layer.rightIndex valid i)) (leftPlaintexts i)
        (rightDecompositions i)) ≤
      gateStep ringDimension publicColumns digitBound oneBound inputBound := by
  intro i
  exact dynamicGateNoise_norm_le q ringDimension publicColumns digitBound oneBound inputBound
      (layer.kinds.get i) oneError (previousErrors.get (layer.leftIndex valid i))
      (previousErrors.get (layer.rightIndex valid i)) (leftPlaintexts i)
      (rightDecompositions i) oneShape (previousShape (layer.leftIndex valid i))
      (previousShape (layer.rightIndex valid i)) (plaintextShape i) (decompositionShape i)
      oneNorm (previousNorm (layer.leftIndex valid i))
      (previousNorm (layer.rightIndex valid i)) (plaintextNorm i) (decompositionNorm i)

/-- Induction principle used by a concrete runtime layer evaluator.  Each step discharges its
gate-local obligations with `dynamicGateNoise_norm_le`; this theorem supplies the arbitrary-depth
recurrence and does not depend on generated graph node identifiers. -/
theorem dynamicBooleanLayers_noise_induction
    (ringDimension publicColumns digitBound oneBound : Nat)
    (layers : List BooleanLayerProgram) (states : Nat → List Mxx.Matrix)
    (initialBound : Nat) (initial : EveryNoiseBounded (states 0) initialBound)
    (step : ∀ depth : Nat, depth < layers.length →
      EveryNoiseBounded (states depth)
        ((List.range depth).foldl
          (fun bound _ ↦ gateStep ringDimension publicColumns digitBound oneBound bound)
          initialBound) →
      EveryNoiseBounded (states (depth + 1))
        (gateStep ringDimension publicColumns digitBound oneBound
          ((List.range depth).foldl
            (fun bound _ ↦ gateStep ringDimension publicColumns digitBound oneBound bound)
            initialBound))) :
    EveryNoiseBounded (states layers.length)
      ((List.range layers.length).foldl
        (fun bound _ ↦ gateStep ringDimension publicColumns digitBound oneBound bound)
        initialBound) := by
  have loop : ∀ depth : Nat, depth ≤ layers.length →
      EveryNoiseBounded (states depth)
        ((List.range depth).foldl
          (fun bound _ ↦ gateStep ringDimension publicColumns digitBound oneBound bound)
          initialBound) := by
    intro depth depthLe
    induction depth with
    | zero => simpa using initial
    | succ depth induction =>
        simp only [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
        exact step depth (Nat.lt_of_succ_le depthLe) (induction (Nat.le_of_lt depthLe))
  exact loop layers.length (Nat.le_refl _)

theorem dynamicBooleanLayers_noise_le_circuitBound
    (ringDimension publicColumns depth digitBound oneBound : Nat)
    (states : Nat → List Mxx.Matrix)
    (initial : EveryNoiseBounded (states 0) (2 * oneBound))
    (step : ∀ layer : Nat, layer < depth →
      EveryNoiseBounded (states layer)
        ((List.range layer).foldl
          (fun bound _ ↦ gateStep ringDimension publicColumns digitBound oneBound bound)
          (2 * oneBound)) →
      EveryNoiseBounded (states (layer + 1))
        (gateStep ringDimension publicColumns digitBound oneBound
          ((List.range layer).foldl
            (fun bound _ ↦ gateStep ringDimension publicColumns digitBound oneBound bound)
            (2 * oneBound)))) :
    EveryNoiseBounded (states depth)
      (circuitBound ringDimension publicColumns depth digitBound oneBound) := by
  simpa [circuitBound] using
    dynamicBooleanLayers_noise_induction ringDimension publicColumns digitBound oneBound
      (List.replicate depth { kinds := [], leftPredecessors := [], rightPredecessors := [] })
      states (2 * oneBound) initial (by
        intro layer layerLt
        simpa using step layer (by simpa using layerLt))

end MxxWe
