import Mxx.Certificate.OperationalNoise.BoundReplay

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.EventReplay

/-- A finite collection of signed coefficients. The enclosing theorem supplies its shape. -/
abbrev Coefficients (count : Nat) := Fin count → Int

def addCoefficients {count : Nat} (left right : Coefficients count) : Coefficients count :=
  fun index => left index + right index

def subtractCoefficients {count : Nat} (left right : Coefficients count) : Coefficients count :=
  fun index => left index - right index

/-- One coefficient class bounds every coefficient selected from a finite value. -/
def CoeffClass.Bounds {count : Nat} (bound : CoeffClass)
    (coefficients : Coefficients count) : Prop :=
  ∀ index, bound.Interprets (coefficients index).natAbs

theorem CoeffClass.Interprets.mono {bound : CoeffClass} {actual upper : Nat}
    (actualLe : actual ≤ upper) (upperSound : bound.Interprets upper) :
    bound.Interprets actual := by
  cases bound with
  | exactZero => simp_all [CoeffClass.Interprets]
  | finite maximum =>
      exact Nat.le_trans actualLe upperSound
  | large => trivial

@[simp]
theorem addCoefficients_apply {count : Nat} (left right : Coefficients count)
    (index : Fin count) :
    addCoefficients left right index = left index + right index := by
  rfl

@[simp]
theorem subtractCoefficients_apply {count : Nat} (left right : Coefficients count)
    (index : Fin count) :
    subtractCoefficients left right index = left index - right index := by
  rfl

/-- Scalar and matrix addition use the same exact coefficient equation and Rust `Sum` bound. -/
theorem addCoefficients_bound {count : Nat} {left right : Coefficients count}
    {leftClass rightClass : CoeffClass} (leftBound : leftClass.Bounds left)
    (rightBound : rightClass.Bounds right) :
    (addKnown leftClass rightClass).Bounds (addCoefficients left right) := by
  intro index
  apply CoeffClass.Interprets.mono (Int.natAbs_add_le (left index) (right index))
  exact addKnown_sound (leftBound index) (rightBound index)

/-- Matrix subtraction has the exact signed equation and the same magnitude-sum bound. -/
theorem subtractCoefficients_bound {count : Nat} {left right : Coefficients count}
    {leftClass rightClass : CoeffClass} (leftBound : leftClass.Bounds left)
    (rightBound : rightClass.Bounds right) :
    (addKnown leftClass rightClass).Bounds (subtractCoefficients left right) := by
  intro index
  apply CoeffClass.Interprets.mono (Int.natAbs_sub_le (left index) (right index))
  exact addKnown_sound (leftBound index) (rightBound index)

/-- The signed contribution of the operand coefficient pairs selected for one output. -/
def productCoefficient : List (Int × Int) → Int
  | [] => 0
  | (left, right) :: terms => left * right + productCoefficient terms

def ProductTermsBounded (leftMaximum rightMaximum : Nat) : List (Int × Int) → Prop
  | [] => True
  | (left, right) :: terms =>
      left.natAbs ≤ leftMaximum ∧ right.natAbs ≤ rightMaximum ∧
        ProductTermsBounded leftMaximum rightMaximum terms

/-- A singleton coefficient product is exactly the signed contribution already used by the
    four-role symbolic product replay. -/
theorem productCoefficient_singleton_operator (left right : ExactTerm)
    (leftScalar rightScalar : Bool) :
    productCoefficient [(left.coefficient, right.coefficient)] =
      (operatorProductContribution left right leftScalar rightScalar).coefficient := by
  simp [productCoefficient, operatorProductContribution_coefficient]

theorem productCoefficient_natAbs_le (terms : List (Int × Int))
    {leftMaximum rightMaximum : Nat}
    (termsBounded : ProductTermsBounded leftMaximum rightMaximum terms) :
    (productCoefficient terms).natAbs ≤ terms.length * leftMaximum * rightMaximum := by
  induction terms with
  | nil => simp [productCoefficient]
  | cons term terms ih =>
      rcases term with ⟨left, right⟩
      simp only [ProductTermsBounded] at termsBounded
      rcases termsBounded with ⟨leftBound, rightBound, tailBound⟩
      have productBound : (left * right).natAbs ≤ leftMaximum * rightMaximum := by
        rw [Int.natAbs_mul]
        exact Nat.mul_le_mul leftBound rightBound
      have tailResult := ih tailBound
      calc
        (productCoefficient ((left, right) :: terms)).natAbs =
            (left * right + productCoefficient terms).natAbs := rfl
        _ ≤ (left * right).natAbs + (productCoefficient terms).natAbs :=
          Int.natAbs_add_le _ _
        _ ≤ leftMaximum * rightMaximum + terms.length * leftMaximum * rightMaximum :=
          Nat.add_le_add productBound tailResult
        _ = ((left, right) :: terms).length * leftMaximum * rightMaximum := by
          simp [Nat.succ_mul, Nat.add_mul, Nat.add_comm]

/-- Replay one matrix-product coefficient using the exact Rust product factor. -/
theorem productCoefficient_withFacts_bound
    {leftRows leftColumns rightRows rightColumns ringDimension factor : Nat}
    {facts : ProductFacts} {leftClass rightClass : CoeffClass}
    {leftMaximum rightMaximum : Nat} {terms : List (Int × Int)}
    (factorExact :
      productFactor leftRows leftColumns rightRows rightColumns ringDimension facts = some factor)
    (termCount : terms.length ≤ factor)
    (termsBounded : ProductTermsBounded leftMaximum rightMaximum terms)
    (leftSound : leftClass.Interprets leftMaximum)
    (rightSound : rightClass.Interprets rightMaximum) :
    productWithFacts leftRows leftColumns rightRows rightColumns ringDimension facts
          leftClass rightClass = some (productWithFactor factor leftClass rightClass) ∧
      (productWithFactor factor leftClass rightClass).Interprets
        (productCoefficient terms).natAbs := by
  have productSound := productWithFacts_sound factorExact leftSound rightSound
  exact ⟨productSound.1,
    CoeffClass.Interprets.mono
      (Nat.le_trans (productCoefficient_natAbs_le terms termsBounded)
        (Nat.mul_le_mul_right rightMaximum (Nat.mul_le_mul_right leftMaximum termCount)))
      productSound.2⟩

/-- Replay one tensor coefficient using the exact G2a tensor factor. -/
theorem tensorCoefficient_bound {ringDimension leftMaximum rightMaximum : Nat}
    {facts : ProductFacts} {leftClass rightClass : CoeffClass}
    {terms : List (Int × Int)}
    (termCount : terms.length ≤ tensorFactor ringDimension facts)
    (termsBounded : ProductTermsBounded leftMaximum rightMaximum terms)
    (leftSound : leftClass.Interprets leftMaximum)
    (rightSound : rightClass.Interprets rightMaximum) :
    (tensorWithFacts ringDimension facts leftClass rightClass).Interprets
      (productCoefficient terms).natAbs := by
  apply CoeffClass.Interprets.mono
    (Nat.le_trans (productCoefficient_natAbs_le terms termsBounded)
      (Nat.mul_le_mul_right rightMaximum
        (Nat.mul_le_mul_right leftMaximum termCount)))
  exact tensorWithFacts_sound leftSound rightSound

/-- Scalar multiplication and matrix scaling use a single coefficient product. -/
theorem scaleCoefficient_bound {value scale : Int} {valueClass scaleClass : CoeffClass}
    (valueBound : valueClass.Interprets value.natAbs)
    (scaleBound : scaleClass.Interprets scale.natAbs) :
    (scaleValue valueClass scaleClass).Interprets (value * scale).natAbs := by
  rw [Int.natAbs_mul]
  exact scaleValue_sound valueBound scaleBound

/-- An identity-routing operator may copy a selected input coefficient or insert exact zero. -/
def RoutesCoefficients {inputCount outputCount : Nat}
    (input : Coefficients inputCount) (output : Coefficients outputCount) : Prop :=
  ∀ outputIndex, output outputIndex = 0 ∨
    ∃ inputIndex, output outputIndex = input inputIndex

/-- Slice, indexed slice, constant lift, and coefficient extraction share this local rule. -/
theorem routesCoefficients_bound {inputCount outputCount : Nat}
    {input : Coefficients inputCount} {output : Coefficients outputCount}
    {bound : CoeffClass} (inputBound : bound.Bounds input)
    (routes : RoutesCoefficients input output) : bound.Bounds output := by
  intro outputIndex
  rcases routes outputIndex with outputZero | ⟨inputIndex, outputSelected⟩
  · rw [outputZero]
    cases bound <;> simp [CoeffClass.Interprets]
  · rw [outputSelected]
    exact inputBound inputIndex

/-- The exact validity evidence for `X ↦ X^index` on the negacyclic coefficient basis. -/
structure RingAutomorphismIndexValid (ringDimension index : Nat) where
  ringPositive : 0 < ringDimension
  indexPositive : 0 < index
  indexBelowTwiceRing : index < 2 * ringDimension
  indexOdd : index % 2 = 1
  inverse : Fin ringDimension → Fin ringDimension
  inverseTarget : ∀ source,
    inverse (⟨(source.val * index) % ringDimension,
      Nat.mod_lt _ ringPositive⟩ : Fin ringDimension) = source
  targetInverse : ∀ target,
    (⟨((inverse target).val * index) % ringDimension,
      Nat.mod_lt _ ringPositive⟩ : Fin ringDimension) = target

/-- Constructs automorphism evidence from a checked modular inverse. Certificate generators emit
    only the two closed inverse equations; the permutation laws are proved here for every index. -/
def ringAutomorphismIndexValidOfInverse (ringDimension index inverseIndex : Nat)
    (ringPositive : 0 < ringDimension) (indexPositive : 0 < index)
    (indexBelowTwiceRing : index < 2 * ringDimension) (indexOdd : index % 2 = 1)
    (rightInverse : (index * inverseIndex) % ringDimension = 1)
    (leftInverse : (inverseIndex * index) % ringDimension = 1) :
    RingAutomorphismIndexValid ringDimension index where
  ringPositive := ringPositive
  indexPositive := indexPositive
  indexBelowTwiceRing := indexBelowTwiceRing
  indexOdd := indexOdd
  inverse := fun target =>
    ⟨(target.val * inverseIndex) % ringDimension, Nat.mod_lt _ ringPositive⟩
  inverseTarget := by
    intro source
    apply Fin.ext
    calc
      (((source.val * index) % ringDimension) * inverseIndex) % ringDimension =
          (source.val * (index * inverseIndex)) % ringDimension := by
            rw [Nat.mod_mul_mod, Nat.mul_assoc]
      _ = (source.val * ((index * inverseIndex) % ringDimension)) % ringDimension := by
            rw [Nat.mul_mod_mod]
      _ = source.val := by
            rw [rightInverse, Nat.mul_one, Nat.mod_eq_of_lt source.isLt]
  targetInverse := by
    intro target
    apply Fin.ext
    calc
      (((target.val * inverseIndex) % ringDimension) * index) % ringDimension =
          (target.val * (inverseIndex * index)) % ringDimension := by
            rw [Nat.mod_mul_mod, Nat.mul_assoc]
      _ = (target.val * ((inverseIndex * index) % ringDimension)) % ringDimension := by
            rw [Nat.mul_mod_mod]
      _ = target.val := by
            rw [leftInverse, Nat.mul_one, Nat.mod_eq_of_lt target.isLt]

def ringAutomorphismTarget {ringDimension : Nat} (index : Nat)
    (ringPositive : 0 < ringDimension) (source : Fin ringDimension) : Fin ringDimension :=
  ⟨(source.val * index) % ringDimension, Nat.mod_lt _ ringPositive⟩

def ringAutomorphismSignedValue {ringDimension : Nat} (index : Nat)
    (input : Coefficients ringDimension) (source : Fin ringDimension) : Int :=
  if (source.val * index) % (2 * ringDimension) < ringDimension then input source
  else -input source

/-- Exact signed coefficient permutation implemented by the raw ring automorphism. -/
def ringAutomorphismCoefficients {ringDimension index : Nat}
    (valid : RingAutomorphismIndexValid ringDimension index)
    (input : Coefficients ringDimension) : Coefficients ringDimension :=
  fun target =>
    let source := valid.inverse target
    ringAutomorphismSignedValue index input source

theorem ringAutomorphismCoefficients_target {ringDimension index : Nat}
    (valid : RingAutomorphismIndexValid ringDimension index)
    (input : Coefficients ringDimension) (source : Fin ringDimension) :
    ringAutomorphismCoefficients valid input
        (ringAutomorphismTarget index valid.ringPositive source) =
      ringAutomorphismSignedValue index input source := by
  unfold ringAutomorphismCoefficients
  rw [show valid.inverse (ringAutomorphismTarget index valid.ringPositive source) = source by
    exact valid.inverseTarget source]

/-- A valid raw automorphism preserves and reflects the maximum coefficient bound exactly. -/
theorem ringAutomorphismCoefficients_bounds_iff {ringDimension index : Nat}
    (valid : RingAutomorphismIndexValid ringDimension index)
    (input : Coefficients ringDimension) (bound : CoeffClass) :
    bound.Bounds (ringAutomorphismCoefficients valid input) ↔ bound.Bounds input := by
  constructor
  · intro outputBound source
    have selected := outputBound (ringAutomorphismTarget index valid.ringPositive source)
    rw [ringAutomorphismCoefficients_target valid input source] at selected
    by_cases positive : (source.val * index) % (2 * ringDimension) < ringDimension
    · simpa [ringAutomorphismSignedValue, positive] using selected
    · simpa [ringAutomorphismSignedValue, positive] using selected
  · intro inputBound target
    let source := valid.inverse target
    have selected := inputBound source
    by_cases positive : (source.val * index) % (2 * ringDimension) < ringDimension
    · simpa [ringAutomorphismCoefficients, source, ringAutomorphismSignedValue, positive]
        using selected
    · simpa [ringAutomorphismCoefficients, source, ringAutomorphismSignedValue, positive]
        using selected

/-- `magnitude` is the attained maximum absolute coefficient, not merely an upper bound. -/
def CoefficientsHaveMagnitude {count : Nat} (coefficients : Coefficients count)
    (magnitude : Nat) : Prop :=
  (∀ coefficient, (coefficients coefficient).natAbs ≤ magnitude) ∧
    ∃ coefficient, (coefficients coefficient).natAbs = magnitude

/-- Semantic replay evidence for one concrete raw automorphism operation. The output coefficients
    are tied to the signed permutation, so an unrelated value cannot justify the transfer. -/
structure RingAutomorphismReplay {ringDimension index : Nat}
    (valid : RingAutomorphismIndexValid ringDimension index)
    (inputMagnitude outputMagnitude : Nat) where
  input : Coefficients ringDimension
  output : Coefficients ringDimension
  outputEquation : output = ringAutomorphismCoefficients valid input
  inputHasMagnitude : CoefficientsHaveMagnitude input inputMagnitude
  outputHasMagnitude : CoefficientsHaveMagnitude output outputMagnitude

theorem RingAutomorphismReplay.bound {ringDimension index : Nat}
    {valid : RingAutomorphismIndexValid ringDimension index}
    {inputMagnitude outputMagnitude : Nat} {bound : CoeffClass}
    (replay : RingAutomorphismReplay valid inputMagnitude outputMagnitude)
    (inputBound : bound.Interprets inputMagnitude) :
    bound.Interprets outputMagnitude := by
  have coefficientsBound : bound.Bounds replay.input := by
    intro coefficient
    exact CoeffClass.Interprets.mono (replay.inputHasMagnitude.1 coefficient) inputBound
  have outputBound : bound.Bounds replay.output := by
    rw [replay.outputEquation]
    exact (ringAutomorphismCoefficients_bounds_iff valid replay.input bound).2 coefficientsBound
  rcases replay.outputHasMagnitude.2 with ⟨coefficient, exactMagnitude⟩
  rw [← exactMagnitude]
  exact outputBound coefficient

end Mxx.Certificate.OperationalNoise.EventReplay
