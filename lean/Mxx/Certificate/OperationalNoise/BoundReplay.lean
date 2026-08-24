import Mxx.Certificate.OperationalNoise.Core

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.EventReplay

/-- A known coefficient class. Absence is represented outside this type and may not be promoted
    to `large`; a finite constructor carries the Rust invariant that its maximum is positive. -/
inductive CoeffClass where
  | exactZero
  | finite (maximum : { value : Nat // 0 < value })
  | large
deriving DecidableEq, Repr

/-- Interpret a coefficient class as a proposition about one actual coefficient magnitude. -/
def CoeffClass.Interprets : CoeffClass → Nat → Prop
  | .exactZero, actual => actual = 0
  | .finite maximum, actual => actual ≤ maximum.val
  | .large, _ => True

def addKnown : CoeffClass → CoeffClass → CoeffClass
  | .exactZero, right => right
  | left, .exactZero => left
  | .large, _ => .large
  | _, .large => .large
  | .finite left, .finite right =>
      .finite ⟨left.val + right.val, Nat.add_pos_left left.property right.val⟩

def maxKnown : CoeffClass → CoeffClass → CoeffClass
  | .large, _ => .large
  | _, .large => .large
  | .exactZero, right => right
  | left, .exactZero => left
  | .finite left, .finite right =>
      .finite
        ⟨Nat.max left.val right.val,
          Nat.lt_of_lt_of_le left.property (Nat.le_max_left _ _)⟩

/-- Multiply two known classes with the exact nonnegative factor selected by Rust. -/
def productWithFactor (factor : Nat) : CoeffClass → CoeffClass → CoeffClass
  | .exactZero, _ => .exactZero
  | _, .exactZero => .exactZero
  | .large, _ => .large
  | _, .large => .large
  | .finite left, .finite right =>
      if hfactor : factor = 0 then .exactZero
      else
        .finite
          ⟨factor * left.val * right.val,
            Nat.mul_pos (Nat.mul_pos (Nat.pos_of_ne_zero hfactor) left.property) right.property⟩

/-- Scaling by a known magnitude has its own zero rule, including `0 * Large = ExactZero`. -/
def scaleMagnitude : Nat → CoeffClass → CoeffClass
  | 0, _ => .exactZero
  | _ + 1, .exactZero => .exactZero
  | factor + 1, .finite bound =>
      .finite ⟨(factor + 1) * bound.val, Nat.mul_pos (Nat.zero_lt_succ factor) bound.property⟩
  | _ + 1, .large => .large

def scaleValue (value scale : CoeffClass) : CoeffClass :=
  productWithFactor 1 value scale

/-- A nonempty coefficient-class product. Nonemptiness mirrors the Rust monomial requirement. -/
def productNonempty : CoeffClass → List CoeffClass → CoeffClass
  | head, [] => head
  | head, next :: tail => productWithFactor 1 head (productNonempty next tail)

def productFoldWithFactor (factor : Nat) (head : CoeffClass) (tail : List CoeffClass) :
    CoeffClass :=
  scaleMagnitude factor (productNonempty head tail)

theorem addKnown_sound {leftActual rightActual : Nat}
    {leftClass rightClass : CoeffClass}
    (leftSound : leftClass.Interprets leftActual)
    (rightSound : rightClass.Interprets rightActual) :
    (addKnown leftClass rightClass).Interprets (leftActual + rightActual) := by
  cases leftClass <;> cases rightClass <;>
    simp_all [CoeffClass.Interprets, addKnown, Nat.add_le_add]

theorem maxKnown_sound {leftActual rightActual : Nat}
    {leftClass rightClass : CoeffClass}
    (leftSound : leftClass.Interprets leftActual)
    (rightSound : rightClass.Interprets rightActual) :
    (maxKnown leftClass rightClass).Interprets (Nat.max leftActual rightActual) := by
  cases leftClass <;> cases rightClass <;>
    simp_all [CoeffClass.Interprets, maxKnown, Nat.max_le]
  exact ⟨Nat.le_trans leftSound (Nat.le_max_left _ _),
    Nat.le_trans rightSound (Nat.le_max_right _ _)⟩

theorem productWithFactor_sound {factor leftActual rightActual : Nat}
    {leftClass rightClass : CoeffClass}
    (leftSound : leftClass.Interprets leftActual)
    (rightSound : rightClass.Interprets rightActual) :
    (productWithFactor factor leftClass rightClass).Interprets
      (factor * leftActual * rightActual) := by
  cases leftClass with
  | exactZero => simp_all [CoeffClass.Interprets, productWithFactor]
  | large =>
      cases rightClass <;> simp_all [CoeffClass.Interprets, productWithFactor]
  | finite left =>
      cases rightClass with
      | exactZero => simp_all [CoeffClass.Interprets, productWithFactor]
      | large => simp [CoeffClass.Interprets, productWithFactor]
      | finite right =>
          simp only [productWithFactor]
          split
          · simp_all [CoeffClass.Interprets]
          · change factor * leftActual * rightActual ≤ factor * left.val * right.val
            exact Nat.mul_le_mul (Nat.mul_le_mul_left factor leftSound) rightSound

theorem scaleMagnitude_sound {factor actual : Nat} {bound : CoeffClass}
    (sound : bound.Interprets actual) :
    (scaleMagnitude factor bound).Interprets (factor * actual) := by
  cases factor with
  | zero => simp [scaleMagnitude, CoeffClass.Interprets]
  | succ factor =>
      cases bound <;> simp_all [scaleMagnitude, CoeffClass.Interprets, Nat.mul_le_mul_left]

theorem scaleValue_sound {valueActual scaleActual : Nat} {value scale : CoeffClass}
    (valueSound : value.Interprets valueActual) (scaleSound : scale.Interprets scaleActual) :
    (scaleValue value scale).Interprets (valueActual * scaleActual) := by
  simpa [scaleValue, Nat.one_mul] using
    productWithFactor_sound (factor := 1) valueSound scaleSound

theorem productNonempty_sound {headActual : Nat} {tailActual : List Nat}
    {head : CoeffClass} {tail : List CoeffClass} (headSound : head.Interprets headActual)
    (tailSound : List.Forall₂ (fun bound actual => bound.Interprets actual) tail tailActual) :
    (productNonempty head tail).Interprets (headActual * tailActual.prod) := by
  induction tailSound generalizing head headActual with
  | nil => simpa [productNonempty]
  | @cons nextActual actuals next rest nextSound restSound ih =>
      simp only [productNonempty, List.prod_cons]
      have tailProductSound := ih nextSound
      simpa [Nat.mul_assoc] using
        productWithFactor_sound (factor := 1) headSound tailProductSound

theorem productFoldWithFactor_sound {factor headActual : Nat} {tailActual : List Nat}
    {head : CoeffClass} {tail : List CoeffClass} (headSound : head.Interprets headActual)
    (tailSound : List.Forall₂ (fun bound actual => bound.Interprets actual) tail tailActual) :
    (productFoldWithFactor factor head tail).Interprets
      (factor * (headActual * tailActual.prod)) := by
  exact scaleMagnitude_sound (productNonempty_sound headSound tailSound)

structure ProductFacts where
  leftConstantPolynomial : Bool
  rightConstantPolynomial : Bool
  rightKnownZeroRows : Option Nat
  leftSupportUpper : Option Nat
  rightSupportUpper : Option Nat
deriving DecidableEq, Repr

def optionalWithin (value : Option Nat) (upper : Nat) : Bool :=
  match value with
  | none => true
  | some actual => decide (actual ≤ upper)

def effectiveSupport (constantPolynomial : Bool) (supportUpper : Option Nat)
    (ringDimension : Nat) : Nat :=
  if constantPolynomial then 1 else supportUpper.getD ringDimension

/-- Exact coefficient factor for the four Rust product shape branches. Invalid support, inner
    shape, or zero-row facts return `none`; no conservative fallback is assigned. -/
def productFactor (leftRows leftColumns rightRows rightColumns ringDimension : Nat)
    (facts : ProductFacts) : Option Nat :=
  if ringDimension = 0 || leftRows = 0 || leftColumns = 0 || rightRows = 0 ||
      rightColumns = 0 then
    none
  else if !optionalWithin facts.leftSupportUpper ringDimension ||
      !optionalWithin facts.rightSupportUpper ringDimension then
    none
  else
    let leftScalar := leftRows = 1 && leftColumns = 1
    let rightScalar := rightRows = 1 && rightColumns = 1
    if leftScalar && rightScalar then
      some (effectiveSupport facts.leftConstantPolynomial facts.leftSupportUpper ringDimension)
    else if leftScalar then
      some (effectiveSupport facts.leftConstantPolynomial facts.leftSupportUpper ringDimension)
    else if rightScalar then
      some (effectiveSupport facts.rightConstantPolynomial facts.rightSupportUpper ringDimension)
    else if leftColumns != rightRows then
      none
    else
      let zeroRows := facts.rightKnownZeroRows.getD 0
      if zeroRows ≤ rightRows then
        some ((leftColumns - zeroRows) *
          if facts.leftConstantPolynomial || facts.rightConstantPolynomial then 1
          else ringDimension)
      else
        none

def productWithFacts (leftRows leftColumns rightRows rightColumns ringDimension : Nat)
    (facts : ProductFacts) (left right : CoeffClass) : Option CoeffClass :=
  (productFactor leftRows leftColumns rightRows rightColumns ringDimension facts).map
    (fun factor => productWithFactor factor left right)

def tensorFactor (ringDimension : Nat) (facts : ProductFacts) : Nat :=
  if facts.leftConstantPolynomial || facts.rightConstantPolynomial then 1 else ringDimension

def tensorWithFacts (ringDimension : Nat) (facts : ProductFacts)
    (left right : CoeffClass) : CoeffClass :=
  productWithFactor (tensorFactor ringDimension facts) left right

theorem productWithFacts_sound
    {leftRows leftColumns rightRows rightColumns ringDimension factor : Nat}
    {facts : ProductFacts} {left right : CoeffClass} {leftActual rightActual : Nat}
    (factorExact :
      productFactor leftRows leftColumns rightRows rightColumns ringDimension facts = some factor)
    (leftSound : left.Interprets leftActual) (rightSound : right.Interprets rightActual) :
    productWithFacts leftRows leftColumns rightRows rightColumns ringDimension facts left right =
        some (productWithFactor factor left right) ∧
      (productWithFactor factor left right).Interprets (factor * leftActual * rightActual) := by
  exact ⟨by simp [productWithFacts, factorExact], productWithFactor_sound leftSound rightSound⟩

theorem tensorWithFacts_sound {ringDimension leftActual rightActual : Nat}
    {facts : ProductFacts} {left right : CoeffClass}
    (leftSound : left.Interprets leftActual) (rightSound : right.Interprets rightActual) :
    (tensorWithFacts ringDimension facts left right).Interprets
      (tensorFactor ringDimension facts * leftActual * rightActual) := by
  exact productWithFactor_sound leftSound rightSound

end Mxx.Certificate.OperationalNoise.EventReplay
