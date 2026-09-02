import MxxPrimitives.Bounds

namespace Mxx.Primitives

variable {q n sourceRows inner targetColumns : Nat}
variable {source : ExactMatrix q n sourceRows inner}
variable {actualPreimage : ExactMatrix q n inner targetColumns}
variable {actualTarget : ExactMatrix q n sourceRows targetColumns}

structure RightPreimage
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns) where
  equation : source * actualPreimage = actualTarget

def PreimageWithin
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (preimageBound : Nat) : Prop :=
  Nonempty (BoundedLift actualPreimage preimageBound)

structure RightPreimageFamily
    (GroupIndex BranchIndex : Type u)
    [Fintype GroupIndex] [Fintype BranchIndex] where
  source : GroupIndex → ExactMatrix q n sourceRows inner
  actualTarget : GroupIndex → BranchIndex → ExactMatrix q n sourceRows targetColumns
  actualPreimage : GroupIndex → BranchIndex → ExactMatrix q n inner targetColumns
  relation : ∀ group branch,
    RightPreimage (source group) (actualPreimage group branch) (actualTarget group branch)
  commonPreimageBound : Nat
  bounded : ∀ group branch,
    PreimageWithin (actualPreimage group branch) commonPreimageBound

structure TargetApproxFamily
    (GroupIndex BranchIndex : Type u)
    [Fintype GroupIndex] [Fintype BranchIndex]
    (actualTarget : GroupIndex → BranchIndex → ExactMatrix q n sourceRows targetColumns) where
  targetIdeal : GroupIndex → BranchIndex → ExactMatrix q n sourceRows targetColumns
  targetApprox : ∀ group branch,
    Approx (actualTarget group branch) (targetIdeal group branch)
  commonTargetNoiseBound : Nat
  bounded : ∀ group branch,
    matrixNorm (targetApprox group branch).error ≤ commonTargetNoiseBound

/- The algebraic core of preimage consumption.  Application layers instantiate this equation
with exact and error witnesses after reducing the latter into the exact ring. -/
theorem consume_right_preimage {R : Type u} [Semiring R]
    (value multiplier source preimage ideal target valueError : R)
    (hvalue : value = multiplier * source + valueError)
    (htarget : source * preimage = ideal + target) :
    value * preimage = multiplier * ideal + (multiplier * target + valueError * preimage) := by
  rw [hvalue, add_mul, mul_assoc multiplier source preimage, htarget]
  rw [mul_add, add_assoc]

theorem rightPreimage_target_eq
    (fact : RightPreimage source actualPreimage actualTarget) :
    source * actualPreimage = actualTarget := fact.equation

theorem consume_rectangular_semiring {R : Type u} [Semiring R]
    {s k c r : Type v} [Fintype s] [Fintype k]
    (b : Matrix s k R) (kk : Matrix k c R) (t p : Matrix s c R)
    (l : Matrix r s R) (x : Matrix r k R)
    (xError : Matrix r k R) (tError : Matrix s c R)
    (hX : x = l * b + xError)
    (hB : b * kk = t)
    (hT : t = p + tError) :
    x * kk = l * p + (l * tError + xError * kk) := by
  rw [hX, Matrix.add_mul, Matrix.mul_assoc l b kk, hB, hT, Matrix.mul_add]
  ac_rfl

/- A rectangular, reduced-witness version of preimage consumption.  The error
returned by this theorem is exactly the integer lift
`l.lift * targetError + xError * preimage.lift`; no large matrix term is
reconstructed or cancelled by the checker. -/
noncomputable def consume_right_preimage_rectangular
    {sourceRows inner targetColumns resultRows : Nat}
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (l : ExactMatrix q n resultRows sourceRows)
    (x : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact l)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (xApprox : ApproxWithin x (l * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
    Approx (x * actualPreimage) (l * idealTarget) := by
  let xError := xApprox.error
  let targetError := targetApprox.error
  let outputError : ErrorMatrix n resultRows targetColumns :=
    leftMagnitude.lift * targetError + xError * preimageLift.witness
  refine { error := outputError, equation := ?_ }
  have hleftEq : reduceMatrix q n resultRows sourceRows leftMagnitude.lift = l :=
    leftMagnitude.reduce_eq
  have hpre : reduceMatrix q n inner targetColumns preimageLift.witness = actualPreimage :=
    preimageLift.reduce_eq
  have hrel : source * actualPreimage = actualTarget := relation.equation
  have hcalc := consume_rectangular_semiring
    (b := source) (kk := actualPreimage) (t := actualTarget) (p := idealTarget)
    (l := l) (x := x) (xError := reduceMatrix q n resultRows inner xError)
    (tError := reduceMatrix q n sourceRows targetColumns targetError)
    (by simpa [xError] using xApprox.equation) hrel
    (by simpa [targetError] using targetApprox.equation)
  rw [hcalc]
  rw [← hleftEq, ← hpre]
  rw [← reduceMatrix_mul, ← reduceMatrix_mul, ← reduceMatrix_add]

/- The bounded companion derives both product estimates from the primitive matrix norm lemma.
The public API therefore carries only the three source bounds and the two approximation facts. -/
noncomputable def consume_right_preimage_rectangular_with_bound
    {sourceRows inner targetColumns resultRows : Nat}
    (hn : 0 < n)
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (l : ExactMatrix q n resultRows sourceRows)
    (x : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact l)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (xApprox : ApproxWithin x (l * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
    ApproxWithin (x * actualPreimage) (l * idealTarget)
      (sourceRows * n * leftMagnitude.bound * targetNoiseBound +
        inner * n * xNoiseBound * preimageBound) := by
  let xError := xApprox.error
  let targetError := targetApprox.error
  let outputError : ErrorMatrix n resultRows targetColumns :=
    leftMagnitude.lift * targetError + xError * preimageLift.witness
  have hleftEq : reduceMatrix q n resultRows sourceRows leftMagnitude.lift = l :=
    leftMagnitude.reduce_eq
  have hpre : reduceMatrix q n inner targetColumns preimageLift.witness = actualPreimage :=
    preimageLift.reduce_eq
  have hcalc := consume_rectangular_semiring
    (b := source) (kk := actualPreimage) (t := actualTarget) (p := idealTarget)
    (l := l) (x := x) (xError := reduceMatrix q n resultRows inner xError)
    (tError := reduceMatrix q n sourceRows targetColumns targetError)
    (by simpa [xError] using xApprox.equation) relation.equation
    (by simpa [targetError] using targetApprox.equation)
  refine { toApprox := { error := outputError, equation := ?_ }, norm_le := ?_ }
  · rw [hcalc, ← hleftEq, ← hpre]
    rw [← reduceMatrix_mul, ← reduceMatrix_mul, ← reduceMatrix_add]
  · apply (matrixNorm_add_le _ _).trans
    apply Nat.add_le_add
    · apply (matrixNorm_mul_le (n := n) hn
        (a := leftMagnitude.lift) (b := targetError)).trans
      calc
        _ ≤ sourceRows * n * (leftMagnitude.bound * targetNoiseBound) := by
          simpa [Nat.mul_assoc] using
            Nat.mul_le_mul_left (sourceRows * n)
              (Nat.mul_le_mul leftMagnitude.norm_le targetApprox.norm_le)
        _ = sourceRows * n * leftMagnitude.bound * targetNoiseBound := by ring
    · apply (matrixNorm_mul_le (n := n) (by omega)
        (a := xError) (b := preimageLift.witness)).trans
      calc
        _ ≤ inner * n * (xNoiseBound * preimageBound) := by
          simpa [Nat.mul_assoc] using
            Nat.mul_le_mul_left (inner * n)
              (Nat.mul_le_mul xApprox.norm_le preimageLift.norm_le)
        _ = inner * n * xNoiseBound * preimageBound := by ring

/- When the left witness is constant in every matrix entry, the constant-side convolution lemma
   removes the ring-dimension factor from the left product. The right product remains generic. -/
noncomputable def consume_right_preimage_rectangular_with_constant_left_bound
    {sourceRows inner targetColumns resultRows : Nat}
    (hn : 0 < n)
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns)
    (l : ExactMatrix q n resultRows sourceRows)
    (x : ExactMatrix q n resultRows inner)
    (idealTarget : ExactMatrix q n sourceRows targetColumns)
    (relation : RightPreimage source actualPreimage actualTarget)
    (leftMagnitude : MagnitudeFact l)
    (hconstant : leftMagnitude.support = SupportClass.constant)
    {preimageBound xNoiseBound targetNoiseBound : Nat}
    (preimageLift : BoundedLift actualPreimage preimageBound)
    (xApprox : ApproxWithin x (l * source) xNoiseBound)
    (targetApprox : ApproxWithin actualTarget idealTarget targetNoiseBound) :
    ApproxWithin (x * actualPreimage) (l * idealTarget)
      (sourceRows * leftMagnitude.bound * targetNoiseBound +
        inner * n * xNoiseBound * preimageBound) := by
  let xError := xApprox.error
  let targetError := targetApprox.error
  let outputError : ErrorMatrix n resultRows targetColumns :=
    leftMagnitude.lift * targetError + xError * preimageLift.witness
  have hleftEq : reduceMatrix q n resultRows sourceRows leftMagnitude.lift = l :=
    leftMagnitude.reduce_eq
  have hpre : reduceMatrix q n inner targetColumns preimageLift.witness = actualPreimage :=
    preimageLift.reduce_eq
  have hcalc := consume_rectangular_semiring
    (b := source) (kk := actualPreimage) (t := actualTarget) (p := idealTarget)
    (l := l) (x := x) (xError := reduceMatrix q n resultRows inner xError)
    (tError := reduceMatrix q n sourceRows targetColumns targetError)
    (by simpa [xError] using xApprox.equation) relation.equation
    (by simpa [targetError] using targetApprox.equation)
  refine { toApprox := { error := outputError, equation := ?_ }, norm_le := ?_ }
  · rw [hcalc, ← hleftEq, ← hpre]
    rw [← reduceMatrix_mul, ← reduceMatrix_mul, ← reduceMatrix_add]
  · apply (matrixNorm_add_le _ _).trans
    apply Nat.add_le_add
    · apply (matrixNorm_mul_left_constant_le (n := n) hn
        (a := leftMagnitude.lift) (b := targetError)
        (leftMagnitude.isConstant_of_support_constant hconstant)).trans
      calc
        _ ≤ sourceRows * (leftMagnitude.bound * targetNoiseBound) := by
          simpa [Nat.mul_assoc] using
            Nat.mul_le_mul_left sourceRows
              (Nat.mul_le_mul leftMagnitude.norm_le targetApprox.norm_le)
        _ = sourceRows * leftMagnitude.bound * targetNoiseBound := by ring
    · apply (matrixNorm_mul_le (n := n) (by omega)
        (a := xError) (b := preimageLift.witness)).trans
      calc
        _ ≤ inner * n * (xNoiseBound * preimageBound) := by
          simpa [Nat.mul_assoc] using
            Nat.mul_le_mul_left (inner * n)
              (Nat.mul_le_mul xApprox.norm_le preimageLift.norm_le)
        _ = inner * n * xNoiseBound * preimageBound := by ring

end Mxx.Primitives
