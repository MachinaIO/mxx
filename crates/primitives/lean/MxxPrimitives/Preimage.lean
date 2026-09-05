import MxxPrimitives.Bounds

namespace Mxx.Primitives

/- An exact right-preimage relation.  The equation refers to the actual source,
   preimage, and target supplied by the primitive invocation. -/
structure RightPreimage {q n sourceRows inner targetColumns : Nat}
    (source : ExactMatrix q n sourceRows inner)
    (preimage : ExactMatrix q n inner targetColumns)
    (target : ExactMatrix q n sourceRows targetColumns) : Prop where
  equation : source * preimage = target

def PreimageWithin {q n rows columns : Nat}
    (actual : ExactMatrix q n rows columns) (bound : Nat) : Prop :=
  ∃ witness : ErrorMatrix n rows columns,
    actual = reduceMatrix q n rows columns witness ∧ CoeffBound witness bound

theorem consume_rectangular {R : Type u} [Semiring R]
    {s k c r : Type v} [Fintype s] [Fintype k]
    (b : Matrix s k R) (kk : Matrix k c R) (t p : Matrix s c R)
    (l : Matrix r s R) (x : Matrix r k R) (xError : Matrix r k R)
    (tError : Matrix s c R)
    (hX : x = l * b + xError) (hB : b * kk = t) (hT : t = p + tError) :
    x * kk = l * p + (l * tError + xError * kk) := by
  rw [hX, Matrix.add_mul, Matrix.mul_assoc l b kk, hB, hT, Matrix.mul_add]
  ac_rfl

/- Consume a nonzero-target-error preimage.  The two summands in the bound
   correspond respectively to the target error and the left approximation error. -/
theorem consume_right_preimage_bound
    {q n sourceRows inner targetColumns resultRows : Nat} (hn : 0 < n)
    {source : ExactMatrix q n sourceRows inner}
    {preimage : ExactMatrix q n inner targetColumns}
    {target ideal : ExactMatrix q n sourceRows targetColumns}
    {left : ExactMatrix q n resultRows sourceRows}
    {value : ExactMatrix q n resultRows inner}
    {leftError : ErrorMatrix n resultRows inner}
    {targetError : ErrorMatrix n sourceRows targetColumns}
    {preimageError : ErrorMatrix n inner targetColumns}
    {leftLift : ErrorMatrix n resultRows sourceRows}
    {leftBound targetBound valueBound preimageBound : Nat}
    (hleft : value = reduceMatrix q n resultRows inner leftError + left * source)
    (htarget : target = ideal + reduceMatrix q n sourceRows targetColumns targetError)
    (hleftLift : left = reduceMatrix q n resultRows sourceRows leftLift)
    (hpreimage : preimage = reduceMatrix q n inner targetColumns preimageError)
    (hrelation : source * preimage = target)
    (hleftBound : CoeffBound leftLift leftBound)
    (htargetBound : CoeffBound targetError targetBound)
    (hvalueBound : CoeffBound leftError valueBound)
    (hpreimageBound : CoeffBound preimageError preimageBound) :
    Approx (value * preimage) (left * ideal)
      (sourceRows * n * leftBound * targetBound +
        inner * n * valueBound * preimageBound) := by
  let outputError := leftLift * targetError + leftError * preimageError
  refine ⟨outputError, ?_, ?_⟩
  · have hleft' : value = left * source + reduceMatrix q n resultRows inner leftError := by
      simpa [add_comm] using hleft
    have hcalc := consume_rectangular
      (b := source) (kk := preimage) (t := target) (p := ideal)
      (l := left) (x := value)
      (xError := reduceMatrix q n resultRows inner leftError)
      (tError := reduceMatrix q n sourceRows targetColumns targetError)
      hleft' hrelation htarget
    rw [hcalc, hleftLift, hpreimage]
    rw [← reduceMatrix_mul, ← reduceMatrix_mul, ← reduceMatrix_add]
  · rw [coeffBound_iff_matrixNorm_le] at *
    apply (matrixNorm_add_le _ _).trans
    apply Nat.add_le_add
    · exact (matrixNorm_mul_le hn).trans <| by
        simpa [Nat.mul_assoc] using Nat.mul_le_mul_left (sourceRows * n)
          (Nat.mul_le_mul hleftBound htargetBound)
    · exact (matrixNorm_mul_le hn).trans <| by
        simpa [Nat.mul_assoc] using Nat.mul_le_mul_left (inner * n)
          (Nat.mul_le_mul hvalueBound hpreimageBound)

end Mxx.Primitives
