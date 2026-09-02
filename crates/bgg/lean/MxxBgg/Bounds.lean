import MxxBgg.Multiplication

namespace Mxx.Bgg

open Mxx.Primitives

variable {q n secretColumns gadgetColumns rows columns preimageBound : Nat}

def scalarLift (value : ErrorPoly n) : ErrorMatrix n 1 1 := fun _ _ => value

theorem scalarLift_mul
    (value : ErrorPoly n) (matrix : ErrorMatrix n 1 gadgetColumns) :
    scalarLift value * matrix = value • matrix := by
  ext row column
  have hrow : row = 0 := Fin.eq_zero row
  subst row
  simp [scalarLift, Matrix.mul_apply, smul_eq_mul]

theorem scalarLift_norm (value : ErrorPoly n) :
    matrixNorm (scalarLift value) = polyNorm value := by
  simp [scalarLift, matrixNorm]

theorem matrixNorm_neg (value : ErrorMatrix n rows columns) :
    matrixNorm (-value) = matrixNorm value := by
  simp [matrixNorm, polyNorm_neg]

theorem matrixNorm_sub_le (left right : ErrorMatrix n rows columns) :
    matrixNorm (left - right) ≤ matrixNorm left + matrixNorm right := by
  rw [sub_eq_add_neg]
  calc
    matrixNorm (left + -right) ≤ matrixNorm left + matrixNorm (-right) :=
      matrixNorm_add_le _ _
    _ = matrixNorm left + matrixNorm right := by rw [matrixNorm_neg]

/- This is the BGG+ noise bound.  The only input bounds are bounds for
   primitive witnesses and source encoding errors; each product estimate is
   discharged by the primitive matrix norm theorem.  No application-provided
   product-bound premise is accepted. -/
theorem multiplication_error_bound
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (targetApprox : Approx actualTarget rightPublic)
    (leftMaskMagnitude : MagnitudeFact leftMask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (leftErrorBound rightErrorBound targetErrorBound messageBound : Nat)
    (left_error_le : matrixNorm left.error ≤ leftErrorBound)
    (right_error_le : matrixNorm right.error ≤ rightErrorBound)
    (target_error_le : matrixNorm targetApprox.error ≤ targetErrorBound)
    (message_le : polyNorm messageLift ≤ messageBound)
    (ring_dimension_pos : 0 < n) :
    matrixNorm
        (left.error * preimageLift.witness + messageLift • right.error -
          messageLift • (leftMaskMagnitude.lift * targetApprox.error)) ≤
      gadgetColumns * n * leftErrorBound * preimageBound +
        (n * messageBound * rightErrorBound +
          n * messageBound * (secretColumns * n * leftMaskMagnitude.bound *
            targetErrorBound)) := by
  have hleft :
    matrixNorm (left.error * preimageLift.witness) ≤
        gadgetColumns * n * leftErrorBound * preimageBound := by
    exact (matrixNorm_mul_le ring_dimension_pos).trans
      (by
        simpa [Nat.mul_assoc] using
          Nat.mul_le_mul_left (gadgetColumns * n)
            (Nat.mul_le_mul left_error_le preimageLift.norm_le))
  have hright :
      matrixNorm (messageLift • right.error) ≤ n * messageBound * rightErrorBound := by
    rw [← scalarLift_mul]
    exact (matrixNorm_mul_le ring_dimension_pos).trans
      (by
        have message_norm_le : matrixNorm (scalarLift messageLift) ≤ messageBound := by
          simpa [scalarLift_norm] using message_le
        simpa [Nat.mul_assoc] using
          Nat.mul_le_mul_left n
            (Nat.mul_le_mul message_norm_le right_error_le))
  have htargetProduct :
      matrixNorm (leftMaskMagnitude.lift * targetApprox.error) ≤
        secretColumns * n * leftMaskMagnitude.bound * targetErrorBound := by
    exact (matrixNorm_mul_le ring_dimension_pos).trans
      (by
        simpa [Nat.mul_assoc] using
          Nat.mul_le_mul_left (secretColumns * n)
            (Nat.mul_le_mul leftMaskMagnitude.norm_le target_error_le))
  have htarget :
        matrixNorm (messageLift • (leftMaskMagnitude.lift * targetApprox.error)) ≤
          n * messageBound *
          (secretColumns * n * leftMaskMagnitude.bound * targetErrorBound) := by
    rw [← scalarLift_mul]
    exact (matrixNorm_mul_le ring_dimension_pos).trans
      (by
        have message_norm_le : matrixNorm (scalarLift messageLift) ≤ messageBound := by
          simpa [scalarLift_norm] using message_le
        simpa [Nat.mul_assoc] using
          Nat.mul_le_mul_left n
            (Nat.mul_le_mul message_norm_le htargetProduct))
  have hadd :
      matrixNorm (left.error * preimageLift.witness + messageLift • right.error) ≤
        gadgetColumns * n * leftErrorBound * preimageBound +
          n * messageBound * rightErrorBound :=
    (matrixNorm_add_le _ _).trans (Nat.add_le_add hleft hright)
  have hcombined := Nat.add_le_add hadd htarget
  exact (matrixNorm_sub_le _ _).trans (by simpa [Nat.add_assoc] using hcombined)

end Mxx.Bgg
