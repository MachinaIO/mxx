import MxxBgg.PublicKey
import MxxPrimitives.Preimage

namespace Mxx.Bgg

open Mxx.Primitives

variable {q n secretColumns gadgetColumns rows columns preimageBound : Nat}

/- A matrix identity over a commutative ring.  This is the complete BGG+
   cancellation step; all later theorems only instantiate it with reduced
   integer witnesses. -/
theorem multiplication_core {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic targetError : Matrix secret gadgetCols R}
    {decomposition : Matrix gadgetCols gadgetCols R}
    {leftCiphertext rightCiphertext : Matrix (Fin 1) gadgetCols R}
    {mask leftPayload rightPayload : Matrix (Fin 1) secret R}
    {leftMessage rightMessage : R}
    {leftError rightError : Matrix (Fin 1) gadgetCols R}
    (leftEquation :
      leftCiphertext =
        mask * leftPublic - leftMessage • (leftPayload * gadget) + leftError)
    (rightEquation :
      rightCiphertext =
        mask * rightPublic - rightMessage • (rightPayload * gadget) + rightError)
    (leftPayload_eq : leftPayload = mask)
    (targetEquation : gadget * decomposition = rightPublic + targetError) :
    leftCiphertext * decomposition + leftMessage • rightCiphertext =
      mask * (leftPublic * decomposition) -
        (leftMessage * rightMessage) • (rightPayload * gadget) +
        (leftError * decomposition + leftMessage • rightError -
          leftMessage • (mask * targetError)) := by
  rw [leftEquation, rightEquation, leftPayload_eq]
  simp only [Matrix.sub_mul, Matrix.add_mul, Matrix.mul_assoc, Matrix.smul_mul,
    smul_add, smul_sub, smul_smul]
  have maskTarget : mask * (gadget * decomposition) =
      mask * rightPublic + mask * targetError := by
    rw [targetEquation, Matrix.mul_add]
  rw [maskTarget]
  simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

theorem linear_add_core {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic : Matrix secret gadgetCols R}
    {leftCiphertext rightCiphertext : Matrix (Fin 1) gadgetCols R}
    {mask payload : Matrix (Fin 1) secret R}
    {leftMessage rightMessage : R}
    {leftError rightError : Matrix (Fin 1) gadgetCols R}
    (leftEquation :
      leftCiphertext =
        mask * leftPublic - leftMessage • (payload * gadget) + leftError)
    (rightEquation :
      rightCiphertext =
        mask * rightPublic - rightMessage • (payload * gadget) + rightError) :
    leftCiphertext + rightCiphertext =
      mask * (leftPublic + rightPublic) -
        (leftMessage + rightMessage) • (payload * gadget) +
        (leftError + rightError) := by
  rw [leftEquation, rightEquation]
  simp [Matrix.mul_add, add_smul, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

theorem linear_sub_core {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic : Matrix secret gadgetCols R}
    {leftCiphertext rightCiphertext : Matrix (Fin 1) gadgetCols R}
    {mask payload : Matrix (Fin 1) secret R}
    {leftMessage rightMessage : R}
    {leftError rightError : Matrix (Fin 1) gadgetCols R}
    (leftEquation :
      leftCiphertext =
        mask * leftPublic - leftMessage • (payload * gadget) + leftError)
    (rightEquation :
      rightCiphertext =
        mask * rightPublic - rightMessage • (payload * gadget) + rightError) :
    leftCiphertext - rightCiphertext =
      mask * (leftPublic - rightPublic) -
        (leftMessage - rightMessage) • (payload * gadget) +
        (leftError - rightError) := by
  rw [leftEquation, rightEquation]
  simp only [sub_eq_add_neg, Matrix.mul_add, add_smul, neg_smul]
  simp [add_assoc, add_left_comm, add_comm]

theorem scalar_two_core {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget pub : Matrix secret gadgetCols R}
    {cipher : Matrix (Fin 1) gadgetCols R}
    {mask payload : Matrix (Fin 1) secret R} {message : R}
    {error : Matrix (Fin 1) gadgetCols R}
    (equation : cipher = mask * pub - message • (payload * gadget) + error) :
    (2 : R) • cipher = mask * ((2 : R) • pub) -
      ((2 : R) * message) • (payload * gadget) + (2 : R) • error := by
  rw [equation]
  simp [Matrix.mul_smul, smul_add, smul_smul, sub_eq_add_neg, add_comm]

/- Reduction preserves scalar multiplication of an integer witness.  This is
   a ring-map fact, not a norm or application-specific rule. -/
theorem reduceMatrix_int_smul
    (scalar : ErrorPoly n) (value : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (scalar • value) =
      reducePoly q n scalar • reduceMatrix q n rows columns value := by
  funext row column
  change reducePoly q n (scalar * value row column) =
    reducePoly q n scalar * reducePoly q n (value row column)
  rw [reducePoly_mul]

theorem reduceMatrix_sub
    (left right : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (left - right) =
      reduceMatrix q n rows columns left - reduceMatrix q n rows columns right := by
  funext row column
  change reducePoly q n (left row column - right row column) =
    reducePoly q n (left row column) - reducePoly q n (right row column)
  rw [sub_eq_add_neg, reducePoly_add, reducePoly_neg]
  simp only [sub_eq_add_neg]

/- BGG+ multiplication with an explicit integer error witness.  The public
   matrix on the right is the ideal target of the gadget preimage.  If the
   target error is zero, the third term in `outputError` disappears by
   construction. -/
noncomputable def multiply
    {leftCiphertext rightCiphertext : ExactMatrix q n 1 gadgetColumns}
    {leftMask leftPayload rightMask rightPayload : ExactMatrix q n 1 secretColumns}
    {leftPublic rightPublic gadget : ExactMatrix q n secretColumns gadgetColumns}
    {leftMessage rightMessage : ExactPoly q n}
    (left : Encoding leftCiphertext leftMask leftPayload leftPublic gadget leftMessage)
    (right : Encoding rightCiphertext rightMask rightPayload rightPublic gadget rightMessage)
    {decomposition : ExactMatrix q n gadgetColumns gadgetColumns}
    {actualTarget : ExactMatrix q n secretColumns gadgetColumns}
    (relation : RightPreimage gadget decomposition actualTarget)
    (targetApprox : Approx actualTarget rightPublic)
    (leftMaskMagnitude : MagnitudeFact leftMask)
    (preimageLift : BoundedLift decomposition preimageBound)
    (messageLift : ErrorPoly n)
    (message_reduce : reducePoly q n messageLift = leftMessage)
    (mask_eq : leftMask = rightMask)
    (leftPayload_eq : leftPayload = leftMask) :
    Encoding
      (leftCiphertext * decomposition + leftMessage • rightCiphertext)
      leftMask rightPayload (leftPublic * decomposition) gadget
      (leftMessage * rightMessage) := by
  let outputError : ErrorMatrix n 1 gadgetColumns :=
    left.error * preimageLift.witness + messageLift • right.error -
      messageLift • (leftMaskMagnitude.lift * targetApprox.error)
  refine ⟨outputError, ?_⟩
  have rightEquation :
      rightCiphertext =
        leftMask * rightPublic - rightMessage • (rightPayload * gadget) +
          reduceMatrix q n 1 gadgetColumns right.error := by
    simpa [← mask_eq] using right.equation
  have leftEquation :
      leftCiphertext =
        leftMask * leftPublic - leftMessage • (leftMask * gadget) +
          reduceMatrix q n 1 gadgetColumns left.error := by
    simpa [leftPayload_eq] using left.equation
  have targetEquation :
      gadget * decomposition = rightPublic +
        reduceMatrix q n secretColumns gadgetColumns targetApprox.error := by
    rw [relation.equation]
    exact targetApprox.equation
  have core := multiplication_core
    (leftEquation := leftEquation)
    (rightEquation := rightEquation)
    (leftPayload_eq := rfl)
    (targetEquation := targetEquation)
  have reducedOutput :
      reduceMatrix q n 1 gadgetColumns outputError =
        reduceMatrix q n 1 gadgetColumns left.error * decomposition +
          leftMessage • reduceMatrix q n 1 gadgetColumns right.error -
          leftMessage •
            (leftMask * reduceMatrix q n secretColumns gadgetColumns targetApprox.error) := by
    unfold outputError
    simp only [reduceMatrix_sub, reduceMatrix_add, reduceMatrix_mul, reduceMatrix_int_smul]
    rw [preimageLift.reduce_eq, message_reduce, leftMaskMagnitude.reduce_eq]
  calc
    leftCiphertext * decomposition + leftMessage • rightCiphertext =
        leftMask * (leftPublic * decomposition) -
          (leftMessage * rightMessage) • (rightPayload * gadget) +
            (reduceMatrix q n 1 gadgetColumns left.error * decomposition +
              leftMessage • reduceMatrix q n 1 gadgetColumns right.error -
              leftMessage •
                (leftMask * reduceMatrix q n secretColumns gadgetColumns targetApprox.error)) := core
    _ = leftMask * (leftPublic * decomposition) -
        (leftMessage * rightMessage) • (rightPayload * gadget) +
          reduceMatrix q n 1 gadgetColumns outputError := by rw [reducedOutput]

end Mxx.Bgg
