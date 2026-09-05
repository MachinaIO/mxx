import MxxBgg.Encoding

namespace Mxx.Bgg

open Mxx.Primitives

/- Algebraic BGG+ multiplication with compatible, but independent, secrets.  The output witness
   contains both terms: the left error times the decomposition and the left message times the
   right error.  Omitting the latter would undercount noise. -/
theorem multiplication_core {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic : Matrix secret gadgetCols R}
    {decomposition : Matrix gadgetCols gadgetCols R}
    {leftCiphertext rightCiphertext : Matrix (Fin 1) gadgetCols R}
    {leftSecret rightSecret rightPayload : Matrix (Fin 1) secret R}
    {leftMessage rightMessage : R}
    {leftError rightError : Matrix (Fin 1) gadgetCols R}
    (leftEquation :
      leftCiphertext = leftSecret * leftPublic - leftMessage • (rightSecret * gadget) + leftError)
    (rightEquation :
      rightCiphertext = rightSecret * rightPublic - rightMessage • (rightPayload * gadget) + rightError)
    (targetEquation : gadget * decomposition = rightPublic)
    : leftCiphertext * decomposition + leftMessage • rightCiphertext =
      leftSecret * (leftPublic * decomposition) -
        (leftMessage * rightMessage) • (rightPayload * gadget) +
          (leftError * decomposition + leftMessage • rightError) := by
  rw [leftEquation, rightEquation]
  simp only [Matrix.sub_mul, Matrix.add_mul, Matrix.mul_assoc, Matrix.smul_mul,
    smul_add, smul_sub, smul_smul]
  rw [targetEquation]
  simp [sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

theorem linear_add_core {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic : Matrix secret gadgetCols R}
    {leftCiphertext rightCiphertext : Matrix (Fin 1) gadgetCols R}
    {mask payload : Matrix (Fin 1) secret R}
    {leftMessage rightMessage : R}
    {leftError rightError : Matrix (Fin 1) gadgetCols R}
    (leftEquation :
      leftCiphertext = mask * leftPublic - leftMessage • (payload * gadget) + leftError)
    (rightEquation :
      rightCiphertext = mask * rightPublic - rightMessage • (payload * gadget) + rightError) :
    leftCiphertext + rightCiphertext =
      mask * (leftPublic + rightPublic) -
        (leftMessage + rightMessage) • (payload * gadget) + (leftError + rightError) := by
  rw [leftEquation, rightEquation]
  simp [Matrix.mul_add, add_smul, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

end Mxx.Bgg
