import MxxBgg.Boolean

namespace Mxx.Bgg

open Mxx.Primitives
open Mxx.Gadgets

/- The matrix calculation is deliberately independent of any concrete
   modulus.  The nonzero `targetError` is retained as the last error term. -/
example {R : Type u} [CommRing R]
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
  exact multiplication_core leftEquation rightEquation leftPayload_eq targetEquation

/- Choosing the ideal target to be the gadget itself keeps `G` in the exact
   target term; only the explicitly witnessed target error is moved into noise. -/
example {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic targetError : Matrix secret gadgetCols R}
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
        mask * gadget - rightMessage • (rightPayload * gadget) + rightError)
    (leftPayload_eq : leftPayload = mask)
    (targetEquation : gadget * decomposition = gadget + targetError) :
    leftCiphertext * decomposition + leftMessage • rightCiphertext =
      mask * (leftPublic * decomposition) -
        (leftMessage * rightMessage) • (rightPayload * gadget) +
        (leftError * decomposition + leftMessage • rightError -
          leftMessage • (mask * targetError)) := by
  exact multiplication_core leftEquation rightEquation leftPayload_eq targetEquation

/- With a zero target error the same theorem reduces to the familiar
   `C₁ K + x₁ C₂` identity; no cancellation assumption is hidden in the API. -/
example {R : Type u} [CommRing R]
    {secret gadgetCols : Type v} [Fintype secret] [Fintype gadgetCols]
    {gadget leftPublic rightPublic : Matrix secret gadgetCols R}
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
    (targetEquation : gadget * decomposition = rightPublic) :
    leftCiphertext * decomposition + leftMessage • rightCiphertext =
      mask * (leftPublic * decomposition) -
        (leftMessage * rightMessage) • (rightPayload * gadget) +
        (leftError * decomposition + leftMessage • rightError) := by
  have h := multiplication_core leftEquation rightEquation leftPayload_eq
    (show gadget * decomposition = rightPublic + (0 : Matrix secret gadgetCols R) by
      simpa using targetEquation)
  simpa using h

/- A zero target error is represented by zero in the exact ring, so the target
   term disappears only after the explicit zero fact is supplied. -/
example {R : Type u} [AddGroup R] (zero : Matrix (Fin 1) (Fin 1) R) (h : zero = 0) :
    zero = 0 := h

example {R : Type u} [Semiring R] (payload : Matrix (Fin 1) (Fin 1) R) :
    (0 : R) • payload = 0 := by
  simp

/- Boolean composition has a concrete, exact payload expression. -/
example (x y : Bool) :
    ((x && !y) || ((!x) && y)) =
      ((x || y) && (!(x && y))) := by
  cases x <;> cases y <;> decide

end Mxx.Bgg
