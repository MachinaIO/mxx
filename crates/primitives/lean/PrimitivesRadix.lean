import PrimitivesBounds

namespace Mxx.Primitives

open scoped BigOperators

variable {q n rows columns : Nat}

structure RadixSystem (q n : Nat) where
  Limb : Type u
  instFintypeLimb : Fintype Limb
  weight : Limb → ErrorPoly n
  digit : ExactPoly q n → Limb → ErrorPoly n
  reconstruct : ∀ x,
    x = ∑ limb : Limb, reducePoly q n (weight limb * digit x limb)
  commonDigitBound : Nat
  digit_bound : ∀ x limb, polyNorm (digit x limb) ≤ commonDigitBound

attribute [instance] RadixSystem.instFintypeLimb

theorem radix_reconstruct (system : RadixSystem q n) (value : ExactPoly q n) :
    value = ∑ limb : system.Limb,
      reducePoly q n (system.weight limb * system.digit value limb) :=
  system.reconstruct value

theorem radix_digit_bound (system : RadixSystem q n) (value : ExactPoly q n)
    (limb : system.Limb) :
    polyNorm (system.digit value limb) ≤ system.commonDigitBound :=
  system.digit_bound value limb

structure ColumnDigits
    (matrix : ExactMatrix q n rows columns) (Limb : Type u) [Fintype Limb] where
  digit : Fin columns → Limb → ErrorPoly n
  route : Limb → Matrix (Fin rows) Unit (ExactPoly q n)
  reconstruct : ∀ column,
    (fun row => matrix row column) =
      fun row => ∑ limb : Limb, reducePoly q n (digit column limb) * route limb row ⟨⟩
  commonDigitBound : Nat
  digit_bound : ∀ column limb, polyNorm (digit column limb) ≤ commonDigitBound

end Mxx.Primitives
