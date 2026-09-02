import MxxGadgets.InjectorInvariant

namespace Mxx.Gadgets

example : (13 : Int) * 3 = 4 * (-1) + (4 * 7 + 5 * 3) := by norm_num

/- The recurrence uses one ring-dimension factor per actual matrix product. -/
example :
    InjectorStateInvariant.transitionNoiseBound 2 8 3 5 7 11 13 =
      2 * 8 * 5 * 7 + 3 * 8 * 11 * 13 := by
  rfl

/- A one-entry instance makes the target-source preservation visible: `P` is not replaced by
   zero, so the output ideal is `L * P` and the target error contributes `L * E`. -/
example (b k p e x l ex : Int)
    (hvalue : x = l * b + ex)
    (htarget : b * k = p + e) :
    x * k = l * p + (l * e + ex * k) := by
  let B : Matrix (Fin 1) (Fin 1) Int := fun _ _ => b
  let K : Matrix (Fin 1) (Fin 1) Int := fun _ _ => k
  let P : Matrix (Fin 1) (Fin 1) Int := fun _ _ => p
  let E : Matrix (Fin 1) (Fin 1) Int := fun _ _ => e
  let X : Matrix (Fin 1) (Fin 1) Int := fun _ _ => x
  let L : Matrix (Fin 1) (Fin 1) Int := fun _ _ => l
  let eX : Matrix (Fin 1) (Fin 1) Int := fun _ _ => ex
  have hv : X = L * B + eX := by
    ext i j
    simpa [X, L, B, eX, Matrix.mul_apply] using hvalue
  have ht : B * K = P + E := by
    ext i j
    simpa [B, K, P, E, Matrix.mul_apply] using htarget
  have h := consume_right_preimage_matrix
    (B := B) (K := K) (P := P) (E := E) (X := X) (L := L) (eX := eX) hv ht
  have hentry := congrFun (congrFun h (0 : Fin 1)) (0 : Fin 1)
  simpa [X, L, B, K, P, E, eX, Matrix.mul_apply] using hentry

end Mxx.Gadgets

namespace Mxx.Gadgets

/- Setting the target ideal to the gadget's public source demonstrates that the source survives
   preimage consumption.  The theorem does not assume that this source is zero. -/
example {R : Type u} [Semiring R]
    {s k r : Type v} [Fintype s] [Fintype k]
    {B : Matrix s k R} {K : Matrix k k R} {E : Matrix s k R}
    {X eX : Matrix r k R} {L : Matrix r s R}
    (hvalue : X = L * B + eX)
    (htarget : B * K = B + E) :
    X * K = L * B + (L * E + eX * K) := by
  exact consume_right_preimage_matrix hvalue htarget

end Mxx.Gadgets
