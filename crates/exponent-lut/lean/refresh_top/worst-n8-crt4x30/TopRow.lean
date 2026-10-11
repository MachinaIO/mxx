import TopSetup

/-! The payload of row `1` is the constant one. -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime

/-- The constant one as the identity `1 × 1` matrix, the form of the claim's row-`1` payload. -/
theorem one_poly :
    (fun _ _ ↦ reducePoly Q0 N (1 : ErrorPoly N) : ExactMatrix Q0 N 1 1) =
      (1 : Matrix (Fin 1) (Fin 1) (ExactPoly Q0 N)) := by
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp

end RefreshTopProof
