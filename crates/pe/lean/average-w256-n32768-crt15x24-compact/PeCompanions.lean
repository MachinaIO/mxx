import PeDigits

/-!
# The expanded companions

Each compact companion over `R_{PQ}` expands to an encoding over `R_Q` of its digit times the mask
pair against the gadget (`expand_rel`), within `companionBound`.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The error bound of an expanded companion. -/
def companionBound : Nat := (31 * (Q / 2) + (N + 2) * (P / 2)) / P

/-- An expanded compact encoding of `P d (σ, 1)` encodes `d (σ, 1) g`. -/
theorem companion_expand {sm d : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {At : Fin 2 → Fin 2 → ExactMatrix PQ N 1 1} {ct : Fin 2 → ExactMatrix PQ N 1 1}
    {e : Fin 2 → ErrorMatrix N 1 1}
    (h : RelE sm At (fun j _ _ ↦ reducePoly PQ N (((P : Nat) : Int) * (d * unitPair sm j))) ct e)
    (he : ∀ j, CoeffBound (e j) 31) :
    Rel sm (expandKey At) (fun r ↦ (reducePoly Q N d * reducePoly Q N (unitPair sm r)) • g0)
      (fun r ↦ expand (ct r)) companionBound := by
  obtain ⟨hr, hb⟩ := expand_rel hsm he h
  refine Rel.target_eq (RelE.toRel hr hb) ?_
  funext r
  rw [map_mul]

end PeProof
