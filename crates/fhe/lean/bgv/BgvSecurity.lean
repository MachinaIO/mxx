import Scheme
import BgvFirstHop

set_option backward.isDefEq.respectTransparency false

/-! IND-CPA security of the generated BGV scheme relative to its declared assumptions. -/

namespace BgvSecurity

open MxxSecurity MxxFhe GeneratedScheme Mxx.Primitives MxxRuntime

/-- A `1 × 2` row as a `2 × 1` column. -/
def reshape {q : Nat} (u : ExactMatrix q n 1 2) : ExactMatrix q n 2 1 := fun r _ ↦ u 0 ⟨r.val, r.isLt⟩

/-- `E` in the second column of a row. -/
noncomputable def padE (E : ExactMatrix Q n 1 1) : ExactMatrix Q n 1 2 :=
  fun _ j ↦ if j.val = 0 then 0 else E 0 0

theorem ctOfRow_eq (u : ExactMatrix Q n 1 2) (E : ExactMatrix Q n 1 1) :
    ctOfRow u E = reshape (matrixScale 1032193 u + padE E) := by
  refine ext21 ?_ ?_ <;> simp [ctOfRow, reshape, matrixScale, tPoly, padE]

theorem ct_uniform (E : ExactMatrix Q n 1 1) (g : ExactMatrix Q n 2 1 → Comp Bool) :
    ((sampleResidue : Comp (ExactMatrix Q n 1 2)) >>= fun u ↦ g (ctOfRow u E)) =ᵈ
      ((sampleResidue : Comp (ExactMatrix Q n 1 2)) >>= fun u ↦ g (reshape u)) := by
  simp only [ctOfRow_eq]
  refine (sampleResidue_scale (by decide) t_coprime
    (fun X ↦ g (reshape (X + padE E)))).trans ?_
  exact sampleResidue_add (by decide) (by decide) _ (fun X ↦ g (reshape X))


theorem last_hop (A : Adversary scheme) :
    randomHop (uses.get ⟨2, by decide⟩) A false =ᵈ randomHop (uses.get ⟨2, by decide⟩) A true := by
  have hgame : ∀ right, randomHop (uses.get ⟨2, by decide⟩) A right =ᵈ
      ((sampleResidue : Comp (ExactMatrix Q n 1 2)) >>= fun a ↦
        (sampleResidue : Comp (ExactMatrix QP n 2 3)) >>= fun ek ↦ A.choose (pkOfRow a) ek >>=
          fun σ ↦ (sampleResidue : Comp (ExactMatrix Q n 1 2)) >>= fun u ↦
            A.guess σ.2.2 (Ciphertext.ct0 (reshape u))) := by
    intro right
    simp only [randomHop, uses, List.get, AssumptionId.random, DecisionRlwe.random]
    unfold MxxFhe.reduce
    simp only [simulator_2, bind_assoc, pure_bind, sim2_setup_eq, sim2_encrypt_eq]
    refine EvalDistEq.bind_right _ fun a ↦ ?_
    refine (swap _ _ _).trans (EvalDistEq.bind_right _ fun ek ↦ ?_)
    refine (swap _ _ _).trans (EvalDistEq.bind_right _ fun σ ↦ ?_)
    exact ct_uniform (encodeSlots (if right = true then σ.2.1 else σ.1))
      (fun c ↦ A.guess σ.2.2 (Ciphertext.ct0 c))
  exact (hgame false).trans (hgame true).symm

theorem security : SimulatedSecurity := by
  refine secure_of_hops uses (by decide) first_hop ?_ last_hop
  intro A right i hi
  have : i = 0 ∨ i = 1 := by
    have : uses.length = 3 := rfl
    omega
  rcases this with rfl | rfl
  · exact circular_to_public_key A right
  · exact public_key_to_ciphertext A right

end BgvSecurity
