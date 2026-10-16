import GenRel

/-!
# A PRF sampler at one modulus

The sampler's members are finished LUTs (`LutMemberFlat.rel`): member `o` encodes `Δ' ν_o`
against `gq`. The sampler places member `o` at `X^(pl o)` and sums the members (refresh_top's
`sampler_carrier_gen` without the move down, whose members end before their finish).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open scoped BigOperators

variable {n : Nat}

/-- **A sampler at one modulus**: the placed and summed members encode
`Δ' Σ_o X^(pl o) ν_o` against `gq`, with the members' bounds summed. -/
theorem sampler_flat (hn : 1 < n) {q m' M : Nat} {sm sp Δ' : ErrorPoly n}
    (ν : Fin M → ErrorPoly n) (pl : Fin M → Nat) (hpl : ∀ o, pl o < 2 * n)
    {gq : ExactMatrix q n 1 m'}
    {Ks : Fin M → Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin M → Fin 2 → ExactMatrix q n 1 m'}
    {Bm : Nat}
    (hmem : ∀ o, Rel sm (Ks o) (fun j ↦ reducePoly q n (Δ' * ν o * unitPair sp j) • gq) (cs o) Bm) :
    Rel sm (fun k j ↦ ∑ o, multiplyMonomial (Ks o k j) (pl o : Int))
      (fun j ↦ reducePoly q n (Δ' * (∑ o, Xe (n := n) (pl o) * ν o) * unitPair sp j) • gq)
      (fun j ↦ ∑ o, multiplyMonomial (cs o j) (pl o : Int)) (M * Bm) := by
  have hmem' : ∀ o ∈ (Finset.univ : Finset (Fin M)), Rel sm
      (fun a j ↦ multiplyMonomial (Ks o a j) (pl o : Int))
      (fun j ↦ reducePoly q n (Xe (n := n) (pl o)) •
        (reducePoly q n (Δ' * ν o * unitPair sp j) • gq))
      (fun j ↦ multiplyMonomial (cs o j) (pl o : Int)) Bm :=
    fun o _ ↦ rel_monomial_gen hn (hmem o) (hpl o)
  have hsum := Rel.sum Finset.univ hmem'
  have hk : (∑ o, fun a j ↦ multiplyMonomial (Ks o a j) (pl o : Int)) =
      fun a j ↦ ∑ o, multiplyMonomial (Ks o a j) (pl o : Int) := by
    funext a j; simp only [Finset.sum_apply]
  have hr : (∑ o, fun j ↦ multiplyMonomial (cs o j) (pl o : Int)) =
      fun j ↦ ∑ o, multiplyMonomial (cs o j) (pl o : Int) := by
    funext j; simp only [Finset.sum_apply]
  have ht : (∑ o, fun j ↦ reducePoly q n (Xe (n := n) (pl o)) •
      (reducePoly q n (Δ' * ν o * unitPair sp j) • gq)) =
      fun j ↦ reducePoly q n (Δ' * (∑ o, Xe (n := n) (pl o) * ν o) * unitPair sp j) • gq := by
    funext j
    rw [Finset.sum_apply]
    simp only [smul_smul, ← map_mul]
    rw [← Finset.sum_smul, ← map_sum]
    congr 2
    rw [Finset.mul_sum, Finset.sum_mul]
    exact Finset.sum_congr rfl fun o _ ↦ by ring
  rw [hk, hr, ht] at hsum
  simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin] using hsum

end Dio2Proof
