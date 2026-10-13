import TopLemmas
import ExponentIterate
import Requirement_0
import Claim

/-!
# The requirement

The requirement checks that every member's selectors sum to one, the label's encodings, the
expected exponent, and the coefficients of `σ`. With the selectors in `[0, 1]` (the externals'
contract), a sum of one is a one-hot vector.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem ite_add_ite_eq_two (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then (1 : Int) else 0) + (if q then 1 else 0) = 2) ↔ p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

theorem familyGet_int {α : Type} {count : Nat} {f : Fin count → α} {z : Int} {out : α}
    (h : familyGetDynamic f z out) (hz : 0 ≤ z) (hlt : z < count) :
    ∃ hk : z.toNat < count, out = f ⟨z.toNat, hk⟩ := by
  obtain ⟨p, hp, rfl⟩ := h
  refine ⟨by omega, ?_⟩
  congr 1
  ext
  simp only
  omega

/-- Selectors in `[0, 1]` summing to one are one-hot. -/
theorem onehot_of_sum {w : Nat} {s : Fin w → Int} (hs : ∀ k, 0 ≤ s k ∧ s k ≤ 1)
    (hsum : ∑ k, s k = 1) : ∃ sel : Fin w, ∀ k, s k = if k = sel then 1 else 0 := by
  by_contra hne
  push_neg at hne
  have h01 : ∀ k, s k = 0 ∨ s k = 1 := fun k ↦ by have := hs k; omega
  by_cases hex : ∃ k, s k = 1
  · obtain ⟨sel, hsel⟩ := hex
    obtain ⟨k, hk⟩ := hne sel
    have hk1 : k ≠ sel := fun h ↦ by subst h; simp [hsel] at hk
    simp only [hk1, if_false] at hk
    have hk' : s k = 1 := (h01 k).resolve_left hk
    have : s sel + s k ≤ ∑ j, s j := by
      rw [← Finset.sum_pair hk1.symm]
      exact Finset.sum_le_sum_of_subset_of_nonneg (by simp) fun j _ _ ↦ (hs j).1
    omega
  · push_neg at hex
    have : ∑ j, s j = 0 := Finset.sum_eq_zero fun j _ ↦ (h01 j).resolve_right (hex j)
    omega

end RefreshTopProof
