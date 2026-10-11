import TopLemmas

/-!
# PRF members

A PRF member sums its cells; with one-hot selectors the sum is the selected cell.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem pair_sum {α ι : Type} [AddCommMonoid α] (s : Finset ι) (x y : ι → α) :
    (∑ k ∈ s, ![x k, y k]) = ![∑ k ∈ s, x k, ∑ k ∈ s, y k] := by
  funext j
  fin_cases j <;> simp [Finset.sum_apply]

theorem blocks_sum {α ι : Type} [AddCommMonoid α] (s : Finset ι) (a b c d : ι → α) :
    (∑ k ∈ s, ![![a k, b k], ![c k, d k]]) =
      ![![∑ k ∈ s, a k, ∑ k ∈ s, b k], ![∑ k ∈ s, c k, ∑ k ∈ s, d k]] := by
  funext i j
  fin_cases i <;> fin_cases j <;> simp [Finset.sum_apply]

/-- The cells of a member with one-hot selectors sum to the selected cell. -/
theorem onehot_collapse {q m w : Nat} (h : ExactMatrix q N 1 m) (sp : ErrorPoly N) (p : Nat)
    (β : Fin w → ErrorPoly N) (s : Fin w → Int) (sel : Fin w)
    (hs : ∀ k, s k = if k = sel then 1 else 0) :
    (∑ k : Fin w, fun j ↦ reducePoly q N (((p : Nat) : Int) * (β k * ((s k : Int) : ErrorPoly N)) *
      unitPair sp j) • h) = fun j ↦ reducePoly q N (((p : Nat) : Int) * β sel * unitPair sp j) • h := by
  rw [Finset.sum_eq_single sel]
  · simp [hs]
  · intro k _ hk
    funext j
    simp [hs, hk]
  · simp

/-- The slot of cell `cellIndex o k`. -/
theorem cellIndex_mod {members w : Nat} (o : Fin members) (k : Fin w) :
    (cellIndex o k).val % w = k.val := by
  simp only [cellIndex]
  rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt k.isLt]

/-- The member of cell `cellIndex o k`. -/
theorem cellIndex_div {members w : Nat} (o : Fin members) (k : Fin w) :
    (cellIndex o k).val / w = o.val := by
  simp only [cellIndex]
  have hw : 0 < w := Nat.lt_of_le_of_lt (Nat.zero_le _) k.isLt
  rw [Nat.add_comm, Nat.add_mul_div_left _ _ hw, Nat.div_eq_of_lt k.isLt, Nat.zero_add]

theorem Xe_mul_Xe (a b : Nat) : Xe (n := N) a * Xe (n := N) b = Xe (n := N) (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

theorem polyNorm_Xe_mul (a b : Nat) : polyNorm (Xe (n := N) a * Xe (n := N) b) ≤ 1 := by
  rw [Xe_mul_Xe]
  exact polyNorm_Xe hN1 _

end RefreshTopProof
