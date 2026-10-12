import TopSigma
import TopLink
import Part_I_Online
import TopLabel
import TopColumn

/-!
# The decode stage's inputs of a part

The PRF members select labels by one-hot selector families `s`, `s'`: member `o` reads
`z_o = Σ_k s_(o,k) a_k + Σ_k s'_(o,k) a_k`. The mask's `σ_o = 1 - ⌊(z_o mod 8) / 4⌋`, the fresh
error `e'` reads the fresh table `(0, 0, 0, 1)` at `readoutExp z_o`, and the source LUT maps
`X^e` to `X^(1 - e)`: all as the decode stage expects them.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Part_I

/-- The label exponent that member `o` of the selector families `s`, `s'` reads. -/
def zOf {m : Nat} (s s' : Fin (m * 5) → Int) (o : Fin m) : Int :=
  ∑ k : Fin 5, s (cellIndex o k) * (labelExp k : Int) + ∑ k : Fin 5, s' (cellIndex o k) * (labelExp k : Int)

/-- The mask's `σ`. -/
def sigmaOf (s s' : Fin (8 * 5) → Int) (o : Fin 8) : Int := 1 - (zOf s s' o % 8) / 4

/-- The fresh table. -/
noncomputable def freshTable (k : Fin 4) : ErrorPoly N :=
  ![((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
    ((1 : Int) : ErrorPoly N)] k

/-- The fresh error `e'`: column `k` reads members `8 k + i`. -/
noncomputable def freshE (s s' : Fin (48 * 5) → Int) : ErrorMatrix N 1 6 := fun _ k ↦
  ∑ i : Fin 8, Xe (n := N) i * sampled freshTable (readoutExp (zOf s s' ⟨i.val + 8 * k.val, by omega⟩).toNat)

theorem sel_sum {m : Nat} {s : Fin (m * 5) → Int} {sel : Fin m → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0) (o : Fin m) :
    ∑ k : Fin 5, s (cellIndex o k) * (labelExp k : Int) = labelExp (sel o) := by
  simp only [hsel, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, if_true]

theorem zOf_eq {m : Nat} {s s' : Fin (m * 5) → Int} {sel sel' : Fin m → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (o : Fin m) :
    zOf s s' o = ((labelExp (sel' o) + labelExp (sel o) : Nat) : Int) := by
  rw [zOf, sel_sum hsel, sel_sum hsel']
  push_cast
  ring

/-- The mask's value is `σ` modulo `q₀`. -/
theorem mask_sigma {s s' : Fin (8 * 5) → Int} {sel sel' : Fin 8 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (sp : ErrorPoly N) :
    reducePoly q0 N (((1 : Int) : ErrorPoly N) * (∑ o : Fin 8, Xe (n := N) o *
      (sampled (fun k ↦ ![((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
        ((536870721 : Int) : ErrorPoly N)] k) (readoutExp (labelExp (sel' o) + labelExp (sel o))) +
       sampled (fun k ↦ ![((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
        ((536870721 : Int) : ErrorPoly N)] k) 1)) * unitPair sp 1) =
      reducePoly q0 N (∑ o : Fin 8, ((sigmaOf s s' o : Int) : ErrorPoly N) * Xe (n := N) o) := by
  have h := beta_sigma (fun o ↦ labelExp (sel' o) + labelExp (sel o))
  rw [Int.cast_one, show unitPair sp 1 = 1 from rfl]
  refine h.trans ?_
  congr 1
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [sigmaOf, zOf_eq hsel hsel']

/-- The fresh values are the fresh error's entries. -/
theorem fresh_entry {s s' : Fin (48 * 5) → Int} {sel sel' : Fin 48 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0) (sp : ErrorPoly N)
    (c : Fin 6) :
    ((1 : Int) : ErrorPoly N) * (∑ i : Fin 8, Xe (n := N) i *
      sampled (fun k ↦ ![((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N),
        ((1 : Int) : ErrorPoly N)] k)
        ((fun o ↦ readoutExp (labelExp (sel' o) + labelExp (sel o))) ⟨i.val + 8 * c.val, by omega⟩)) *
      unitPair sp 1 = freshE s s' 0 c := by
  rw [Int.cast_one, one_mul, show unitPair sp 1 = 1 from rfl, mul_one, freshE]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [zOf_eq hsel hsel', Int.toNat_natCast]
  rfl

/-- The source LUT maps `X^e` to `c X^(1 - e)`, `c = 1`. -/
theorem source_message {e e' c : Int} (he : 0 ≤ e ∧ e ≤ 1) (he' : e' = 1 - e) (hc : c = 1) (r : ErrorPoly N) :
    Xe (n := N) (e.toNat - e.toNat % 2) * sourceTable ⟨e.toNat % 2, Nat.mod_lt _ (by norm_num)⟩ * r =
      message c e' * r := by
  subst he' hc
  rcases (show e = 0 ∨ e = 1 by omega) with rfl | rfl
  · simp [sourceTable, message, Xe]
  · simp [sourceTable, message, Xe]

/-- A sampled value of the fresh table is `0` or `±1`. -/
theorem sampled_fresh (z : Nat) : ∃ C : Int, C.natAbs ≤ 1 ∧ sampled freshTable z = (C : ErrorPoly N) := by
  have hv : ∀ f : Fin 4, ∃ t : Int, t.natAbs ≤ 1 ∧ freshTable f = (t : ErrorPoly N) := by
    intro f
    fin_cases f
    · exact ⟨0, by decide, rfl⟩
    · exact ⟨0, by decide, rfl⟩
    · exact ⟨0, by decide, rfl⟩
    · exact ⟨1, by decide, rfl⟩
  obtain ⟨t, ht, hf⟩ := hv ⟨z % 4, Nat.mod_lt _ (by norm_num)⟩
  refine ⟨(-1) ^ (z / 4) * t, ?_, ?_⟩
  · rw [Int.natAbs_mul, Int.natAbs_pow]
    simpa using ht
  · rw [sampled, hf, Int.cast_mul]

/-- The fresh error is small. -/
theorem freshE_bound (s s' : Fin (48 * 5) → Int) (c : Fin 6) (i : Fin N) :
    4 * |((freshE s s' 0 c).coeff i)| + 4 < (q0 : Int) := by
  have hnorm : polyNorm (freshE s s' 0 c) ≤ 8 := by
    unfold freshE
    refine (polyNorm_sum_le _ _).trans ?_
    calc ∑ j : Fin 8, polyNorm (Xe (n := N) j * sampled freshTable
          (readoutExp (zOf s s' ⟨j.val + 8 * c.val, by omega⟩).toNat))
        ≤ ∑ _j : Fin 8, 1 := Finset.sum_le_sum fun j _ ↦ by
          obtain ⟨C, hC, hs⟩ := sampled_fresh (readoutExp (zOf s s' ⟨j.val + 8 * c.val, by omega⟩).toNat)
          rw [hs]
          exact (polyNorm_Xe_int _ C).trans hC
      _ = 8 := by simp
  have h := (coeff_natAbs_le_polyNorm _ i).trans hnorm
  have : |((freshE s s' 0 c).coeff i)| ≤ 8 := by
    rw [Int.abs_eq_natAbs]
    exact_mod_cast h
  have hQ : (100 : Int) < q0 := by norm_num
  linarith

end RefreshTopProof
