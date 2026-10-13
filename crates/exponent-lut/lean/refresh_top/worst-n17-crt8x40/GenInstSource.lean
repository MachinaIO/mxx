import GenFinalValues
import GenBridge_LutKey

/-!
# The source LUT of the action input (WP D)

The two-branch exponent table `keyTableG [a] [b] (Σ_j keyTermE σ 1 t 1 (-1) j)` with `t = (1, 0)`
and automorphisms `σ ∈ {1, n + 1}`: its coefficients (`src_coef`) and its Walsh sum on a
plaintext `X^e`, `e ∈ {0, 1}`: `2 a b X^(1 - e)` (`src_walsh`).
-/

namespace RefreshTopProof.GenInst

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenIface RefreshTopProof.GenIface.BrLutKey
  RefreshTopProof.Gen
open scoped BigOperators

variable {n : Nat}

/-- **A coefficient of the source table**: `a b (X + σ(X^(2n-1)))`. -/
theorem src_coef {q : Nat} (hq : 1 < q) (hn : 0 < n) {σ : Int} (hodd : Odd σ.toNat) (a b : Int) :
    (keyTableG [a] [b] (∑ k ∈ Finset.range ((2 : Int).toNat), keyTermE σ 1
      (fun j : Fin 2 ↦ MxxIR.selectInt (Int.ofNat (j : Nat)) [1, 0] + 0)
      (matrixPolynomial [1] : ExactMatrix q n 1 1) (-1) k) : ExactMatrix q n 1 1) 0 0 =
      (a : ExactPoly q n) * (b : ExactPoly q n) * (X q ^ 1 + X q ^ ((2 * n - 1) * σ.toNat)) := by
  have h2 : ((2 : Int).toNat) = 2 := rfl
  rw [h2, Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_zero, zero_add]
  have e1 : ((1 : Int) % (2 * (n : Int))).toNat = 1 := by
    rw [Int.emod_eq_of_lt (by norm_num) (by omega)]; rfl
  have e0 : ((0 : Int) % (2 * (n : Int))).toNat = 0 := by simp
  have em : ((-1 : Int) % (2 * (n : Int))).toNat = 2 * n - 1 := by
    have : (-1 : Int) % (2 * (n : Int)) = 2 * (n : Int) - 1 := by
      rw [show (-1 : Int) = (2 * (n : Int) - 1) + (-1) * (2 * (n : Int)) by ring,
        Int.add_mul_emod_self_right, Int.emod_eq_of_lt (by omega) (by omega)]
    rw [this]; omega
  have mp : ∀ k : Int, (matrixPolynomial [k] : ExactMatrix q n 1 1) 0 0 = (k : ExactPoly q n) := by
    intro k; simp [matrixPolynomial]
  simp only [keyTableG, keyTermE, dif_pos (show (0 : Nat) < 2 by norm_num),
    dif_pos (show (1 : Nat) < 2 by norm_num), matrixMulScalarLeft, Matrix.add_apply, multiplyMonomial,
    ringAutomorphism_eq hq hn σ.toNat hodd, mp]
  simp only [Int.ofNat_eq_natCast, Nat.cast_zero, Nat.cast_one, zero_mul, one_mul, mul_one,
    show MxxIR.selectInt 0 [1, 0] = 1 from rfl, show MxxIR.selectInt 1 [1, 0] = 0 from rfl,
    add_zero, e0, e1, em, pow_zero, Int.cast_one]
  rw [substitution_root_pow, map_one]
  ring

/-- **The Walsh sum of the source table** on the plaintext `X^e`, `e ∈ {0, 1}`, with the
automorphisms `1` and `n + 1`: `2 a b X^(1 - e)`. -/
theorem src_walsh {q : Nat} (hq : 1 < q) (hn : 0 < n) (σs : Fin 2 → Int) (hodd : ∀ k, Odd (σs k).toNat)
    (h0 : σs 0 = 1) (h1 : σs 1 = (n : Int) + 1) (a b : Int) {e : Int} (he : e = 0 ∨ e = 1)
    (C : Fin 2 → ExactPoly q n)
    (hC : ∀ k, C k = (a : ExactPoly q n) * (b : ExactPoly q n) * (X q ^ 1 + X q ^ ((2 * n - 1) * (σs k).toNat))) :
    ∑ k, reducePoly q n (substitution (σs k).toNat (hodd k) (message (n := n) 1 e)) * C k =
      reducePoly q n (((2 * a * b : Int) : ErrorPoly n) * Xe (n := n) (1 - e).toNat) := by
  have t0 : (σs 0).toNat = 1 := by rw [h0]; rfl
  have t1 : (σs 1).toNat = n + 1 := by rw [h1]; omega
  have hr : (X q : ExactPoly q n) ^ n = -1 := Negacyclic.root_pow_n
  have r2n : (X q : ExactPoly q n) ^ (2 * n) = 1 := by rw [mul_comm, pow_mul, hr]; norm_num
  have rodd : (X q : ExactPoly q n) ^ ((2 * n - 1) * n) = -1 := by
    rw [mul_comm, pow_mul, hr]; exact Odd.neg_one_pow ⟨n - 1, by omega⟩
  have hsub : ∀ (σ : Nat) (ho : Odd σ) (k : Nat),
      reducePoly q n (substitution σ ho (Xe (n := n) k)) = X q ^ (k * σ) := by
    intro σ ho k
    rw [reducePoly_substitution, reducePoly_Xe, substitution_root_pow]
  have hmsg : ∀ k : Nat, k < 2 * n → message (n := n) 1 (k : Int) = Xe (n := n) k := by
    intro k hk
    unfold message
    rw [Int.emod_eq_of_lt (by omega) (by push_cast; omega)]
    simp [Xe]
  rw [Fin.sum_univ_two, hC 0, hC 1]
  rcases he with rfl | rfl
  · rw [show ((0 : Int)) = ((0 : Nat) : Int) from rfl, hmsg 0 (by omega), hsub, hsub, t0, t1]
    rw [map_mul, map_intCast, show ((1 - ((0 : Nat) : Int)).toNat) = 1 from rfl, reducePoly_Xe]
    have e1 : (X q : ExactPoly q n) ^ ((2 * n - 1) * (n + 1)) = -(X q ^ ((2 * n - 1) * 1)) := by
      rw [Nat.mul_add, pow_add, rodd]; ring
    rw [e1]; push_cast; ring
  · have m1 : message (n := n) 1 1 = Xe (n := n) 1 := hmsg 1 (by omega)
    rw [m1, hsub, hsub, t0, t1]
    rw [map_mul, map_intCast, show ((1 - (1 : Int)).toNat) = 0 from rfl, reducePoly_Xe]
    have e1 : (X q : ExactPoly q n) ^ (1 * (n + 1)) = -(X q ^ 1) := by
      rw [one_mul, pow_add, hr]; ring
    have e2 : (X q : ExactPoly q n) ^ (1 * (n + 1)) * (X q ^ ((2 * n - 1) * (n + 1))) = 1 := by
      rw [← pow_add, show 1 * (n + 1) + (2 * n - 1) * (n + 1) = 2 * n * (n + 1) by
        rw [← Nat.add_mul]; congr 1; omega, pow_mul, r2n, one_pow]
    have e3 : (X q : ExactPoly q n) ^ (1 * 1) * X q ^ ((2 * n - 1) * 1) = 1 := by
      rw [← pow_add, show 1 * 1 + (2 * n - 1) * 1 = 2 * n by omega, r2n]
    push_cast
    linear_combination ((a : ExactPoly q n) * (b : ExactPoly q n)) * e2 +
      ((a : ExactPoly q n) * (b : ExactPoly q n)) * e3 +
      ((a : ExactPoly q n) * (b : ExactPoly q n) * X q) * e1

/-- A selection from a list of zeros is zero. -/
theorem selectInt_zeros {l : List Int} (hl : ∀ x ∈ l, x = 0) (s : Int) : MxxIR.selectInt s l = 0 := by
  unfold MxxIR.selectInt
  cases h : l[s.toNat]? with
  | none => rfl
  | some x => exact hl x (List.mem_of_getElem? h)

end RefreshTopProof.GenInst
