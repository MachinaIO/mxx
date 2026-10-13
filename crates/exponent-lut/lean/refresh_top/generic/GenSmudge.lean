import ExponentRuntime

/-!
# The decoder smudge for any digit count and base

The smudge of one coefficient is read from `d` PRF digits `δ_r`, each an odd value of
`[-(P - 1), P - 1]` (the digit table `2 f - 1` through the negacyclic wrap, `decoder/decoding.rs`),
weighted by `2^(-1) P^r` modulo the odd decoding modulus: the weight of digit `0` is
`h = 2^(-1)` (`2 h ≡ 1`) and the weight of digit `r ≥ 1` is the integer `P^r / 2`. After the
public constant `h` is subtracted, the coefficient is

  `smudgeInt P d δ = (Σ_r P^r δ_r - 1) / 2 = U - P^d / 2`, `U = Σ_r P^r u_r`, `u_r = (δ_r + P - 1) / 2`,

with `0 ≤ U < P^d`, so `-P^d / 2 ≤ smudgeInt < P^d / 2` exactly.

* `smudge_two_mul`, `smudgeInt_eq_U`, `smudgeU_range`, `smudge_bound`: the value and its bound.
* `smudge_split`: the weighted form `(δ_0 - 1) / 2 + Σ_(r ≥ 1) (P^r / 2) δ_r` of the scheme.
* `half_digit_gen`, `smudge_weighted`: the subtraction of `h` modulo the decoding modulus.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

/-- The smudge of the digits `δ` in base `P`: `(Σ_r P^r δ_r - 1) / 2`. -/
def smudgeInt (P d : Nat) (δ : Fin d → Int) : Int := (∑ r : Fin d, (P : Int) ^ r.val * δ r - 1) / 2

/-- The uniform value `U = Σ_r P^r u_r` of the digits `u_r = (δ_r + P - 1) / 2 ∈ [0, P)`. -/
def smudgeU (P d : Nat) (δ : Fin d → Int) : Int :=
  ∑ r : Fin d, (P : Int) ^ r.val * ((δ r + ((P : Int) - 1)) / 2)

/-- The digit sum is odd: `Σ_r P^r δ_r = 2 smudgeInt + 1`. -/
theorem smudge_two_mul {P d : Nat} (hP : P % 2 = 0) (δ : Fin (d + 1) → Int) (hδ : δ 0 % 2 = 1) :
    ∑ r : Fin (d + 1), (P : Int) ^ r.val * δ r = 2 * smudgeInt P (d + 1) δ + 1 := by
  have hodd : (∑ r : Fin (d + 1), (P : Int) ^ r.val * δ r) % 2 = 1 := by
    rw [Fin.sum_univ_succ]
    obtain ⟨k, hk⟩ : (2 : Int) ∣ ∑ r : Fin d, (P : Int) ^ (r.succ : Fin (d + 1)).val * δ r.succ := by
      refine Finset.dvd_sum fun r _ ↦ Dvd.dvd.mul_right ?_ _
      rw [Fin.val_succ, pow_succ]
      exact Dvd.dvd.mul_left (by omega) _
    rw [hk]
    simp only [Fin.val_zero, pow_zero, one_mul]
    omega
  unfold smudgeInt
  omega

/-- The smudge is `U - P^d / 2`. -/
theorem smudgeInt_eq_U {P d : Nat} (hP : P % 2 = 0) (δ : Fin (d + 1) → Int)
    (hδ : ∀ r, δ r % 2 = 1) :
    smudgeInt P (d + 1) δ = smudgeU P (d + 1) δ - (P : Int) ^ (d + 1) / 2 := by
  have h2 := smudge_two_mul hP δ (hδ 0)
  have hu : ∀ r, 2 * ((δ r + ((P : Int) - 1)) / 2) = δ r + ((P : Int) - 1) := by
    intro r
    have := hδ r
    have hP' : (P : Int) % 2 = 0 := by omega
    omega
  have hU : 2 * smudgeU P (d + 1) δ = ∑ r : Fin (d + 1), (P : Int) ^ r.val * δ r + ((P : Int) ^ (d + 1) - 1) := by
    unfold smudgeU
    rw [Finset.mul_sum]
    have hgeom : ∑ r : Fin (d + 1), (P : Int) ^ r.val * ((P : Int) - 1) = (P : Int) ^ (d + 1) - 1 := by
      rw [← Finset.sum_mul, Fin.sum_univ_eq_sum_range (fun i ↦ (P : Int) ^ i), geom_sum_mul]
    rw [← hgeom, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun r _ ↦ ?_
    rw [mul_left_comm, hu]
    ring
  have hPd : (P : Int) ^ (d + 1) % 2 = 0 := by
    rw [pow_succ]
    have hP' : (P : Int) % 2 = 0 := by omega
    rw [Int.mul_emod, hP', mul_zero, Int.zero_emod]
  omega

/-- `0 ≤ U < P^d` for digits `u_r ∈ [0, P)`. -/
theorem smudgeU_range {P d : Nat} (δ : Fin d → Int)
    (hu : ∀ r, 0 ≤ (δ r + ((P : Int) - 1)) / 2 ∧ (δ r + ((P : Int) - 1)) / 2 ≤ (P : Int) - 1) :
    0 ≤ smudgeU P d δ ∧ smudgeU P d δ ≤ (P : Int) ^ d - 1 := by
  unfold smudgeU
  constructor
  · exact Finset.sum_nonneg fun r _ ↦ mul_nonneg (pow_nonneg (by omega) _) (hu r).1
  · calc ∑ r : Fin d, (P : Int) ^ r.val * ((δ r + ((P : Int) - 1)) / 2)
        ≤ ∑ r : Fin d, (P : Int) ^ r.val * ((P : Int) - 1) :=
          Finset.sum_le_sum fun r _ ↦ mul_le_mul_of_nonneg_left (hu r).2 (pow_nonneg (by omega) _)
      _ = (P : Int) ^ d - 1 := by
          rw [← Finset.sum_mul, Fin.sum_univ_eq_sum_range (fun i ↦ (P : Int) ^ i), geom_sum_mul]

/-- **The smudge bound.** For odd digits `|δ_r| ≤ P - 1` with `P` even, the smudge lies in
`[-P^d / 2, P^d / 2)`. -/
theorem smudge_bound {P d : Nat} (hP : P % 2 = 0) (δ : Fin (d + 1) → Int)
    (hδ : ∀ r, δ r % 2 = 1) (hδb : ∀ r, |δ r| ≤ (P : Int) - 1) :
    -((P : Int) ^ (d + 1) / 2) ≤ smudgeInt P (d + 1) δ ∧
      smudgeInt P (d + 1) δ < (P : Int) ^ (d + 1) / 2 := by
  have hr := smudgeU_range (P := P) δ fun r ↦ by
    have := abs_le.1 (hδb r); constructor <;> omega
  have he := smudgeInt_eq_U hP δ hδ
  have hPd : (P : Int) ^ (d + 1) % 2 = 0 := by
    rw [pow_succ]
    have hP' : (P : Int) % 2 = 0 := by omega
    rw [Int.mul_emod, hP', mul_zero, Int.zero_emod]
  constructor <;> omega

theorem smudge_natAbs_le {P d : Nat} (hP : P % 2 = 0) (δ : Fin (d + 1) → Int)
    (hδ : ∀ r, δ r % 2 = 1) (hδb : ∀ r, |δ r| ≤ (P : Int) - 1) :
    (smudgeInt P (d + 1) δ).natAbs ≤ P ^ (d + 1) / 2 := by
  have h := smudge_bound hP δ hδ hδb
  have : ((P ^ (d + 1) / 2 : Nat) : Int) = (P : Int) ^ (d + 1) / 2 := by push_cast; rfl
  omega

/-- The scheme's weighted form: `(δ_0 - 1) / 2 + Σ_(r ≥ 1) (P^r / 2) δ_r` is the smudge. -/
theorem smudge_split {P d : Nat} (hP : P % 2 = 0) (δ : Fin (d + 1) → Int) (hδ : δ 0 % 2 = 1) :
    (δ 0 - 1) / 2 + ∑ r : Fin d, ((P : Int) ^ (r.val + 1) / 2) * δ r.succ = smudgeInt P (d + 1) δ := by
  have h2 := smudge_two_mul hP δ hδ
  rw [Fin.sum_univ_succ] at h2
  have hw : ∀ r : Fin d, (P : Int) ^ (r.val + 1) = 2 * ((P : Int) ^ (r.val + 1) / 2) := by
    intro r
    have hP' : (P : Int) % 2 = 0 := by omega
    have : (P : Int) ^ (r.val + 1) % 2 = 0 := by
      rw [pow_succ, Int.mul_emod, hP', mul_zero, Int.zero_emod]
    omega
  have hs : ∑ r : Fin d, (P : Int) ^ (r.succ : Fin (d + 1)).val * δ r.succ =
      2 * ∑ r : Fin d, ((P : Int) ^ (r.val + 1) / 2) * δ r.succ := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl fun r _ ↦ ?_
    rw [Fin.val_succ]
    conv_lhs => rw [hw r]
    ring
  rw [hs] at h2
  simp only [Fin.val_zero, pow_zero, one_mul] at h2
  omega

/-- The half constant cancels: with `2 h ≡ 1 (mod q)`, `h t - h ≡ (t - 1) / 2` for odd `t`. -/
theorem half_digit_gen {n q : Nat} {h : Int} (h2 : ((2 * h : Int) : ZMod q) = 1) (t : Int)
    (ht : t % 2 = 1) :
    reducePoly q n (((h * t - h : Int) : ErrorPoly n)) = reducePoly q n (((t - 1) / 2 : Int) : ErrorPoly n) := by
  obtain ⟨k, hk⟩ : ∃ k, t = 2 * k + 1 := ⟨t / 2, by omega⟩
  subst hk
  rw [show (2 * k + 1 - 1) / 2 = k by omega, reducePoly_intCast, reducePoly_intCast,
    intCast_eq_algebraMap, intCast_eq_algebraMap]
  congr 1
  push_cast at h2 ⊢
  linear_combination k * h2

/-- **The weighted smudge modulo the decoding modulus.** With the weights `w_0 = h`
(`2 h ≡ 1 (mod q)`) and `w_r = P^r / 2` (`r ≥ 1`), `Σ_r w_r δ_r - h ≡ smudgeInt P d δ`. -/
theorem smudge_weighted {n q P d : Nat} (hP : P % 2 = 0) {h : Int} (h2 : ((2 * h : Int) : ZMod q) = 1)
    (w : Fin (d + 1) → Int) (hw0 : w 0 = h) (hw : ∀ r : Fin d, w r.succ = (P : Int) ^ (r.val + 1) / 2)
    (δ : Fin (d + 1) → Int) (hδ : δ 0 % 2 = 1) :
    reducePoly q n (((∑ r, w r * δ r - h : Int) : ErrorPoly n)) =
      reducePoly q n ((smudgeInt P (d + 1) δ : Int) : ErrorPoly n) := by
  rw [← smudge_split hP δ hδ, Fin.sum_univ_succ, hw0,
    show h * δ 0 + ∑ r : Fin d, w r.succ * δ r.succ - h =
      (h * δ 0 - h) + ∑ r : Fin d, w r.succ * δ r.succ by ring,
    Int.cast_add, map_add, half_digit_gen h2 _ hδ, ← map_add, ← Int.cast_add]
  simp only [hw]

end RefreshTopProof.Gen
