import TfheRun

/-!
# The phase of an extracted and key-switched ciphertext

`ks_phase`: when the ring ciphertext `(c₀, c₁)` has the constant phase `y` under the ring secret
`z` (`c₁ - c₀ z ≡ y` in the constant coefficient), the key switch of its sample extraction
decrypts under the LWE secret `s` to `η ≡ (q/Q) y - Σ dᵢ εᵢ` up to the rounding allowance: the
switch from `Q` to `q` and the rounding of each extracted mask entry to its top 16 bits.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity

/-- The deterministic rounding allowance of one extraction and key switch: the switch from `Q`
to `q` (`1025 (Q - 1) / (2 Q)`) and the rounding of the extracted mask (`1024 · 2^15`). -/
abbrev allowance : ℝ := 33554945

theorem cast_emod_q (a : Int) : ((a % 4294967296 : Int) : ZMod q) = a := ZMod.intCast_mod a q

theorem cast_emod_Q (a : Int) : ((a % 5234636801 : Int) : ZMod Q) = a := ZMod.intCast_mod a Q

/-- Rounding `x q / Q` to the nearest integer moves `Q` times the result by at most `Q / 2`. -/
theorem switch_bound (x : Int) :
    |5234636801 * ((x * 4294967296 + 2617318400) / 5234636801) - 4294967296 * x| ≤
      2617318400 := by
  rw [abs_le]
  omega

/-- Eight base-four digits rebuild a 16-bit value. -/
theorem digits_four {t : Int} (h0 : 0 ≤ t) (h1 : t < 65536) :
    ∑ k : Fin 8, t / 4 ^ k.val % 4 * 4 ^ k.val = t := by
  simp only [Fin.sum_univ_eight, Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two]
  norm_num
  omega

/-- Regroup a sum over flat key-switching digits by ring coefficient. -/
theorem sum_digits_reindex (f : Fin 8192 → Int) (g : Fin 1024 → Int) :
    ∑ i : Fin 8192, f i * (g ⟨i.val / 8, by omega⟩ * 4 ^ (i.val % 8)) =
      ∑ j : Fin 1024, g j * ∑ k : Fin 8, f ⟨j.val * 8 + k.val, by omega⟩ * 4 ^ k.val := by
  rw [← (finProdFinEquiv : Fin 1024 × Fin 8 ≃ Fin 8192).sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  have hjk : (finProdFinEquiv (j, k) : Fin 8192) = ⟨j.val * 8 + k.val, by omega⟩ := by
    apply Fin.ext
    simp only [finProdFinEquiv_apply_val]
    ring
  rw [hjk]
  have hdiv : (⟨(j.val * 8 + k.val) / 8, by omega⟩ : Fin 1024) = j := Fin.ext (by simp only; omega)
  have hmod : (j.val * 8 + k.val) % 8 = k.val := by omega
  simp only [hdiv, hmod]
  ring

/-- The top 16 bits of a mask entry, rounded. -/
def ksTop (x : Int) : Int := (x + 32768) / 65536 % 65536

theorem digitAt_eq (a : Fin 1024 → Int) (i : Fin 8192) :
    digitAt a i = ksTop (a ⟨i.val / 8, by omega⟩) / 4 ^ (i.val % 8) % 4 := by
  unfold digitAt ksTop
  congr 3

theorem extractAt_range (c : Fin 1024 → Int) (k : Fin 1024) :
    0 ≤ extractAt c k ∧ extractAt c k < 4294967296 := by
  unfold extractAt rescale
  omega

theorem ks_phase {s : Fin 720 → Int} {z : Fin 1024 → Int} (hz : ∀ k, z k = 0 ∨ z k = 1)
    {A : Fin 5898240 → Int} {B ε : Fin 8192 → Int}
    (hB : ∀ i, B i = (intMatrixVectorProduct false A s i + z ⟨i.val / 8, by omega⟩ *
      kskPow ⟨i.val % 8, by omega⟩ + ε i) % 4294967296)
    (c0 c1 : ExactMatrix Q N 1 1) {zp : ExactPoly Q N} (hzp : ∀ k, zp.coeff k = (z k : ZMod Q))
    {y : Int} (hy : (y : ZMod Q) = (c1 0 0).coeff 0 - (c0 0 0 * zp).coeff 0) :
    ∃ η : Int,
      ((ksBody B (rescale (coeffVals c1 0)) (digitAt (extractAt (coeffVals c0))) -
          ∑ j, ksMask A (digitAt (extractAt (coeffVals c0))) j * s j : Int) : ZMod q) = η ∧
      |(η : ℝ) - ((4294967296 / 5234636801 : ℝ) * y -
        ∑ i, (digitAt (extractAt (coeffVals c0)) i : ℝ) * ε i)| ≤ allowance := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  set c0v := coeffVals c0 with hc0v
  set c1v0 := coeffVals c1 0 with hc1v0
  set a' := extractAt c0v with ha'
  set d := digitAt a' with hd
  -- The constant coefficient of `c₀ z` over the integers.
  set sg : Fin 1024 → Int := fun k ↦ if k.val = 0 then 1 else -1 with hsg
  set γ : Fin 1024 → Int := fun k ↦ (c0v (extractIndex k) * sg k) % 5234636801 with hγ
  set rγ : Fin 1024 → Int := fun k ↦ (γ k * 4294967296 + 2617318400) / 5234636801 with hrγ
  have ha'γ (k : Fin 1024) : a' k = rγ k % 4294967296 := by
    simp only [ha', extractAt, rescale, hrγ, hγ, hsg]
    congr 4
    by_cases hk : k.val = 0 <;> simp [hk]
  have hzp' : zp = reducePoly Q N (intPoly z) := by
    apply negacyclic_ext (by decide)
    intro k
    rw [hzp, reducePoly_coeff (by decide) (by decide), intPoly_coeff (by decide)]
  have hc0 : c0 0 0 = reducePoly Q N (canonicalLift (c0 0 0)) :=
    (reducePoly_canonicalLift (by decide) (by decide) _).symm
  have hcz : (c0 0 0 * zp).coeff 0 = ((∑ k : Fin 1024, γ k * z k : Int) : ZMod Q) := by
    rw [hzp', hc0, ← map_mul, reducePoly_coeff (by decide) (by decide)]
    have h0 := coeff_zero_mul (n := 1024) (by decide) (canonicalLift (c0 0 0)) (intPoly z)
    simp only [canonicalLift_coeff (by decide : 0 < 1024), intPoly_coeff (by decide : 0 < 1024)]
      at h0
    rw [show (0 : Fin 1024) = ⟨0, by decide⟩ from rfl, h0]
    push_cast
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    simp only [hγ, cast_emod_Q]
    push_cast
    rw [hc0v]
    simp only [coeffVals, extractIndex, hsg]
    by_cases hk : k.val = 0 <;> simp [hk] <;> ring
  have hy' : ((c1v0 - ∑ k : Fin 1024, γ k * z k : Int) : ZMod Q) = (y : ZMod Q) := by
    rw [hy, hcz]
    push_cast
    simp [hc1v0, coeffVals]
  obtain ⟨l, hl⟩ := exists_of_cast_eq hy'
  -- Switching from `Q` to `q`.
  have hc1range : 0 ≤ c1v0 ∧ c1v0 < 5234636801 := by
    have h : ((c1 0 0).coeff 0).val < 5234636801 := ZMod.val_lt _
    simp only [hc1v0, coeffVals]
    omega
  set rbQ : Int := (c1v0 * 4294967296 + 2617318400) / 5234636801 with hrbQ
  have hresc : rescale c1v0 = rbQ % 4294967296 := by
    simp only [rescale, hrbQ, Int.emod_eq_of_lt hc1range.1 hc1range.2]
  set P : Int := rbQ - ∑ k, rγ k * z k - 4294967296 * l with hP
  set D : Int := (5234636801 * rbQ - 4294967296 * c1v0) -
    ∑ k, (5234636801 * rγ k - 4294967296 * γ k) * z k with hD
  have hσ : ∑ k, (5234636801 * rγ k - 4294967296 * γ k) * z k =
      5234636801 * ∑ k, rγ k * z k - 4294967296 * ∑ k, γ k * z k := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun k _ ↦ by ring
  have hPD : 5234636801 * P = 4294967296 * y + D := by
    rw [hD, hσ, hP]
    linear_combination 4294967296 * hl
  have hDbound : |D| ≤ 1025 * 2617318400 := by
    have hσb : |∑ k, (5234636801 * rγ k - 4294967296 * γ k) * z k| ≤ 1024 * 2617318400 := by
      refine (abs_sum_le_card _ 2617318400 fun k ↦ ?_).trans (by simp)
      rw [mul_comm]
      exact abs_binary_mul_le (hz k) (switch_bound _) (by norm_num)
    have hb := switch_bound c1v0
    rw [hD]
    linarith [abs_sub (5234636801 * rbQ - 4294967296 * c1v0)
      (∑ k, (5234636801 * rγ k - 4294967296 * γ k) * z k)]
  -- Key switching.
  set tt : Fin 1024 → Int := fun k ↦ ksTop (a' k) with htt
  have hdigits (k : Fin 1024) :
      ∑ t : Fin 8, d ⟨k.val * 8 + t.val, by omega⟩ * 4 ^ t.val = tt k := by
    have hterm (t : Fin 8) : d ⟨k.val * 8 + t.val, by omega⟩ = tt k / 4 ^ t.val % 4 := by
      rw [hd, digitAt_eq]
      have hdiv : (⟨(k.val * 8 + t.val) / 8, by omega⟩ : Fin 1024) = k :=
        Fin.ext (by simp only; omega)
      have hmod : (k.val * 8 + t.val) % 8 = t.val := by omega
      simp only [hdiv, hmod, htt]
    simp only [hterm]
    have hr := extractAt_range c0v k
    exact digits_four (by simp only [htt, ksTop]; omega) (by simp only [htt, ksTop]; omega)
  have hreindex : ∑ i : Fin 8192, d i * (z ⟨i.val / 8, by omega⟩ * kskPow ⟨i.val % 8, by omega⟩ +
      ε i) = ∑ k : Fin 1024, z k * (65536 * tt k) + ∑ i, d i * ε i := by
    have h := sum_digits_reindex d z
    simp only [hdigits] at h
    calc _ = (∑ i : Fin 8192, d i * (z ⟨i.val / 8, by omega⟩ * 4 ^ (i.val % 8))) * 65536 +
          ∑ i, d i * ε i := by
          rw [Finset.sum_mul, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl fun i _ ↦ by simp only [kskPow]; ring
      _ = _ := by
          rw [h, Finset.sum_mul]
          exact congrArg₂ (· + ·) (Finset.sum_congr rfl fun k _ ↦ by ring) rfl
  set ρ : Fin 1024 → Int := fun k ↦ 65536 * ((a' k + 32768) / 65536) - a' k with hρdef
  have hρbound (k : Fin 1024) : |ρ k| ≤ 32768 := by
    have := extractAt_range c0v k
    simp only [hρdef, ha']
    rw [abs_le]
    omega
  have hq0 : ((4294967296 : Int) : ZMod q) = 0 := by exact_mod_cast ZMod.natCast_self q
  have ht_cast (k : Fin 1024) : ((65536 * tt k : Int) : ZMod q) = ((rγ k + ρ k : Int) : ZMod q) := by
    have h1 : 65536 * tt k = a' k + ρ k - 4294967296 * ((a' k + 32768) / 65536 / 65536) := by
      simp only [htt, ksTop, hρdef]
      omega
    rw [h1, Int.cast_sub, Int.cast_mul, hq0, zero_mul, sub_zero, Int.cast_add, Int.cast_add,
      ha'γ k, cast_emod_q]
  have hBd : intMatrixVectorProduct (outer := 1) true B d 0 = ∑ i, B i * d i := by
    rw [intMatrixVectorProduct_columns (by rfl)]
    exact Finset.sum_congr rfl fun i _ ↦ by simp
  have hAd (j : Fin 720) : intMatrixVectorProduct true A d j =
      ∑ i : Fin 8192, A ⟨i.val * 720 + j.val, by omega⟩ * d i := by
    rw [intMatrixVectorProduct_columns (by rfl)]
  have hAs (i : Fin 8192) : intMatrixVectorProduct false A s i =
      ∑ j : Fin 720, A ⟨i.val * 720 + j.val, by omega⟩ * s j := by
    rw [intMatrixVectorProduct_rows (by rfl)]
  have hkB (i : Fin 8192) : ((B i : Int) : ZMod q) =
      ∑ j : Fin 720, (A ⟨i.val * 720 + j.val, by omega⟩ : ZMod q) * (s j : ZMod q) +
        ((z ⟨i.val / 8, by omega⟩ * kskPow ⟨i.val % 8, by omega⟩ : Int) : ZMod q) +
        (ε i : ZMod q) := by
    rw [hB i, cast_emod_q, hAs]
    push_cast
    ring
  have hphase : ((ksBody B (rescale c1v0) d - ∑ j, ksMask A d j * s j : Int) : ZMod q) =
      ((rbQ - ∑ i : Fin 8192, d i * (z ⟨i.val / 8, by omega⟩ * kskPow ⟨i.val % 8, by omega⟩ +
        ε i) : Int) : ZMod q) := by
    have halg := ks_algebra (R := ZMod q) (rbQ : ZMod q)
      (fun i j ↦ (A ⟨i.val * 720 + j.val, by omega⟩ : ZMod q)) (fun j ↦ (s j : ZMod q))
      (fun i ↦ (d i : ZMod q))
      (fun i ↦ (z ⟨i.val / 8, by omega⟩ : ZMod q) * (kskPow ⟨i.val % 8, by omega⟩ : ZMod q))
      (fun i ↦ (ε i : ZMod q))
    simp only [ksBody, ksMask, hresc, hBd, hAd, Int.cast_sub, Int.cast_sum, Int.cast_mul,
      cast_emod_q, Int.cast_zero, hkB]
    rw [halg]
    push_cast
    ring
  refine ⟨P - ∑ k, z k * ρ k - ∑ i, d i * ε i, ?_, ?_⟩
  · rw [hphase, hreindex]
    have hsumt : ((∑ k : Fin 1024, z k * (65536 * tt k) : Int) : ZMod q) =
        ∑ k : Fin 1024, (z k : ZMod q) * ((rγ k : Int) : ZMod q) +
          ∑ k : Fin 1024, (z k : ZMod q) * ((ρ k : Int) : ZMod q) := by
      rw [Int.cast_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun k _ ↦ by rw [Int.cast_mul, ht_cast, Int.cast_add]; ring
    have hq0' : (4294967296 : ZMod q) = 0 := by simpa using hq0
    have hPcast : ((P : Int) : ZMod q) =
        (rbQ : ZMod q) - ∑ k, (z k : ZMod q) * ((rγ k : Int) : ZMod q) := by
      rw [hP]
      push_cast
      rw [hq0']
      simp only [mul_comm (z _ : ZMod q)]
      ring
    rw [Int.cast_sub, Int.cast_add, hsumt]
    push_cast
    rw [hPcast]
    ring
  · have hPDr : (5234636801 : ℝ) * (P : ℝ) = 4294967296 * (y : ℝ) + (D : ℝ) := by
      exact_mod_cast hPD
    have hZρ : |∑ k, z k * ρ k| ≤ 1024 * 32768 :=
      (abs_sum_le_card _ _ fun k ↦ abs_binary_mul_le (hz k) (hρbound k) (by norm_num)).trans
        (by simp)
    have hdiff : ((P - ∑ k, z k * ρ k - ∑ i, d i * ε i : Int) : ℝ) -
        ((4294967296 / 5234636801 : ℝ) * y - ∑ i, (d i : ℝ) * ε i) =
        (D : ℝ) / 5234636801 - ((∑ k, z k * ρ k : Int) : ℝ) := by
      push_cast
      field_simp
      linear_combination hPDr
    rw [hdiff]
    have hDr : |(D : ℝ)| ≤ 1025 * 2617318400 := by exact_mod_cast hDbound
    have hZρr : |((∑ k, z k * ρ k : Int) : ℝ)| ≤ 1024 * 32768 := by exact_mod_cast hZρ
    calc |(D : ℝ) / 5234636801 - ((∑ k, z k * ρ k : Int) : ℝ)|
        ≤ |(D : ℝ) / 5234636801| + |((∑ k, z k * ρ k : Int) : ℝ)| := abs_sub _ _
      _ = |(D : ℝ)| / 5234636801 + |((∑ k, z k * ρ k : Int) : ℝ)| := by
          rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 5234636801)]
      _ ≤ allowance := by
          have : |(D : ℝ)| / 5234636801 ≤ 1025 * 2617318400 / 5234636801 :=
            div_le_div_of_nonneg_right hDr (by norm_num)
          simp only [allowance]
          linarith

end MxxFheTfhe
