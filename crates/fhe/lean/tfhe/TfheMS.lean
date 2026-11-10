import TfheBase
import MxxSecurity.TapeBounds

/-!
# The rounding error of the switch to `2 N` is sub-Gaussian

The rounding of a mask entry `w` to a multiple of `2^21` moves it by `msRound w ∈ (-2^20, 2^20]`,
a function of `w mod 2^21`. When `w = x D + C` with `x` uniform and `1 ≤ D ≤ 6`, the reflection
`x ↦ K - x` pairs the values of `msRound` around a center within `2` of zero, so the exponential
moment of `msRound` is at most `exp (2 |λ| + λ² (2^20 + 2)² / 2)`. A uniform residue modulo
`Qmask` is uniform modulo `2^21` up to the factor `1 + 2^21 / Qmask`.
-/

namespace MxxFheTfhe

open MeasureTheory MxxRuntime MxxSecurity

/-- The rounding error `2^20 - ((w + 2^20) mod 2^21)` of the switch of `w` to a multiple of
`2^21`. -/
def msRound (w : Int) : Int := 1048576 - (w + 1048576) % 2097152

theorem msRound_bound (w : Int) : -1048576 < msRound w ∧ msRound w ≤ 1048576 := by
  unfold msRound; omega

theorem msRound_periodic (w k : Int) : msRound (w + 2097152 * k) = msRound w := by
  unfold msRound; omega

/-- The reflection that pairs the rounding errors of `x D + C`. -/
theorem ms_pair (D : Int) (hD : 1 ≤ D ∧ D ≤ 6) (C : Int) :
    ∃ K M : Int, |M| ≤ 4 ∧ ∀ y : Int,
      msRound ((K - y) % 2097152 * D + C) = M - msRound (y * D + C) := by
  obtain ⟨hD1, hD6⟩ := hD
  interval_cases D
  · refine ⟨2 * ((C + 1048576) % 1) - 1 - 2 * C, 1 - 2 * ((C + 1048576) % 1), ?_, fun y ↦ ?_⟩
    · rw [abs_le]; omega
    · unfold msRound; omega
  · refine ⟨(2 * ((C + 1048576) % 2) - 2 - 2 * C) / 2, 2 - 2 * ((C + 1048576) % 2), ?_,
      fun y ↦ ?_⟩
    · rw [abs_le]; omega
    · unfold msRound; omega
  · refine ⟨(2 * ((C + 1048576) % 1) - 1 - 2 * C) * 699051, 1 - 2 * ((C + 1048576) % 1), ?_,
      fun y ↦ ?_⟩
    · rw [abs_le]; omega
    · unfold msRound; omega
  · refine ⟨(2 * ((C + 1048576) % 4) - 4 - 2 * C) / 4, 4 - 2 * ((C + 1048576) % 4), ?_,
      fun y ↦ ?_⟩
    · rw [abs_le]; omega
    · unfold msRound; omega
  · refine ⟨(2 * ((C + 1048576) % 1) - 1 - 2 * C) * 838861, 1 - 2 * ((C + 1048576) % 1), ?_,
      fun y ↦ ?_⟩
    · rw [abs_le]; omega
    · unfold msRound; omega
  · refine ⟨(2 * ((C + 1048576) % 2) - 2 - 2 * C) / 2 * 699051, 2 - 2 * ((C + 1048576) % 2), ?_,
      fun y ↦ ?_⟩
    · rw [abs_le]; omega
    · unfold msRound; omega

/-- The exponent of the moment bound of one rounding error. -/
noncomputable def msExp (a : ℝ) : ℝ := 2 * |a| + a ^ 2 * 1048578 ^ 2 / 2

theorem msExp_nonneg (a : ℝ) : 0 ≤ msExp a := by unfold msExp; positivity

/-- Two paired values contribute at most `2 exp (msExp a)`. -/
theorem pair_exp_le (a : ℝ) {r M : Int} (hr : -1048576 < r ∧ r ≤ 1048576) (hM : |M| ≤ 4) :
    Real.exp (a * r) + Real.exp (a * (M - r)) ≤ 2 * Real.exp (msExp a) := by
  set x : ℝ := a * ((r : ℝ) - (M : ℝ) / 2)
  have hsum : Real.exp (a * r) + Real.exp (a * (M - r)) =
      2 * Real.exp (a * M / 2) * Real.cosh x := by
    rw [Real.cosh_eq]
    have h1 : a * (r : ℝ) = a * M / 2 + x := by simp only [x]; ring
    have h2 : a * ((M : ℝ) - r) = a * M / 2 + -x := by simp only [x]; ring
    rw [h1, h2, Real.exp_add, Real.exp_add]
    ring
  rw [hsum]
  have hMr : |(M : ℝ)| ≤ 4 := by exact_mod_cast hM
  have hrM : |(r : ℝ) - (M : ℝ) / 2| ≤ 1048578 := by
    have h1 : (-1048576 : ℝ) < r := by exact_mod_cast hr.1
    have h2 : (r : ℝ) ≤ 1048576 := by exact_mod_cast hr.2
    rw [abs_le] at hMr ⊢
    constructor <;> linarith [hMr.1, hMr.2]
  have hcosh : Real.cosh x ≤ Real.exp (a ^ 2 * 1048578 ^ 2 / 2) := by
    refine (Real.cosh_le_exp_half_sq x).trans (Real.exp_le_exp.mpr ?_)
    have : x ^ 2 ≤ a ^ 2 * 1048578 ^ 2 := by
      have hsq : ((r : ℝ) - (M : ℝ) / 2) ^ 2 ≤ 1048578 ^ 2 := by
        rw [← sq_abs]
        exact pow_le_pow_left₀ (abs_nonneg _) hrM 2
      have hx : x ^ 2 = a ^ 2 * ((r : ℝ) - (M : ℝ) / 2) ^ 2 := by simp only [x]; ring
      rw [hx]
      exact mul_le_mul_of_nonneg_left hsq (sq_nonneg a)
    linarith
  have hlin : a * M / 2 ≤ 2 * |a| := by
    have := abs_mul a (M : ℝ)
    have h := le_abs_self (a * M)
    rw [this] at h
    nlinarith [abs_nonneg a, abs_nonneg (M : ℝ)]
  calc 2 * Real.exp (a * M / 2) * Real.cosh x
      ≤ 2 * Real.exp (2 * |a|) * Real.exp (a ^ 2 * 1048578 ^ 2 / 2) := by
        gcongr
    _ = 2 * Real.exp (msExp a) := by
        rw [mul_assoc, ← Real.exp_add]
        rfl

/-- The moment of the rounding error of `y D + C` over `y < 2^21`. -/
theorem ms_sum_le (D : Int) (hD : 1 ≤ D ∧ D ≤ 6) (C : Int) (a : ℝ) :
    ∑ y ∈ Finset.range 2097152, Real.exp (a * msRound ((y : Int) * D + C)) ≤
      2097152 * Real.exp (msExp a) := by
  obtain ⟨K, M, hM, hpair⟩ := ms_pair D hD C
  set f : Nat → ℝ := fun y ↦ Real.exp (a * msRound ((y : Int) * D + C)) with hf
  set σ : Nat → Nat := fun y ↦ ((K - y) % 2097152).toNat with hσ
  have hσcast (y : Nat) : ((σ y : Nat) : Int) = (K - y) % 2097152 := by
    simp only [hσ]
    omega
  have hσmem (y : Nat) : σ y ∈ Finset.range 2097152 := by
    rw [Finset.mem_range]
    have := hσcast y
    omega
  have hσσ (y : Nat) (hy : y ∈ Finset.range 2097152) : σ (σ y) = y := by
    rw [Finset.mem_range] at hy
    have h1 := hσcast y
    have h2 := hσcast (σ y)
    omega
  have hreflect : ∑ y ∈ Finset.range 2097152, f y = ∑ y ∈ Finset.range 2097152, f (σ y) :=
    Finset.sum_nbij' σ σ (fun y _ ↦ hσmem y) (fun y _ ↦ hσmem y) (fun y hy ↦ hσσ y hy)
      (fun y hy ↦ hσσ y hy) (fun y hy ↦ by rw [hσσ y hy])
  have hfσ (y : Nat) : f (σ y) = Real.exp (a * ((M : Int) - msRound ((y : Int) * D + C) : Int)) := by
    simp only [hf, hσcast, hpair]
  have htwo : 2 * ∑ y ∈ Finset.range 2097152, f y ≤
      2 * (2097152 * Real.exp (msExp a)) := by
    calc 2 * ∑ y ∈ Finset.range 2097152, f y
        = ∑ y ∈ Finset.range 2097152, (f y + f (σ y)) := by
          rw [Finset.sum_add_distrib, ← hreflect]; ring
      _ ≤ ∑ _y ∈ Finset.range 2097152, 2 * Real.exp (msExp a) := by
          refine Finset.sum_le_sum fun y _ ↦ ?_
          rw [hfσ]
          simp only [hf]
          push_cast
          exact pair_exp_le a (msRound_bound _) hM
      _ = 2 * (2097152 * Real.exp (msExp a)) := by
          rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]; push_cast; ring
  linarith

/-- Periodic sums over whole periods. -/
theorem sum_range_periodic (g : Nat → ℝ) (hg : ∀ n, g (n + 2097152) = g n) (k : Nat) :
    ∑ n ∈ Finset.range (k * 2097152), g n = k * ∑ n ∈ Finset.range 2097152, g n := by
  have hshift : ∀ k n, g (k * 2097152 + n) = g n := by
    intro k
    induction k with
    | zero => intro n; simp
    | succ k ih => intro n; rw [show (k + 1) * 2097152 + n = (k * 2097152 + n) + 2097152 by ring,
        hg, ih]
  induction k with
  | zero => simp
  | succ k ih =>
    rw [show (k + 1) * 2097152 = k * 2097152 + 2097152 by ring, Finset.sum_range_add, ih]
    simp only [hshift]
    push_cast
    ring

/-- The moment of the rounding error of `y D + C` for `y` uniform below `M`. -/
theorem ms_sum_mask_le (D : Int) (hD : 1 ≤ D ∧ D ≤ 6) (C : Int) (a : ℝ) (M : Nat) :
    ∑ y ∈ Finset.range M, Real.exp (a * msRound ((y : Int) * D + C)) ≤
      (M / 2097152 + 1 : Nat) * (2097152 * Real.exp (msExp a)) := by
  set g : Nat → ℝ := fun y ↦ Real.exp (a * msRound ((y : Int) * D + C))
  have hg : ∀ n, g (n + 2097152) = g n := by
    intro n
    simp only [g]
    congr 2
    rw [show (((n + 2097152 : Nat) : Int) * D + C) = ((n : Int) * D + C) + 2097152 * D by
      push_cast; ring, msRound_periodic]
  calc ∑ y ∈ Finset.range M, g y
      ≤ ∑ y ∈ Finset.range ((M / 2097152 + 1) * 2097152), g y := by
        refine Finset.sum_le_sum_of_subset_of_nonneg (fun y hy ↦ ?_) fun _ _ _ ↦ (Real.exp_pos _).le
        rw [Finset.mem_range] at hy ⊢
        have := Nat.lt_div_mul_add (a := M) (b := 2097152) (by norm_num)
        nlinarith
    _ = (M / 2097152 + 1 : Nat) * ∑ y ∈ Finset.range 2097152, g y := sum_range_periodic g hg _
    _ ≤ (M / 2097152 + 1 : Nat) * (2097152 * Real.exp (msExp a)) := by
        gcongr
        exact ms_sum_le D hD C a

/-- The excess factor of a uniform residue modulo `Qmask` over a uniform residue modulo `2^21`. -/
noncomputable abbrev maskExcess : ℝ := 2097152 / (Qmask : ℝ)

theorem msRound_emod_q (y D C : Int) :
    msRound (y % 4294967296 * D + C) = msRound (y * D + C) := by
  rw [show y % 4294967296 * D + C = (y * D + C) + 2097152 * (-(2048 * (y / 4294967296) * D)) by
    rw [Int.emod_def]; ring, msRound_periodic]

/-- The moment of the rounding error of `x D + C` for a uniform residue `x` modulo `Qmask`. -/
theorem ms_lintegral_le (D : Int) (hD : 1 ≤ D ∧ D ≤ 6) (C : Int) (a : ℝ) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (a * msRound ((SamplerLaw.residue Qmask).clamp x %
      4294967296 * D + C))) ∂(SamplerLaw.residue Qmask).measure ≤
      ENNReal.ofReal (Real.exp (msExp a + maskExcess)) := by
  have hM : 0 < Qmask := by decide
  set e : Int → ℝ := fun x ↦ Real.exp (a * msRound (x * D + C)) with he
  have hint : ∀ x ∈ residues Qmask, ENNReal.ofReal (Real.exp (a * msRound
      ((SamplerLaw.residue Qmask).clamp x % 4294967296 * D + C))) = ENNReal.ofReal (e x) := by
    intro x hx
    have hx' := mem_residues.mp hx
    simp only [SamplerLaw.clamp, if_pos hx', msRound_emod_q, he]
  rw [residue_measure_eq hM, lintegral_countable']
  simp_rw [PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _), uniformPMF_apply]
  rw [tsum_eq_sum (s := residues Qmask) fun x hx ↦ by rw [if_neg hx, mul_zero]]
  rw [Finset.sum_congr rfl fun x hx ↦ by rw [if_pos hx, hint x hx], ← Finset.sum_mul,
    ← ENNReal.ofReal_sum_of_nonneg fun _ _ ↦ (Real.exp_pos _).le]
  have hcard : (residues Qmask).card = Qmask := by
    rw [Finset.card_image_of_injective _ (fun a b h ↦ by exact_mod_cast h), Finset.card_range]
  have hsum : ∑ x ∈ residues Qmask, e x = ∑ n ∈ Finset.range Qmask, e n :=
    Finset.sum_image fun a _ b _ h ↦ by exact_mod_cast h
  rw [hcard, hsum]
  have hbound := ms_sum_mask_le D hD C a Qmask
  have hMr : (0 : ℝ) < Qmask := by exact_mod_cast hM
  have hdiv : ((Qmask / 2097152 + 1 : Nat) : ℝ) * 2097152 ≤ (Qmask : ℝ) + 2097152 := by
    have := Nat.div_mul_le_self Qmask 2097152
    have h : ((Qmask / 2097152 : Nat) : ℝ) * 2097152 ≤ (Qmask : ℝ) := by exact_mod_cast this
    push_cast
    linarith
  rw [show ((Qmask : Nat) : ENNReal)⁻¹ = ENNReal.ofReal ((Qmask : ℝ)⁻¹) by
    rw [ENNReal.ofReal_inv_of_pos hMr]; simp, ← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  have hE := Real.exp_pos (msExp a)
  calc (∑ n ∈ Finset.range Qmask, e n) * (Qmask : ℝ)⁻¹
      ≤ ((Qmask / 2097152 + 1 : Nat) * (2097152 * Real.exp (msExp a))) * (Qmask : ℝ)⁻¹ := by
        gcongr
    _ ≤ ((Qmask : ℝ) + 2097152) * Real.exp (msExp a) * (Qmask : ℝ)⁻¹ := by
        have h2 : ((Qmask / 2097152 + 1 : Nat) : ℝ) * (2097152 * Real.exp (msExp a)) ≤
            ((Qmask : ℝ) + 2097152) * Real.exp (msExp a) := by
          rw [← mul_assoc]
          exact mul_le_mul_of_nonneg_right hdiv hE.le
        exact mul_le_mul_of_nonneg_right h2 (by positivity)
    _ = (1 + maskExcess) * Real.exp (msExp a) := by
        simp only [maskExcess]
        field_simp
    _ ≤ Real.exp maskExcess * Real.exp (msExp a) := by
        gcongr
        linarith [Real.add_one_le_exp maskExcess]
    _ = Real.exp (msExp a + maskExcess) := by rw [← Real.exp_add]; ring_nf

end MxxFheTfhe
