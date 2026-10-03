import RuntimeProbability
import Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation
import Mathlib.NumberTheory.ModularForms.JacobiTheta.TwoVariable
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series

/-!
# The exponential moment of the truncated discrete Gaussian

`SamplerLaw.gaussian sigma cutoff` is sub-Gaussian with variance proxy `sigma²`:
`E[exp(a X)] ≤ exp(sigma² a² / 2)`. The untruncated bound is the theta-function inequality
`∑ exp(-(n - c)² / (2 s)) ≤ ∑ exp(-n² / (2 s))`, from Poisson summation; truncating to
`|x| ≤ cutoff` removes the largest values of `cosh (a x)` and can only lower the moment.
-/

namespace MxxRuntime

open Real Complex MeasureTheory

/-- The unnormalized Gaussian over the integers, shifted by `c`, is summable. -/
theorem summable_gaussian_shift {s : ℝ} (hs : 0 < s) (c : ℝ) :
    Summable fun n : ℤ ↦ Real.exp (-((n : ℝ) - c) ^ 2 / (2 * s)) := by
  have hT : 0 < 1 / (2 * π * s) := by positivity
  refine (summable_pow_mul_jacobiTheta₂_term_bound (|c| / (2 * π * s)) hT 0).of_nonneg_of_le
    (fun _ ↦ (Real.exp_pos _).le) fun n ↦ ?_
  simp only [pow_zero, one_mul, Int.cast_abs]
  apply Real.exp_le_exp.mpr
  have hn : -(((n : ℝ) - c) ^ 2) / (2 * s) ≤ -(n : ℝ) ^ 2 / (2 * s) + |c| * |(n : ℝ)| / s := by
    have : -(((n : ℝ) - c) ^ 2) ≤ -(n : ℝ) ^ 2 + 2 * (|c| * |(n : ℝ)|) := by
      nlinarith [abs_mul_abs_self c, abs_mul_abs_self (n : ℝ), abs_nonneg c, abs_nonneg (n : ℝ),
        neg_abs_le (c * n), le_abs_self (c * n), abs_mul c (n : ℝ)]
    calc -(((n : ℝ) - c) ^ 2) / (2 * s) ≤ (-(n : ℝ) ^ 2 + 2 * (|c| * |(n : ℝ)|)) / (2 * s) :=
          div_le_div_of_nonneg_right this (by positivity)
      _ = _ := by field_simp
  calc -(((n : ℝ) - c) ^ 2) / (2 * s) ≤ -(n : ℝ) ^ 2 / (2 * s) + |c| * |(n : ℝ)| / s := hn
    _ = -π * (1 / (2 * π * s) * (n : ℝ) ^ 2 - 2 * (|c| / (2 * π * s)) * |((n : ℤ) : ℝ)|) := by
        field_simp
        ring

/-- The theta-function inequality: shifting the Gaussian never increases its integer sum. -/
theorem tsum_gaussian_shift_le {s : ℝ} (hs : 0 < s) (c : ℝ) :
    ∑' n : ℤ, Real.exp (-((n : ℝ) - c) ^ 2 / (2 * s)) ≤
      ∑' n : ℤ, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) := by
  set a : ℝ := 1 / (2 * π * s) with ha_def
  set b : ℝ := c / (2 * π * s) with hb_def
  have ha : 0 < a := by positivity
  -- The Gaussian at shift `c` is `exp (-c² / (2 s))` times the quadratic-exponential sum.
  have hsplit : ∀ n : ℤ, Real.exp (-((n : ℝ) - c) ^ 2 / (2 * s)) =
      Real.exp (-c ^ 2 / (2 * s)) * Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n) := by
    intro n
    rw [← Real.exp_add]
    congr 1
    simp only [ha_def, hb_def]
    field_simp
    ring
  have hquad : Summable fun n : ℤ ↦ Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n) := by
    refine ((summable_gaussian_shift hs c).mul_left (Real.exp (c ^ 2 / (2 * s)))).congr
      fun n ↦ ?_
    rw [hsplit, ← mul_assoc, ← Real.exp_add,
      show c ^ 2 / (2 * s) + -c ^ 2 / (2 * s) = 0 by ring, Real.exp_zero, one_mul]
  -- Poisson summation, in norm.
  have hpoisson := Complex.tsum_exp_neg_quadratic (a := (a : ℂ)) (by simpa using ha) (b : ℂ)
  have hnorm_term : ∀ n : ℤ, ‖cexp (-π / (a : ℂ) * ((n : ℂ) + I * b) ^ 2)‖ =
      Real.exp (π * b ^ 2 / a) * Real.exp (-π / a * (n : ℝ) ^ 2) := by
    intro n
    rw [Complex.norm_exp, ← Real.exp_add]
    congr 1
    have : -π / (a : ℂ) * ((n : ℂ) + I * b) ^ 2 =
        ((-π / a * ((n : ℝ) ^ 2 - b ^ 2) : ℝ) : ℂ) + ((-π / a * (2 * n * b) : ℝ) : ℂ) * I := by
      push_cast
      ring_nf
      rw [I_sq]
      ring
    rw [this, add_re, ofReal_re, re_ofReal_mul, I_re, mul_zero, add_zero]
    ring
  have hsum0 : Summable fun n : ℤ ↦ Real.exp (-π / a * (n : ℝ) ^ 2) := by
    have h := summable_gaussian_shift (s := a / (2 * π)) (by positivity) 0
    refine h.congr fun n ↦ ?_
    congr 1
    field_simp
    ring
  have hnorm_summable : Summable fun n : ℤ ↦ ‖cexp (-π / (a : ℂ) * ((n : ℂ) + I * b) ^ 2)‖ := by
    simp_rw [hnorm_term]
    exact hsum0.mul_left _
  have hsqrt : ‖(1 : ℂ) / (a : ℂ) ^ (1 / 2 : ℂ)‖ = 1 / a ^ (1 / 2 : ℝ) := by
    rw [show (1 / 2 : ℂ) = ((1 / 2 : ℝ) : ℂ) by push_cast; ring, ← ofReal_cpow ha.le,
      ← ofReal_one, ← ofReal_div, norm_real, Real.norm_of_nonneg (by positivity)]
  have hquad_le : ∑' n : ℤ, Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n) ≤
      1 / a ^ (1 / 2 : ℝ) * (Real.exp (π * b ^ 2 / a) * ∑' n : ℤ, Real.exp (-π / a * (n : ℝ) ^ 2)) := by
    have hcast : ((∑' n : ℤ, Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n) : ℝ) : ℂ) =
        ∑' n : ℤ, cexp (-π * (a : ℂ) * (n : ℂ) ^ 2 + 2 * π * (b : ℂ) * n) := by
      rw [ofReal_tsum]
      congr 1
      funext n
      rw [ofReal_exp]
      push_cast
      ring_nf
    calc ∑' n : ℤ, Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n)
        = ‖((∑' n : ℤ, Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n) : ℝ) : ℂ)‖ := by
          rw [norm_real, Real.norm_of_nonneg (tsum_nonneg fun _ ↦ (Real.exp_pos _).le)]
      _ = ‖(1 : ℂ) / (a : ℂ) ^ (1 / 2 : ℂ) *
            ∑' n : ℤ, cexp (-π / (a : ℂ) * ((n : ℂ) + I * b) ^ 2)‖ := by
          rw [hcast, hpoisson]
      _ ≤ 1 / a ^ (1 / 2 : ℝ) * ∑' n : ℤ, ‖cexp (-π / (a : ℂ) * ((n : ℂ) + I * b) ^ 2)‖ := by
          rw [norm_mul, hsqrt]
          gcongr
          exact norm_tsum_le_tsum_norm hnorm_summable
      _ = _ := by
          simp_rw [hnorm_term]
          rw [tsum_mul_left]
  -- The unshifted sum by the real Poisson formula.
  have hzero : ∑' n : ℤ, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) =
      1 / a ^ (1 / 2 : ℝ) * ∑' n : ℤ, Real.exp (-π / a * (n : ℝ) ^ 2) := by
    rw [← Real.tsum_exp_neg_mul_int_sq ha]
    congr 1
    funext n
    congr 1
    simp only [ha_def]
    field_simp
    ring
  have hb : π * b ^ 2 / a = c ^ 2 / (2 * s) := by
    simp only [ha_def, hb_def]
    field_simp
  calc ∑' n : ℤ, Real.exp (-((n : ℝ) - c) ^ 2 / (2 * s))
      = Real.exp (-c ^ 2 / (2 * s)) *
          ∑' n : ℤ, Real.exp (-π * a * (n : ℝ) ^ 2 + 2 * π * b * n) := by
        simp_rw [hsplit]
        rw [tsum_mul_left]
    _ ≤ Real.exp (-c ^ 2 / (2 * s)) * (1 / a ^ (1 / 2 : ℝ) *
          (Real.exp (π * b ^ 2 / a) * ∑' n : ℤ, Real.exp (-π / a * (n : ℝ) ^ 2))) := by
        gcongr
    _ = ∑' n : ℤ, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) := by
        rw [hzero, hb, ← mul_assoc, ← mul_assoc, mul_comm (Real.exp _) (1 / _), mul_assoc,
          mul_assoc, ← mul_assoc (Real.exp _), ← Real.exp_add,
          show -c ^ 2 / (2 * s) + c ^ 2 / (2 * s) = 0 by ring, Real.exp_zero, one_mul]

/-- The untruncated exponential moment: `∑ ρ(n) exp(a n) ≤ exp(s a² / 2) ∑ ρ(n)`. -/
theorem tsum_gaussian_mul_exp_le {s : ℝ} (hs : 0 < s) (a : ℝ) :
    ∑' n : ℤ, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) * Real.exp (a * n) ≤
      Real.exp (s * a ^ 2 / 2) * ∑' n : ℤ, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) := by
  have hterm : ∀ n : ℤ, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) * Real.exp (a * n) =
      Real.exp (s * a ^ 2 / 2) * Real.exp (-((n : ℝ) - s * a) ^ 2 / (2 * s)) := by
    intro n
    rw [← Real.exp_add, ← Real.exp_add]
    congr 1
    field_simp
    ring
  simp_rw [hterm]
  rw [tsum_mul_left]
  exact mul_le_mul_of_nonneg_left (tsum_gaussian_shift_le hs (s * a)) (Real.exp_pos _).le

theorem summable_gaussian_mul_exp {s : ℝ} (hs : 0 < s) (a : ℝ) :
    Summable fun n : ℤ ↦ Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) * Real.exp (a * n) := by
  refine ((summable_gaussian_shift hs (s * a)).mul_left (Real.exp (s * a ^ 2 / 2))).congr
    fun n ↦ ?_
  rw [← Real.exp_add, ← Real.exp_add]
  congr 1
  field_simp
  ring

/-- Truncating a symmetric weight to `|n| ≤ B` removes the largest values of `cosh (a n)`, so a
bound on the full exponential moment ratio also bounds the truncated one. -/
theorem sum_mul_exp_le_of_tsum {ρ : ℤ → ℝ} (hpos : ∀ n, 0 < ρ n) (hneg : ∀ n, ρ (-n) = ρ n)
    (hsumρ : Summable ρ) {a : ℝ} (hsum : Summable fun n : ℤ ↦ ρ n * Real.exp (a * n))
    (hsumneg : Summable fun n : ℤ ↦ ρ n * Real.exp (-a * n)) {E : ℝ}
    (hfull : ∑' n : ℤ, ρ n * Real.exp (a * n) ≤ E * ∑' n : ℤ, ρ n) {B : ℤ} (hB : 0 ≤ B) :
    ∑ n ∈ Finset.Icc (-B) B, ρ n * Real.exp (a * n) ≤ E * ∑ n ∈ Finset.Icc (-B) B, ρ n := by
  set I := Finset.Icc (-B) B
  have hsymI : ∑ n ∈ I, ρ n * Real.exp (a * n) = ∑ n ∈ I, ρ n * Real.cosh (a * n) := by
    have hnegI : ∑ n ∈ I, ρ n * Real.exp (a * n) = ∑ n ∈ I, ρ n * Real.exp (-(a * n)) := by
      refine Finset.sum_nbij' (fun n ↦ -n) (fun n ↦ -n) (fun n hn ↦ ?_) (fun n hn ↦ ?_)
        (fun n _ ↦ neg_neg n) (fun n _ ↦ neg_neg n) fun n _ ↦ ?_
      · simp only [I, Finset.mem_Icc] at hn ⊢; omega
      · simp only [I, Finset.mem_Icc] at hn ⊢; omega
      · rw [hneg]; push_cast; ring_nf
    have : 2 * ∑ n ∈ I, ρ n * Real.exp (a * n) = 2 * ∑ n ∈ I, ρ n * Real.cosh (a * n) := by
      rw [two_mul]
      conv_lhs => arg 2; rw [hnegI]
      rw [← Finset.sum_add_distrib, Finset.mul_sum]
      refine Finset.sum_congr rfl fun n _ ↦ ?_
      rw [Real.cosh_eq]
      ring
    linarith
  have hcoshsum : Summable fun n : ℤ ↦ ρ n * Real.cosh (a * n) := by
    refine ((hsum.add hsumneg).div_const 2).congr fun n ↦ ?_
    rw [Real.cosh_eq]
    ring_nf
  have hsymZ : ∑' n : ℤ, ρ n * Real.cosh (a * n) = ∑' n : ℤ, ρ n * Real.exp (a * n) := by
    have hnegZ : ∑' n : ℤ, ρ n * Real.exp (-a * n) = ∑' n : ℤ, ρ n * Real.exp (a * n) := by
      rw [← (Equiv.neg ℤ).tsum_eq]
      congr 1
      funext n
      simp only [Equiv.neg_apply, hneg]
      push_cast
      ring_nf
    have h2 : ∑' n : ℤ, ρ n * Real.cosh (a * n) =
        (∑' n : ℤ, ρ n * Real.exp (a * n) + ∑' n : ℤ, ρ n * Real.exp (-a * n)) / 2 := by
      rw [← hsum.tsum_add hsumneg, ← tsum_div_const]
      congr 1
      funext n
      rw [Real.cosh_eq]
      ring_nf
    rw [h2, hnegZ]
    ring
  set T := ∑' n : ℤ, ρ n * Real.cosh (a * n)
  set Z := ∑' n : ℤ, ρ n
  have hT : T ≤ E * Z := hsymZ ▸ hfull
  set Ain := ∑ n ∈ I, ρ n * Real.cosh (a * n)
  set ain := ∑ n ∈ I, ρ n
  set Aout := ∑' n : ↑((I : Set ℤ)ᶜ), ρ n * Real.cosh (a * n)
  set aout := ∑' n : ↑((I : Set ℤ)ᶜ), ρ n
  have hTsplit : Ain + Aout = T := hcoshsum.sum_add_tsum_compl
  have hZsplit : ain + aout = Z := hsumρ.sum_add_tsum_compl
  have hBreal : (0 : ℝ) ≤ B := by exact_mod_cast hB
  have hAin : Ain ≤ Real.cosh (a * B) * ain := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun n hn ↦ ?_
    rw [mul_comm (Real.cosh _)]
    refine mul_le_mul_of_nonneg_left (Real.cosh_le_cosh.2 ?_) (hpos n).le
    simp only [I, Finset.mem_Icc] at hn
    have hnB : |(n : ℝ)| ≤ |(B : ℝ)| := by
      rw [abs_of_nonneg hBreal, abs_le]
      constructor <;> exact_mod_cast (by omega)
    rw [abs_mul, abs_mul]
    exact mul_le_mul_of_nonneg_left hnB (abs_nonneg a)
  have hAout : Real.cosh (a * B) * aout ≤ Aout := by
    rw [← tsum_mul_left]
    refine ((hsumρ.subtype _).mul_left _).tsum_le_tsum (fun n ↦ ?_) (hcoshsum.subtype _)
    rw [mul_comm]
    refine mul_le_mul_of_nonneg_left (Real.cosh_le_cosh.2 ?_) (hpos n).le
    have hn := n.2
    simp only [Set.mem_compl_iff, Finset.mem_coe, I, Finset.mem_Icc, not_and_or, not_le] at hn
    have hBn : |(B : ℝ)| ≤ |((n : ℤ) : ℝ)| := by
      rw [abs_of_nonneg hBreal]
      rcases hn with h | h
      · have : ((n : ℤ) : ℝ) < -B := by exact_mod_cast h
        rw [abs_of_neg (by linarith)]; linarith
      · have : (B : ℝ) < ((n : ℤ) : ℝ) := by exact_mod_cast h
        rw [abs_of_pos (by linarith)]; linarith
    rw [abs_mul, abs_mul]
    exact mul_le_mul_of_nonneg_left hBn (abs_nonneg a)
  have hZpos : 0 < Z := hsumρ.tsum_pos (fun n ↦ (hpos n).le) 0 (hpos 0)
  have hain : 0 ≤ ain := Finset.sum_nonneg fun n _ ↦ (hpos n).le
  have haout : 0 ≤ aout := tsum_nonneg fun n ↦ (hpos n).le
  have h1 : Ain * aout ≤ Real.cosh (a * B) * ain * aout := mul_le_mul_of_nonneg_right hAin haout
  have h2 : Real.cosh (a * B) * aout * ain ≤ Aout * ain := mul_le_mul_of_nonneg_right hAout hain
  have h3 : T * ain ≤ E * Z * ain := mul_le_mul_of_nonneg_right hT hain
  have key : Ain * Z ≤ E * ain * Z := by
    rw [← hZsplit] at h3 ⊢
    rw [← hTsplit] at h3
    nlinarith
  rw [hsymI]
  exact le_of_mul_le_mul_right key hZpos

/-- The truncated discrete Gaussian moment: `∑_{|n| ≤ B} ρ(n) exp(a n) ≤ exp(s a²/2) ∑_{|n| ≤ B} ρ(n)`. -/
theorem sum_gaussian_mul_exp_le {s : ℝ} (hs : 0 < s) {B : ℤ} (hB : 0 ≤ B) (a : ℝ) :
    ∑ n ∈ Finset.Icc (-B) B, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) * Real.exp (a * n) ≤
      Real.exp (s * a ^ 2 / 2) *
        ∑ n ∈ Finset.Icc (-B) B, Real.exp (-((n : ℝ) - 0) ^ 2 / (2 * s)) :=
  sum_mul_exp_le_of_tsum (fun _ ↦ Real.exp_pos _) (fun n ↦ by push_cast; ring_nf)
    (summable_gaussian_shift hs 0) (summable_gaussian_mul_exp hs a)
    (summable_gaussian_mul_exp hs (-a)) (tsum_gaussian_mul_exp_le hs a) hB

/-- The sampler's Gaussian law is sub-Gaussian with variance proxy `sigma²`. -/
theorem SamplerLaw.lintegral_exp_gaussian_le (sigma : Rat) (cutoff : Int) (a : ℝ) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (a * x)) ∂(SamplerLaw.gaussian sigma cutoff).measure ≤
      ENNReal.ofReal (Real.exp ((sigma : ℝ) ^ 2 * a ^ 2 / 2)) := by
  simp only [SamplerLaw.measure, SamplerLaw.pmf]
  split_ifs with h
  · obtain ⟨hsigma, hcutoff⟩ := h
    set s : ℝ := (sigma : ℝ) ^ 2
    have hs : 0 < s := by positivity
    set I := Finset.Icc (-cutoff) cutoff
    have hweight : ∀ x ∈ I, gaussianWeight sigma cutoff x =
        ENNReal.ofReal (Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s))) := by
      intro x hx
      simp only [I, Finset.mem_Icc] at hx
      simp only [gaussianWeight, s, sub_zero]
      rw [if_pos (abs_le.mpr hx)]
    have htotal : ∑' x, gaussianWeight sigma cutoff x =
        ENNReal.ofReal (∑ x ∈ I, Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s))) := by
      rw [tsum_eq_sum (s := I) fun x hx ↦ gaussianWeight_eq_zero hx,
        ENNReal.ofReal_sum_of_nonneg fun _ _ ↦ (Real.exp_pos _).le]
      exact Finset.sum_congr rfl hweight
    rw [lintegral_countable']
    simp_rw [PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _), PMF.normalize_apply,
      ← mul_assoc]
    rw [ENNReal.tsum_mul_right, tsum_eq_sum (s := I) fun x hx ↦ by
      rw [gaussianWeight_eq_zero hx, mul_zero]]
    rw [Finset.sum_congr rfl fun x hx ↦ by rw [hweight x hx],
      ← Finset.sum_congr rfl fun x _ ↦ (ENNReal.ofReal_mul (Real.exp_pos _).le),
      ← ENNReal.ofReal_sum_of_nonneg fun _ _ ↦ by positivity, htotal]
    have hpos : 0 < ∑ x ∈ I, Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s)) :=
      Finset.sum_pos (fun _ _ ↦ Real.exp_pos _) ⟨0, by simp [I, hcutoff]⟩
    have hmoment := sum_gaussian_mul_exp_le hs hcutoff a
    calc ENNReal.ofReal (∑ x ∈ I, Real.exp (a * x) * Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s))) *
          (ENNReal.ofReal (∑ x ∈ I, Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s))))⁻¹
        ≤ ENNReal.ofReal (Real.exp (s * a ^ 2 / 2) *
            ∑ x ∈ I, Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s))) *
          (ENNReal.ofReal (∑ x ∈ I, Real.exp (-((x : ℝ) - 0) ^ 2 / (2 * s))))⁻¹ := by
          gcongr
          simpa only [mul_comm (Real.exp (a * _))] using hmoment
      _ = ENNReal.ofReal (Real.exp (s * a ^ 2 / 2)) := by
          rw [ENNReal.ofReal_mul (Real.exp_pos _).le, mul_assoc,
            ENNReal.mul_inv_cancel (by simpa using hpos) ENNReal.ofReal_ne_top, mul_one]
  · rw [PMF.toMeasure_pure, lintegral_dirac]
    simp only [Int.cast_zero, mul_zero, Real.exp_zero]
    exact ENNReal.ofReal_le_ofReal (Real.one_le_exp (by positivity))

/-- A uniform bit: `E[exp(a X)] = (1 + exp a) / 2 ≤ exp(a/2 + a²/8)`. -/
theorem SamplerLaw.lintegral_exp_bit_le (a : ℝ) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (a * x)) ∂(SamplerLaw.interval 0 1).measure ≤
      ENNReal.ofReal (Real.exp (a / 2 + a ^ 2 / 8)) := by
  simp only [SamplerLaw.measure, SamplerLaw.pmf, zero_le_one, dite_true]
  rw [lintegral_countable']
  simp_rw [PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _),
    PMF.uniformOfFinset_apply]
  rw [tsum_eq_sum (s := Finset.Icc 0 1) fun x hx ↦ by rw [if_neg hx, mul_zero]]
  have hIcc : Finset.Icc (0 : ℤ) 1 = {0, 1} := by decide
  simp only [hIcc, Finset.mem_insert, Finset.mem_singleton, zero_ne_one, not_false_eq_true,
    Finset.sum_insert, Finset.sum_singleton, if_true, or_true, true_or, Int.cast_zero,
    mul_zero, Real.exp_zero, Int.cast_one, mul_one, Finset.card_insert_of_notMem,
    Finset.card_singleton]
  rw [← add_mul, ← ENNReal.ofReal_add zero_le_one (Real.exp_pos _).le,
    show ((1 + 1 : ℕ) : ENNReal)⁻¹ = ENNReal.ofReal (1 / 2) by
      rw [one_div, ENNReal.ofReal_inv_of_pos two_pos]; norm_num,
    ← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  have hcosh : (1 + Real.exp a) * (1 / 2) = Real.exp (a / 2) * Real.cosh (a / 2) := by
    have h1 : Real.exp (a / 2) * Real.exp (a / 2) = Real.exp a := by
      rw [← Real.exp_add, add_halves]
    have h2 : Real.exp (a / 2) * Real.exp (-(a / 2)) = 1 := by
      rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
    rw [Real.cosh_eq]
    linear_combination (-1 / 2) * h1 + (-1 / 2) * h2
  rw [hcosh, Real.exp_add]
  gcongr
  calc Real.cosh (a / 2) ≤ Real.exp ((a / 2) ^ 2 / 2) := Real.cosh_le_exp_half_sq _
    _ = Real.exp (a ^ 2 / 8) := by ring_nf

end MxxRuntime
