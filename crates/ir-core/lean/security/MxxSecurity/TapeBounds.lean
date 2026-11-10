import MxxSecurity.Tape
import RuntimeSubGaussian

/-!
# Tail-bound tools over tapes

* `measure_clamp_ne`, `ae_clamp_eq`: every draw is in the support of its law almost surely, so
  clamping a coordinate changes nothing almost surely.
* `lintegral_mul_exp_block_le`: a block of coordinates that the weight does not read contributes
  at most its conditional exponential moment, for any function of the block.
-/

namespace MxxSecurity

open MeasureTheory MxxRuntime Function

set_option linter.unusedSectionVars false

theorem measure_clamp_ne (law : SamplerLaw) : law.measure {x | law.clamp x ≠ x} = 0 := by
  rw [SamplerLaw.measure, PMF.toMeasure_apply_eq_zero_iff _ MeasurableSet.of_discrete,
    Set.disjoint_left]
  intro x hx hne
  apply hne
  rw [PMF.mem_support_iff] at hx
  cases law with
  | gaussian sigma cutoff =>
    simp only [SamplerLaw.pmf] at hx
    simp only [SamplerLaw.clamp]
    split_ifs at hx with h
    · rw [PMF.normalize_apply] at hx
      have hw : gaussianWeight sigma cutoff x ≠ 0 := fun h0 ↦ hx (by rw [h0, zero_mul])
      unfold gaussianWeight at hw
      split_ifs at hw with hc
      · rw [if_pos ⟨h.1, h.2, hc⟩]
      · exact absurd rfl hw
    · rw [PMF.pure_apply] at hx
      split_ifs at hx with hx0
      · subst hx0; split_ifs <;> rfl
      · exact absurd rfl hx
  | interval minimum maximum =>
    simp only [SamplerLaw.pmf] at hx
    simp only [SamplerLaw.clamp]
    split_ifs at hx with h
    · rw [uniformPMF_apply] at hx
      split_ifs at hx with hm
      · rw [Finset.mem_Icc] at hm
        rw [if_pos hm]
      · exact absurd rfl hx
    · rw [PMF.pure_apply] at hx
      split_ifs at hx with hx0
      · subst hx0; split_ifs <;> rfl
      · exact absurd rfl hx
  | residue modulus =>
    simp only [SamplerLaw.pmf] at hx
    simp only [SamplerLaw.clamp]
    split_ifs at hx with h
    · rw [uniformPMF_apply] at hx
      split_ifs at hx with hm
      · obtain ⟨v, hv, rfl⟩ := Finset.mem_image.mp hm
        rw [if_pos ⟨by omega, by simpa using hv⟩]
      · exact absurd rfl hx
    · rw [PMF.pure_apply] at hx
      split_ifs at hx with hx0
      · subst hx0; split_ifs <;> rfl
      · exact absurd rfl hx

variable {ι : Type} [DecidableEq ι] [Countable ι] {μ : ι → Measure ℤ} [∀ i, IsProbabilityMeasure (μ i)]

/-- Every coordinate is in the support of its law almost surely. -/
theorem ae_clamp_eq (law : ι → SamplerLaw) (hμ : ∀ i, μ i = (law i).measure) :
    ∀ᵐ ω ∂Measure.infinitePi μ, ∀ i, (law i).clamp (ω i) = ω i := by
  rw [ae_all_iff]
  intro i
  rw [ae_iff]
  have hmap : (Measure.infinitePi μ) {ω | ¬(law i).clamp (ω i) = ω i} =
      (Measure.infinitePi μ).map (fun ω ↦ ω i) {x | (law i).clamp x ≠ x} := by
    rw [Measure.map_apply (measurable_pi_apply i) MeasurableSet.of_discrete]
    rfl
  rw [hmap, Measure.infinitePi_map_eval, hμ]
  exact measure_clamp_ne _

/-- A block `B` of coordinates that the weight `F` does not read contributes at most `exp K`,
when for every tape the integral over a fresh draw of the block is at most `exp K`. -/
theorem lintegral_mul_exp_block_le (B S : Finset ι) (hdisj : Disjoint S B)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {g : (ι → ℤ) → ℝ} (hg : DependsOn g (S ∪ B : Finset ι))
    {K : ℝ} (hK : ∀ ω, ∫⁻ ω', ENNReal.ofReal (Real.exp (g (B.piecewise ω' ω))) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp K)) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (g ω)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp K) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
  have hFm : Measurable F := measurable_of_dependsOn S hF
  have hgm : Measurable g := measurable_of_dependsOn _ hg
  have hm : Measurable fun ω ↦ F ω * ENNReal.ofReal (Real.exp (g ω)) :=
    hFm.mul (ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp hgm))
  conv_lhs => rw [infinitePi_eq_map_piecewise μ B]
  have hpm : Measurable fun p : (ι → ℤ) × (ι → ℤ) ↦
      F (B.piecewise p.1 p.2) * ENNReal.ofReal (Real.exp (g (B.piecewise p.1 p.2))) :=
    hm.comp (measurable_piecewise_pair B)
  rw [lintegral_map hm (measurable_piecewise_pair B), lintegral_prod _ hpm.aemeasurable,
    lintegral_lintegral_swap (f := fun x y ↦ F (B.piecewise x y) *
      ENNReal.ofReal (Real.exp (g (B.piecewise x y)))) hpm.aemeasurable]
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  refine lintegral_mono fun ω ↦ ?_
  have hFω : ∀ ω', F (B.piecewise ω' ω) = F ω := fun ω' ↦
    hF fun i hi ↦ Finset.piecewise_eq_of_notMem _ _ _
      (Finset.disjoint_left.mp hdisj (Finset.mem_coe.mp hi))
  simp only [hFω]
  have hxm : Measurable fun x : ι → ℤ ↦ ENNReal.ofReal (Real.exp (g (B.piecewise x ω))) := by
    refine measurable_of_dependsOn B fun x x' h ↦ ?_
    have : B.piecewise x ω = B.piecewise x' ω := by
      funext i
      by_cases hi : i ∈ B
      · simp only [Finset.piecewise_eq_of_mem _ _ _ hi]; exact h i (Finset.mem_coe.mpr hi)
      · simp only [Finset.piecewise_eq_of_notMem _ _ _ hi]
    simp only [this]
  rw [lintegral_const_mul _ hxm, mul_comm (ENNReal.ofReal _)]
  gcongr
  exact hK ω

/-- A moment bound with a constant and a linear term gives a two-sided tail bound: if
`E exp (a Y) ≤ exp (c + b |a| + V a² / 2)` for every `a`, then `P(t ≤ |Y|) ≤ 2 exp (c - (t - b)² / (2 V))`
for `t ≥ b`. -/
theorem measure_le_abs_le_lin {Ω : Type} [MeasurableSpace Ω] (ν : Measure Ω) {Y : Ω → ℝ}
    (hY : Measurable Y) {V b c : ℝ} (hV : 0 < V)
    (hmgf : ∀ a : ℝ, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * Y ω)) ∂ν ≤
      ENNReal.ofReal (Real.exp (c + b * |a| + V * a ^ 2 / 2))) {t : ℝ} (ht : b ≤ t) :
    ν {ω | t ≤ |Y ω|} ≤ ENNReal.ofReal (2 * Real.exp (c - (t - b) ^ 2 / (2 * V))) := by
  have tail : ∀ Z : Ω → ℝ, Measurable Z →
      (∀ a : ℝ, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * Z ω)) ∂ν ≤
        ENNReal.ofReal (Real.exp (c + b * |a| + V * a ^ 2 / 2))) →
      ν {ω | t ≤ Z ω} ≤ ENNReal.ofReal (Real.exp (c - (t - b) ^ 2 / (2 * V))) := by
    intro Z hZ hmgfZ
    set a := (t - b) / V
    have ha : 0 ≤ a := div_nonneg (by linarith) hV.le
    have hmarkov := mul_meas_ge_le_lintegral₀ (μ := ν)
      (f := fun ω ↦ ENNReal.ofReal (Real.exp (a * Z ω)))
      (ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
        (hZ.const_mul a))).aemeasurable (ENNReal.ofReal (Real.exp (a * t)))
    have hsub : {ω | t ≤ Z ω} ⊆
        {ω | ENNReal.ofReal (Real.exp (a * t)) ≤ ENNReal.ofReal (Real.exp (a * Z ω))} :=
      fun ω hω ↦ ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr
        (mul_le_mul_of_nonneg_left hω ha))
    have hbound : ENNReal.ofReal (Real.exp (a * t)) * ν {ω | t ≤ Z ω} ≤
        ENNReal.ofReal (Real.exp (c + b * |a| + V * a ^ 2 / 2)) :=
      ((mul_le_mul_right (measure_mono hsub) _).trans hmarkov).trans (hmgfZ a)
    have hexp : Real.exp (c - (t - b) ^ 2 / (2 * V)) * Real.exp (a * t) =
        Real.exp (c + b * |a| + V * a ^ 2 / 2) := by
      rw [← Real.exp_add, abs_of_nonneg ha]
      congr 1
      simp only [a]
      field_simp
      ring
    have hpos : ENNReal.ofReal (Real.exp (a * t)) ≠ 0 := by
      simpa using Real.exp_pos (a * t)
    calc ν {ω | t ≤ Z ω}
        = (ENNReal.ofReal (Real.exp (a * t)))⁻¹ * (ENNReal.ofReal (Real.exp (a * t)) *
            ν {ω | t ≤ Z ω}) := by
          rw [← mul_assoc, ENNReal.inv_mul_cancel hpos ENNReal.ofReal_ne_top, one_mul]
      _ ≤ (ENNReal.ofReal (Real.exp (a * t)))⁻¹ *
            ENNReal.ofReal (Real.exp (c + b * |a| + V * a ^ 2 / 2)) := by
          gcongr
      _ = ENNReal.ofReal (Real.exp (c - (t - b) ^ 2 / (2 * V))) := by
          rw [← hexp, ENNReal.ofReal_mul (Real.exp_pos _).le, mul_comm, mul_assoc,
            ENNReal.mul_inv_cancel hpos ENNReal.ofReal_ne_top, mul_one]
  have hupper := tail Y hY hmgf
  have hlower := tail (fun ω ↦ -Y ω) hY.neg fun a ↦ by
    have h := hmgf (-a)
    simp only [neg_mul, mul_neg, neg_sq, abs_neg] at h ⊢
    exact h
  calc ν {ω | t ≤ |Y ω|} ≤ ν ({ω | t ≤ Y ω} ∪ {ω | t ≤ -Y ω}) := by
        refine measure_mono fun ω hω ↦ ?_
        simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
        rcases le_abs'.mp hω with h | h
        · right; linarith
        · left; exact h
    _ ≤ ν {ω | t ≤ Y ω} + ν {ω | t ≤ -Y ω} := measure_union_le _ _
    _ ≤ ENNReal.ofReal (Real.exp (c - (t - b) ^ 2 / (2 * V))) +
          ENNReal.ofReal (Real.exp (c - (t - b) ^ 2 / (2 * V))) := add_le_add hupper hlower
    _ = ENNReal.ofReal (2 * Real.exp (c - (t - b) ^ 2 / (2 * V))) := by
        rw [← ENNReal.ofReal_add (Real.exp_pos _).le (Real.exp_pos _).le]
        congr 1
        ring

end MxxSecurity
