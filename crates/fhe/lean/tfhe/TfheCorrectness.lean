import TfheOutput
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The correctness of one public-key NAND gate

A run fails only if the input noise exceeds `inThreshold` or the output noise at the actual
initial rotation exceeds `outThreshold` (`run_correct`). The input noise has a Gaussian tail with
proxy `inProxy` (`inNoise_tail`); the output noise at each of the `2048` initial rotations has one
with proxy `outProxy`, and the actual rotation is one of them. Together the failure probability is
at most `2^-100`.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open MeasureTheory Mxx.Primitives MxxRuntime MxxSecurity Function GeneratedScheme

theorem outNoiseRaw_measurable (messages : Fin 2 → Int) (v : Int) :
    Measurable (outNoiseRaw messages v) := by
  refine measurable_of_dependsOn (readKeys 720 ∪ Finset.univ.image kkey) fun ω ω' h ↦ ?_
  have h1 : ∀ k ∈ (readKeys 720 : Set Key), ω k = ω' k :=
    fun k hk ↦ h k (by simp only [Finset.coe_union, Set.mem_union] at hk ⊢; exact Or.inl hk)
  have hk : ∀ i, ω (kkey i) = ω' (kkey i) :=
    fun i ↦ h _ (by simp only [Finset.coe_union, Set.mem_union, Finset.coe_image,
      Finset.coe_univ, Set.image_univ, Set.mem_range]; exact Or.inr ⟨i, rfl⟩)
  unfold outNoiseRaw
  rw [brSumRaw_congr messages v 720 h1, outDigits_congr messages v h1]
  simp only [hk]

theorem outNoise_tail (messages : Fin 2 → Int) (v : Int) :
    Measure.infinitePi μ {ω | outThreshold < |outNoise messages ω v|} ≤
      ENNReal.ofReal (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy))) := by
  have hae : {ω | outThreshold < |outNoise messages ω v|} =ᵐ[Measure.infinitePi μ]
      {ω | outThreshold < |outNoiseRaw messages v ω|} := by
    filter_upwards [ae_supp] with ω hω
    simp only [outNoise_eq_raw messages v ω hω]
  rw [measure_congr hae]
  refine (measure_mono (t := {ω | outThreshold ≤ |outNoiseRaw messages v ω|})
    fun ω (hω : outThreshold < |outNoiseRaw messages v ω|) ↦ le_of_lt hω).trans ?_
  exact measure_le_abs_le (Measure.infinitePi μ) (outNoiseRaw_measurable messages v)
    (by unfold outProxy; positivity) (outNoiseRaw_mgf messages v) (by norm_num [outThreshold])

/-- The output noise at the actual initial rotation exceeds the threshold only if it does at one
of the `2048` rotations. -/
theorem outNoise_union (messages : Fin 2 → Int) :
    Measure.infinitePi μ {ω | outThreshold < |outNoise messages ω (nandV messages ω)|} ≤
      2048 * ENNReal.ofReal (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy))) := by
  have hsub : {ω | outThreshold < |outNoise messages ω (nandV messages ω)|} ⊆
      ⋃ v ∈ (Finset.univ : Finset (Fin 2048)), {ω | outThreshold < |outNoise messages ω v.val|} := by
    intro ω hω
    have hr := nandV_range messages ω
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, Finset.mem_univ, exists_true_left]
    refine ⟨⟨(nandV messages ω).toNat, by omega⟩, ?_⟩
    have : (((nandV messages ω).toNat : Nat) : Int) = nandV messages ω := Int.toNat_of_nonneg hr.1
    simp only [this]
    exact hω
  calc Measure.infinitePi μ {ω | outThreshold < |outNoise messages ω (nandV messages ω)|}
      ≤ ∑ v ∈ (Finset.univ : Finset (Fin 2048)),
          Measure.infinitePi μ {ω | outThreshold < |outNoise messages ω v.val|} :=
        (measure_mono hsub).trans (measure_biUnion_finset_le _ _)
    _ ≤ ∑ _v ∈ (Finset.univ : Finset (Fin 2048)),
          ENNReal.ofReal (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy))) :=
        Finset.sum_le_sum fun v _ ↦ outNoise_tail messages v.val
    _ = 2048 * ENNReal.ofReal (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy))) := by
        simp

theorem exp_neg_le_half_pow {x : ℝ} {k : ℕ} (h : (k : ℝ) * 0.6931471808 ≤ x) :
    Real.exp (-x) ≤ (1 / 2) ^ k := by
  have hlog := Real.log_two_lt_d9
  have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hle : Real.exp (-x) ≤ Real.exp (-(k * Real.log 2)) :=
    Real.exp_le_exp.mpr (by nlinarith)
  have hpow : Real.exp (-(k * Real.log 2)) = (1 / 2) ^ k := by
    rw [Real.exp_neg, Real.exp_nat_mul, Real.exp_log two_pos, one_div, inv_pow]
  exact hle.trans_eq hpow

theorem final_numeric :
    ENNReal.ofReal (2 * Real.exp (720 * maskExcess - (inThreshold - 1440) ^ 2 / (2 * inProxy))) +
      2048 * ENNReal.ofReal (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy))) ≤
      (2 : ENNReal)⁻¹ ^ 100 := by
  have hin : Real.exp (720 * maskExcess - (inThreshold - 1440) ^ 2 / (2 * inProxy)) ≤
      (1 / 2) ^ 102 := by
    rw [show 720 * maskExcess - (inThreshold - 1440) ^ 2 / (2 * inProxy) =
      -((inThreshold - 1440) ^ 2 / (2 * inProxy) - 720 * maskExcess) by ring]
    refine exp_neg_le_half_pow ?_
    simp only [maskExcess, inThreshold, inProxy, Qmask]
    norm_num
  have hout : Real.exp (-outThreshold ^ 2 / (2 * outProxy)) ≤ (1 / 2) ^ 114 := by
    rw [neg_div]
    refine exp_neg_le_half_pow ?_
    simp only [outThreshold, outProxy]
    norm_num
  rw [show (2 : ENNReal)⁻¹ ^ 100 = ENNReal.ofReal ((1 / 2) ^ 100) by
      rw [ENNReal.ofReal_pow (by norm_num), one_div, ENNReal.ofReal_inv_of_pos two_pos]
      simp,
    show (2048 : ENNReal) = ENNReal.ofReal 2048 by simp,
    ← ENNReal.ofReal_mul (by norm_num),
    ← ENNReal.ofReal_add (by positivity) (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  calc 2 * Real.exp (720 * maskExcess - (inThreshold - 1440) ^ 2 / (2 * inProxy)) +
        2048 * (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy)))
      ≤ 2 * (1 / 2) ^ 102 + 2048 * (2 * (1 / 2) ^ 114) := by gcongr
    _ ≤ (1 / 2) ^ 100 := by norm_num

/-- The failure probability of one NAND gate on fresh encryptions is at most `2^-100`. -/
theorem correctness : CorrectnessClaim := by
  refine fun (messages : Fin 2 → Int) hvalid ↦ ?_
  have hm0 : messages 0 = 0 ∨ messages 0 = 1 := by
    have h : (0 : Int) ≤ messages 0 ∧ messages 0 ≤ 1 := hvalid 0
    omega
  have hm1 : messages 1 = 0 ∨ messages 1 = 1 := by
    have h : (0 : Int) ≤ messages 1 ∧ messages 1 ≤ 1 := hvalid 1
    omega
  rw [probTrue_evaluationFails]
  have hsub : {ω | failR messages ω = true} ⊆
      {ω | inThreshold < |inNoise messages ω|} ∪
        {ω | outThreshold < |outNoise messages ω (nandV messages ω)|} := by
    intro ω hω
    by_contra hc
    simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_lt] at hc
    have := run_correct hm0 hm1 ω hc.1 hc.2
    simp only [Set.mem_setOf_eq] at hω
    rw [this] at hω
    exact Bool.false_ne_true hω
  calc Measure.infinitePi μ {ω | failR messages ω = true}
      ≤ Measure.infinitePi μ {ω | inThreshold < |inNoise messages ω|} +
          Measure.infinitePi μ {ω | outThreshold < |outNoise messages ω (nandV messages ω)|} :=
        (measure_mono hsub).trans (measure_union_le _ _)
    _ ≤ ENNReal.ofReal (2 * Real.exp (720 * maskExcess - (inThreshold - 1440) ^ 2 / (2 * inProxy))) +
          2048 * ENNReal.ofReal (2 * Real.exp (-outThreshold ^ 2 / (2 * outProxy))) :=
        add_le_add (inNoise_tail messages) (outNoise_union messages)
    _ ≤ (2 : ENNReal)⁻¹ ^ 100 := final_numeric

end MxxFheTfhe
