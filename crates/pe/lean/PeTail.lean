import RuntimeSubGaussian
import RuntimeSampling

/-!
# The tail of a decryption

A phase error that is a fixed bias plus a linear form in fresh Gaussian keys, with coefficients
that read no key and have bounded energy, exceeds its budget at some coefficient with probability
at most a union of Gaussian tails (`tapeMeasure_linear_tail`).
-/

namespace MxxPe

open MeasureTheory MxxRuntime

/-- For `rows` linear forms in the fresh Gaussian keys `key x`, with coefficients `c ω i x` that
read only keys of `T` (which holds no `key x`) and have energy at most `E`, some form reaches `m`
in absolute value with probability at most `rows · 2 exp (-m² / (2 σ² E))`. -/
theorem tapeMeasure_linear_tail {α : Type} [Fintype α] [DecidableEq α] {rows : Nat}
    (key : α → SampleKey) (hinj : Function.Injective key) {sigma : Rat} (hsigma : sigma ≠ 0)
    {cutoff : Int} (hlaw : ∀ x, (key x).law = .gaussian sigma cutoff)
    (c : SampleTape → Fin rows → α → ℝ) (T : Finset SampleKey) (hT : ∀ x, key x ∉ T)
    (hc : ∀ i, DependsOn (fun ω ↦ c ω i) (T : Set SampleKey)) {E : ℝ} (hE : 0 < E)
    (henergy : ∀ ω i, ∑ x, c ω i x ^ 2 ≤ E) {m : ℝ} (hm : 0 ≤ m) :
    tapeMeasure {ω | ∃ i, m ≤ |∑ x, c ω i x * ω (key x)|} ≤
      (rows : ENNReal) * ENNReal.ofReal (2 * Real.exp (-m ^ 2 / (2 * ((sigma : ℝ) ^ 2 * E)))) := by
  classical
  have hV : 0 < (sigma : ℝ) ^ 2 := by
    have : (sigma : ℝ) ≠ 0 := by exact_mod_cast hsigma
    positivity
  have hsets : {ω : SampleTape | ∃ i, m ≤ |∑ x, c ω i x * ω (key x)|} =
      ⋃ i, {ω | m ≤ |∑ x, c ω i x * ω (key x)|} := by
    ext ω
    simp
  rw [hsets]
  refine (measure_iUnion_fintype_le _ _).trans ?_
  have hone : ∀ i : Fin rows, tapeMeasure {ω | m ≤ |∑ x, c ω i x * ω (key x)|} ≤
      ENNReal.ofReal (2 * Real.exp (-m ^ 2 / (2 * ((sigma : ℝ) ^ 2 * E)))) := by
    intro i
    have h := tail_linear_truncated (μ := fun k : SampleKey ↦ k.law.measure) key hinj hV
      (fun x a ↦ by dsimp only; rw [hlaw x]; exact SamplerLaw.lintegral_exp_gaussian_le sigma cutoff a)
      (c := fun ω ↦ c ω i) T hT (hc i) hE hm
    have hempty : {ω : SampleKey → ℤ | E < ∑ x, c ω i x ^ 2} = ∅ := by
      ext ω
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_lt]
      exact henergy ω i
    rw [hempty, measure_empty, zero_add] at h
    exact h
  calc ∑ i : Fin rows, tapeMeasure {ω | m ≤ |∑ x, c ω i x * ω (key x)|}
      ≤ ∑ _i : Fin rows, ENNReal.ofReal (2 * Real.exp (-m ^ 2 / (2 * ((sigma : ℝ) ^ 2 * E)))) :=
        Finset.sum_le_sum fun i _ ↦ hone i
    _ = _ := by simp

end MxxPe
