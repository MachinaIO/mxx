import RuntimeSampling
import RuntimeSubGaussian

/-!
# The rounding-residue replacement hypothesis (H2)

A ModDown of a ciphertext row rounds every coefficient; its residue is a deterministic function of
the execution. The average-case noise analysis of a scheme commonly treats such residues as
independent uniforms (the modulus-switching noise formulas of TFHE-rs do so for the LWE body). A
certificate that relies on it takes `ResidueReplacement` as an explicit hypothesis: the tail of a
noise form `L + Σ j, g j * ρ j` is at most the tail of the same form with every residue replaced
by an independent uniform integer on `[-h j, h j]`.

The replacements are tape coordinates at the reserved site `residueSite`, which no generated
stage samples (a stage samples at sites that start with its position in the workflow), each with
the law `SamplerLaw.interval (-h j) (h j)`, one coordinate per site through an injective numbering. The hypothesis is an inequality between tails, not an
equality of laws, and has no antecedent; a certificate proves separately that `L` and `g` read no
replacement coordinate and that every residue lies in `[-h j, h j]`.
-/

namespace MxxRuntime

/-- The reserved site of the replacement coordinates. -/
def residueSite : List Nat := [1000000]

/-- The replacement coordinate of residue `j`, uniform on `[-h, h]`. -/
def residueKey (h : Nat) (j : Nat) : SampleKey :=
  ⟨residueSite, 0, 0, j, .interval (-(h : Int)) h⟩

/-- H2: replacing the residues `ρ` of the sites `ι` by independent uniforms does not decrease the
tail of the noise form `L + Σ j, g j * ρ j` at `m`. Site `j` is replaced by the coordinate
`residueKey (h j) (idx j)`, for an injective numbering `idx` of the sites. -/
def ResidueReplacement {ι : Type} [Fintype ι] (idx : ι → Nat) (ρ : ι → SampleTape → Int)
    (h : ι → Nat) (L : SampleTape → ℝ) (g : ι → SampleTape → ℝ) (m : ℝ) : Prop :=
  tapeMeasure {t | m ≤ |L t + ∑ j, g j t * (ρ j t : ℝ)|} ≤
    tapeMeasure {t | m ≤ |L t + ∑ j, g j t * (t (residueKey (h j) (idx j)) : ℝ)|}

theorem residueKey_injective {ι : Type} {idx : ι → Nat} (hidx : Function.Injective idx)
    (h : ι → Nat) : Function.Injective fun j : ι ↦ residueKey (h j) (idx j) := by
  intro a b hab
  simp only [residueKey, SampleKey.mk.injEq] at hab
  exact hidx hab.2.2.2.1

/-- Under H2, the tail of `L + Σ j, g j ρ j` at `m` is at most the probability that the energy
`Σ h² g²` of the coefficients exceeds `E` plus the Gaussian tail of proxy `E` at `m - Lmax`, when
`|L| ≤ Lmax` on every tape and the coefficients read no replacement coordinate. -/
theorem tail_of_residueReplacement {ι : Type} [Fintype ι] [DecidableEq ι] {idx : ι → Nat}
    (hidx : Function.Injective idx) {ρ : ι → SampleTape → Int} {h : ι → Nat}
    {L : SampleTape → ℝ} {g : ι → SampleTape → ℝ} {m : ℝ}
    (hH2 : ResidueReplacement idx ρ h L g m) {Lmax : ℝ} (hL : ∀ t, |L t| ≤ Lmax) (hm : Lmax ≤ m)
    (T : Finset SampleKey) (hT : ∀ j, residueKey (h j) (idx j) ∉ T)
    (hg : DependsOn (fun t j ↦ g j t) T) {E : ℝ} (hE : 0 < E) :
    tapeMeasure {t | m ≤ |L t + ∑ j, g j t * (ρ j t : ℝ)|} ≤
      tapeMeasure {t | E < ∑ j, ((h j : ℝ) ^ 2) * g j t ^ 2} +
        ENNReal.ofReal (2 * Real.exp (-(m - Lmax) ^ 2 / (2 * E))) := by
  refine hH2.trans ?_
  have hsub : {t | m ≤ |L t + ∑ j, g j t * (t (residueKey (h j) (idx j)) : ℝ)|} ⊆
      {t | m - Lmax ≤ |∑ j, g j t * (t (residueKey (h j) (idx j)) : ℝ)|} := by
    intro t ht
    simp only [Set.mem_setOf_eq] at ht ⊢
    have h1 := abs_add_le (L t) (∑ j, g j t * (t (residueKey (h j) (idx j)) : ℝ))
    have h2 := hL t
    linarith
  refine (MeasureTheory.measure_mono hsub).trans ?_
  have hlaw : ∀ (j : ι) (a : ℝ), ∫⁻ y, ENNReal.ofReal (Real.exp (a * y))
      ∂(residueKey (h j) (idx j)).law.measure ≤
      ENNReal.ofReal (Real.exp ((h j : ℝ) ^ 2 * a ^ 2 / 2)) := fun j a ↦ by
    simpa [residueKey] using SamplerLaw.lintegral_exp_interval_symmetric_le (h j) a
  exact tail_linear_truncated' (μ := fun key : SampleKey ↦ key.law.measure)
    (fun j : ι ↦ residueKey (h j) (idx j)) (residueKey_injective hidx h) hlaw T hT hg hE
    (by linarith)

end MxxRuntime
