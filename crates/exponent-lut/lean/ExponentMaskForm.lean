import ExponentErrorForm
import RuntimeSubGaussian

/-!
# Mask-weighted terms as sub-Gaussian linear forms

A ModDown adds the term `s₀ r` to the error, for the ternary mask `s₀` and a residue `r` of the
public key. Coefficient `i` of `s₀ w` is the linear form `Σ_k s₀_k c_k` in the mask coefficients,
`c_k = Σ_j w_j ε(k, j, i)` with the negacyclic sign `ε`. The mask coefficients are independent
tape coordinates uniform on `[-1, 1]`, so when `w` reads no mask coordinate the form is
sub-Gaussian with the proxy `Σ_k c_k²` (Hoeffding), whatever `w` is.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime MeasureTheory
open scoped BigOperators

variable {n : Nat}

/-- The sign of `X^i X^j` at `X^k` in `ℤ[X]/(X^n + 1)`. -/
def negSign (n : Nat) (i j k : Fin n) : Int :=
  if (i.val + j.val) % n = k.val then (if i.val + j.val < n then 1 else -1) else 0

/-- The coefficients of a negacyclic product. -/
theorem coeff_mul_eq (hn : 0 < n) (x y : ErrorPoly n) (k : Fin n) :
    (x * y).coeff k = ∑ i, ∑ j, x.coeff i * y.coeff j * negSign n i j k := by
  conv_lhs => rw [Negacyclic.expansion hn x, Negacyclic.expansion hn y]
  rw [Finset.sum_mul_sum, Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  have h := coeff_scaled_basis_mul hn (x.coeff i) (y.coeff j) i j k
  unfold scaledBasis at h
  rw [h, negSign]
  split_ifs <;> ring

/-- The tape coordinate of coefficient `k` of the mask, sampled at `site`. -/
def maskKey (site : List Nat) (k : Nat) : SampleKey := ⟨site, 0, 0, k, .interval (-1) 1⟩

theorem maskKey_injective (site : List Nat) : Function.Injective (maskKey site) := by
  intro a b h
  simp only [maskKey, SampleKey.mk.injEq] at h
  exact h.2.2.2.1

/-- The mask read from the tape. -/
noncomputable def maskPoly (n : Nat) (site : List Nat) (tape : SampleTape) : ErrorPoly n :=
  intPoly fun k : Fin n ↦ tape (maskKey site k.val)

/-- The coefficient of `X^k` of the mask in coefficient `i` of `s₀ w`. -/
noncomputable def maskCoefficient (w : ErrorPoly n) (i k : Fin n) : Int :=
  ∑ j, w.coeff j * negSign n k j i

theorem maskPoly_mul_coeff (hn : 0 < n) (site : List Nat) (tape : SampleTape) (w : ErrorPoly n)
    (i : Fin n) :
    (maskPoly n site tape * w).coeff i =
      ∑ k : Fin n, maskCoefficient w i k * tape (maskKey site k.val) := by
  rw [coeff_mul_eq hn]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [maskPoly, intPoly_coeff hn, maskCoefficient, Finset.sum_mul]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  ring

/-- The mask uniform on `[-1, 1]` has the Hoeffding proxy `1`. -/
theorem maskKey_law (site : List Nat) (k : Nat) (a : ℝ) :
    ∫⁻ y, ENNReal.ofReal (Real.exp (a * y)) ∂(maskKey site k).law.measure ≤
      ENNReal.ofReal (Real.exp (1 * a ^ 2 / 2)) := by
  have h := SamplerLaw.lintegral_exp_interval_symmetric_le 1 a
  simpa [maskKey] using h

/-- The mask-weighted term `s₀ w` with `w` reading only coordinates outside the mask: every
coefficient `i` is sub-Gaussian over the mask keys with the proxy `e i` of its coefficient energy. -/
theorem maskForm_csubG (hn : 0 < n) (site : List Nat) {w : SampleTape → ErrorPoly n}
    (T : Finset SampleKey) (hT : ∀ k : Fin n, maskKey site k.val ∉ T) (hw : DependsOn w T)
    {e : Fin n → ℝ} (he : ∀ tape i, ∑ k, (maskCoefficient (w tape) i k : ℝ) ^ 2 ≤ e i) :
    CSubG (fun key : SampleKey ↦ key.law.measure)
      (fun tape i ↦ (((maskPoly n site tape * w tape).coeff i : Int) : ℝ))
      (Finset.univ.image fun k : Fin n ↦ maskKey site k.val) (fun i ↦ 1 * e i) := by
  have hinj : Function.Injective fun k : Fin n ↦ maskKey site k.val :=
    fun a b h ↦ Fin.ext (maskKey_injective site h)
  have hc : DependsOn (fun tape i k ↦ (maskCoefficient (w tape) i k : ℝ)) T := by
    intro t t' h
    simp only [hw h]
  have hlin := CSubG.linear (μ := fun key : SampleKey ↦ key.law.measure)
    (fun k : Fin n ↦ maskKey site k.val) hinj zero_le_one
    (fun k a ↦ maskKey_law site k.val a) T hT hc he
  refine (show (fun tape i ↦ (((maskPoly n site tape * w tape).coeff i : Int) : ℝ)) =
      fun tape i ↦ ∑ k, (maskCoefficient (w tape) i k : ℝ) * (tape (maskKey site k.val) : ℝ) from
    ?_) ▸ hlin
  funext tape i
  rw [maskPoly_mul_coeff hn]
  push_cast
  rfl

end MxxExponentLut
