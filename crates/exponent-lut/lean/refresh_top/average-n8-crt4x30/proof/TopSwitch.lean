import TopLut2

/-!
# Switching coefficients

The preprocessing computes the coefficient `C_σ = W⁻¹ Σ_k T(k) σ(Y^(-k))` of a LUT with the inverse
powers `Y^(-k) = -X^(2N/W·(W-k))`; it is the coefficient `lutCoefficient` of the LUT lemmas.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem root_pow_eight {q : Nat} : (X (n := N) q) ^ 8 = -1 := Negacyclic.root_pow_n

/-- `X^(8 m) = (-1)^m`. -/
theorem root_pow_mul_eight {q : Nat} (m : Nat) : (X (n := N) q) ^ (8 * m) = (-1) ^ m := by
  rw [pow_mul, root_pow_eight]

theorem neg_root_subst {q : Nat} (σ : Nat) (hodd : Odd σ) (a : Nat) :
    substitution (n := N) (R := ZMod q) σ hodd (-((X (n := N) q) ^ a)) = (X (n := N) q) ^ (8 * σ + a * σ) := by
  rw [map_neg, substitution_root_pow, pow_add, root_pow_mul_eight, Odd.neg_one_pow hodd]
  ring

/-- The width-`4` coefficient. -/
theorem switch_coeff4 {q : Nat} (σ : Nat) (hodd : Odd σ) (T : Fin 4 → ExactPoly q N)
    (inv : ExactPoly q N) :
    inv * (T 0 * substitution σ hodd ((X (n := N) q) ^ 0) + T 1 * substitution σ hodd (-((X (n := N) q) ^ 6)) +
      (T 2 * substitution σ hodd (-((X (n := N) q) ^ 4)) + T 3 * substitution σ hodd (-((X (n := N) q) ^ 2)))) =
      lutFourierCoefficient 4 (Y q) inv T σ := by
  unfold lutFourierCoefficient
  rw [Fin.sum_univ_four, neg_root_subst, neg_root_subst, neg_root_subst, substitution_root_pow]
  simp only [Y, ← pow_mul, Fin.val_zero, Fin.val_one, Fin.val_two]
  have h3 : ((3 : Fin 4) : Nat) = 3 := rfl
  rw [h3]
  have e0 : (X (n := N) q) ^ (0 * σ) = (X (n := N) q) ^ (2 * (σ * (2 * 4 - 0))) := by
    rw [show 2 * (σ * (2 * 4 - 0)) = 8 * (2 * σ) by ring, root_pow_mul_eight, pow_mul]
    simp
  have e1 : (X (n := N) q) ^ (8 * σ + 6 * σ) = (X (n := N) q) ^ (2 * (σ * (2 * 4 - 1))) := by
    congr 1; omega
  have e2 : (X (n := N) q) ^ (8 * σ + 4 * σ) = (X (n := N) q) ^ (2 * (σ * (2 * 4 - 2))) := by
    congr 1; omega
  have e3 : (X (n := N) q) ^ (8 * σ + 2 * σ) = (X (n := N) q) ^ (2 * (σ * (2 * 4 - 3))) := by
    congr 1; omega
  rw [e0, e1, e2, e3]
  ring

/-- The width-`2` coefficient. -/
theorem switch_coeff2 {q : Nat} (σ : Nat) (hodd : Odd σ) (T : Fin 2 → ExactPoly q N)
    (inv : ExactPoly q N) :
    inv * (T 0 * substitution σ hodd ((X (n := N) q) ^ 0) + T 1 * substitution σ hodd (-((X (n := N) q) ^ 7))) =
      lutFourierCoefficient 8 (X (n := N) q) inv T σ := by
  unfold lutFourierCoefficient
  rw [Fin.sum_univ_two, neg_root_subst, substitution_root_pow]
  simp only [Fin.val_zero, Fin.val_one]
  have e0 : (X (n := N) q) ^ (0 * σ) = (X (n := N) q) ^ (σ * (2 * 8 - 0)) := by
    rw [show σ * (2 * 8 - 0) = 8 * (2 * σ) by ring, root_pow_mul_eight, pow_mul]
    simp
  have e1 : (X (n := N) q) ^ (8 * σ + 7 * σ) = (X (n := N) q) ^ (σ * (2 * 8 - 1)) := by
    congr 1; omega
  rw [e0, e1]

end RefreshTopProof
