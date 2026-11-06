import PeOnlineBase

/-!
# The radix LUT identity

The `256` branches of a step are the automorphisms `σ_{1 + 256 i}`, `i < 256`, of the subgroup
`H = ⟨257⟩` of `(ℤ/2N)^×`: the identity (`i = 0`), the internal branches `kSw a` (`i = a + 1`),
the first-depth leaves `kSw (15 + b)` (`i = 16 (b + 1)`), and the second-depth leaves `kLeaf2 l`
(`i = l % 15 + 1 + 16 (l / 15 + 1)`). The scalar action of branch `k` carries
`W⁻¹ Σ_j X^(t_j) σ_k(X^(-j))`, and the branches together interpolate the table.
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The automorphism `1 + 256 i` of branch `i` of `H`. -/
def hIndex (i : Nat) : Nat := 1 + 256 * i

theorem kSw_internal (a : Nat) (ha : a < 15) : kSw a = hIndex (a + 1) := by
  interval_cases a <;> rfl

theorem kSw_leaf (b : Nat) (hb : b < 15) : kSw (15 + b) = hIndex (16 * (b + 1)) := by
  interval_cases b <;> rfl

theorem kLeaf2_eq (l : Nat) (hl : l < 225) : kLeaf2 l = hIndex (l % 15 + 1 + 16 * (l / 15 + 1)) := by
  interval_cases l <;> rfl

/-- The second-depth automorphism is the composition of its parent's and its member's. -/
theorem kLeaf2_comp (l : Nat) (hl : l < 225) :
    kLeaf2 l % (2 * N) = (kSw (l % 15) * kSw (15 + l / 15)) % (2 * N) := by
  interval_cases l <;> rfl

/-- The sum over `H` splits into the identity, the internal branches, the first-depth leaves,
and the second-depth leaves. -/
theorem sum_H {M : Type} [AddCommMonoid M] (F : Nat → M) :
    ∑ i : Fin 256, F i = F 0 + ∑ a : Fin 15, F (a + 1) + ∑ b : Fin 15, F (16 * (b + 1)) +
      ∑ l : Fin 225, F (l % 15 + 1 + 16 * (l / 15 + 1)) := by
  have hsplit : ∀ G : Nat → M, ∑ i : Fin 256, G i = ∑ b : Fin 16, ∑ a : Fin 16, G (a + 16 * b) := by
    intro G
    rw [← (finProdFinEquiv : Fin 16 × Fin 16 ≃ Fin 256).sum_comp, Fintype.sum_prod_type]
    rfl
  have h105 : ∀ G : Nat → M, ∑ l : Fin 225, G l = ∑ b : Fin 15, ∑ a : Fin 15, G (a + 15 * b) := by
    intro G
    rw [← (finProdFinEquiv : Fin 15 × Fin 15 ≃ Fin 225).sum_comp, Fintype.sum_prod_type]
    rfl
  rw [hsplit, h105 (fun l ↦ F (l % 15 + 1 + 16 * (l / 15 + 1)))]
  rw [Fin.sum_univ_succ, Fin.sum_univ_succ]
  simp only [Fin.val_zero, Fin.val_succ, mul_zero, add_zero, zero_add, Fin.sum_univ_succ (n := 15)]
  rw [Finset.sum_add_distrib]
  have hl : ∀ (b : Fin 15) (a : Fin 15), (a + 15 * b : Nat) % 15 + 1 + 16 * ((a + 15 * b : Nat) / 15 + 1) =
      a + 1 + 16 * (b + 1) := by
    intro b a
    have : (a + 15 * b : Nat) % 15 = a := by omega
    have : (a + 15 * b : Nat) / 15 = b := by omega
    omega
  simp only [hl]
  abel

/-- The table entry `j` of the step at `offset` as the monomial it encodes. -/
noncomputable def tableMono (tables : Fin 512 → Int) (offset : Nat) (ho : offset + 256 ≤ 512) (j : Fin 256) :
    ExactPoly Q N :=
  X Q ^ (tables ⟨offset + j, by omega⟩).toNat

/-- The table entry `j` of the last step as the constant it encodes. -/
noncomputable def tableConst (tables : Fin 512 → Int) (offset : Nat) (ho : offset + 256 ≤ 512) (j : Fin 256) :
    ExactPoly Q N :=
  (tables ⟨offset + j, by omega⟩ : ExactPoly Q N)

theorem monomial_entry (e : Int) (he0 : 0 ≤ e) (he : e < 2 * N) :
    multiplyMonomial (1 : ExactMatrix Q N 1 1) (e * 1) 0 0 = X Q ^ e.toNat := by
  simp only [multiplyMonomial, Matrix.one_apply_eq, one_mul, mul_one]
  have h2 : (2 * (N : Int)) = 65536 := rfl
  rw [h2] at he
  rw [show (2 * ((N : Nat) : Int)) = 65536 from rfl, Int.emod_eq_of_lt he0 he]

theorem autMonomial_entry (k : Nat) (hk : Odd k) (j : Fin 256) :
    ringAutomorphism k (multiplyMonomial (1 : ExactMatrix Q N 1 1) ((65536 - (j : Int)) * 1)) 0 0 =
      X Q ^ (k * (2 * N - j)) := by
  rw [ringAutomorphism_eq hQ hN k hk]
  simp only [multiplyMonomial, Matrix.one_apply_eq, one_mul, mul_one]
  rw [substitution_root_pow, root_pow_mod_two_n, root_pow_mod_two_n (k := k * (2 * N - j))]
  congr 1
  have hj := j.isLt
  have h2 : 2 * N = 65536 := rfl
  have h2' : (2 * ((N : Nat) : Int)) = 65536 := rfl
  rw [h2, h2']
  have : ((65536 - (j : Int)) % 65536).toNat = (65536 - (j : Nat)) % 65536 := by omega
  rw [this, Nat.mul_mod, Nat.mod_mod, ← Nat.mul_mod, Nat.mul_comm]

set_option maxRecDepth 100000 in
/-- The coefficient of branch `k`, scaled by `W⁻¹`, is the Fourier coefficient of the table. -/
theorem branchCoef_fourier (tables : Fin 512 → Int) (offset : Nat) (ho : offset + 256 ≤ 512)
    (ht : ∀ j : Fin 256, 0 ≤ tables ⟨offset + j, by omega⟩ ∧ tables ⟨offset + j, by omega⟩ < 2 * N)
    (k : Nat) (hk : Odd k) :
    (Winv : ExactPoly Q N) * branchCoef tables offset k ho 0 0 =
      lutFourierCoefficient N (X Q) (Winv : ExactPoly Q N) (tableMono tables offset ho) k := by
  unfold lutFourierCoefficient branchCoef
  refine congrArg (fun x ↦ (Winv : ExactPoly Q N) * x) ?_
  rw [Matrix.sum_apply]
  apply Finset.sum_congr rfl
  intro j _
  simp only [coefTerm, matrixMulScalarLeft_eq, Matrix.smul_apply, smul_eq_mul]
  rw [monomial_entry _ (ht j).1 (ht j).2, autMonomial_entry k hk j]
  rfl

set_option maxRecDepth 100000 in
/-- The constant-coefficient form for the last step. -/
theorem branchCoefC_fourier (tables : Fin 512 → Int) (offset : Nat) (ho : offset + 256 ≤ 512)
    (k : Nat) (hk : Odd k) :
    (Winv : ExactPoly Q N) * branchCoefC tables offset k ho 0 0 =
      lutFourierCoefficient N (X Q) (Winv : ExactPoly Q N) (tableConst tables offset ho) k := by
  unfold lutFourierCoefficient branchCoefC
  refine congrArg (fun x ↦ (Winv : ExactPoly Q N) * x) ?_
  rw [Matrix.sum_apply]
  apply Finset.sum_congr rfl
  intro j _
  simp only [coefTermC, matrixMulScalarLeft_eq, Matrix.smul_apply, smul_eq_mul]
  rw [autMonomial_entry k hk j]
  simp [liftInteger, tableConst]

theorem winv_mul : (Winv : ExactPoly Q N) * ((2 ^ 8 : Nat) : ExactPoly Q N) = 1 :=
  inverse_mul (k := 71413264965178608184357752453475574199477050220518708989523601544437844709191912168638523104835678117098241 * 256 / 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489) (by norm_num)

/-- The branches of `H` interpolate the table on the lookup domain. -/
theorem lut_H {R : Type} [CommRing R] (root inverse : R) (hroot : root ^ N = -1)
    (hinverse : inverse * ((2 ^ 8 : Nat) : R) = 1) (table : Fin (2 ^ 8) → R) (e : Fin (2 ^ 8)) :
    ∑ i : Fin 256, root ^ (hIndex i * e) * lutFourierCoefficient N root inverse table (hIndex i) =
      table e :=
  lut_fourier_table_value 8 N 128 root inverse table e (by norm_num) (by norm_num) hroot hinverse
    (fun i ↦ hIndex i) (Equiv.refl _) (fun i ↦ by
      have := i.isLt
      simp only [hIndex, Equiv.refl_apply, show 2 * N = 65536 from rfl]
      omega)

theorem hIndex_odd (i : Nat) : Odd (hIndex i) := ⟨128 * i, by unfold hIndex; ring⟩

theorem kLeaf2_odd (l : Nat) (hl : l < 225) : Odd (kLeaf2 l) := by
  rw [kLeaf2_eq l hl]; exact hIndex_odd _

/-- `X^(e p m) = X^(e kLeaf2 l)`: the second-depth automorphism composes its parent's and its
member's. -/
theorem X_pow_leaf2 {R : Type} [CommRing R] (e l : Nat) (hl : l < 225) :
    AdjoinRoot.root (negacyclicModulus N R) ^ (e * (kSw (l % 15) * kSw (15 + l / 15))) =
      AdjoinRoot.root (negacyclicModulus N R) ^ (e * kLeaf2 l) := by
  rw [root_pow_mod_two_n, root_pow_mod_two_n (k := e * kLeaf2 l), Nat.mul_mod,
    Nat.mul_mod e (kLeaf2 l), kLeaf2_comp l hl]

/-- The `256` branches of a step with table entries in `[0, 2N)` interpolate the table at `e`. -/
theorem step_interp {R : Type} [CommRing R] (root inverse : R) (hroot : root ^ N = -1)
    (hinverse : inverse * ((2 ^ 8 : Nat) : R) = 1) (table : Fin (2 ^ 8) → R)
    (coef : Nat → R) (hcoef : ∀ k, Odd k → coef k = lutFourierCoefficient N root inverse table k)
    (e : Fin 256) :
    root ^ e.val * coef 1 + ∑ a : Fin 15, root ^ (e.val * kSw a) * coef (kSw a) +
      ∑ b : Fin 15, root ^ (e.val * kSw (15 + b)) * coef (kSw (15 + b)) +
      ∑ l : Fin 225, root ^ (e.val * kLeaf2 l) * coef (kLeaf2 l) = table e := by
  rw [← lut_H root inverse hroot hinverse table e, sum_H (fun i ↦ root ^ (hIndex i * e) *
    lutFourierCoefficient N root inverse table (hIndex i))]
  have h1 : hIndex 0 = 1 := rfl
  have e1 : ∑ a : Fin 15, root ^ (e.val * kSw a) * coef (kSw a) =
      ∑ a : Fin 15, root ^ (hIndex ((a : Nat) + 1) * e) *
        lutFourierCoefficient N root inverse table (hIndex ((a : Nat) + 1)) :=
    Finset.sum_congr rfl fun a _ ↦ by rw [kSw_internal a a.isLt, hcoef _ (hIndex_odd _), Nat.mul_comm]
  have e2 : ∑ b : Fin 15, root ^ (e.val * kSw (15 + b)) * coef (kSw (15 + b)) =
      ∑ b : Fin 15, root ^ (hIndex (16 * ((b : Nat) + 1)) * e) *
        lutFourierCoefficient N root inverse table (hIndex (16 * ((b : Nat) + 1))) :=
    Finset.sum_congr rfl fun b _ ↦ by rw [kSw_leaf b b.isLt, hcoef _ (hIndex_odd _), Nat.mul_comm]
  have e3 : ∑ l : Fin 225, root ^ (e.val * kLeaf2 l) * coef (kLeaf2 l) =
      ∑ l : Fin 225, root ^ (hIndex ((l : Nat) % 15 + 1 + 16 * ((l : Nat) / 15 + 1)) * e) *
        lutFourierCoefficient N root inverse table (hIndex ((l : Nat) % 15 + 1 + 16 * ((l : Nat) / 15 + 1))) :=
    Finset.sum_congr rfl fun l _ ↦ by rw [kLeaf2_eq l l.isLt, hcoef _ (hIndex_odd _), Nat.mul_comm]
  rw [e1, e2, e3, h1, hcoef 1 (by decide), Nat.one_mul]

end PeProof
