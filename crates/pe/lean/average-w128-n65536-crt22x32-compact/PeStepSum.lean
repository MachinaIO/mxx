import PeBranch

/-!
# The sum of the branches of a step

Every branch's encoding is acted on by its public scalar `V_b = Dec(W⁻¹ α_b g)`; the sum over the
128 branches of `H` interpolates the table.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

/-- The public scalar action of a coefficient runs. -/
theorem scalarV_runs (coef : ExactMatrix Q N 1 1) :
    gadgetDecomposeRuns Backend.backend 65536 44
      (matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g0) (scalarV coef) := by
  obtain ⟨layout', hl, hb, hd, hw, -⟩ := g0_runs
  rw [layoutQ] at hl
  cases hl
  have hw2 : 44 = 1 * Backend.layout0.digitCount := by simp at hw; omega
  let X := matrixMulScalarLeft (matrixMulScalarLeft winvPoly coef) g0
  have hD : gadgetDecomposeRuns Backend.backend 65536 44 X
      (castMatrixRows hw2.symm (regularDecomposeMatrix Backend.layout0 X)) :=
    ⟨Backend.layout0, layoutQ, hb, hd, hw2, rfl,
      regularGadgetMatrix_residual_exact Backend.layout0 X hQ hN rfl⟩
  exact Classical.epsilon_spec ⟨_, hD⟩

/-- The scalar action of a branch with coefficient `α`. -/
theorem branch_scalar {s μ : ErrorPoly N} {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    {c : Fin 2 → ExactMatrix Q N 1 44} {B : Nat}
    (h : Rel s A (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * μ * unitPair s j) • g0) c B)
    (coef : ExactMatrix Q N 1 1) :
    Rel s (fun r j ↦ A r j * scalarV coef)
      (fun j ↦ (reducePoly Q N (unitPair s j) * (reducePoly Q N μ * ((Winv : ExactPoly Q N) * coef 0 0))) • g0)
      (fun j ↦ c j * scalarV coef) (scalarBound B) := by
  refine Rel.target_eq (scalar_rel h (scalarV_runs coef)) ?_
  funext j
  congr 1
  simp only [matrixMulScalarLeft_eq, Matrix.smul_apply, smul_eq_mul, winvPoly, matrixPolynomial]
  simp
  ring

/-- The monomial `X^e` routed by `σ_k` reduces to `X^(e k)`. -/
theorem reduce_subst_Xe (k : Nat) (hk : Odd k) (e : Nat) :
    reducePoly Q N (substitution k hk (Xe (n := N) e)) = X Q ^ (e * k) := by
  rw [Xe, substitution_root_pow, ← reducePoly_Xe]

/-- The error bound of a step. -/
def stepBound (Bp B1 B2 : Nat) : Nat :=
  scalarBound Bp + 15 * scalarBound B1 + 7 * scalarBound B1 + 105 * scalarBound B2

set_option maxRecDepth 200000 in
set_option maxHeartbeats 64000000 in
/-- The sum of the 128 branches of a step: for `μ = X^e` with `e < 128` the branch encodings, each
acted on by its scalar, sum to an encoding of the table value `T e`. -/
theorem step_sum {s : ErrorPoly N} {e : Nat} (he : e < 128) (table : Fin (2 ^ 7) → ExactPoly Q N)
    (coef : Nat → ExactMatrix Q N 1 1)
    (hcoef : ∀ k, Odd k → (Winv : ExactPoly Q N) * coef k 0 0 =
      lutFourierCoefficient N (X Q) (Winv : ExactPoly Q N) table k)
    {Aid : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {cid : Fin 2 → ExactMatrix Q N 1 44} {Bp : Nat}
    (hid : Rel s Aid (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) * Xe e * unitPair s j) • g0) cid Bp)
    {Kint : Fin 15 → Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {rint : Fin 15 → Fin 2 → ExactMatrix Q N 1 44}
    {B1 : Nat}
    (hint : ∀ b : Fin 15, Rel s (Kint b) (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      substitution (kSw b) (kSw_odd ⟨b, by omega⟩) (Xe e) * unitPair s j) • g0) (rint b) B1)
    {Kl1 : Fin 7 → Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {rl1 : Fin 7 → Fin 2 → ExactMatrix Q N 1 44}
    (hl1 : ∀ i : Fin 7, Rel s (Kl1 i) (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      substitution (kSw (15 + i)) (kSw_odd ⟨15 + i, by omega⟩) (Xe e) * unitPair s j) • g0) (rl1 i) B1)
    {Kl2 : Fin 105 → Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {rl2 : Fin 105 → Fin 2 → ExactMatrix Q N 1 44}
    {B2 : Nat}
    (hl2 : ∀ l : Fin 105, Rel s (Kl2 l) (fun j ↦ reducePoly Q N ((((1 : Nat) : Int) : ErrorPoly N) *
      substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt))
        (substitution (kSw (l % 15)) (kSw_odd ⟨l % 15, by omega⟩) (Xe e)) * unitPair s j) • g0) (rl2 l) B2) :
    Rel s (fun r j ↦ Aid r j * scalarV (coef 1) + ∑ b : Fin 15, Kint b r j * scalarV (coef (kSw b)) +
        ∑ i : Fin 7, Kl1 i r j * scalarV (coef (kSw (15 + i))) + ∑ l : Fin 105, Kl2 l r j * scalarV (coef (kLeaf2 l)))
      (fun j ↦ (reducePoly Q N (unitPair s j) * table ⟨e, by norm_num; omega⟩) • g0)
      (fun j ↦ cid j * scalarV (coef 1) + ∑ b : Fin 15, rint b j * scalarV (coef (kSw b)) +
        ∑ i : Fin 7, rl1 i j * scalarV (coef (kSw (15 + i))) + ∑ l : Fin 105, rl2 l j * scalarV (coef (kLeaf2 l)))
      (stepBound Bp B1 B2) := by
  have s0 := branch_scalar hid (coef 1)
  have s1 := Rel.sum Finset.univ (fun b _ ↦ branch_scalar (hint b) (coef (kSw b)))
  have s2 := Rel.sum Finset.univ (fun i _ ↦ branch_scalar (hl1 i) (coef (kSw (15 + i))))
  have s3 := Rel.sum Finset.univ (fun l _ ↦ branch_scalar (hl2 l) (coef (kLeaf2 l)))
  have hsum := ((s0.add s1).add s2).add s3
  refine Rel.mono (Rel.target_eq hsum ?_) (le_of_eq ?_)
  rotate_left
  · simp [stepBound]
  have hint_val := step_interp (X Q) (Winv : ExactPoly Q N) Negacyclic.root_pow_n winv_mul table
    (fun k ↦ (Winv : ExactPoly Q N) * coef k 0 0) hcoef ⟨e, he⟩
  funext j
  simp only [Pi.add_apply, Finset.sum_apply]
  rw [← Finset.sum_smul, ← Finset.sum_smul, ← Finset.sum_smul, ← add_smul, ← add_smul, ← add_smul]
  congr 1
  rw [← Finset.mul_sum, ← Finset.mul_sum, ← Finset.mul_sum, ← mul_add, ← mul_add, ← mul_add]
  congr 1
  rw [← hint_val]
  simp only [reducePoly_Xe, reduce_subst_Xe]
  congr 1
  refine Finset.sum_congr rfl fun l _ ↦ ?_
  have hin : substitution (kSw (l % 15)) (kSw_odd ⟨l % 15, by omega⟩) (Xe (n := N) e) =
      Xe (e * kSw (l % 15)) := substitution_root_pow _ _ _
  rw [hin, reduce_subst_Xe, Nat.mul_assoc]
  congr 1
  exact X_pow_leaf2 e l l.isLt

end PeProof
