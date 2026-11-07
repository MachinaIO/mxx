import RuntimeMatrixOps

/-! The all-coefficients constant (`ConstantMatrix::AllCoefficients`). A generated module imports
this module only when it uses the constant, so the modules built on `MxxRuntime` alone do not
depend on it. -/

namespace MxxRuntime

open Mxx.Primitives
open scoped BigOperators

/-- Every entry is the polynomial whose `n` coefficients all equal `value`: one value, so the
exported term does not grow with the ring dimension. -/
noncomputable def allCoefficients {q n rows columns : Nat} (value : Int) :
    ExactMatrix q n rows columns :=
  fun _ _ ↦ polynomialOfCoefficients fun _ ↦ value

theorem allCoefficients_coeff {q n rows columns : Nat} (hq : 1 < q) (hn : 0 < n) (value : Int)
    (row : Fin rows) (column : Fin columns) (i : Fin n) :
    ((allCoefficients (q := q) (n := n) value : ExactMatrix q n rows columns) row column).coeff i =
      (value : ZMod q) :=
  polynomialOfCoefficients_coeff hq hn _ i

/-- A literal list repeating `value` `n` times is the all-coefficients constant. -/
theorem matrixPolynomial_replicate {q n : Nat} (value : Int) (row column : Fin 1) :
    (matrixPolynomial (List.replicate n value) : ExactMatrix q n 1 1) row column =
      (allCoefficients value : ExactMatrix q n 1 1) row column := by
  have hfold : ∀ k, (matrixPolynomial (List.replicate k value) : ExactMatrix q n 1 1) row column =
      ∑ i ∈ Finset.range k,
        (value : ExactPoly q n) * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ i := by
    intro k
    induction k with
    | zero => simp [matrixPolynomial]
    | succ k ih =>
      simp only [matrixPolynomial, List.replicate_succ] at ih ⊢
      simp only [bind, pure, List.flatMap_cons, List.singleton_append, List.foldr_cons] at ih ⊢
      rw [ih, Finset.sum_range_succ', Finset.mul_sum, add_comm]
      simp only [pow_succ, pow_zero, mul_one]
      congr 1
      apply Finset.sum_congr rfl
      intro i _
      ring
  rw [hfold]
  exact Fin.sum_univ_eq_sum_range
    (fun i ↦ (value : ExactPoly q n) * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ i) n |>.symm

end MxxRuntime
