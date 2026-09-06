import ExponentBounds

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- The target gain sums absolute integer coefficients. In particular the table
need not be Boolean: a canonical table coefficient of two contributes three. -/
theorem nativeTarget_polynomialL1_sum_le {n : Nat} {α : Type} [Fintype α]
    (values : α → ErrorPoly n) :
    polynomialL1 (∑ i, values i) ≤ ∑ i, polynomialL1 (values i) := by
  classical
  have add_bound (x y : ErrorPoly n) :
      polynomialL1 (x + y) ≤ polynomialL1 x + polynomialL1 y := by
    unfold polynomialL1
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun i _ => by
      rw [Negacyclic.coeff_add]
      exact Int.natAbs_add_le _ _
  have bound (s : Finset α) :
      polynomialL1 (∑ i ∈ s, values i) ≤ ∑ i ∈ s, polynomialL1 (values i) := by
    induction s using Finset.induction_on with
    | empty => simp [polynomialL1]
    | @insert a s ha ih =>
      simp only [Finset.sum_insert ha]
      exact (add_bound _ _).trans (Nat.add_le_add_left ih _)
  exact bound Finset.univ

/-- A signed basis lift retains the evaluator's negacyclic sign without making
any assumption about the signs or magnitudes of the source table. -/
theorem target_basis_gain {n width : Nat} (hn : 0 < n)
    (coefficient : Fin width → Int) (position : Fin width → Fin n) :
    polynomialL1 (∑ k, scaledBasis (coefficient k) (position k)) ≤
      ∑ k, (coefficient k).natAbs := by
  exact (nativeTarget_polynomialL1_sum_le _).trans_eq (by
    simp only [polynomialL1_scaledBasis hn])

theorem reduce_scaled_basis {n : Nat} (q : Nat) (a : Int) (i : Fin n) :
    reducePoly q n (scaledBasis a i) = (a : ExactPoly q n) *
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ i.val := by
  simp [scaledBasis, reducePoly]

theorem coefficient_integer_basis {q n : Nat} [Fact (1 < q)] (hn : 0 < n)
    (a : Int) (i j : Fin n) :
    ((a : ExactPoly q n) * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ i.val).coeff j =
      if i = j then (a : ZMod q) else 0 := by
  change (algebraMap (ZMod q) (ExactPoly q n) (a : ZMod q) * _).coeff j = _
  rw [Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  split_ifs <;> simp

end MxxExponentLut
