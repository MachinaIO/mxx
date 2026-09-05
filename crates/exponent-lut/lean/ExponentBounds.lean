import MxxPrimitives.Bounds

namespace MxxExponentLut

open Mxx.Primitives
open scoped BigOperators

/-! The simulator's `l1` gain is a maximum column sum of absolute integer
coefficients, not a coefficient maximum. These lemmas retain that distinction. -/

noncomputable def polynomialL1 {n : Nat} (x : ErrorPoly n) : Nat :=
  ∑ i : Fin n, (x.coeff i).natAbs

private theorem coefficient_sum {n : Nat} {α : Type} [DecidableEq α]
    (s : Finset α) (f : α → ErrorPoly n) (k : Fin n) :
    (∑ i ∈ s, f i).coeff k = ∑ i ∈ s, (f i).coeff k := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih => simp only [Finset.sum_insert ha, Negacyclic.coeff_add, ih]

private theorem absolute_sum {α : Type} [DecidableEq α]
    (s : Finset α) (f : α → Int) :
    (∑ i ∈ s, f i).natAbs ≤ ∑ i ∈ s, (f i).natAbs := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    simp only [Finset.sum_insert ha]
    exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add_left ih _)

theorem polynomial_action_bound {n : Nat} (hn : 0 < n) (x y : ErrorPoly n) :
    polyNorm (x * y) ≤ polynomialL1 x * polyNorm y := by
  let term : Fin n → Fin n → ErrorPoly n := fun i j =>
    scaledBasis (x.coeff i) i * scaledBasis (y.coeff j) j
  have hsum : x * y = ∑ i : Fin n, ∑ j : Fin n, term i j := by
    calc
      x * y = (∑ i : Fin n, scaledBasis (x.coeff i) i) *
          (∑ j : Fin n, scaledBasis (y.coeff j) j) := by
        conv_lhs => rw [Negacyclic.expansion hn x, Negacyclic.expansion hn y]
        simp only [scaledBasis]
      _ = _ := by rw [Finset.sum_mul_sum]
  rw [hsum]
  apply Finset.sup_le
  intro k hk
  rw [coefficient_sum]
  calc
    (∑ i : Fin n, (∑ j : Fin n, term i j).coeff k).natAbs ≤
        ∑ i : Fin n, ((∑ j : Fin n, term i j).coeff k).natAbs :=
      absolute_sum Finset.univ _
    _ ≤ ∑ i : Fin n, (x.coeff i).natAbs * polyNorm y := by
      apply Finset.sum_le_sum
      intro i hi
      rw [coefficient_sum, Finset.sum_eq_single (matchingIndex i k)]
      · exact (coeff_natAbs_le_polyNorm _ _).trans
          ((polyNorm_scaled_root_mul_le hn _ _ _ _).trans
            (Nat.mul_le_mul_left _ (coeff_natAbs_le_polyNorm y _)))
      · intro j hj hne
        dsimp [term]
        rw [coeff_scaled_basis_mul hn]
        have hnon : ¬(i.val + j.val) % n = k.val := by
          intro h
          exact hne ((matchingIndex_unique i j k).mp h)
        simp [hnon]
      · simp
    _ = polynomialL1 x * polyNorm y := by rw [polynomialL1, Finset.sum_mul]

/-- The actual integer matrix is checked column by column. -/
def ActionGain {n inner columns : Nat} (action : ErrorMatrix n inner columns)
    (gain : Nat) : Prop :=
  ∀ column : Fin columns, ∑ index : Fin inner, polynomialL1 (action index column) ≤ gain

theorem polynomialL1_le {n bound : Nat} {x : ErrorPoly n}
    (bounded : ∀ i : Fin n, (x.coeff i).natAbs ≤ bound) :
    polynomialL1 x ≤ n * bound := by
  calc
    _ ≤ ∑ i : Fin n, bound := Finset.sum_le_sum (fun i _ => bounded i)
    _ = n * bound := by simp

/-- The compact bounded-matrix plan uses every row and every ring coefficient. -/
theorem bounded_matrix_action_gain {n inner columns bound : Nat}
    {action : ErrorMatrix n inner columns} (bounded : CoeffBound action bound) :
    ActionGain action (inner * n * bound) := by
  intro column
  calc
    _ ≤ ∑ index : Fin inner, n * bound := by
      apply Finset.sum_le_sum
      intro index hi
      exact polynomialL1_le (bounded index column)
    _ = inner * n * bound := by simp [Nat.mul_assoc]

theorem right_action_bound {n rows inner columns bound gain : Nat} (hn : 0 < n)
    {error : ErrorMatrix n rows inner} {action : ErrorMatrix n inner columns}
    (he : CoeffBound error bound) (ha : ActionGain action gain) :
    CoeffBound (error * action) (gain * bound) := by
  intro row column coefficient
  have hpoly (index : Fin inner) : polyNorm (error row index) ≤ bound := by
    apply Finset.sup_le
    intro k hk
    exact he row index k
  apply (coeff_natAbs_le_polyNorm _ _).trans
  change polyNorm (∑ index : Fin inner, error row index * action index column) ≤ _
  calc
    _ ≤ ∑ index : Fin inner, polyNorm (error row index * action index column) :=
      polyNorm_sum_le Finset.univ _
    _ ≤ ∑ index : Fin inner, polynomialL1 (action index column) * bound := by
      apply Finset.sum_le_sum
      intro index hi
      rw [mul_comm]
      exact (polynomial_action_bound hn _ _).trans (Nat.mul_le_mul_left _ (hpoly index))
    _ = (∑ index : Fin inner, polynomialL1 (action index column)) * bound := by
      rw [Finset.sum_mul]
    _ ≤ gain * bound := Nat.mul_le_mul_right bound (ha column)

theorem left_polynomial_bound {n rows columns bound gain : Nat} (hn : 0 < n)
    {error : ErrorMatrix n rows columns} {scalar : ErrorPoly n}
    (he : CoeffBound error bound) (hs : polynomialL1 scalar ≤ gain) :
    CoeffBound (scalar • error) (gain * bound) := by
  intro row column coefficient
  have hp : polyNorm (error row column) ≤ bound := by
    apply Finset.sup_le
    intro k hk
    exact he row column k
  change ((scalar * error row column).coeff coefficient).natAbs ≤ _
  exact (coeff_natAbs_le_polyNorm _ _).trans
    ((polynomial_action_bound hn _ _).trans (Nat.mul_le_mul hs hp))

theorem reduce_polynomial_action {q n rows columns : Nat}
    (scalar : ErrorPoly n) (error : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (scalar • error) =
      reducePoly q n scalar • reduceMatrix q n rows columns error := by
  funext row column
  exact (reducePoly q n).map_mul scalar (error row column)

theorem polynomialL1_scaledBasis {n : Nat} (hn : 0 < n) (a : Int) (i : Fin n) :
    polynomialL1 (scaledBasis a i) = a.natAbs := by
  simp only [polynomialL1, scaledBasis, Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  rw [Finset.sum_eq_single i]
  · simp
  · intro j hj hji
    simp [Ne.symm hji]
  · simp

end MxxExponentLut
