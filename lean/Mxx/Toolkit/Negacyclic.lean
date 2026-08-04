import Mxx.Toolkit.Norms
import Mathlib.RingTheory.AdjoinRoot

open Polynomial

namespace Mxx.Toolkit

/-- The quotient ring used to interpret one negacyclic polynomial entry. -/
abbrev Negacyclic (q n : Nat) := AdjoinRoot (X ^ n + 1 : Polynomial (ZMod q))

noncomputable def negacyclicRoot (q n : Nat) : Negacyclic q n :=
  AdjoinRoot.root (X ^ n + 1 : Polynomial (ZMod q))

/-- The canonical coefficient basis of the negacyclic quotient. -/
noncomputable def negacyclicPowerBasis (q n : Nat) [Fact (1 < q)] [NeZero n] :
    PowerBasis (ZMod q) (Negacyclic q n) :=
  AdjoinRoot.powerBasis' (Polynomial.monic_X_pow_add_C (1 : ZMod q) (NeZero.ne n))

@[simp] theorem negacyclicPowerBasis_dim (q n : Nat) [Fact (1 < q)] [NeZero n] :
    (negacyclicPowerBasis q n).dim = n := by
  rw [negacyclicPowerBasis, AdjoinRoot.powerBasis'_dim]
  exact Polynomial.natDegree_X_pow_add_C

@[simp] theorem negacyclicPowerBasis_gen (q n : Nat) [Fact (1 < q)] [NeZero n] :
    (negacyclicPowerBasis q n).gen = negacyclicRoot q n := by
  rw [negacyclicPowerBasis, AdjoinRoot.powerBasis'_gen]
  rfl

theorem negacyclicRoot_pow (q n : Nat) : negacyclicRoot q n ^ n = -1 := by
  have relation := AdjoinRoot.eval₂_root (X ^ n + 1 : Polynomial (ZMod q))
  have reduced : negacyclicRoot q n ^ n + 1 = 0 := by
    simpa [negacyclicRoot] using relation
  exact eq_neg_of_add_eq_zero_left reduced

noncomputable def negacyclicEmbed (q n : Nat) (value : Int) : Negacyclic q n :=
  AdjoinRoot.of (X ^ n + 1 : Polynomial (ZMod q)) (value : ZMod q)

private theorem listRangeSum_eq_finSum {α : Type} [AddCommMonoid α]
    (n : Nat) (f : Nat → α) :
    ((List.range n).map f).sum = ∑ index : Fin n, f index := by
  rw [Fin.sum_univ_eq_sum_range f n]
  rw [← List.sum_toFinset _ (List.nodup_range : (List.range n).Nodup)]
  rw [List.toFinset_range]

private theorem embed_negacyclicFold (q n : Nat) (left right : Nat → Int)
    (coefficient : Fin n) (indices : List Nat) (initial : Int) :
    negacyclicEmbed q n (indices.foldl (fun total index ↦
      if index ≤ coefficient.val then
        total + left index * right (coefficient.val - index)
      else total - left index * right (n + coefficient.val - index)) initial) =
      negacyclicEmbed q n initial + (indices.map fun index ↦
        if index ≤ coefficient.val then
          negacyclicEmbed q n (left index * right (coefficient.val - index))
        else -negacyclicEmbed q n
          (left index * right (n + coefficient.val - index))).sum := by
  induction indices generalizing initial with
  | nil => simp [negacyclicEmbed]
  | cons head tail induction =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      rw [induction]
      split <;> simp [negacyclicEmbed, add_assoc, sub_eq_add_neg]

private theorem embed_negacyclicCoefficient (q n : Nat) [NeZero n]
    (left right : Nat → Int) (coefficient : Fin n) :
    negacyclicEmbed q n (Mxx.negacyclicCoefficient n left right coefficient) =
      ((List.range n).map fun index ↦ if index ≤ coefficient.val then
        negacyclicEmbed q n (left index * right (coefficient.val - index))
      else -negacyclicEmbed q n
        (left index * right (n + coefficient.val - index))).sum := by
  unfold Mxx.negacyclicCoefficient
  simpa [negacyclicEmbed] using
    embed_negacyclicFold q n left right coefficient (List.range n) 0

private def shiftedIndex (n : Nat) [NeZero n] (output left : Fin n) : Fin n :=
  ⟨if left.val ≤ output.val then output.val - left.val
    else n + output.val - left.val, by
      split
      · omega
      · omega⟩

private def negacyclicIndexEquiv (n : Nat) [NeZero n] :
    Fin n × Fin n ≃ Fin n × Fin n where
  toFun pair := (pair.2, shiftedIndex n pair.1 pair.2)
  invFun pair :=
    (⟨(pair.1.val + pair.2.val) % n, Nat.mod_lt _ (NeZero.pos n)⟩, pair.1)
  left_inv pair := by
    rcases pair with ⟨output, left⟩
    apply Prod.ext
    · simp only
      unfold shiftedIndex
      dsimp only
      apply Fin.ext
      change (left.val + (if left.val ≤ output.val then output.val - left.val
        else n + output.val - left.val)) % n = output.val
      split <;> rename_i before
      · rw [show left.val + (output.val - left.val) = output.val by omega]
        rw [Nat.mod_eq_of_lt output.isLt]
      · rw [show left.val + (n + output.val - left.val) = n + output.val by omega]
        simp [Nat.mod_eq_of_lt output.isLt]
    · rfl
  right_inv pair := by
    rcases pair with ⟨left, right⟩
    apply Prod.ext
    · rfl
    · simp only
      unfold shiftedIndex
      dsimp only
      apply Fin.ext
      change (if left.val ≤ (left.val + right.val) % n then
        (left.val + right.val) % n - left.val
      else n + (left.val + right.val) % n - left.val) = right.val
      by_cases below : left.val + right.val < n
      · simp [Nat.mod_eq_of_lt below]
      · have sumLt : left.val + right.val < 2 * n := by omega
        have reduced : (left.val + right.val) % n = left.val + right.val - n := by
          rw [Nat.mod_eq_sub_mod (by omega)]
          rw [Nat.mod_eq_of_lt (by omega)]
        have notBefore : ¬ left.val ≤ left.val + right.val - n := by omega
        simpa [reduced, notBefore] using
          (show n + (left.val + right.val - n) - left.val = right.val by omega)

private theorem negacyclicTerm_mul (q n : Nat) [NeZero n]
    (left right : Nat → Int) (output index : Fin n) :
    (if index.val ≤ output.val then
        negacyclicEmbed q n (left index * right (output.val - index.val))
      else -negacyclicEmbed q n
        (left index * right (n + output.val - index.val))) *
        negacyclicRoot q n ^ output.val =
      (negacyclicEmbed q n (left index) * negacyclicRoot q n ^ index.val) *
        (negacyclicEmbed q n (right (shiftedIndex n output index).val) *
          negacyclicRoot q n ^ (shiftedIndex n output index).val) := by
  unfold shiftedIndex
  split <;> rename_i before
  · have exponent : index.val + (output.val - index.val) = output.val := by omega
    simp only [negacyclicEmbed, map_mul, Int.cast_mul]
    have powers : negacyclicRoot q n ^ index.val *
        negacyclicRoot q n ^ (output.val - index.val) =
        negacyclicRoot q n ^ output.val := by rw [← pow_add, exponent]
    rw [show (AdjoinRoot.of (X ^ n + 1)) ↑(left index) * negacyclicRoot q n ^ index.val *
      ((AdjoinRoot.of (X ^ n + 1)) ↑(right (output.val - index.val)) *
        negacyclicRoot q n ^ (output.val - index.val)) =
      (AdjoinRoot.of (X ^ n + 1)) ↑(left index) *
        (AdjoinRoot.of (X ^ n + 1)) ↑(right (output.val - index.val)) *
          (negacyclicRoot q n ^ index.val *
            negacyclicRoot q n ^ (output.val - index.val)) by ring,
      powers]
  · have exponent : index.val + (n + output.val - index.val) = n + output.val := by omega
    simp only [negacyclicEmbed, map_mul, Int.cast_mul]
    have powers : negacyclicRoot q n ^ index.val *
        negacyclicRoot q n ^ (n + output.val - index.val) =
        -(negacyclicRoot q n ^ output.val) := by
      rw [← pow_add, exponent, pow_add, negacyclicRoot_pow]
      ring
    rw [show (AdjoinRoot.of (X ^ n + 1)) ↑(left index) * negacyclicRoot q n ^ index.val *
      ((AdjoinRoot.of (X ^ n + 1)) ↑(right (n + output.val - index.val)) *
        negacyclicRoot q n ^ (n + output.val - index.val)) =
      (AdjoinRoot.of (X ^ n + 1)) ↑(left index) *
        (AdjoinRoot.of (X ^ n + 1)) ↑(right (n + output.val - index.val)) *
          (negacyclicRoot q n ^ index.val *
            negacyclicRoot q n ^ (n + output.val - index.val)) by ring,
      powers]
    ring

/-- Interpret one coefficient function as a polynomial in `Z_q[X]/(X^n+1)`. -/
noncomputable def negacyclicValue (q n : Nat) (coefficients : Nat → Int) :
    Negacyclic q n :=
  ∑ index : Fin n,
    negacyclicEmbed q n (coefficients index) * negacyclicRoot q n ^ index.val

/-- The executable coefficient convolution is multiplication in the negacyclic quotient ring. -/
theorem negacyclicValue_convolution (q n : Nat) [NeZero n]
    (left right : Nat → Int) :
    negacyclicValue q n (Mxx.negacyclicCoefficient n left right) =
      negacyclicValue q n left * negacyclicValue q n right := by
  have embedded (output : Fin n) :
      negacyclicEmbed q n (Mxx.negacyclicCoefficient n left right output) =
        ∑ index : Fin n, if index.val ≤ output.val then
          negacyclicEmbed q n (left index * right (output.val - index.val))
        else -negacyclicEmbed q n
          (left index * right (n + output.val - index.val)) := by
    rw [← listRangeSum_eq_finSum n (fun index ↦
      if index ≤ output.val then
        negacyclicEmbed q n (left index * right (output.val - index))
      else -negacyclicEmbed q n (left index * right (n + output.val - index)))]
    exact embed_negacyclicCoefficient q n left right output
  unfold negacyclicValue
  let productTerm := fun pair : Fin n × Fin n ↦
    (negacyclicEmbed q n (left pair.1) * negacyclicRoot q n ^ pair.1.val) *
      (negacyclicEmbed q n (right pair.2) * negacyclicRoot q n ^ pair.2.val)
  calc
    (∑ output : Fin n, negacyclicEmbed q n
      (Mxx.negacyclicCoefficient n left right output) * negacyclicRoot q n ^ output.val) =
        ∑ output : Fin n, ∑ index : Fin n,
          (if index.val ≤ output.val then
              negacyclicEmbed q n (left index * right (output.val - index.val))
            else -negacyclicEmbed q n
              (left index * right (n + output.val - index.val))) *
              negacyclicRoot q n ^ output.val := by
          apply Finset.sum_congr rfl
          intro output _
          rw [embedded output, Finset.sum_mul]
    _ = ∑ pair : Fin n × Fin n,
      (if pair.2.val ≤ pair.1.val then
          negacyclicEmbed q n (left pair.2 * right (pair.1.val - pair.2.val))
        else -negacyclicEmbed q n
          (left pair.2 * right (n + pair.1.val - pair.2.val))) *
          negacyclicRoot q n ^ pair.1.val := by
      rw [Fintype.sum_prod_type]
    _ = ∑ pair : Fin n × Fin n, productTerm (negacyclicIndexEquiv n pair) := by
      apply Finset.sum_congr rfl
      intro pair _
      exact negacyclicTerm_mul q n left right pair.1 pair.2
    _ = ∑ pair : Fin n × Fin n, productTerm pair :=
      Equiv.sum_comp (negacyclicIndexEquiv n) productTerm
    _ = (∑ leftIndex : Fin n,
        negacyclicEmbed q n (left leftIndex) * negacyclicRoot q n ^ leftIndex.val) *
        ∑ rightIndex : Fin n,
          negacyclicEmbed q n (right rightIndex) * negacyclicRoot q n ^ rightIndex.val := by
      rw [Fintype.sum_prod_type]
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro leftIndex _
      rw [Finset.mul_sum]

/-- Two canonical coefficient vectors represent the same negacyclic polynomial exactly when all
their coefficients agree modulo the ciphertext modulus. -/
theorem negacyclicValue_eq_iff (q n : Nat) [Fact (1 < q)] [NeZero n]
    (left right : Nat → Int) :
    negacyclicValue q n left = negacyclicValue q n right ↔
      ∀ index : Fin n, (left index : ZMod q) = (right index : ZMod q) := by
  let basis := (negacyclicPowerBasis q n).basis
  have basisEq (index : Fin n) :
      basis (Fin.cast (negacyclicPowerBasis_dim q n).symm index) =
        negacyclicRoot q n ^ index.val := by
    rw [show basis (Fin.cast (negacyclicPowerBasis_dim q n).symm index) =
      (negacyclicPowerBasis q n).gen ^
        (Fin.cast (negacyclicPowerBasis_dim q n).symm index).val by
          exact PowerBasis.basis_eq_pow _ _]
    simp
  have valueEq (values : Nat → Int) :
      negacyclicValue q n values = ∑ index : Fin n,
        (values index : ZMod q) •
          basis (Fin.cast (negacyclicPowerBasis_dim q n).symm index) := by
    apply Finset.sum_congr rfl
    intro index _
    rw [basisEq]
    simp [negacyclicEmbed, Algebra.smul_def]
  constructor
  · intro equal index
    have represented := congrArg (fun value ↦ basis.repr value
      (Fin.cast (negacyclicPowerBasis_dim q n).symm index)) equal
    rw [valueEq] at represented
    rw [valueEq] at represented
    simp only [map_sum] at represented
    simpa [Finsupp.single_apply] using represented
  · intro coefficients
    unfold negacyclicValue negacyclicEmbed
    apply Finset.sum_congr rfl
    intro index _
    rw [show (left index : ZMod q) = (right index : ZMod q) from coefficients index]

theorem negacyclicEmbed_reduce (q n : Nat) [NeZero q] (value : Int) :
    negacyclicEmbed q n (Mxx.reduceCoefficient q value) = negacyclicEmbed q n value := by
  simp [negacyclicEmbed, Mxx.reduceCoefficient, NeZero.ne q]

private theorem negacyclicEmbed_indexFold (q n : Nat) (indices : List Nat)
    (term : Nat → Int) (initial : Int) :
    negacyclicEmbed q n
      (indices.foldl (fun total index ↦ total + term index) initial) =
      negacyclicEmbed q n initial +
        (indices.map fun index ↦ negacyclicEmbed q n (term index)).sum := by
  induction indices generalizing initial with
  | nil => simp [negacyclicEmbed]
  | cons head tail induction =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      rw [induction]
      simp [negacyclicEmbed, add_assoc]

theorem negacyclicValue_add (q n : Nat) (left right : Nat → Int) :
    negacyclicValue q n (fun index ↦ left index + right index) =
      negacyclicValue q n left + negacyclicValue q n right := by
  unfold negacyclicValue negacyclicEmbed
  simp_rw [Int.cast_add, map_add, add_mul]
  exact Finset.sum_add_distrib

theorem negacyclicValue_subtract (q n : Nat) (left right : Nat → Int) :
    negacyclicValue q n (fun index ↦ left index - right index) =
      negacyclicValue q n left - negacyclicValue q n right := by
  unfold negacyclicValue negacyclicEmbed
  simp_rw [Int.cast_sub, map_sub, sub_mul]
  rw [Finset.sum_sub_distrib]

theorem negacyclicValue_negate (q n : Nat) (matrix : Nat → Int) :
    negacyclicValue q n (fun index ↦ -matrix index) = -negacyclicValue q n matrix := by
  unfold negacyclicValue negacyclicEmbed
  simp_rw [Int.cast_neg, map_neg, neg_mul]
  rw [Finset.sum_neg_distrib]

theorem negacyclicValue_scale (q n : Nat) (scalar : Int) (matrix : Nat → Int) :
    negacyclicValue q n (fun index ↦ scalar * matrix index) =
      (scalar : Negacyclic q n) • negacyclicValue q n matrix := by
  unfold negacyclicValue negacyclicEmbed
  simp_rw [Int.cast_mul, map_mul]
  simp [Finset.mul_sum, mul_assoc]

theorem addCoefficients_getD (left right : List Int) (index : Nat) :
    (Mxx.addCoefficients left right).getD index 0 =
      left.getD index 0 + right.getD index 0 := by
  induction left generalizing right index with
  | nil => cases right <;> cases index <;> simp [Mxx.addCoefficients]
  | cons leftHead leftTail induction =>
      cases right with
      | nil => cases index <;> simp [Mxx.addCoefficients]
      | cons rightHead rightTail =>
          cases index with
          | zero => simp [Mxx.addCoefficients]
          | succ index =>
              simpa only [Mxx.addCoefficients, List.getD_cons_succ] using
                induction rightTail index

theorem subtractCoefficients_getD (left right : List Int) (index : Nat) :
    (Mxx.subtractCoefficients left right).getD index 0 =
      left.getD index 0 - right.getD index 0 := by
  induction left generalizing right index with
  | nil =>
      rw [show Mxx.subtractCoefficients [] right = right.map (-·) by rfl]
      rw [show (0 : Int) = -0 by simp, List.getD_map]
      simp
  | cons leftHead leftTail induction =>
      cases right with
      | nil => cases index <;> simp [Mxx.subtractCoefficients]
      | cons rightHead rightTail =>
          cases index with
          | zero => simp [Mxx.subtractCoefficients]
          | succ index =>
              simpa only [Mxx.subtractCoefficients, List.getD_cons_succ] using
                induction rightTail index

theorem cast_reduce (q : Nat) [NeZero q] (value : Int) :
    ((Mxx.reduceCoefficient q value : Int) : ZMod q) = (value : ZMod q) := by
  simp [Mxx.reduceCoefficient, NeZero.ne q]

/-- Equality modulo `q` implies equality of the canonical nonnegative residues. -/
theorem int_emod_eq_of_zmod_eq (q : Nat) (left right : Int)
    (equal : (left : ZMod q) = (right : ZMod q)) :
    left % q = right % q := by
  have divides : (q : Int) ∣ right - left :=
    (ZMod.intCast_eq_intCast_iff_dvd_sub left right q).mp equal
  apply Int.emod_eq_emod_iff_emod_sub_eq_zero.mpr
  apply Int.emod_eq_zero_of_dvd
  simpa [sub_eq_add_neg, add_comm] using dvd_neg.mpr divides

/-- A coefficient already normalized by `reduceCoefficient` is the canonical representative of
any congruent integer. -/
theorem canonical_eq_emod_of_zmod_eq (q : Nat) [NeZero q] (actual expected : Int)
    (canonical : actual = Mxx.reduceCoefficient q actual)
    (equal : (actual : ZMod q) = (expected : ZMod q)) :
    actual = expected % q := by
  calc
    actual = actual % q := by
      simpa [Mxx.reduceCoefficient, NeZero.ne q] using canonical
    _ = expected % q := int_emod_eq_of_zmod_eq q actual expected equal

/-- Reading a valid coefficient from executable matrix multiplication returns the corresponding
row/column inner product of exact negacyclic convolutions. -/
theorem matrixMul_coefficient (left right : Mxx.Matrix) (row column coefficient : Nat)
    (compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows)
    (rowLt : row < left.rows) (columnLt : column < right.columns)
    (coefficientLt : coefficient < left.ringDimension) :
    (Mxx.matrixMul left right).coefficient row column coefficient =
      Mxx.reduceCoefficient left.modulus (
        (List.range left.columns).foldl (fun total inner ↦
          total + Mxx.negacyclicCoefficient left.ringDimension
            (left.coefficient row inner) (right.coefficient inner column) coefficient) 0) := by
  have entryLt : row * right.columns + column < left.rows * right.columns := by
    calc
      row * right.columns + column < row * right.columns + right.columns :=
        Nat.add_lt_add_left columnLt _
      _ = (row + 1) * right.columns := by
        rw [Nat.add_mul, Nat.one_mul]
      _ ≤ left.rows * right.columns :=
        Nat.mul_le_mul_right right.columns (Nat.succ_le_iff.mpr rowLt)
  have linearLt : (row * right.columns + column) * left.ringDimension + coefficient <
      left.rows * right.columns * left.ringDimension := by
    calc
      (row * right.columns + column) * left.ringDimension + coefficient <
          (row * right.columns + column) * left.ringDimension + left.ringDimension :=
        Nat.add_lt_add_left coefficientLt _
      _ = (row * right.columns + column + 1) * left.ringDimension := by ring
      _ ≤ left.rows * right.columns * left.ringDimension :=
        Nat.mul_le_mul_right left.ringDimension (Nat.succ_le_iff.mpr entryLt)
  simp only [Mxx.matrixMul, if_pos compatible, Mxx.Matrix.coefficient]
  rw [List.getD_eq_getElem _ _ (by simpa using linearLt), List.getElem_ofFn]
  have ringPositive : 0 < left.ringDimension := by omega
  have columnsPositive : 0 < right.columns := by omega
  have entryDiv : ((row * right.columns + column) * left.ringDimension + coefficient) /
      left.ringDimension = row * right.columns + column := by
    rw [Nat.mul_comm (row * right.columns + column), Nat.mul_add_div ringPositive,
      Nat.div_eq_of_lt coefficientLt, Nat.add_zero]
  have coefficientMod :
      ((row * right.columns + column) * left.ringDimension + coefficient) %
        left.ringDimension = coefficient := by
    rw [Nat.mul_add_mod_of_lt coefficientLt]
  rw [entryDiv, coefficientMod]
  have rowDiv : (row * right.columns + column) / right.columns = row := by
    rw [Nat.mul_comm row, Nat.mul_add_div columnsPositive, Nat.div_eq_of_lt columnLt,
      Nat.add_zero]
  have columnMod : (row * right.columns + column) % right.columns = column := by
    rw [Nat.mul_add_mod_of_lt columnLt]
  rw [rowDiv, columnMod]

/-- Reading a valid coefficient from executable polynomial-scalar broadcast returns the exact
negacyclic convolution of the scalar polynomial and that matrix entry. -/
theorem matrixPolynomialScale_coefficient (scalar matrix : Mxx.Matrix)
    (row column coefficient : Nat) (rowLt : row < matrix.rows)
    (columnLt : column < matrix.columns) (coefficientLt : coefficient < matrix.ringDimension) :
    (Mxx.matrixPolynomialScale scalar matrix).coefficient row column coefficient =
      Mxx.reduceCoefficient matrix.modulus
        (Mxx.negacyclicCoefficient matrix.ringDimension
          (scalar.coefficient 0 0) (matrix.coefficient row column) coefficient) := by
  have entryLt : row * matrix.columns + column < matrix.rows * matrix.columns := by
    calc
      row * matrix.columns + column < row * matrix.columns + matrix.columns :=
        Nat.add_lt_add_left columnLt _
      _ = (row + 1) * matrix.columns := by rw [Nat.add_mul, Nat.one_mul]
      _ ≤ matrix.rows * matrix.columns :=
        Nat.mul_le_mul_right matrix.columns (Nat.succ_le_iff.mpr rowLt)
  have linearLt : (row * matrix.columns + column) * matrix.ringDimension + coefficient <
      matrix.rows * matrix.columns * matrix.ringDimension := by
    calc
      _ < (row * matrix.columns + column) * matrix.ringDimension + matrix.ringDimension :=
        Nat.add_lt_add_left coefficientLt _
      _ = (row * matrix.columns + column + 1) * matrix.ringDimension := by ring
      _ ≤ matrix.rows * matrix.columns * matrix.ringDimension :=
        Nat.mul_le_mul_right matrix.ringDimension (Nat.succ_le_iff.mpr entryLt)
  simp only [Mxx.matrixPolynomialScale, Mxx.Matrix.coefficient]
  rw [List.getD_eq_getElem _ _ (by simpa using linearLt), List.getElem_ofFn]
  have ringPositive : 0 < matrix.ringDimension := by omega
  have columnsPositive : 0 < matrix.columns := by omega
  have entryDiv : ((row * matrix.columns + column) * matrix.ringDimension + coefficient) /
      matrix.ringDimension = row * matrix.columns + column := by
    rw [Nat.mul_comm (row * matrix.columns + column), Nat.mul_add_div ringPositive,
      Nat.div_eq_of_lt coefficientLt, Nat.add_zero]
  have coefficientMod :
      ((row * matrix.columns + column) * matrix.ringDimension + coefficient) %
        matrix.ringDimension = coefficient := by
    rw [Nat.mul_add_mod_of_lt coefficientLt]
  rw [entryDiv, coefficientMod]
  have rowDiv : (row * matrix.columns + column) / matrix.columns = row := by
    rw [Nat.mul_comm row, Nat.mul_add_div columnsPositive, Nat.div_eq_of_lt columnLt,
      Nat.add_zero]
  have columnMod : (row * matrix.columns + column) % matrix.columns = column := by
    rw [Nat.mul_add_mod_of_lt columnLt]
  rw [rowDiv, columnMod]

/-- Interpret every polynomial entry of an executable matrix in the exact negacyclic quotient. -/
noncomputable def matrixValue (q n rows columns : Nat) (matrix : Mxx.Matrix) :
    _root_.Matrix (Fin rows) (Fin columns) (Negacyclic q n) :=
  fun row column ↦ negacyclicValue q n (matrix.coefficient row column)

/-- Equality of interpreted `1 × 1` matrices implies equality modulo `q` of their first stored
coefficients. This bridge uses coefficient uniqueness in the negacyclic quotient and does not
enumerate the ring dimension. -/
theorem matrixValue_headD_zmod_eq (q n : Nat) [Fact (1 < q)] [NeZero n]
    (left right : Mxx.Matrix)
    (equal : matrixValue q n 1 1 left = matrixValue q n 1 1 right) :
    (left.coefficients.headD 0 : ZMod q) =
      (right.coefficients.headD 0 : ZMod q) := by
  have entryEqual := congrFun (congrFun equal (0 : Fin 1)) (0 : Fin 1)
  have coefficientEqual :=
    (negacyclicValue_eq_iff q n (left.coefficient 0 0) (right.coefficient 0 0)).mp
      entryEqual (0 : Fin n)
  simpa [matrixValue, Mxx.Matrix.coefficient, List.head?_eq_getElem?] using coefficientEqual

/-- Runtime matrix dimensions and coefficient ring, independent of any magnitude claim. -/
structure MatrixShape (matrix : Mxx.Matrix) (modulus : Int) (n rows columns : Nat) : Prop where
  modulus : matrix.modulus = modulus
  ringDimension : matrix.ringDimension = n
  rows : matrix.rows = rows
  columns : matrix.columns = columns

/-- A shaped executable matrix whose flat coefficient storage contains exactly one full
coefficient vector for every matrix entry.  Concat operations use this storage invariant when
locating the first entry of a later block. -/
structure MatrixLayout (matrix : Mxx.Matrix) (modulus : Int)
    (n rowCount columnCount : Nat) : Prop
    extends MatrixShape matrix modulus n rowCount columnCount where
  coefficientCount : matrix.coefficients.length = rowCount * columnCount * n

theorem withSamplerParams_shape (matrix : Mxx.Matrix) (params : Mxx.SamplerParams) :
    MatrixShape (matrix.withSamplerParams params) params.modulus params.ringDimension
      params.rows params.columns := by
  exact ⟨rfl, rfl, rfl, rfl⟩

theorem withSamplerParams_layout (matrix : Mxx.Matrix) (params : Mxx.SamplerParams) :
    MatrixLayout (matrix.withSamplerParams params) params.modulus params.ringDimension
      params.rows params.columns := by
  refine { toMatrixShape := withSamplerParams_shape matrix params, coefficientCount := ?_ }
  simp only [Mxx.Matrix.withSamplerParams, List.length_append, List.length_take,
    List.length_replicate]
  omega

/-- Every matrix in a uniform sampler support has the sampler's complete matrix layout. -/
theorem uniformMatrixSupport_layout (params : Mxx.SamplerParams) (minimum maximum : Int)
    (matrix : Mxx.Matrix) (member : matrix ∈ Mxx.Ir.uniformMatrixSupport params minimum maximum) :
    MatrixLayout matrix params.modulus params.ringDimension params.rows params.columns := by
  rw [Mxx.Ir.uniformMatrixSupport] at member
  obtain ⟨coefficients, _, rfl⟩ := List.mem_map.mp member
  exact withSamplerParams_layout { coefficients } params

/-- The gadget matrix has one block row per sampler row and one digit column per block row. -/
theorem gadgetMatrix_layout (params : Mxx.SamplerParams) (base : Int) (digitCount : Nat) :
    MatrixLayout (Mxx.gadgetMatrix params base digitCount) params.modulus
      params.ringDimension params.rows (params.rows * digitCount) := by
  unfold Mxx.gadgetMatrix
  exact withSamplerParams_layout _ _

/-- Place two one-row matrices above one another. -/
noncomputable def pairRows {q n columns : Nat}
    (top bottom : _root_.Matrix (Fin 1) (Fin columns) (Negacyclic q n)) :
    _root_.Matrix (Fin 2) (Fin columns) (Negacyclic q n) :=
  fun row column ↦ Fin.cases (top 0 column) (fun _ ↦ bottom 0 column) row

/-- Place two polynomial scalars next to one another. -/
noncomputable def pairColumns {q n : Nat}
    (left right : _root_.Matrix (Fin 1) (Fin 1) (Negacyclic q n)) :
    _root_.Matrix (Fin 1) (Fin 2) (Negacyclic q n) :=
  fun _ column ↦ Fin.cases (left 0 0) (fun _ ↦ right 0 0) column

/-- Place two polynomial scalars on the diagonal of a `2 × 2` matrix. -/
noncomputable def diagonalPair {q n : Nat}
    (top bottom : _root_.Matrix (Fin 1) (Fin 1) (Negacyclic q n)) :
    _root_.Matrix (Fin 2) (Fin 2) (Negacyclic q n) :=
  fun row column ↦
    Fin.cases (Fin.cases (top 0 0) (fun _ ↦ 0) column)
      (fun _ ↦ Fin.cases 0 (fun _ ↦ bottom 0 0) column) row

private theorem map_range_getD {α : Type} (n : Nat) (f : Nat → α) (fallback : α)
    (index : Fin n) :
    ((List.range n).map f).getD index.val fallback = f index.val := by
  rw [List.getD_eq_getElem _ _ (by simp [index.isLt])]
  simp

private theorem flatMap_range_getD {α : Type} (outerCount chunkLength : Nat)
    (f : Nat → List α) (fallback : α) (outer : Fin outerCount) (inner : Fin chunkLength)
    (length : ∀ index, (f index).length = chunkLength) :
    ((List.range outerCount).flatMap f).getD
        (outer.val * chunkLength + inner.val) fallback =
      (f outer.val).getD inner.val fallback := by
  induction outerCount with
  | zero => exact Fin.elim0 outer
  | succ count induction =>
      rw [List.range_succ, List.flatMap_append]
      simp only [List.flatMap_singleton]
      have outerLt := outer.isLt
      have innerLt := inner.isLt
      have prefixLength : ((List.range count).flatMap f).length = count * chunkLength := by
        simp [length]
      by_cases before : outer.val < count
      · have indexLt : outer.val * chunkLength + inner.val <
            ((List.range count).flatMap f).length := by
          rw [prefixLength]
          calc
            outer.val * chunkLength + inner.val < outer.val * chunkLength + chunkLength :=
              Nat.add_lt_add_left innerLt _
            _ = (outer.val + 1) * chunkLength := by ring
            _ ≤ count * chunkLength :=
              Nat.mul_le_mul_right chunkLength (Nat.succ_le_iff.mpr before)
        rw [List.getD_append _ _ _ _ indexLt]
        exact induction ⟨outer.val, before⟩
      · have atLast : outer.val = count := by omega
        have indexGe : ((List.range count).flatMap f).length ≤
            outer.val * chunkLength + inner.val := by
          rw [prefixLength, atLast]
          exact Nat.le_add_right _ _
        rw [List.getD_append_right _ _ _ _ indexGe, prefixLength,
          atLast, Nat.add_sub_cancel_left]

private theorem getD_append_four_first {α : Type} (n : Nat) (a b c d : List α)
    (fallback : α) (index : Fin n) (aLength : a.length = n) :
    (a ++ b ++ c ++ d).getD index.val fallback = a.getD index.val fallback := by
  simp only [List.append_assoc]
  have indexLt : index.val < a.length := by rw [aLength]; exact index.isLt
  rw [List.getD_append _ _ _ _ indexLt]

private theorem getD_append_four_second {α : Type} (n : Nat) (a b c d : List α)
    (fallback : α) (index : Fin n) (aLength : a.length = n) (bLength : b.length = n) :
    (a ++ b ++ c ++ d).getD (n + index.val) fallback = b.getD index.val fallback := by
  simp only [List.append_assoc]
  rw [List.getD_append_right _ _ _ _ (by simp [aLength]), aLength,
    Nat.add_sub_cancel_left]
  have indexLt : index.val < b.length := by rw [bLength]; exact index.isLt
  rw [List.getD_append _ _ _ _ indexLt]

private theorem getD_append_four_third {α : Type} (n : Nat) (a b c d : List α)
    (fallback : α) (index : Fin n) (aLength : a.length = n) (bLength : b.length = n)
    (cLength : c.length = n) :
    (a ++ b ++ c ++ d).getD (2 * n + index.val) fallback = c.getD index.val fallback := by
  simp only [List.append_assoc]
  rw [List.getD_append_right _ _ _ _ (by rw [aLength]; omega), aLength]
  have firstOffset : 2 * n + index.val - n = n + index.val := by omega
  rw [firstOffset, List.getD_append_right _ _ _ _ (by simp [bLength]), bLength,
    Nat.add_sub_cancel_left]
  have indexLt : index.val < c.length := by rw [cLength]; exact index.isLt
  rw [List.getD_append _ _ _ _ indexLt]

private theorem getD_append_four_fourth {α : Type} (n : Nat) (a b c d : List α)
    (fallback : α) (index : Fin n) (aLength : a.length = n) (bLength : b.length = n)
    (cLength : c.length = n) :
    (a ++ b ++ c ++ d).getD (3 * n + index.val) fallback = d.getD index.val fallback := by
  simp only [List.append_assoc]
  rw [List.getD_append_right _ _ _ _ (by rw [aLength]; omega), aLength]
  have firstOffset : 3 * n + index.val - n = 2 * n + index.val := by omega
  rw [firstOffset, List.getD_append_right _ _ _ _ (by rw [bLength]; omega), bLength]
  have secondOffset : 2 * n + index.val - n = n + index.val := by omega
  rw [secondOffset, List.getD_append_right _ _ _ _ (by simp [cLength]), cLength,
    Nat.add_sub_cancel_left]

theorem matrixConcatRows_two_layout {q : Int} {n columns : Nat} (top bottom : Mxx.Matrix)
    (topLayout : MatrixLayout top q n 1 columns)
    (bottomLayout : MatrixLayout bottom q n 1 columns) :
    MatrixLayout (Mxx.matrixConcatRows [top, bottom]) q n 2 columns := by
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · simp [Mxx.matrixConcatRows, topLayout.modulus]
  · simp [Mxx.matrixConcatRows, topLayout.ringDimension]
  · simp [Mxx.matrixConcatRows, topLayout.rows, bottomLayout.rows]
  · simp [Mxx.matrixConcatRows, topLayout.columns]
  · simp [Mxx.matrixConcatRows, topLayout.coefficientCount,
      bottomLayout.coefficientCount]
    ring

theorem matrixConcatColumns_two_layout {q : Int} {n : Nat} (left right : Mxx.Matrix)
    (leftLayout : MatrixLayout left q n 1 1)
    (rightLayout : MatrixLayout right q n 1 1) :
    MatrixLayout (Mxx.matrixConcatColumns [left, right]) q n 1 2 := by
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · simp [Mxx.matrixConcatColumns, leftLayout.modulus]
  · simp [Mxx.matrixConcatColumns, leftLayout.ringDimension]
  · simp [Mxx.matrixConcatColumns, leftLayout.rows]
  · simp [Mxx.matrixConcatColumns, leftLayout.columns, rightLayout.columns]
  · simp [Mxx.matrixConcatColumns, leftLayout.rows, leftLayout.columns,
      rightLayout.columns, leftLayout.ringDimension]
    omega

theorem matrixConcatDiagonal_two_layout {q : Int} {n : Nat} (top bottom : Mxx.Matrix)
    (topLayout : MatrixLayout top q n 1 1)
    (bottomLayout : MatrixLayout bottom q n 1 1) :
    MatrixLayout (Mxx.matrixConcatDiagonal [top, bottom]) q n 2 2 := by
  refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
  · simp [Mxx.matrixConcatDiagonal, topLayout.modulus]
  · simp [Mxx.matrixConcatDiagonal, topLayout.ringDimension]
  · simp [Mxx.matrixConcatDiagonal, topLayout.rows, bottomLayout.rows]
  · simp [Mxx.matrixConcatDiagonal, topLayout.columns, bottomLayout.columns]
  · simp [Mxx.matrixConcatDiagonal, topLayout.rows, bottomLayout.rows,
      topLayout.columns, bottomLayout.columns, topLayout.ringDimension]
    omega

theorem matrixValue_concatRows_two {q n columns : Nat} (top bottom : Mxx.Matrix)
    (topLayout : MatrixLayout top q n 1 columns)
    (bottomLayout : MatrixLayout bottom q n 1 columns) :
    matrixValue q n 2 columns (Mxx.matrixConcatRows [top, bottom]) =
      pairRows (matrixValue q n 1 columns top) (matrixValue q n 1 columns bottom) := by
  rcases top with ⟨topCoefficients, topModulus, topRing, topRows, topColumns⟩
  rcases bottom with
    ⟨bottomCoefficients, bottomModulus, bottomRing, bottomRows, bottomColumns⟩
  rcases topLayout with
    ⟨⟨topModulusEq, topRingEq, topRowsEq, topColumnsEq⟩, topLength⟩
  rcases bottomLayout with
    ⟨⟨bottomModulusEq, bottomRingEq, bottomRowsEq, bottomColumnsEq⟩, bottomLength⟩
  simp only at *
  subst topModulus
  subst topRing
  subst topRows
  subst topColumns
  subst bottomModulus
  subst bottomRing
  subst bottomRows
  subst bottomColumns
  simp only [Nat.one_mul] at topLength bottomLength
  ext row column
  fin_cases row
  · unfold matrixValue negacyclicValue
    apply Finset.sum_congr rfl
    intro coefficient _
    have indexLt : column.val * n + coefficient < topCoefficients.length := by
      rw [topLength]
      calc
        column.val * n + coefficient.val < column.val * n + n :=
          Nat.add_lt_add_left coefficient.isLt _
        _ = (column.val + 1) * n := by ring
        _ ≤ columns * n := Nat.mul_le_mul_right n (Nat.succ_le_iff.mpr column.isLt)
    simp only [Mxx.matrixConcatRows, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, Mxx.Matrix.coefficient, Nat.zero_mul, Nat.zero_add]
    rw [List.getD_append _ _ _ _ indexLt]
    simp
  · unfold matrixValue negacyclicValue
    apply Finset.sum_congr rfl
    intro coefficient _
    simp only [Mxx.matrixConcatRows, List.flatMap_cons, List.flatMap_nil,
      List.append_nil, Mxx.Matrix.coefficient, Nat.one_mul]
    have indexEq : (columns + column.val) * n + coefficient =
        topCoefficients.length + (column.val * n + coefficient) := by
      rw [topLength, Nat.add_mul]
      omega
    rw [indexEq, List.getD_append_right _ _ _ _ (Nat.le_add_right _ _),
      Nat.add_sub_cancel_left]
    simp

theorem matrixValue_concatColumns_two {q n : Nat} (left right : Mxx.Matrix)
    (leftLayout : MatrixLayout left q n 1 1)
    (rightLayout : MatrixLayout right q n 1 1) :
    matrixValue q n 1 2 (Mxx.matrixConcatColumns [left, right]) =
      pairColumns (matrixValue q n 1 1 left) (matrixValue q n 1 1 right) := by
  rcases left with ⟨leftCoefficients, leftModulus, leftRing, leftRows, leftColumns⟩
  rcases right with ⟨rightCoefficients, rightModulus, rightRing, rightRows, rightColumns⟩
  rcases leftLayout with
    ⟨⟨leftModulusEq, leftRingEq, leftRowsEq, leftColumnsEq⟩, leftLength⟩
  rcases rightLayout with
    ⟨⟨rightModulusEq, rightRingEq, rightRowsEq, rightColumnsEq⟩, rightLength⟩
  simp only at *
  subst leftModulus
  subst leftRing
  subst leftRows
  subst leftColumns
  subst rightModulus
  subst rightRing
  subst rightRows
  subst rightColumns
  ext row column
  fin_cases row
  fin_cases column
  · unfold matrixValue negacyclicValue
    apply Finset.sum_congr rfl
    intro coefficient _
    simp only [Mxx.matrixConcatColumns, Mxx.Matrix.coefficient, Nat.zero_mul,
      Nat.zero_add, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      List.range_one, List.map_cons, List.map_nil]
    rw [List.getD_append _ _ _ _ (by simp [coefficient.isLt])]
    rw [map_range_getD]
    simp
  · unfold matrixValue negacyclicValue
    apply Finset.sum_congr rfl
    intro coefficient _
    simp only [Mxx.matrixConcatColumns, Mxx.Matrix.coefficient, Nat.zero_mul,
      Nat.zero_add, Nat.one_mul, List.flatMap_cons, List.flatMap_nil, List.append_nil,
      List.range_one, List.map_cons, List.map_nil]
    rw [List.getD_append_right _ _ _ _ (by simp)]
    simp only [List.length_map, List.length_range]
    rw [Nat.add_sub_cancel_left, map_range_getD]
    simp

theorem matrixValue_concatDiagonal_two {q n : Nat} (top bottom : Mxx.Matrix)
    (topLayout : MatrixLayout top q n 1 1)
    (bottomLayout : MatrixLayout bottom q n 1 1) :
    matrixValue q n 2 2 (Mxx.matrixConcatDiagonal [top, bottom]) =
      diagonalPair (matrixValue q n 1 1 top) (matrixValue q n 1 1 bottom) := by
  rcases top with ⟨topCoefficients, topModulus, topRing, topRows, topColumns⟩
  rcases bottom with
    ⟨bottomCoefficients, bottomModulus, bottomRing, bottomRows, bottomColumns⟩
  rcases topLayout with
    ⟨⟨topModulusEq, topRingEq, topRowsEq, topColumnsEq⟩, topLength⟩
  rcases bottomLayout with
    ⟨⟨bottomModulusEq, bottomRingEq, bottomRowsEq, bottomColumnsEq⟩, bottomLength⟩
  simp only at *
  subst topModulus
  subst topRing
  subst topRows
  subst topColumns
  subst bottomModulus
  subst bottomRing
  subst bottomRows
  subst bottomColumns
  simp only [Nat.one_mul] at topLength bottomLength
  have diagonalCoefficients :
      (Mxx.matrixConcatDiagonal
        [{ coefficients := topCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 },
          { coefficients := bottomCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 }]).coefficients =
        (List.range n).map (fun index ↦ topCoefficients.getD index 0) ++
        (List.range n).map (fun _ ↦ 0) ++
        (List.range n).map (fun _ ↦ 0) ++
        (List.range n).map (fun index ↦ bottomCoefficients.getD index 0) := by
    simp [Mxx.matrixConcatDiagonal, Mxx.diagonalCoefficient, Mxx.Matrix.coefficient,
      List.range_succ]
    rw [List.replicate_add]
    rw [List.append_assoc]
  have output00 (coefficient : Fin n) :
      (Mxx.matrixConcatDiagonal
        [{ coefficients := topCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 },
          { coefficients := bottomCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 }]).coefficient 0 0 coefficient =
        topCoefficients.getD coefficient.val 0 := by
    unfold Mxx.Matrix.coefficient
    rw [diagonalCoefficients]
    simp only [Mxx.matrixConcatDiagonal, Nat.add_zero, Nat.zero_mul, Nat.zero_add]
    rw [getD_append_four_first n]
    · rw [map_range_getD]
    · simp
  have output01 (coefficient : Fin n) :
      (Mxx.matrixConcatDiagonal
        [{ coefficients := topCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 },
          { coefficients := bottomCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 }]).coefficient 0 1 coefficient = 0 := by
    unfold Mxx.Matrix.coefficient
    rw [diagonalCoefficients]
    simp only [Mxx.matrixConcatDiagonal, Nat.zero_mul, Nat.zero_add, Nat.one_mul]
    rw [getD_append_four_second n]
    · rw [map_range_getD]
    · simp
    · simp
  have output10 (coefficient : Fin n) :
      (Mxx.matrixConcatDiagonal
        [{ coefficients := topCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 },
          { coefficients := bottomCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 }]).coefficient 1 0 coefficient = 0 := by
    unfold Mxx.Matrix.coefficient
    rw [diagonalCoefficients]
    simp only [Mxx.matrixConcatDiagonal, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, Nat.add_zero, Nat.one_mul]
    rw [getD_append_four_third n]
    · rw [map_range_getD]
    · simp
    · simp
    · simp
  have output11 (coefficient : Fin n) :
      (Mxx.matrixConcatDiagonal
        [{ coefficients := topCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 },
          { coefficients := bottomCoefficients, modulus := q, ringDimension := n,
            rows := 1, columns := 1 }]).coefficient 1 1 coefficient =
        bottomCoefficients.getD coefficient.val 0 := by
    unfold Mxx.Matrix.coefficient
    rw [diagonalCoefficients]
    simp only [Mxx.matrixConcatDiagonal, List.map_cons, List.map_nil, List.sum_cons,
      List.sum_nil, Nat.add_zero, Nat.one_mul]
    rw [getD_append_four_fourth n]
    · rw [map_range_getD]
    · simp
    · simp
    · simp
  ext row column
  fin_cases row <;> fin_cases column
  · change negacyclicValue q n _ = negacyclicValue q n _
    unfold negacyclicValue
    apply Finset.sum_congr rfl
    intro coefficient _
    rw [output00]
    simp [Mxx.Matrix.coefficient]
  · change negacyclicValue q n _ = 0
    unfold negacyclicValue
    simp_rw [output01]
    simp [negacyclicEmbed]
  · change negacyclicValue q n _ = 0
    unfold negacyclicValue
    simp_rw [output10]
    simp [negacyclicEmbed]
  · change negacyclicValue q n _ = negacyclicValue q n _
    unfold negacyclicValue
    apply Finset.sum_congr rfl
    intro coefficient _
    rw [output11]
    simp [Mxx.Matrix.coefficient]

theorem matrixMul_shape {q : Int} {n rows inner columns : Nat} (left right : Mxx.Matrix)
    (leftShape : MatrixShape left q n rows inner)
    (rightShape : MatrixShape right q n inner columns) :
    MatrixShape (Mxx.matrixMul left right) q n rows columns := by
  have compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows := by
    exact ⟨leftShape.modulus.trans rightShape.modulus.symm,
      leftShape.ringDimension.trans rightShape.ringDimension.symm,
      leftShape.columns.trans rightShape.rows.symm⟩
  rw [Mxx.matrixMul, if_pos compatible]
  exact ⟨leftShape.modulus, leftShape.ringDimension, leftShape.rows, rightShape.columns⟩

theorem matrixAdd_shape {q : Int} {n rows columns : Nat} (left right : Mxx.Matrix)
    (leftShape : MatrixShape left q n rows columns)
    (_rightShape : MatrixShape right q n rows columns) :
    MatrixShape (Mxx.matrixAdd left right) q n rows columns := by
  exact ⟨leftShape.modulus, leftShape.ringDimension, leftShape.rows, leftShape.columns⟩

theorem matrixSubtract_shape {q : Int} {n rows columns : Nat} (left right : Mxx.Matrix)
    (leftShape : MatrixShape left q n rows columns)
    (_rightShape : MatrixShape right q n rows columns) :
    MatrixShape (Mxx.matrixSubtract left right) q n rows columns := by
  exact ⟨leftShape.modulus, leftShape.ringDimension, leftShape.rows, leftShape.columns⟩

theorem matrixNegate_shape {q : Int} {n rows columns : Nat} (matrix : Mxx.Matrix)
    (shape : MatrixShape matrix q n rows columns) :
    MatrixShape (Mxx.matrixNegate matrix) q n rows columns := by
  exact ⟨shape.modulus, shape.ringDimension, shape.rows, shape.columns⟩

theorem matrixScale_shape {q : Int} {n rows columns : Nat} (scalar : Int) (matrix : Mxx.Matrix)
    (shape : MatrixShape matrix q n rows columns) :
    MatrixShape (Mxx.matrixScale scalar matrix) q n rows columns := by
  exact ⟨shape.modulus, shape.ringDimension, shape.rows, shape.columns⟩

theorem matrixPolynomialScale_shape {q : Int} {n rows columns : Nat}
    (scalar matrix : Mxx.Matrix) (shape : MatrixShape matrix q n rows columns) :
    MatrixShape (Mxx.matrixPolynomialScale scalar matrix) q n rows columns := by
  exact ⟨shape.modulus, shape.ringDimension, shape.rows, shape.columns⟩

private theorem addCoefficients_length_of_eq (left right : List Int)
    (lengthEq : left.length = right.length) :
    (Mxx.addCoefficients left right).length = left.length := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons _ _ => simp at lengthEq
  | cons leftHead leftTail induction =>
      cases right with
      | nil => simp at lengthEq
      | cons rightHead rightTail =>
          simp only [Mxx.addCoefficients, List.length_cons]
          congr 1
          apply induction
          exact Nat.succ.inj (by simpa using lengthEq)

private theorem subtractCoefficients_length_of_eq (left right : List Int)
    (lengthEq : left.length = right.length) :
    (Mxx.subtractCoefficients left right).length = left.length := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons _ _ => simp at lengthEq
  | cons leftHead leftTail induction =>
      cases right with
      | nil => simp at lengthEq
      | cons rightHead rightTail =>
          simp only [Mxx.subtractCoefficients, List.length_cons]
          congr 1
          apply induction
          exact Nat.succ.inj (by simpa using lengthEq)

theorem matrixMul_layout {q : Int} {n rows inner columns : Nat} (left right : Mxx.Matrix)
    (leftLayout : MatrixLayout left q n rows inner)
    (rightLayout : MatrixLayout right q n inner columns) :
    MatrixLayout (Mxx.matrixMul left right) q n rows columns := by
  refine ⟨matrixMul_shape left right leftLayout.toMatrixShape rightLayout.toMatrixShape, ?_⟩
  have compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows := by
    exact ⟨leftLayout.modulus.trans rightLayout.modulus.symm,
      leftLayout.ringDimension.trans rightLayout.ringDimension.symm,
      leftLayout.columns.trans rightLayout.rows.symm⟩
  rw [Mxx.matrixMul, if_pos compatible]
  simp [leftLayout.rows, rightLayout.columns, leftLayout.ringDimension]

theorem matrixAdd_layout {q : Int} {n rows columns : Nat} (left right : Mxx.Matrix)
    (leftLayout : MatrixLayout left q n rows columns)
    (rightLayout : MatrixLayout right q n rows columns) :
    MatrixLayout (Mxx.matrixAdd left right) q n rows columns := by
  refine ⟨matrixAdd_shape left right leftLayout.toMatrixShape rightLayout.toMatrixShape, ?_⟩
  simp only [Mxx.matrixAdd, List.length_map]
  rw [addCoefficients_length_of_eq left.coefficients right.coefficients]
  · exact leftLayout.coefficientCount
  · rw [leftLayout.coefficientCount, rightLayout.coefficientCount]

theorem matrixSubtract_layout {q : Int} {n rows columns : Nat} (left right : Mxx.Matrix)
    (leftLayout : MatrixLayout left q n rows columns)
    (rightLayout : MatrixLayout right q n rows columns) :
    MatrixLayout (Mxx.matrixSubtract left right) q n rows columns := by
  refine ⟨matrixSubtract_shape left right leftLayout.toMatrixShape rightLayout.toMatrixShape, ?_⟩
  simp only [Mxx.matrixSubtract, List.length_map]
  rw [subtractCoefficients_length_of_eq left.coefficients right.coefficients]
  · exact leftLayout.coefficientCount
  · rw [leftLayout.coefficientCount, rightLayout.coefficientCount]

theorem matrixNegate_layout {q : Int} {n rows columns : Nat} (matrix : Mxx.Matrix)
    (layout : MatrixLayout matrix q n rows columns) :
    MatrixLayout (Mxx.matrixNegate matrix) q n rows columns := by
  refine ⟨matrixNegate_shape matrix layout.toMatrixShape, ?_⟩
  simpa [Mxx.matrixNegate] using layout.coefficientCount

theorem matrixScale_layout {q : Int} {n rows columns : Nat} (scalar : Int)
    (matrix : Mxx.Matrix) (layout : MatrixLayout matrix q n rows columns) :
    MatrixLayout (Mxx.matrixScale scalar matrix) q n rows columns := by
  refine ⟨matrixScale_shape scalar matrix layout.toMatrixShape, ?_⟩
  simpa [Mxx.matrixScale] using layout.coefficientCount

theorem matrixPolynomialScale_layout {q : Int} {n rows columns : Nat}
    (scalar matrix : Mxx.Matrix) (layout : MatrixLayout matrix q n rows columns) :
    MatrixLayout (Mxx.matrixPolynomialScale scalar matrix) q n rows columns := by
  refine ⟨matrixPolynomialScale_shape scalar matrix layout.toMatrixShape, ?_⟩
  simp [Mxx.matrixPolynomialScale, layout.rows, layout.columns, layout.ringDimension]

theorem matrixSlice_layout {q : Int} {n rows columns : Nat} (matrix : Mxx.Matrix)
    (layout : MatrixLayout matrix q n rows columns)
    (rowStart rowEnd columnStart columnEnd : Nat) :
    MatrixLayout (Mxx.matrixSlice matrix rowStart rowEnd columnStart columnEnd) q n
      (rowEnd - rowStart) (columnEnd - columnStart) := by
  refine ⟨⟨?_, ?_, rfl, rfl⟩, ?_⟩
  · exact layout.modulus
  · exact layout.ringDimension
  · simp [Mxx.matrixSlice, layout.ringDimension, Nat.mul_assoc]

theorem matrixReshape_layout {q : Int} {n rows columns newRows newColumns : Nat}
    (matrix : Mxx.Matrix) (layout : MatrixLayout matrix q n rows columns)
    (entryCount : rows * columns = newRows * newColumns) :
    MatrixLayout (Mxx.matrixReshape matrix newRows newColumns) q n newRows newColumns := by
  refine ⟨⟨layout.modulus, layout.ringDimension, rfl, rfl⟩, ?_⟩
  simp only [Mxx.matrixReshape]
  rw [layout.coefficientCount, entryCount]

theorem matrixMultiply_leftScalar {q : Int} {n columns : Nat}
    (scalar matrix : Mxx.Matrix) (scalarShape : MatrixShape scalar q n 1 1)
    (matrixShape : MatrixShape matrix q n 1 columns) :
    Mxx.matrixMultiply scalar matrix = Mxx.matrixMul scalar matrix := by
  simp [Mxx.matrixMultiply, scalarShape.rows, scalarShape.columns, matrixShape.rows]

theorem matrixMultiply_rightScalar {q : Int} {n columns : Nat}
    (matrix scalar : Mxx.Matrix) (matrixShape : MatrixShape matrix q n 1 columns)
    (scalarShape : MatrixShape scalar q n 1 1) (columnsNotOne : columns ≠ 1) :
    Mxx.matrixMultiply matrix scalar = Mxx.matrixMul scalar matrix := by
  have matrixColumns : matrix.columns ≠ 1 := by
    intro equal
    apply columnsNotOne
    exact matrixShape.columns.symm.trans equal
  simp [Mxx.matrixMultiply, matrixShape.rows, matrixColumns, scalarShape.rows,
    scalarShape.columns]

theorem matrixMultiply_nonscalar {q : Int} {n rows inner columns : Nat}
    (left right : Mxx.Matrix) (leftShape : MatrixShape left q n rows inner)
    (rightShape : MatrixShape right q n inner columns)
    (leftNotScalar : rows ≠ 1 ∨ inner ≠ 1)
    (rightNotScalar : inner ≠ 1 ∨ columns ≠ 1) :
    Mxx.matrixMultiply left right = Mxx.matrixMul left right := by
  have leftCondition : ¬(left.rows = 1 ∧ left.columns = 1) := by
    intro condition
    rcases condition with ⟨leftRows, leftColumns⟩
    rcases leftNotScalar with rowsNotOne | innerNotOne
    · exact rowsNotOne (leftShape.rows.symm.trans leftRows)
    · exact innerNotOne (leftShape.columns.symm.trans leftColumns)
  have rightCondition : ¬(right.rows = 1 ∧ right.columns = 1) := by
    intro condition
    rcases condition with ⟨rightRows, rightColumns⟩
    rcases rightNotScalar with innerNotOne | columnsNotOne
    · exact innerNotOne (rightShape.rows.symm.trans rightRows)
    · exact columnsNotOne (rightShape.columns.symm.trans rightColumns)
  simp [Mxx.matrixMultiply, leftCondition, rightCondition]

/-- Runtime multiplication preserves the ordinary compatible matrix layout, including both
polynomial-scalar broadcast branches. -/
theorem matrixMultiply_layout {q : Int} {n rows inner columns : Nat}
    (left right : Mxx.Matrix) (leftLayout : MatrixLayout left q n rows inner)
    (rightLayout : MatrixLayout right q n inner columns) :
    MatrixLayout (Mxx.matrixMultiply left right) q n rows columns := by
  by_cases leftScalar : left.rows = 1 ∧ left.columns = 1
  · have rightRows : right.rows = 1 :=
      rightLayout.rows.trans (leftLayout.columns.symm.trans leftScalar.2)
    simp only [Mxx.matrixMultiply, if_pos leftScalar, if_pos rightRows]
    exact matrixMul_layout left right leftLayout rightLayout
  · by_cases rightScalar : right.rows = 1 ∧ right.columns = 1
    · have leftRowsNotOne : left.rows ≠ 1 := by
        intro leftRows
        apply leftScalar
        exact ⟨leftRows,
          leftLayout.columns.trans (rightLayout.rows.symm.trans rightScalar.1)⟩
      have innerColumns : inner = columns :=
        (rightLayout.rows.symm.trans rightScalar.1).trans
          (rightLayout.columns.symm.trans rightScalar.2).symm
      simp only [Mxx.matrixMultiply, if_neg leftScalar, if_pos rightScalar,
        if_neg leftRowsNotOne]
      simpa [innerColumns] using matrixPolynomialScale_layout right left leftLayout
    · simp only [Mxx.matrixMultiply, if_neg leftScalar, if_neg rightScalar]
      exact matrixMul_layout left right leftLayout rightLayout

/-- Polynomial-scalar broadcast in the executable semantics is entrywise multiplication in the
exact negacyclic quotient. -/
theorem matrixValue_polynomialScale (q n rows columns : Nat) [NeZero q] [NeZero n]
    (scalar matrix : Mxx.Matrix)
    (matrixType : matrix.modulus = q ∧ matrix.ringDimension = n ∧
      matrix.rows = rows ∧ matrix.columns = columns) :
    matrixValue q n rows columns (Mxx.matrixPolynomialScale scalar matrix) =
      fun row column ↦ (matrixValue q n 1 1 scalar) 0 0 *
        (matrixValue q n rows columns matrix) row column := by
  funext row column
  change negacyclicValue q n
      ((Mxx.matrixPolynomialScale scalar matrix).coefficient row column) =
    negacyclicValue q n (scalar.coefficient 0 0) *
      negacyclicValue q n (matrix.coefficient row column)
  have outputCoefficient (coefficient : Fin n) :
      (Mxx.matrixPolynomialScale scalar matrix).coefficient row column coefficient =
        Mxx.reduceCoefficient q
          (Mxx.negacyclicCoefficient n (scalar.coefficient 0 0)
            (matrix.coefficient row column) coefficient) := by
    have rowLt : row.val < matrix.rows := by rw [matrixType.2.2.1]; exact row.isLt
    have columnLt : column.val < matrix.columns := by
      rw [matrixType.2.2.2]
      exact column.isLt
    have coefficientLt : coefficient.val < matrix.ringDimension := by
      rw [matrixType.2.1]
      exact coefficient.isLt
    rw [matrixPolynomialScale_coefficient scalar matrix row column coefficient rowLt columnLt
      coefficientLt, matrixType.1, matrixType.2.1]
  unfold negacyclicValue
  simp_rw [outputCoefficient]
  simp_rw [negacyclicEmbed_reduce]
  exact negacyclicValue_convolution q n _ _

/-- Executable `Mxx.matrixMul` is ordinary matrix multiplication over the negacyclic quotient. -/
theorem matrixValue_mul (q n rows inner columns : Nat) [NeZero q] [NeZero n]
    (left right : Mxx.Matrix)
    (leftType : left.modulus = q ∧ left.ringDimension = n ∧
      left.rows = rows ∧ left.columns = inner)
    (rightType : right.modulus = q ∧ right.ringDimension = n ∧
      right.rows = inner ∧ right.columns = columns) :
    matrixValue q n rows columns (Mxx.matrixMul left right) =
      matrixValue q n rows inner left * matrixValue q n inner columns right := by
  funext row column
  simp only [matrixValue, _root_.Matrix.mul_apply]
  have compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows := by
    exact ⟨leftType.1.trans rightType.1.symm,
      leftType.2.1.trans rightType.2.1.symm,
      leftType.2.2.2.trans rightType.2.2.1.symm⟩
  have outputCoefficient (coefficient : Fin n) :
      (Mxx.matrixMul left right).coefficient row column coefficient =
        Mxx.reduceCoefficient left.modulus (
          (List.range left.columns).foldl (fun total innerIndex ↦
            total + Mxx.negacyclicCoefficient left.ringDimension
              (left.coefficient row innerIndex)
              (right.coefficient innerIndex column) coefficient) 0) := by
    apply matrixMul_coefficient left right row column coefficient compatible
    · rw [leftType.2.2.1]
      exact row.isLt
    · rw [rightType.2.2.2]
      exact column.isLt
    · rw [leftType.2.1]
      exact coefficient.isLt
  unfold negacyclicValue
  simp_rw [outputCoefficient]
  rw [leftType.1, leftType.2.1, leftType.2.2.2]
  simp_rw [negacyclicEmbed_reduce]
  simp_rw [negacyclicEmbed_indexFold]
  simp only [negacyclicEmbed]
  simp
  simp_rw [listRangeSum_eq_finSum]
  simp_rw [Finset.sum_mul]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro innerIndex _
  rw [← Finset.sum_mul]
  change negacyclicValue q n
      (Mxx.negacyclicCoefficient n (left.coefficient row innerIndex)
        (right.coefficient innerIndex column)) =
    negacyclicValue q n (left.coefficient row innerIndex) *
      negacyclicValue q n (right.coefficient innerIndex column)
  exact negacyclicValue_convolution q n _ _

/-- Runtime `matrixMultiply`, including both `1 × 1` polynomial-scalar broadcast branches, is
ordinary matrix multiplication over the negacyclic quotient for compatible matrix layouts. -/
theorem matrixValue_matrixMultiply (q n rows inner columns : Nat) [NeZero q] [NeZero n]
    (left right : Mxx.Matrix)
    (leftLayout : MatrixLayout left q n rows inner)
    (rightLayout : MatrixLayout right q n inner columns) :
    matrixValue q n rows columns (Mxx.matrixMultiply left right) =
      matrixValue q n rows inner left * matrixValue q n inner columns right := by
  have leftType : left.modulus = q ∧ left.ringDimension = n ∧
      left.rows = rows ∧ left.columns = inner :=
    ⟨leftLayout.modulus, leftLayout.ringDimension, leftLayout.rows, leftLayout.columns⟩
  have rightType : right.modulus = q ∧ right.ringDimension = n ∧
      right.rows = inner ∧ right.columns = columns :=
    ⟨rightLayout.modulus, rightLayout.ringDimension, rightLayout.rows, rightLayout.columns⟩
  by_cases leftScalar : left.rows = 1 ∧ left.columns = 1
  · have rightRows : right.rows = 1 :=
      rightLayout.rows.trans (leftLayout.columns.symm.trans leftScalar.2)
    rw [show Mxx.matrixMultiply left right = Mxx.matrixMul left right by
      simp [Mxx.matrixMultiply, leftScalar, rightRows]]
    exact matrixValue_mul q n rows inner columns left right leftType rightType
  · by_cases rightScalar : right.rows = 1 ∧ right.columns = 1
    · have leftRowsNotOne : left.rows ≠ 1 := by
        intro leftRows
        apply leftScalar
        exact ⟨leftRows,
          leftLayout.columns.trans (rightLayout.rows.symm.trans rightScalar.1)⟩
      have innerOne : inner = 1 := rightLayout.rows.symm.trans rightScalar.1
      have columnsOne : columns = 1 := rightLayout.columns.symm.trans rightScalar.2
      subst inner
      subst columns
      rw [show Mxx.matrixMultiply left right = Mxx.matrixPolynomialScale right left by
        simp [Mxx.matrixMultiply, rightScalar, leftRowsNotOne]]
      rw [matrixValue_polynomialScale q n rows 1 right left leftType]
      ext row column
      fin_cases column
      simp [_root_.Matrix.mul_apply, mul_comm]
    · rw [show Mxx.matrixMultiply left right = Mxx.matrixMul left right by
        simp [Mxx.matrixMultiply, leftScalar, rightScalar]]
      exact matrixValue_mul q n rows inner columns left right leftType rightType

/-- Reshaping a matrix back to its existing dimensions preserves every polynomial entry. -/
theorem matrixValue_reshape_same (q n rows columns : Nat) (matrix : Mxx.Matrix)
    (layout : MatrixLayout matrix q n rows columns) :
    matrixValue q n rows columns (Mxx.matrixReshape matrix rows columns) =
      matrixValue q n rows columns matrix := by
  rcases matrix with ⟨coefficients, modulus, ringDimension, matrixRows, matrixColumns⟩
  rcases layout with
    ⟨⟨modulusEq, ringDimensionEq, rowsEq, columnsEq⟩, coefficientCount⟩
  simp only at *
  subst modulus
  subst ringDimension
  subst matrixRows
  subst matrixColumns
  rfl

theorem matrixValue_add (q n rows columns : Nat) [Fact (1 < q)] [NeZero n]
    (left right : Mxx.Matrix)
    (leftType : left.modulus = q ∧ left.ringDimension = n ∧
      left.rows = rows ∧ left.columns = columns)
    (rightType : right.modulus = q ∧ right.ringDimension = n ∧
      right.rows = rows ∧ right.columns = columns) :
    matrixValue q n rows columns (Mxx.matrixAdd left right) =
      matrixValue q n rows columns left + matrixValue q n rows columns right := by
  funext row column
  change negacyclicValue q n ((Mxx.matrixAdd left right).coefficient row column) =
    negacyclicValue q n (left.coefficient row column) +
      negacyclicValue q n (right.coefficient row column)
  rw [← negacyclicValue_add]
  apply (negacyclicValue_eq_iff q n _ _).2
  intro coefficient
  simp only [Mxx.matrixAdd, Mxx.Matrix.coefficient]
  rw [leftType.1]
  conv_lhs =>
    rw [show (0 : Int) = Mxx.reduceCoefficient q 0 by
      simp [Mxx.reduceCoefficient, NeZero.ne q]]
    rw [List.getD_map]
  rw [cast_reduce, addCoefficients_getD]
  simp [leftType.2.1, rightType.2.1, leftType.2.2.2, rightType.2.2.2]

theorem matrixValue_subtract (q n rows columns : Nat) [Fact (1 < q)] [NeZero n]
    (left right : Mxx.Matrix)
    (leftType : left.modulus = q ∧ left.ringDimension = n ∧
      left.rows = rows ∧ left.columns = columns)
    (rightType : right.modulus = q ∧ right.ringDimension = n ∧
      right.rows = rows ∧ right.columns = columns) :
    matrixValue q n rows columns (Mxx.matrixSubtract left right) =
      matrixValue q n rows columns left - matrixValue q n rows columns right := by
  funext row column
  change negacyclicValue q n ((Mxx.matrixSubtract left right).coefficient row column) =
    negacyclicValue q n (left.coefficient row column) -
      negacyclicValue q n (right.coefficient row column)
  rw [← negacyclicValue_subtract]
  apply (negacyclicValue_eq_iff q n _ _).2
  intro coefficient
  simp only [Mxx.matrixSubtract, Mxx.Matrix.coefficient]
  rw [leftType.1]
  conv_lhs =>
    rw [show (0 : Int) = Mxx.reduceCoefficient q 0 by
      simp [Mxx.reduceCoefficient, NeZero.ne q]]
    rw [List.getD_map]
  rw [cast_reduce, subtractCoefficients_getD]
  simp [leftType.2.1, rightType.2.1, leftType.2.2.2, rightType.2.2.2]

theorem matrixValue_negate (q n rows columns : Nat) [Fact (1 < q)] [NeZero n]
    (matrix : Mxx.Matrix)
    (matrixType : matrix.modulus = q ∧ matrix.ringDimension = n ∧
      matrix.rows = rows ∧ matrix.columns = columns) :
    matrixValue q n rows columns (Mxx.matrixNegate matrix) =
      -matrixValue q n rows columns matrix := by
  funext row column
  change negacyclicValue q n ((Mxx.matrixNegate matrix).coefficient row column) =
    -negacyclicValue q n (matrix.coefficient row column)
  rw [← negacyclicValue_negate]
  apply (negacyclicValue_eq_iff q n _ _).2
  intro coefficient
  simp only [Mxx.matrixNegate, Mxx.Matrix.coefficient]
  rw [matrixType.1]
  conv_lhs =>
    rw [show (0 : Int) = Mxx.reduceCoefficient q (-0) by
      simp [Mxx.reduceCoefficient, NeZero.ne q]]
    rw [List.getD_map]
  rw [cast_reduce]

theorem matrixValue_scale (q n rows columns : Nat) [Fact (1 < q)] [NeZero n]
    (scalar : Int) (matrix : Mxx.Matrix)
    (matrixType : matrix.modulus = q ∧ matrix.ringDimension = n ∧
      matrix.rows = rows ∧ matrix.columns = columns) :
    matrixValue q n rows columns (Mxx.matrixScale scalar matrix) =
      (scalar : Negacyclic q n) • matrixValue q n rows columns matrix := by
  funext row column
  change negacyclicValue q n ((Mxx.matrixScale scalar matrix).coefficient row column) =
    (scalar : Negacyclic q n) • negacyclicValue q n (matrix.coefficient row column)
  rw [← negacyclicValue_scale]
  apply (negacyclicValue_eq_iff q n _ _).2
  intro coefficient
  simp only [Mxx.matrixScale, Mxx.Matrix.coefficient]
  rw [matrixType.1]
  conv_lhs =>
    rw [show (0 : Int) = Mxx.reduceCoefficient q (scalar * 0) by
      simp [Mxx.reduceCoefficient, NeZero.ne q]]
    rw [List.getD_map]
  rw [cast_reduce]

/-- A full zero coefficient buffer remains the zero matrix after applying its sampler layout. -/
theorem matrixValue_withSamplerParams_zero (q n rows columns bound : Nat) :
    matrixValue q n rows columns
      (Mxx.Matrix.withSamplerParams
        { coefficients := List.replicate (rows * columns * n) 0 }
        { maxCoefficientBound := bound, modulus := q, ringDimension := n, rows, columns }) = 0 := by
  funext row column
  unfold matrixValue negacyclicValue Mxx.Matrix.coefficient Mxx.Matrix.withSamplerParams
  simp [negacyclicEmbed]

/-- The standard row-major diagonal coefficient buffer denotes the identity matrix after
applying its sampler layout. -/
theorem matrixValue_withSamplerParams_identity (q n size bound : Nat) [NeZero n] :
    matrixValue q n size size
      (Mxx.Matrix.withSamplerParams
        { coefficients :=
            (List.range size).flatMap fun row =>
              (List.range size).flatMap fun column =>
                (List.range n).map fun coefficient =>
                  if row = column ∧ coefficient = 0 then 1 else 0 }
        { maxCoefficientBound := bound, modulus := q, ringDimension := n,
          rows := size, columns := size }) = 1 := by
  let coefficients : List Int :=
    (List.range size).flatMap fun row =>
      (List.range size).flatMap fun column =>
        (List.range n).map fun coefficient =>
          if row = column ∧ coefficient = 0 then 1 else 0
  change matrixValue q n size size
    (Mxx.Matrix.withSamplerParams { coefficients := coefficients }
      { maxCoefficientBound := bound, modulus := q, ringDimension := n,
        rows := size, columns := size }) = 1
  have coefficientsLength : coefficients.length = size * size * n := by
    simp [coefficients]
    ring
  have normalized :
      Mxx.Matrix.withSamplerParams { coefficients := coefficients }
        { maxCoefficientBound := bound, modulus := q, ringDimension := n,
          rows := size, columns := size } =
        { coefficients, modulus := q, ringDimension := n, rows := size, columns := size } := by
    simp [Mxx.Matrix.withSamplerParams, coefficientsLength]
  rw [normalized]
  funext row column
  have outputCoefficient (coefficient : Fin n) :
      coefficients.getD ((row.val * size + column.val) * n + coefficient.val) 0 =
        if row.val = column.val ∧ coefficient.val = 0 then 1 else 0 := by
    let combinedInner : Fin (size * n) :=
      ⟨column.val * n + coefficient.val, by
        calc
          column.val * n + coefficient.val < column.val * n + n :=
            Nat.add_lt_add_left coefficient.isLt _
          _ = (column.val + 1) * n := by ring
          _ ≤ size * n := Nat.mul_le_mul_right n (Nat.succ_le_iff.mpr column.isLt)⟩
    have combinedInnerValue : combinedInner.val = column.val * n + coefficient.val := rfl
    have indexEq : (row.val * size + column.val) * n + coefficient.val =
        row.val * (size * n) + combinedInner.val := by
      rw [combinedInnerValue]
      ring
    rw [indexEq]
    unfold coefficients
    rw [flatMap_range_getD size (size * n) _ _ row combinedInner (by simp)]
    rw [combinedInnerValue]
    rw [flatMap_range_getD size n _ _ column coefficient (by simp)]
    rw [map_range_getD]
  unfold matrixValue negacyclicValue Mxx.Matrix.coefficient
  simp_rw [outputCoefficient]
  by_cases same : row = column
  · subst column
    rw [Matrix.one_apply, if_pos rfl]
    simp only [true_and]
    rw [Finset.sum_eq_single (0 : Fin n)]
    · simp [negacyclicEmbed]
    · intro index _ indexNotZero
      have valueNotZero : index.val ≠ 0 := by
        intro valueZero
        exact indexNotZero (Fin.ext valueZero)
      simp [valueNotZero, negacyclicEmbed]
    · simp
  · have valuesDiffer : row.val ≠ column.val := by
      intro equal
      exact same (Fin.ext equal)
    rw [Matrix.one_apply, if_neg same]
    simp [valuesDiffer, negacyclicEmbed]

end Mxx.Toolkit
