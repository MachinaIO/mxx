import Mxx.Ir
import Mathlib.Data.List.GetD
import Mathlib.Data.ZMod.ValMinAbs
import Mathlib.Data.Matrix.Mul

namespace Mxx.Toolkit

theorem centeredCoefficient_eq_valMinAbs (q : Nat) [NeZero q] (value : Int) :
    Mxx.centeredCoefficient q value = (value : ZMod q).valMinAbs := by
  have qpos : (0 : Int) < q := by exact_mod_cast NeZero.pos q
  have qnot : ¬ (q : Int) ≤ 0 := by omega
  have residueNonnegative : 0 ≤ value % (q : Int) := Int.emod_nonneg _ (by omega)
  have residueLt : value % (q : Int) < q := Int.emod_lt_of_pos _ qpos
  rw [Mxx.centeredCoefficient, if_neg qnot, Mxx.reduceCoefficient, if_neg qnot,
    ZMod.valMinAbs_def_pos]
  have valEq : ((value : ZMod q).val : Int) = value % (q : Int) := ZMod.val_intCast value
  by_cases lower : (value : ZMod q).val ≤ q / 2
  · have lowerInt : value % (q : Int) ≤ (q : Int) / 2 := by
      rw [← valEq]
      exact_mod_cast lower
    rw [if_pos lower, if_neg (by omega)]
    omega
  · have lowerInt : (q : Int) / 2 < value % (q : Int) := by
      rw [← valEq]
      exact_mod_cast Nat.lt_of_not_ge lower
    rw [if_neg lower, if_pos (by omega)]
    omega

theorem centeredCoefficient_natAbs_le (q : Nat) [NeZero q] (value : Int) :
    (Mxx.centeredCoefficient q value).natAbs ≤ value.natAbs := by
  rw [centeredCoefficient_eq_valMinAbs]
  apply ZMod.natAbs_min_of_le_div_two q (value : ZMod q).valMinAbs value
  · exact ZMod.coe_valMinAbs _
  · exact ZMod.natAbs_valMinAbs_le _

theorem centeredCoefficient_add_natAbs_le (q : Nat) [NeZero q] (left right : Int) :
    (Mxx.centeredCoefficient q (left + right)).natAbs ≤
      (Mxx.centeredCoefficient q left).natAbs +
        (Mxx.centeredCoefficient q right).natAbs := by
  rw [centeredCoefficient_eq_valMinAbs, centeredCoefficient_eq_valMinAbs,
    centeredCoefficient_eq_valMinAbs]
  have reduced := ZMod.natAbs_valMinAbs_add_le (left : ZMod q) (right : ZMod q)
  rw [← Int.cast_add] at reduced
  exact le_trans reduced (Int.natAbs_add_le _ _)

theorem centeredCoefficient_mul_natAbs_le (q : Nat) [NeZero q] (left right : Int) :
    (Mxx.centeredCoefficient q (left * right)).natAbs ≤
      (Mxx.centeredCoefficient q left).natAbs *
        (Mxx.centeredCoefficient q right).natAbs := by
  rw [centeredCoefficient_eq_valMinAbs, centeredCoefficient_eq_valMinAbs,
    centeredCoefficient_eq_valMinAbs]
  have reduced := ZMod.natAbs_min_of_le_div_two q
    ((left : ZMod q) * (right : ZMod q)).valMinAbs
    ((left : ZMod q).valMinAbs * (right : ZMod q).valMinAbs)
    (by simp) (ZMod.natAbs_valMinAbs_le _)
  rw [← Int.cast_mul, Int.natAbs_mul] at reduced
  exact reduced

theorem centeredCoefficient_neg_natAbs (q : Nat) [NeZero q] (value : Int) :
    (Mxx.centeredCoefficient q (-value)).natAbs =
      (Mxx.centeredCoefficient q value).natAbs := by
  rw [centeredCoefficient_eq_valMinAbs, centeredCoefficient_eq_valMinAbs]
  simpa only [Int.cast_neg] using ZMod.natAbs_valMinAbs_neg (value : ZMod q)

theorem centeredFoldlAdd_natAbs_le (q : Nat) [NeZero q] (initial : Int)
    (values : List Int) :
    (Mxx.centeredCoefficient q (values.foldl (· + ·) initial)).natAbs ≤
      (Mxx.centeredCoefficient q initial).natAbs +
        (values.map fun value ↦ (Mxx.centeredCoefficient q value).natAbs).sum := by
  induction values generalizing initial with
  | nil => simp
  | cons head tail induction =>
      simp only [List.foldl_cons, List.map_cons, List.sum_cons]
      exact le_trans (induction (initial + head)) <| by
        have added := centeredCoefficient_add_natAbs_le q initial head
        omega

/-- Convolution by the canonical polynomial one is exact.  This deterministic fact is used by
transition-selector execution lifting; unlike a generic product bound, it introduces no ring
dimension factor. -/
theorem negacyclicCoefficient_rightUnit
    (ringDimension : Nat) (left : Nat → Int) (coefficient : Nat)
    (coefficientLt : coefficient < ringDimension) :
    Mxx.negacyclicCoefficient ringDimension left
        (fun index ↦ if index = 0 then 1 else 0) coefficient = left coefficient := by
  let term : Nat → Int := fun index ↦
    if index ≤ coefficient then
      left index * (if coefficient - index = 0 then 1 else 0)
    else
      -(left index * (if ringDimension + coefficient - index = 0 then 1 else 0))
  have foldEq : Mxx.negacyclicCoefficient ringDimension left
      (fun index ↦ if index = 0 then 1 else 0) coefficient =
      (List.range ringDimension).foldl (fun total index ↦ total + term index) 0 := by
    unfold Mxx.negacyclicCoefficient
    congr 1
    funext total index
    by_cases before : index ≤ coefficient <;> simp [term, before, sub_eq_add_neg]
  rw [foldEq]
  have foldSumAux : ∀ (values : List Nat) (initial : Int),
      values.foldl (fun total index ↦ total + term index) initial =
        initial + (values.map term).sum := by
    intro values initial
    induction values generalizing initial with
    | nil => simp
    | cons head tail induction =>
        rw [List.foldl_cons, induction]
        simp [Int.add_assoc]
  have foldSum := foldSumAux (List.range ringDimension) 0
  rw [foldSum]
  simp only [zero_add]
  have mappedSumAux : ∀ (count : Nat) (value : Nat → Int),
      ((List.range count).map value).sum =
        ∑ index ∈ Finset.range count, value index := by
    intro count value
    induction count with
    | zero => simp
    | succ n induction =>
        rw [List.range_succ, List.map_append, List.sum_append]
        simp [Finset.sum_range_succ, induction]
  have mappedSum := mappedSumAux ringDimension term
  rw [mappedSum]
  rw [Finset.sum_eq_single coefficient]
  · simp [term]
  · intro index member different
    have indexLt : index < ringDimension := List.mem_range.mp member
    by_cases before : index ≤ coefficient
    · have difference : coefficient - index ≠ 0 := by omega
      simp [term, before, difference]
    · have difference : ringDimension + coefficient - index ≠ 0 := by omega
      simp [term, before, difference]
  · simp [coefficientLt]

/-- The exact `1 × 1` identity value constructed by an executable IR `IdentityMatrix` node. -/
def unitPolynomialMatrix (q ringDimension : Nat) : Mxx.Matrix :=
  Mxx.Matrix.withSamplerParams
    { coefficients := (List.range ringDimension).map fun coefficient ↦
        if coefficient = 0 then 1 else 0 }
    { modulus := q, ringDimension, rows := 1, columns := 1, maxCoefficientBound := 0 }

/-- Sound coefficient bound for the exact negacyclic convolution used by `Mxx.matrixMul`. -/
theorem negacyclicCoefficient_natAbs_le (q ringDimension : Nat) [NeZero q]
    (left right : Nat → Int) (coefficient leftBound rightBound : Nat)
    (coefficient_lt : coefficient < ringDimension)
    (left_le : ∀ index < ringDimension,
      (Mxx.centeredCoefficient q (left index)).natAbs ≤ leftBound)
    (right_le : ∀ index < ringDimension,
      (Mxx.centeredCoefficient q (right index)).natAbs ≤ rightBound) :
    (Mxx.centeredCoefficient q
      (Mxx.negacyclicCoefficient ringDimension left right coefficient)).natAbs ≤
        ringDimension * leftBound * rightBound := by
  let term := fun index ↦
    if index ≤ coefficient then left index * right (coefficient - index)
    else -(left index * right (ringDimension + coefficient - index))
  have term_le : ∀ index ∈ List.range ringDimension,
      (Mxx.centeredCoefficient q (term index)).natAbs ≤ leftBound * rightBound := by
    intro index member
    have index_lt : index < ringDimension := List.mem_range.mp member
    by_cases before : index ≤ coefficient
    · rw [show term index = left index * right (coefficient - index) by simp [term, before]]
      exact le_trans (centeredCoefficient_mul_natAbs_le q _ _) <|
        Nat.mul_le_mul (left_le index index_lt) (right_le _ (by omega))
    · rw [show term index = -(left index * right (ringDimension + coefficient - index)) by
        simp [term, before]]
      rw [centeredCoefficient_neg_natAbs]
      exact le_trans (centeredCoefficient_mul_natAbs_le q _ _) <|
        Nat.mul_le_mul (left_le index index_lt) (right_le _ (by omega))
  have foldEq : Mxx.negacyclicCoefficient ringDimension left right coefficient =
      (List.range ringDimension).foldl (fun total index ↦ total + term index) 0 := by
    unfold Mxx.negacyclicCoefficient
    congr 1
    funext total index
    by_cases before : index ≤ coefficient <;> simp [term, before, sub_eq_add_neg]
  rw [foldEq]
  calc
    _ ≤ (Mxx.centeredCoefficient q 0).natAbs +
        ((List.range ringDimension).map fun index ↦
          (Mxx.centeredCoefficient q (term index)).natAbs).sum :=
      by simpa [List.foldl_map, Function.comp_def] using
        centeredFoldlAdd_natAbs_le q 0 (List.range ringDimension |>.map term)
    _ ≤ 0 + ((List.range ringDimension).map fun _ ↦ leftBound * rightBound).sum := by
      apply Nat.add_le_add
      · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient]
      · apply List.sum_le_sum
        intro index member
        exact term_le index member
    _ = ringDimension * leftBound * rightBound := by simp [Nat.mul_assoc]

theorem coefficientNorm_le {coefficients : List Int} {bound : Nat}
    (bounded : ∀ coefficient ∈ coefficients, coefficient.natAbs ≤ bound) :
    Mxx.coefficientNorm coefficients ≤ bound := by
  induction coefficients with
  | nil => simp [Mxx.coefficientNorm]
  | cons head tail induction =>
      simp only [Mxx.coefficientNorm]
      apply max_le
      · exact bounded head (by simp)
      · exact induction fun coefficient member ↦ bounded coefficient (by simp [member])

/-- Every stored coefficient is bounded by the centered coefficient norm. The statement also
covers an out-of-range access, whose `getD` value is zero. -/
theorem centeredGetD_natAbs_le_norm (matrix : Mxx.Matrix) (index : Nat) :
    (Mxx.centeredCoefficient matrix.modulus (matrix.coefficients.getD index 0)).natAbs ≤
      Mxx.maxCenteredCoefficientNorm matrix := by
  by_cases inRange : index < matrix.coefficients.length
  · apply Mxx.coefficient_natAbs_le_norm
    rw [List.getD_eq_getElem _ _ inRange]
    exact List.mem_map_of_mem (List.getElem_mem inRange)
  · rw [List.getD_eq_default _ _ (Nat.le_of_not_gt inRange)]
    have centeredZero : Mxx.centeredCoefficient matrix.modulus 0 = 0 := by
      by_cases nonpositive : matrix.modulus ≤ 0
      · simp [Mxx.centeredCoefficient, nonpositive]
      · have positive : 0 < matrix.modulus := lt_of_not_ge nonpositive
        simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, nonpositive,
          positive.ne', positive.le]
    simp [centeredZero]

theorem centeredEntry_natAbs_le_norm (matrix : Mxx.Matrix)
    (row column coefficient : Nat) :
    (Mxx.centeredCoefficient matrix.modulus
      (matrix.coefficient row column coefficient)).natAbs ≤
      Mxx.maxCenteredCoefficientNorm matrix := by
  exact centeredGetD_natAbs_le_norm matrix
    ((row * matrix.columns + column) * matrix.ringDimension + coefficient)

/-- Runtime multiplication by the canonical polynomial one preserves a scalar matrix's exact
coefficient norm.  In particular, the bound has no generic negacyclic ring-dimension factor. -/
theorem matrixMultiply_unitPolynomial_norm_le
    (q ringDimension bound : Nat) [NeZero q] [NeZero ringDimension]
    (left : Mxx.Matrix)
    (leftModulus : left.modulus = q)
    (leftRingDimension : left.ringDimension = ringDimension)
    (leftRows : left.rows = 1) (leftColumns : left.columns = 1)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ bound) :
    Mxx.maxCenteredCoefficientNorm
      (Mxx.matrixMultiply left (unitPolynomialMatrix q ringDimension)) ≤ bound := by
  have unitRows : (unitPolynomialMatrix q ringDimension).rows = 1 := by
    simp [unitPolynomialMatrix, Mxx.Matrix.withSamplerParams]
  have unitColumns : (unitPolynomialMatrix q ringDimension).columns = 1 := by
    simp [unitPolynomialMatrix, Mxx.Matrix.withSamplerParams]
  have compatible : left.modulus = (unitPolynomialMatrix q ringDimension).modulus ∧
      left.ringDimension = (unitPolynomialMatrix q ringDimension).ringDimension ∧
      left.columns = (unitPolynomialMatrix q ringDimension).rows := by
    simp [unitPolynomialMatrix, Mxx.Matrix.withSamplerParams, leftModulus,
      leftRingDimension, leftColumns]
  rw [Mxx.matrixMultiply, if_pos ⟨leftRows, leftColumns⟩, if_pos unitRows,
    Mxx.matrixMul, if_pos compatible]
  unfold Mxx.maxCenteredCoefficientNorm
  apply coefficientNorm_le
  intro outputCoefficient member
  obtain ⟨reduced, reducedMember, rfl⟩ := List.mem_map.mp member
  obtain ⟨linear, rfl⟩ := List.mem_ofFn.mp reducedMember
  let coefficient := linear.val % left.ringDimension
  have coefficientLt : coefficient < left.ringDimension := Nat.mod_lt _ (by
    simpa [leftRingDimension] using NeZero.pos ringDimension)
  have unitCoefficient (index : Nat) :
      (unitPolynomialMatrix q ringDimension).coefficient 0 0 index =
        if index = 0 then 1 else 0 := by
    unfold unitPolynomialMatrix Mxx.Matrix.withSamplerParams Mxx.Matrix.coefficient
    simp only [Nat.one_mul, Nat.zero_mul, Nat.zero_add, List.length_map, List.length_range,
      Nat.sub_self, List.replicate_zero, List.append_nil]
    rw [List.take_of_length_le (by simp)]
    by_cases indexLt : index < ringDimension
    · rw [List.getD_eq_getElem _ _ (by simpa using indexLt)]
      simp [List.getElem_map]
    · rw [List.getD_eq_default _ _ (by simpa using Nat.le_of_not_gt indexLt)]
      have ringPositive : 0 < ringDimension := NeZero.pos ringDimension
      simp [show index ≠ 0 by omega]
  have linearLtRing : linear.val < left.ringDimension := by
    simpa [leftRows, unitColumns] using linear.isLt
  have entryZero : linear.val / left.ringDimension = 0 := Nat.div_eq_of_lt linearLtRing
  simp only [leftColumns, List.range_one, List.foldl_cons, List.foldl_nil, zero_add]
  simp only [unitColumns, entryZero, Nat.zero_div, Nat.zero_mod]
  rw [show (unitPolynomialMatrix q ringDimension).coefficient 0 0 =
      (fun index ↦ if index = 0 then 1 else 0) by funext index; exact unitCoefficient index]
  rw [negacyclicCoefficient_rightUnit left.ringDimension (left.coefficient 0 0) coefficient
    coefficientLt]
  have centeredReduce :
      Mxx.centeredCoefficient left.modulus
          (Mxx.reduceCoefficient left.modulus (left.coefficient 0 0 coefficient)) =
        Mxx.centeredCoefficient left.modulus (left.coefficient 0 0 coefficient) := by
    rw [leftModulus]
    rw [centeredCoefficient_eq_valMinAbs q, centeredCoefficient_eq_valMinAbs q]
    congr 1
    simp [Mxx.reduceCoefficient]
  rw [centeredReduce]
  exact le_trans (centeredEntry_natAbs_le_norm left 0 0 coefficient) leftNorm

theorem coefficientVectors_member {values coefficients : List Int} {count : Nat}
    (member : coefficients ∈ Mxx.Ir.coefficientVectors values count) :
    ∀ coefficient ∈ coefficients, coefficient ∈ values := by
  induction count generalizing coefficients with
  | zero =>
      simp [Mxx.Ir.coefficientVectors] at member
      subst coefficients
      simp
  | succ count induction =>
      rw [Mxx.Ir.coefficientVectors] at member
      obtain ⟨head, headMember, generatedMember⟩ := List.mem_flatMap.mp member
      obtain ⟨tail, tailMember, rfl⟩ := List.mem_map.mp generatedMember
      intro coefficient coefficientMember
      rcases List.mem_cons.mp coefficientMember with rfl | coefficientMember
      · exact headMember
      · exact induction tailMember coefficient coefficientMember

/-- Every coefficient sampled uniformly from `{-1, 0, 1}` has deterministic centered norm at
most one. This proof is independent of the matrix dimensions and does not enumerate the support. -/
theorem uniformMatrixSupport_minusOneOne_norm_le (params : Mxx.SamplerParams)
    (modulus_ge : 2 ≤ params.modulus) (matrix : Mxx.Matrix)
    (member : matrix ∈ Mxx.Ir.uniformMatrixSupport params (-1) 1) :
    Mxx.maxCenteredCoefficientNorm matrix ≤ 1 := by
  have range : Mxx.Ir.integerRange (-1) 1 = [-1, 0, 1] := by decide
  rw [Mxx.Ir.uniformMatrixSupport, range] at member
  obtain ⟨coefficients, coefficientsMember, rfl⟩ := List.mem_map.mp member
  have modulus_eq : (params.modulus.toNat : Int) = params.modulus := by omega
  letI : NeZero params.modulus.toNat := ⟨by omega⟩
  unfold Mxx.maxCenteredCoefficientNorm
  apply coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append, List.mem_replicate] at coefficientMember
  rcases coefficientMember with coefficientMember | ⟨_, rfl⟩
  · have sourceMember : coefficient ∈ coefficients := List.mem_of_mem_take coefficientMember
    have coefficientRange := coefficientVectors_member coefficientsMember coefficient sourceMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at coefficientRange
    change (Mxx.centeredCoefficient params.modulus coefficient).natAbs ≤ 1
    rw [← modulus_eq]
    exact le_trans (centeredCoefficient_natAbs_le params.modulus.toNat coefficient) (by
      rcases coefficientRange with rfl | rfl | rfl <;> decide)
  · change (Mxx.centeredCoefficient params.modulus 0).natAbs ≤ 1
    rw [← modulus_eq]
    exact le_trans (centeredCoefficient_natAbs_le params.modulus.toNat 0) (by decide)

theorem centeredCoefficient_reduce (q : Nat) [NeZero q] (value : Int) :
    Mxx.centeredCoefficient q (Mxx.reduceCoefficient q value) =
      Mxx.centeredCoefficient q value := by
  rw [centeredCoefficient_eq_valMinAbs, centeredCoefficient_eq_valMinAbs]
  congr 1
  simp [Mxx.reduceCoefficient, NeZero.ne q]

private def centeredListNorm (q : Nat) (coefficients : List Int) : Nat :=
  Mxx.coefficientNorm (coefficients.map (Mxx.centeredCoefficient q))

@[simp] private theorem centeredListNorm_nil (q : Nat) : centeredListNorm q [] = 0 := rfl

@[simp] private theorem centeredListNorm_cons (q : Nat) (head : Int) (tail : List Int) :
    centeredListNorm q (head :: tail) =
      max (Mxx.centeredCoefficient q head).natAbs (centeredListNorm q tail) := rfl

private theorem centeredListNorm_reduce (q : Nat) [NeZero q] (coefficients : List Int) :
    centeredListNorm q (coefficients.map (Mxx.reduceCoefficient q)) =
      centeredListNorm q coefficients := by
  induction coefficients with
  | nil => rfl
  | cons head tail induction =>
      simp only [List.map_cons, centeredListNorm_cons]
      rw [centeredCoefficient_reduce, induction]

private theorem centeredListNorm_neg (q : Nat) [NeZero q] (coefficients : List Int) :
    centeredListNorm q (coefficients.map (-·)) = centeredListNorm q coefficients := by
  induction coefficients with
  | nil => rfl
  | cons head tail induction =>
      simp only [List.map_cons, centeredListNorm_cons]
      rw [centeredCoefficient_neg_natAbs, induction]

private theorem centeredListNorm_append (q : Nat) (left right : List Int) :
    centeredListNorm q (left ++ right) =
      max (centeredListNorm q left) (centeredListNorm q right) := by
  induction left with
  | nil => simp
  | cons head tail induction =>
      simp only [List.cons_append, centeredListNorm_cons, induction]
      rw [max_assoc]

private theorem addCoefficients_norm_le (q : Nat) [NeZero q] (left right : List Int) :
    centeredListNorm q (Mxx.addCoefficients left right) ≤
      centeredListNorm q left + centeredListNorm q right := by
  induction left generalizing right with
  | nil => simp [Mxx.addCoefficients]
  | cons leftHead leftTail induction =>
      cases right with
      | nil => simp [Mxx.addCoefficients]
      | cons rightHead rightTail =>
          simp only [Mxx.addCoefficients, centeredListNorm_cons]
          apply max_le
          · have added := centeredCoefficient_add_natAbs_le q leftHead rightHead
            exact le_trans added (by omega)
          · exact le_trans (induction rightTail) (by omega)

private theorem subtractCoefficients_norm_le (q : Nat) [NeZero q]
    (left right : List Int) :
    centeredListNorm q (Mxx.subtractCoefficients left right) ≤
      centeredListNorm q left + centeredListNorm q right := by
  induction left generalizing right with
  | nil => simp [Mxx.subtractCoefficients, centeredListNorm_neg]
  | cons leftHead leftTail induction =>
      cases right with
      | nil => simp [Mxx.subtractCoefficients]
      | cons rightHead rightTail =>
          simp only [Mxx.subtractCoefficients, centeredListNorm_cons]
          apply max_le
          · have added := centeredCoefficient_add_natAbs_le q leftHead (-rightHead)
            rw [centeredCoefficient_neg_natAbs] at added
            simpa [sub_eq_add_neg] using le_trans added (by omega)
          · exact le_trans (induction rightTail) (by omega)

private theorem scaleCoefficients_norm_le (q : Nat) [NeZero q]
    (scalar : Int) (coefficients : List Int) :
    centeredListNorm q (coefficients.map (scalar * ·)) ≤
      scalar.natAbs * centeredListNorm q coefficients := by
  induction coefficients with
  | nil => simp
  | cons head tail induction =>
      simp only [List.map_cons, centeredListNorm_cons]
      apply max_le
      · have multiplied := centeredCoefficient_mul_natAbs_le q scalar head
        have scalarBound := centeredCoefficient_natAbs_le q scalar
        exact le_trans multiplied <| le_trans
          (Nat.mul_le_mul_right _ scalarBound)
          (Nat.mul_le_mul_left _ (le_max_left _ _))
      · exact le_trans induction (Nat.mul_le_mul_left _ (le_max_right _ _))

private theorem centeredInnerSum_natAbs_le (q inner bound : Nat) [NeZero q]
    (term : Nat → Int)
    (term_le : ∀ index < inner,
      (Mxx.centeredCoefficient q (term index)).natAbs ≤ bound) :
    (Mxx.centeredCoefficient q
      ((List.range inner).foldl (fun total index ↦ total + term index) 0)).natAbs ≤
        inner * bound := by
  calc
    _ ≤ (Mxx.centeredCoefficient q 0).natAbs +
        ((List.range inner).map fun index ↦
          (Mxx.centeredCoefficient q (term index)).natAbs).sum :=
      by simpa [List.foldl_map, Function.comp_def] using
        centeredFoldlAdd_natAbs_le q 0 (List.range inner |>.map term)
    _ ≤ 0 + ((List.range inner).map fun _ ↦ bound).sum := by
      apply Nat.add_le_add
      · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient]
      · apply List.sum_le_sum
        intro index member
        exact term_le index (List.mem_range.mp member)
    _ = inner * bound := by simp

/-- Full worst-case coefficient norm for the exact dynamic-size matrix multiplication used by
the executable IR.  The bound contains both the matrix contraction dimension and the complete
negacyclic ring dimension; it makes no independence or CLT assumption. -/
theorem matrixMul_norm_le (q ringDimension inner leftBound rightBound : Nat) [NeZero q]
    (left right : Mxx.Matrix)
    (leftModulus : left.modulus = q) (rightModulus : right.modulus = q)
    (leftRing : left.ringDimension = ringDimension)
    (rightRing : right.ringDimension = ringDimension)
    (leftColumns : left.columns = inner) (rightRows : right.rows = inner)
    (leftNorm : Mxx.maxCenteredCoefficientNorm left ≤ leftBound)
    (rightNorm : Mxx.maxCenteredCoefficientNorm right ≤ rightBound) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixMul left right) ≤
      ringDimension * inner * leftBound * rightBound := by
  have compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows := by
    omega
  rw [Mxx.matrixMul, if_pos compatible]
  unfold Mxx.maxCenteredCoefficientNorm
  apply coefficientNorm_le
  intro outputCoefficient member
  simp only [List.mem_map] at member
  obtain ⟨reduced, reducedMember, rfl⟩ := member
  obtain ⟨linear, rfl⟩ := List.mem_ofFn.mp reducedMember
  let coefficient := linear.val % left.ringDimension
  let entry := linear.val / left.ringDimension
  let column := entry % right.columns
  let row := entry / right.columns
  have ringDimensionPositive : 0 < left.ringDimension := by
    by_contra nonpositive
    have zero : left.ringDimension = 0 := Nat.eq_zero_of_not_pos nonpositive
    have linearLt := linear.isLt
    simp [zero] at linearLt
  have coefficientLt : coefficient < left.ringDimension :=
    Nat.mod_lt _ ringDimensionPositive
  rw [leftModulus]
  rw [centeredCoefficient_reduce]
  let convolution := fun index ↦ Mxx.negacyclicCoefficient ringDimension
    (left.coefficient row index) (right.coefficient index column) coefficient
  have convolution_le : ∀ index < inner,
      (Mxx.centeredCoefficient q (convolution index)).natAbs ≤
        ringDimension * leftBound * rightBound := by
    intro index indexLt
    apply negacyclicCoefficient_natAbs_le q ringDimension _ _ coefficient
    · simpa [leftRing] using coefficientLt
    · intro coefficientIndex coefficientIndexLt
      exact le_trans
        (by simpa [leftModulus] using
          centeredEntry_natAbs_le_norm left row index coefficientIndex)
        leftNorm
    · intro coefficientIndex coefficientIndexLt
      exact le_trans
        (by simpa [rightModulus] using
          centeredEntry_natAbs_le_norm right index column coefficientIndex)
        rightNorm
  have summed := centeredInnerSum_natAbs_le q inner
    (ringDimension * leftBound * rightBound) convolution convolution_le
  simpa [convolution, row, column, entry, coefficient, leftRing, leftColumns,
    Nat.mul_assoc, Nat.mul_left_comm, Nat.mul_comm] using summed

theorem matrixAdd_norm_le (q : Nat) [NeZero q] (left right : Mxx.Matrix)
    (leftModulus : left.modulus = q) (rightModulus : right.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixAdd left right) ≤
      Mxx.maxCenteredCoefficientNorm left + Mxx.maxCenteredCoefficientNorm right := by
  unfold Mxx.matrixAdd Mxx.maxCenteredCoefficientNorm
  rw [leftModulus]
  change centeredListNorm q
      ((Mxx.addCoefficients left.coefficients right.coefficients).map
        (Mxx.reduceCoefficient q)) ≤ _
  rw [centeredListNorm_reduce]
  simpa [centeredListNorm, leftModulus, rightModulus] using
    addCoefficients_norm_le q left.coefficients right.coefficients

theorem matrixSubtract_norm_le (q : Nat) [NeZero q] (left right : Mxx.Matrix)
    (leftModulus : left.modulus = q) (rightModulus : right.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixSubtract left right) ≤
      Mxx.maxCenteredCoefficientNorm left + Mxx.maxCenteredCoefficientNorm right := by
  unfold Mxx.matrixSubtract Mxx.maxCenteredCoefficientNorm
  rw [leftModulus]
  change centeredListNorm q
      ((Mxx.subtractCoefficients left.coefficients right.coefficients).map
        (Mxx.reduceCoefficient q)) ≤ _
  rw [centeredListNorm_reduce]
  simpa [centeredListNorm, leftModulus, rightModulus] using
    subtractCoefficients_norm_le q left.coefficients right.coefficients

theorem matrixNegate_norm (q : Nat) [NeZero q] (matrix : Mxx.Matrix)
    (modulus : matrix.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixNegate matrix) =
      Mxx.maxCenteredCoefficientNorm matrix := by
  unfold Mxx.matrixNegate Mxx.maxCenteredCoefficientNorm
  rw [modulus]
  change centeredListNorm q
      (matrix.coefficients.map fun coefficient ↦ Mxx.reduceCoefficient q (-coefficient)) = _
  rw [show (matrix.coefficients.map fun coefficient ↦ Mxx.reduceCoefficient q (-coefficient)) =
    (matrix.coefficients.map (-·)).map (Mxx.reduceCoefficient q) by
      simp [List.map_map, Function.comp_def]]
  rw [centeredListNorm_reduce, centeredListNorm_neg]
  simp [centeredListNorm]

theorem matrixScale_norm_le (q : Nat) [NeZero q] (scalar : Int) (matrix : Mxx.Matrix)
    (modulus : matrix.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixScale scalar matrix) ≤
      scalar.natAbs * Mxx.maxCenteredCoefficientNorm matrix := by
  unfold Mxx.matrixScale Mxx.maxCenteredCoefficientNorm
  rw [modulus]
  change centeredListNorm q
      (matrix.coefficients.map fun coefficient ↦
        Mxx.reduceCoefficient q (scalar * coefficient)) ≤ _
  rw [show (matrix.coefficients.map fun coefficient ↦
      Mxx.reduceCoefficient q (scalar * coefficient)) =
    (matrix.coefficients.map (scalar * ·)).map (Mxx.reduceCoefficient q) by
      simp [List.map_map, Function.comp_def]]
  rw [centeredListNorm_reduce]
  simpa [centeredListNorm, modulus] using
    scaleCoefficients_norm_le q scalar matrix.coefficients

/-- Vertical concatenation cannot exceed the larger centered coefficient norm of its inputs. -/
theorem matrixConcatRows_two_norm_le (q : Nat) (top bottom : Mxx.Matrix)
    (topModulus : top.modulus = q) (bottomModulus : bottom.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatRows [top, bottom]) ≤
      max (Mxx.maxCenteredCoefficientNorm top)
        (Mxx.maxCenteredCoefficientNorm bottom) := by
  unfold Mxx.matrixConcatRows Mxx.maxCenteredCoefficientNorm
  simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil]
  rw [topModulus]
  change centeredListNorm q (top.coefficients ++ bottom.coefficients) ≤ _
  rw [centeredListNorm_append]
  simp [centeredListNorm, bottomModulus]

/-- Horizontal concatenation cannot exceed the larger centered coefficient norm of its inputs. -/
theorem matrixConcatColumns_two_norm_le (q : Nat) (left right : Mxx.Matrix)
    (leftModulus : left.modulus = q) (rightModulus : right.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatColumns [left, right]) ≤
      max (Mxx.maxCenteredCoefficientNorm left)
        (Mxx.maxCenteredCoefficientNorm right) := by
  unfold Mxx.maxCenteredCoefficientNorm
  simp only [Mxx.matrixConcatColumns]
  rw [leftModulus]
  apply coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficientValue, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  obtain ⟨row, _, rowMember⟩ := List.mem_flatMap.mp coefficientMember
  obtain ⟨matrix, matrixMember, matrixCoefficientMember⟩ := List.mem_flatMap.mp rowMember
  obtain ⟨column, _, columnMember⟩ := List.mem_flatMap.mp matrixCoefficientMember
  obtain ⟨coefficient, _, rfl⟩ := List.mem_map.mp columnMember
  simp only [List.mem_cons, List.not_mem_nil, or_false] at matrixMember
  rcases matrixMember with matrixEq | matrixEq
  · exact le_trans
      (by simpa [Mxx.maxCenteredCoefficientNorm, matrixEq, leftModulus] using
        centeredEntry_natAbs_le_norm left row column coefficient)
      (le_max_left _ _)
  · exact le_trans
      (by simpa [Mxx.maxCenteredCoefficientNorm, matrixEq, rightModulus] using
        centeredEntry_natAbs_le_norm right row column coefficient)
      (le_max_right _ _)

/-- Binary block-diagonal concatenation only copies coefficients from one input or inserts zero,
so its centered coefficient norm is bounded by the larger input norm. -/
theorem matrixConcatDiagonal_two_norm_le (q : Nat) (left right : Mxx.Matrix)
    (leftModulus : left.modulus = q) (rightModulus : right.modulus = q) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatDiagonal [left, right]) ≤
      max (Mxx.maxCenteredCoefficientNorm left)
        (Mxx.maxCenteredCoefficientNorm right) := by
  unfold Mxx.maxCenteredCoefficientNorm
  simp only [Mxx.matrixConcatDiagonal]
  rw [leftModulus]
  apply coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficientValue, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  obtain ⟨row, _, rowMember⟩ := List.mem_flatMap.mp coefficientMember
  obtain ⟨column, _, columnMember⟩ := List.mem_flatMap.mp rowMember
  obtain ⟨coefficient, _, rfl⟩ := List.mem_map.mp columnMember
  by_cases inLeft : row < left.rows ∧ column < left.columns
  · simp [Mxx.diagonalCoefficient, inLeft]
    exact Or.inl (by simpa [Mxx.maxCenteredCoefficientNorm, leftModulus] using
      (centeredEntry_natAbs_le_norm left row column coefficient))
  · by_cases inRight : left.rows ≤ row ∧ row < left.rows + right.rows ∧
        left.columns ≤ column ∧ column < left.columns + right.columns
    · simp [Mxx.diagonalCoefficient, inLeft, inRight]
      exact Or.inr (by simpa [Mxx.maxCenteredCoefficientNorm, leftModulus, rightModulus] using
        (centeredEntry_natAbs_le_norm right
          (row - left.rows) (column - left.columns) coefficient))
    · have centeredZero : Mxx.centeredCoefficient q 0 = 0 := by
        by_cases qZero : q = 0
        · simp [qZero, Mxx.centeredCoefficient]
        · simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, qZero]
      simp [Mxx.diagonalCoefficient, inLeft, inRight, centeredZero]

def addBound (left right : Nat) : Nat := left + right
def subtractBound (left right : Nat) : Nat := left + right
def scaleBound (scalarAbs bound : Nat) : Nat := scalarAbs * bound
def multiplyBound (left right innerDimension ringDimension : Nat) : Nat :=
  left * right * innerDimension * ringDimension
def concatBound : List Nat → Nat
  | [] => 0
  | bound :: bounds => max bound (concatBound bounds)
def selectBound (bounds : List Nat) : Nat := concatBound bounds
def sliceBound (bound : Nat) : Nat := bound
def reshapeBound (bound : Nat) : Nat := bound
def centeredModulusCap (modulus : Nat) (bound : Nat) : Nat := min bound (modulus / 2)

theorem multiplyBound_uses_full_contraction
    (left right innerDimension ringDimension : Nat) :
    multiplyBound left right innerDimension ringDimension =
      left * right * innerDimension * ringDimension := rfl

theorem addBound_sound {left right actual : Nat}
    (actual_le : actual ≤ left + right) : actual ≤ addBound left right := actual_le

theorem subtractBound_sound {left right actual : Nat}
    (actual_le : actual ≤ left + right) : actual ≤ subtractBound left right := actual_le

theorem scaleBound_sound {scalarAbs bound actual : Nat}
    (actual_le : actual ≤ scalarAbs * bound) : actual ≤ scaleBound scalarAbs bound := actual_le

theorem multiplyBound_sound {left right innerDimension ringDimension actual : Nat}
    (actual_le : actual ≤ left * right * innerDimension * ringDimension) :
    actual ≤ multiplyBound left right innerDimension ringDimension := actual_le

/-- Deterministic worst-case bound for a finite dot product. No independence
or central-limit assumption is used. -/
theorem dotProduct_natAbs_le {ι : Type} [Fintype ι] [DecidableEq ι]
    (left right : ι → Int) (leftBound rightBound : Nat)
    (left_le : ∀ index, (left index).natAbs ≤ leftBound)
    (right_le : ∀ index, (right index).natAbs ≤ rightBound) :
    (∑ index, left index * right index).natAbs ≤
      Fintype.card ι * leftBound * rightBound := by
  calc
    (∑ index, left index * right index).natAbs ≤
        ∑ index, (left index * right index).natAbs :=
      Int.natAbs_sum_le Finset.univ _
    _ ≤ ∑ _index : ι, leftBound * rightBound := by
      apply Finset.sum_le_sum
      intro index _
      rw [Int.natAbs_mul]
      exact Nat.mul_le_mul (left_le index) (right_le index)
    _ = Fintype.card ι * leftBound * rightBound := by
      simp [Nat.mul_assoc]

/-- Entrywise form of the full contraction rule for ordinary integer
matrices. Polynomial convolution contributes the separate ring-dimension
factor in the IR-level rule; the fixed coefficient matrices used below have
already expanded that convolution index. -/
theorem matrixMulEntry_natAbs_le {rows inner columns : Nat}
    (left : _root_.Matrix (Fin rows) (Fin inner) Int)
    (right : _root_.Matrix (Fin inner) (Fin columns) Int)
    (leftBound rightBound : Nat)
    (left_le : ∀ row column, (left row column).natAbs ≤ leftBound)
    (right_le : ∀ row column, (right row column).natAbs ≤ rightBound)
    (row : Fin rows) (column : Fin columns) :
    ((left * right) row column).natAbs ≤ inner * leftBound * rightBound := by
  rw [Matrix.mul_apply]
  simpa using dotProduct_natAbs_le
    (fun index => left row index) (fun index => right index column)
    leftBound rightBound (left_le row) (fun index => right_le index column)

theorem concatBound_contains (bounds : List Nat) (bound : Nat) (member : bound ∈ bounds) :
    bound ≤ concatBound bounds := by
  induction bounds generalizing bound with
  | nil => simp at member
  | cons head tail induction =>
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · exact le_max_left _ _
      · exact le_trans (induction _ member) (le_max_right _ _)

theorem selectBound_contains (bounds : List Nat) (bound : Nat) (member : bound ∈ bounds) :
    bound ≤ selectBound bounds := concatBound_contains bounds bound member

@[simp] theorem sliceBound_eq (bound : Nat) : sliceBound bound = bound := rfl
@[simp] theorem reshapeBound_eq (bound : Nat) : reshapeBound bound = bound := rfl

theorem centeredModulusCap_le_bound (modulus bound : Nat) :
    centeredModulusCap modulus bound ≤ bound := min_le_left _ _

theorem centeredModulusCap_le_radius (modulus bound : Nat) :
    centeredModulusCap modulus bound ≤ modulus / 2 := min_le_right _ _

end Mxx.Toolkit
