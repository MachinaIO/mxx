import MxxWe.Certificate.ExecutionBridge
import Mxx.Toolkit.Norms

namespace MxxWe.Certificate

/-! Exact structural forms produced by the checked Diamond transition-selector scan. -/

/-- Canonical zero matrix used by the fixed selector body. -/
def transitionSelectorZeroMatrix (q ringDimension rows columns : Nat) : Mxx.Matrix :=
  Mxx.Matrix.withSamplerParams
    { coefficients := List.replicate (rows * columns * ringDimension) 0 }
    { modulus := q, ringDimension, rows, columns, maxCoefficientBound := 0 }

/-- The bit branch in the fixed child body can only produce canonical polynomial zero or one. -/
inductive TransitionSelectorBitMatrix (q ringDimension : Nat) : Mxx.Matrix → Prop
  | zero : TransitionSelectorBitMatrix q ringDimension
      (transitionSelectorZeroMatrix q ringDimension 1 1)
  | one : TransitionSelectorBitMatrix q ringDimension
      (Mxx.Toolkit.unitPolynomialMatrix q ringDimension)

/-- The selector scan starts with one of the two diagonal forms and can replace it by the special
two-row form.  The second top-row entry is retained exactly because it is the runtime product of
the ternary secret and a selected zero/identity polynomial. -/
inductive TransitionSelectorForm (q ringDimension : Nat) (secret : Mxx.Matrix) :
    Mxx.Matrix → Prop
  | regular : TransitionSelectorForm q ringDimension secret
      (Mxx.matrixConcatDiagonal [secret, secret])
  | distinguished : TransitionSelectorForm q ringDimension secret
      (Mxx.matrixConcatDiagonal
        [secret, Mxx.Toolkit.unitPolynomialMatrix q ringDimension])
  | special (second : Mxx.Matrix)
      (secondLayout : Mxx.Toolkit.MatrixLayout second q ringDimension 1 1)
      (secondNorm : Mxx.maxCenteredCoefficientNorm second ≤ 1) :
      TransitionSelectorForm q ringDimension secret
        (Mxx.matrixConcatRows [
          Mxx.matrixConcatColumns [secret, second],
          transitionSelectorZeroMatrix q ringDimension 1 2])

private theorem transitionSelectorZeroMatrix_norm_le_one
    (q ringDimension rows columns : Nat) [NeZero q] :
    Mxx.maxCenteredCoefficientNorm
        (transitionSelectorZeroMatrix q ringDimension rows columns) ≤ 1 := by
  unfold transitionSelectorZeroMatrix Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append] at coefficientMember
  rcases coefficientMember with sourceMember | paddingMember
  · have replicatedMember := List.mem_of_mem_take sourceMember
    simp only [List.mem_replicate] at replicatedMember
    rcases replicatedMember with ⟨_, rfl⟩
    change (Mxx.centeredCoefficient q 0).natAbs ≤ 1
    have qNe : q ≠ 0 := NeZero.ne q
    have qPos : (0 : Int) < q := by exact_mod_cast NeZero.pos q
    simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, qNe,
      show ¬ (q : Int) < 0 by omega]
  · simp only [List.mem_replicate] at paddingMember
    rcases paddingMember with ⟨_, rfl⟩
    change (Mxx.centeredCoefficient q 0).natAbs ≤ 1
    have qNe : q ≠ 0 := NeZero.ne q
    have qPos : (0 : Int) < q := by exact_mod_cast NeZero.pos q
    simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, qNe,
      show ¬ (q : Int) < 0 by omega]

private theorem transitionSelectorZeroMatrix_norm_zero
    (q ringDimension rows columns : Nat) [NeZero q] :
    Mxx.maxCenteredCoefficientNorm
        (transitionSelectorZeroMatrix q ringDimension rows columns) ≤ 0 := by
  unfold transitionSelectorZeroMatrix Mxx.maxCenteredCoefficientNorm
  apply Mxx.Toolkit.coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  simp only [Mxx.Matrix.withSamplerParams, List.mem_append] at coefficientMember
  rcases coefficientMember with sourceMember | paddingMember
  · have replicatedMember := List.mem_of_mem_take sourceMember
    simp only [List.mem_replicate] at replicatedMember
    rcases replicatedMember with ⟨_, rfl⟩
    change (Mxx.centeredCoefficient q 0).natAbs ≤ 0
    have qNe : q ≠ 0 := NeZero.ne q
    have qPos : (0 : Int) < q := by exact_mod_cast NeZero.pos q
    simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, qNe,
      show ¬ (q : Int) < 0 by omega]
  · simp only [List.mem_replicate] at paddingMember
    rcases paddingMember with ⟨_, rfl⟩
    change (Mxx.centeredCoefficient q 0).natAbs ≤ 0
    have qNe : q ≠ 0 := NeZero.ne q
    have qPos : (0 : Int) < q := by exact_mod_cast NeZero.pos q
    simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, qNe,
      show ¬ (q : Int) < 0 by omega]

private theorem scalarMatrixMultiply_layout
    {q ringDimension : Nat} {left right : Mxx.Matrix}
    (leftLayout : Mxx.Toolkit.MatrixLayout left q ringDimension 1 1)
    (rightLayout : Mxx.Toolkit.MatrixLayout right q ringDimension 1 1) :
    Mxx.Toolkit.MatrixLayout (Mxx.matrixMultiply left right) q ringDimension 1 1 := by
  have compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows := by
    simp [leftLayout.modulus, rightLayout.modulus, leftLayout.ringDimension,
      rightLayout.ringDimension, leftLayout.columns, rightLayout.rows]
  have runtimeEq : Mxx.matrixMultiply left right = Mxx.matrixMul left right := by
    simp [Mxx.matrixMultiply, leftLayout.rows, leftLayout.columns, rightLayout.rows]
  rw [runtimeEq]
  refine {
    toMatrixShape := Mxx.Toolkit.matrixMul_shape left right leftLayout.toMatrixShape
      rightLayout.toMatrixShape
    coefficientCount := ?_
  }
  unfold Mxx.matrixMul
  rw [if_pos compatible]
  simp [leftLayout.rows, rightLayout.columns, leftLayout.ringDimension]

/-- Multiplying the ternary secret by the selected canonical bit keeps a complete scalar layout
and coefficient norm at most one. -/
theorem transitionSelectorBitProduct_layout_and_norm
    {q ringDimension : Nat} [NeZero q] [NeZero ringDimension]
    {secret bit : Mxx.Matrix}
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension 1 1)
    (secretNorm : Mxx.maxCenteredCoefficientNorm secret ≤ 1)
    (bitForm : TransitionSelectorBitMatrix q ringDimension bit) :
    Mxx.Toolkit.MatrixLayout (Mxx.matrixMultiply secret bit) q ringDimension 1 1 ∧
      Mxx.maxCenteredCoefficientNorm (Mxx.matrixMultiply secret bit) ≤ 1 := by
  cases bitForm with
  | zero =>
      have zeroLayout : Mxx.Toolkit.MatrixLayout
          (transitionSelectorZeroMatrix q ringDimension 1 1) q ringDimension 1 1 := by
        unfold transitionSelectorZeroMatrix
        exact Mxx.Toolkit.withSamplerParams_layout _ _
      constructor
      · exact scalarMatrixMultiply_layout secretLayout zeroLayout
      · have runtimeEq : Mxx.matrixMultiply secret
            (transitionSelectorZeroMatrix q ringDimension 1 1) =
            Mxx.matrixMul secret (transitionSelectorZeroMatrix q ringDimension 1 1) := by
          simp [Mxx.matrixMultiply, secretLayout.rows, secretLayout.columns, zeroLayout.rows]
        rw [runtimeEq]
        exact le_trans
          (Mxx.Toolkit.matrixMul_norm_le q ringDimension 1 1 0 secret
            (transitionSelectorZeroMatrix q ringDimension 1 1) secretLayout.modulus
            zeroLayout.modulus secretLayout.ringDimension zeroLayout.ringDimension
            secretLayout.columns zeroLayout.rows secretNorm
            (transitionSelectorZeroMatrix_norm_zero q ringDimension 1 1))
          (by simp)
  | one =>
      have unitLayout : Mxx.Toolkit.MatrixLayout
          (Mxx.Toolkit.unitPolynomialMatrix q ringDimension) q ringDimension 1 1 := by
        unfold Mxx.Toolkit.unitPolynomialMatrix
        exact Mxx.Toolkit.withSamplerParams_layout _ _
      constructor
      · exact scalarMatrixMultiply_layout secretLayout unitLayout
      · exact Mxx.Toolkit.matrixMultiply_unitPolynomial_norm_le q ringDimension 1 secret
          secretLayout.modulus secretLayout.ringDimension secretLayout.rows secretLayout.columns
          secretNorm

private theorem transitionSelectorDiagonal_norm_le
    (q ringDimension : Nat) (top bottom : Mxx.Matrix)
    (topLayout : Mxx.Toolkit.MatrixLayout top q ringDimension 1 1)
    (bottomLayout : Mxx.Toolkit.MatrixLayout bottom q ringDimension 1 1) :
    Mxx.maxCenteredCoefficientNorm (Mxx.matrixConcatDiagonal [top, bottom]) ≤
      max (Mxx.maxCenteredCoefficientNorm top)
        (Mxx.maxCenteredCoefficientNorm bottom) := by
  change Mxx.coefficientNorm
      ((Mxx.matrixConcatDiagonal [top, bottom]).coefficients.map
        (Mxx.centeredCoefficient (Mxx.matrixConcatDiagonal [top, bottom]).modulus)) ≤ _
  unfold Mxx.matrixConcatDiagonal
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero]
  rw [topLayout.modulus]
  apply Mxx.Toolkit.coefficientNorm_le
  intro centered centeredMember
  obtain ⟨coefficientValue, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
  obtain ⟨row, rowMember, rowCoefficients⟩ := List.mem_flatMap.mp coefficientMember
  obtain ⟨column, columnMember, columnCoefficients⟩ := List.mem_flatMap.mp rowCoefficients
  obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp columnCoefficients
  have rowLt : row < top.rows + bottom.rows := by simpa using List.mem_range.mp rowMember
  have columnLt : column < top.columns + bottom.columns := by
    simpa using List.mem_range.mp columnMember
  have topRows : top.rows = 1 := topLayout.rows
  have bottomRows : bottom.rows = 1 := bottomLayout.rows
  have topColumns : top.columns = 1 := topLayout.columns
  have bottomColumns : bottom.columns = 1 := bottomLayout.columns
  rw [topRows, bottomRows] at rowLt
  rw [topColumns, bottomColumns] at columnLt
  unfold Mxx.diagonalCoefficient
  split_ifs with inTop
  · exact le_trans
      (by simpa [topLayout.modulus] using
        Mxx.Toolkit.centeredEntry_natAbs_le_norm top row column coefficient)
      (le_max_left _ _)
  · simp only [Mxx.diagonalCoefficient]
    split_ifs with inBottom
    · have rowInBottom : row - top.rows < bottom.rows := by omega
      have columnInBottom : column - top.columns < bottom.columns := by omega
      exact le_trans
        (by simpa [bottomLayout.modulus] using
          (Mxx.Toolkit.centeredEntry_natAbs_le_norm bottom (row - top.rows)
            (column - top.columns) coefficient))
        (le_max_right _ _)
    · have centeredZero : Mxx.centeredCoefficient q 0 = 0 := by
        simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient]
      rw [centeredZero]
      exact Nat.zero_le _

/-- Every structural selector form has the exact runtime `2 × 2` shape. -/
theorem TransitionSelectorForm.shape
    {q ringDimension : Nat} {secret selector : Mxx.Matrix}
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension 1 1)
    (form : TransitionSelectorForm q ringDimension secret selector) :
    Mxx.Toolkit.MatrixShape selector q ringDimension 2 2 := by
  have unitLayout : Mxx.Toolkit.MatrixLayout
      (Mxx.Toolkit.unitPolynomialMatrix q ringDimension) q ringDimension 1 1 := by
    unfold Mxx.Toolkit.unitPolynomialMatrix
    exact Mxx.Toolkit.withSamplerParams_layout _ _
  cases form with
  | regular =>
      exact (Mxx.Toolkit.matrixConcatDiagonal_two_layout secret secret secretLayout
        secretLayout).toMatrixShape
  | distinguished =>
      exact (Mxx.Toolkit.matrixConcatDiagonal_two_layout secret
        (Mxx.Toolkit.unitPolynomialMatrix q ringDimension) secretLayout unitLayout).toMatrixShape
  | special second secondLayout secondNorm =>
      have topLayout := Mxx.Toolkit.matrixConcatColumns_two_layout secret second secretLayout
        secondLayout
      have bottomLayout : Mxx.Toolkit.MatrixLayout
          (transitionSelectorZeroMatrix q ringDimension 1 2) q ringDimension 1 2 := by
        unfold transitionSelectorZeroMatrix
        exact Mxx.Toolkit.withSamplerParams_layout _ _
      exact (Mxx.Toolkit.matrixConcatRows_two_layout
        (Mxx.matrixConcatColumns [secret, second])
        (transitionSelectorZeroMatrix q ringDimension 1 2) topLayout bottomLayout).toMatrixShape

/-- Every structural selector form has deterministic centered coefficient norm at most one. -/
theorem TransitionSelectorForm.norm_le_one
    {q ringDimension : Nat} [NeZero q]
    {secret selector : Mxx.Matrix}
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension 1 1)
    (secretNorm : Mxx.maxCenteredCoefficientNorm secret ≤ 1)
    (form : TransitionSelectorForm q ringDimension secret selector) :
    Mxx.maxCenteredCoefficientNorm selector ≤ 1 := by
  have unitNorm : Mxx.maxCenteredCoefficientNorm
      (Mxx.Toolkit.unitPolynomialMatrix q ringDimension) ≤ 1 := by
    unfold Mxx.Toolkit.unitPolynomialMatrix Mxx.maxCenteredCoefficientNorm
    apply Mxx.Toolkit.coefficientNorm_le
    intro centered centeredMember
    obtain ⟨coefficient, coefficientMember, rfl⟩ := List.mem_map.mp centeredMember
    simp only [Mxx.Matrix.withSamplerParams, List.mem_append] at coefficientMember
    rcases coefficientMember with sourceMember | paddingMember
    · have originalMember := List.mem_of_mem_take sourceMember
      obtain ⟨index, _, rfl⟩ := List.mem_map.mp originalMember
      change (Mxx.centeredCoefficient q (if index = 0 then 1 else 0)).natAbs ≤ 1
      split
      · exact le_trans (Mxx.Toolkit.centeredCoefficient_natAbs_le q 1) (by decide)
      · exact le_trans (Mxx.Toolkit.centeredCoefficient_natAbs_le q 0) (by decide)
    · simp only [List.mem_replicate] at paddingMember
      rcases paddingMember with ⟨_, rfl⟩
      change (Mxx.centeredCoefficient q 0).natAbs ≤ 1
      have qNe : q ≠ 0 := NeZero.ne q
      have qPos : (0 : Int) < q := by exact_mod_cast NeZero.pos q
      simp [Mxx.centeredCoefficient, Mxx.reduceCoefficient, qNe,
        show ¬ (q : Int) < 0 by omega]
  cases form with
  | regular =>
      exact le_trans
        (transitionSelectorDiagonal_norm_le q ringDimension secret secret
          secretLayout secretLayout)
        (by simpa using secretNorm)
  | distinguished =>
      exact le_trans
        (transitionSelectorDiagonal_norm_le q ringDimension secret
          (Mxx.Toolkit.unitPolynomialMatrix q ringDimension)
          secretLayout
          (by unfold Mxx.Toolkit.unitPolynomialMatrix
              exact Mxx.Toolkit.withSamplerParams_layout _ _))
        (max_le secretNorm unitNorm)
  | special second secondLayout secondNorm =>
      have topNorm := Mxx.Toolkit.matrixConcatColumns_two_norm_le q secret second
        secretLayout.modulus secondLayout.modulus
      have topModulus : (Mxx.matrixConcatColumns [secret, second]).modulus = q := by
        simp [Mxx.matrixConcatColumns, secretLayout.modulus]
      have bottomModulus :
          (transitionSelectorZeroMatrix q ringDimension 1 2).modulus = q := by
        simp [transitionSelectorZeroMatrix, Mxx.Matrix.withSamplerParams]
      exact le_trans
        (Mxx.Toolkit.matrixConcatRows_two_norm_le q
          (Mxx.matrixConcatColumns [secret, second])
          (transitionSelectorZeroMatrix q ringDimension 1 2) topModulus bottomModulus)
        (max_le (le_trans topNorm (max_le secretNorm secondNorm))
          (transitionSelectorZeroMatrix_norm_le_one q ringDimension 1 2))

/-- One semantic bit-scan step preserves the selector's exact structural form.  The zero and
identity alternatives are obtained from the executed matrix constructors, not from a caller
supplied coefficient bound. -/
theorem transitionSelectorStepValue_form
    {q ringDimension : Nat} [NeZero q] [NeZero ringDimension]
    {unitParams bottomParams : Mxx.SamplerParams}
    {carried secret : Mxx.Matrix} {stateIndex specialIndex digit bitIndex : Int}
    (unitParamsEq : unitParams = {
      modulus := q, ringDimension, rows := 1, columns := 1, maxCoefficientBound := 0 })
    (bottomParamsEq : bottomParams = {
      modulus := q, ringDimension, rows := 1, columns := 2, maxCoefficientBound := 0 })
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension 1 1)
    (secretNorm : Mxx.maxCenteredCoefficientNorm secret ≤ 1)
    (carriedForm : TransitionSelectorForm q ringDimension secret carried) :
    TransitionSelectorForm q ringDimension secret
      (transitionSelectorStepValue unitParams bottomParams carried secret stateIndex
        specialIndex digit bitIndex) := by
  subst unitParams
  subst bottomParams
  by_cases stateMatches : stateIndex = specialIndex + bitIndex
  · simp only [transitionSelectorStepValue, stateMatches, if_pos]
    let bitMatrix := if ((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0 then
      transitionSelectorIdentityValue
        { modulus := q, ringDimension, rows := 1, columns := 1,
          maxCoefficientBound := 0 }
      else transitionSelectorZeroValue
        { modulus := q, ringDimension, rows := 1, columns := 1,
          maxCoefficientBound := 0 }
    have bitForm : TransitionSelectorBitMatrix q ringDimension bitMatrix := by
      by_cases bitSet : ((digit / (2 ^ bitIndex.toNat)) % 2) ≠ 0
      · rw [show bitMatrix = transitionSelectorIdentityValue
            { modulus := q, ringDimension, rows := 1, columns := 1,
              maxCoefficientBound := 0 } by simp [bitMatrix, bitSet]]
        simpa [transitionSelectorIdentityValue, Mxx.Toolkit.unitPolynomialMatrix,
          Mxx.Matrix.withSamplerParams] using
          (TransitionSelectorBitMatrix.one (q := q) (ringDimension := ringDimension))
      · rw [show bitMatrix = transitionSelectorZeroValue
            { modulus := q, ringDimension, rows := 1, columns := 1,
              maxCoefficientBound := 0 } by
            have remainderZero : (digit / (2 ^ bitIndex.toNat)) % 2 = 0 :=
              not_ne_iff.mp bitSet
            simp [bitMatrix, remainderZero]]
        simpa [transitionSelectorZeroValue, transitionSelectorZeroMatrix] using
          (TransitionSelectorBitMatrix.zero (q := q) (ringDimension := ringDimension))
    obtain ⟨secondLayout, secondNorm⟩ :=
      transitionSelectorBitProduct_layout_and_norm secretLayout secretNorm bitForm
    apply TransitionSelectorForm.special (second := Mxx.matrixMultiply secret bitMatrix)
      secondLayout secondNorm
  · simp [transitionSelectorStepValue, stateMatches]
    exact carriedForm

/-- One selected child-support member of the checked bit scan returns exactly one matrix and
preserves the selector form. -/
theorem transitionSelectorBitChildOutcome
    {workflow : Mxx.Ir.Workflow} {reference : TransitionSelectorLayout}
    {stage : Mxx.Ir.Stage} {body : Mxx.Ir.Scope} {fuel index : Nat}
    {samplers : Mxx.MxxSamplerFamily} {params : Mxx.Ir.ParamEnvironment}
    {unitParams bottomParams : Mxx.SamplerParams}
    {values : List Mxx.Ir.Value} {carried secret : Mxx.Matrix}
    {stateIndex specialIndex digit : Int}
    (verified : verifyTransitionSelector workflow reference = true)
    (bodyResolved : resolveScope workflow
      { reference.bitScan.operation with scope := reference.bitScan.bodyScope } = some body)
    (definitionFound : Mxx.Ir.lookupDefinition reference.bitScan.bodyScope.definitionName
      stage.program.definitions = some body)
    (childMember : values ∈ Mxx.Ir.childRunnerWithFuel samplers stage.program (fuel + 1)
      reference.bitScan.bodyScope.definitionName
      ((.loopIndex reference.bitScan.indexSlot, .integer index) :: params)
      [.matrix carried, .integer digit, .integer stateIndex, .integer specialIndex,
        .matrix secret])
    (unitTypeEvaluate :
      ({ modulus := .parameter "diamond_modulus",
         ringDimension := .parameter "diamond_ring_dimension",
         rows := .constant 1, columns := .constant 1 } : Mxx.Ir.MatrixTypeExpr).evaluate
        ((.loopIndex reference.bitScan.indexSlot, .integer index) :: params) = some unitParams)
    (bottomTypeEvaluate :
      ({ modulus := .parameter "diamond_modulus",
         ringDimension := .parameter "diamond_ring_dimension",
         rows := .constant 1, columns := .constant 2 } : Mxx.Ir.MatrixTypeExpr).evaluate
        ((.loopIndex reference.bitScan.indexSlot, .integer index) :: params) = some bottomParams)
    {q ringDimension : Nat} [NeZero q] [NeZero ringDimension]
    (unitParamsEq : unitParams = {
      modulus := q, ringDimension, rows := 1, columns := 1, maxCoefficientBound := 0 })
    (bottomParamsEq : bottomParams = {
      modulus := q, ringDimension, rows := 1, columns := 2, maxCoefficientBound := 0 })
    (secretLayout : Mxx.Toolkit.MatrixLayout secret q ringDimension 1 1)
    (secretNorm : Mxx.maxCenteredCoefficientNorm secret ≤ 1)
    (carriedForm : TransitionSelectorForm q ringDimension secret carried) :
    ∃ next, values = [.matrix next] ∧
      TransitionSelectorForm q ringDimension secret next := by
  have checked := verified
  unfold verifyTransitionSelector at checked
  simp only [Bool.and_eq_true, decide_eq_true_eq] at checked
  have loopChecked : verifySequentialLoopRef workflow reference.bitScan = true := by aesop
  have indexSlotEq : reference.bitScan.indexSlot = 1 := by aesop
  have bodyOutputRef : reference.bitScan.bodyOutputs.map wireRef =
      [({ node := 18, port := 0 } : Mxx.Ir.WireRef)] := by aesop
  have namesChecked : verifyTransitionSelectorBitBodyInputNames workflow reference = true := by aesop
  unfold verifyTransitionSelectorBitBodyInputNames at namesChecked
  rw [bodyResolved] at namesChecked
  simp only [decide_eq_true_eq] at namesChecked
  have bodyOutputWires : scopeOutputWires body = reference.bitScan.bodyOutputs.map wireRef := by
    unfold verifySequentialLoopRef at loopChecked
    rw [bodyResolved] at loopChecked
    simp only [Bool.and_eq_true, decide_eq_true_eq] at loopChecked
    aesop
  obtain ⟨childPath⟩ := childExecutionPath_of_outcome definitionFound childMember
  have outputValue := verifyTransitionSelector_bitPathOutput (bitIndex := Int.ofNat index)
    verified bodyResolved childPath.path
    (by simp [namesChecked, Mxx.Ir.lookupEnvironment])
    (by simp [namesChecked, Mxx.Ir.lookupEnvironment])
    (by simp [namesChecked, Mxx.Ir.lookupEnvironment])
    (by simp [namesChecked, Mxx.Ir.lookupEnvironment])
    (by simp [namesChecked, Mxx.Ir.lookupEnvironment])
    (by simp [indexSlotEq, Mxx.Ir.IntExpr.evaluate, Mxx.Ir.lookupLoopIndex])
    (Int.ofNat_nonneg index) unitTypeEvaluate bottomTypeEvaluate
  let next := transitionSelectorStepValue unitParams bottomParams carried secret stateIndex
    specialIndex digit (Int.ofNat index)
  refine ⟨next, ?_, transitionSelectorStepValue_form unitParamsEq bottomParamsEq
    secretLayout secretNorm carriedForm⟩
  rw [childPath.outputs]
  have bodyOutputExact : scopeOutputWires body =
      [({ node := 18, port := 0 } : Mxx.Ir.WireRef)] :=
    bodyOutputWires.trans bodyOutputRef
  cases outputs : body.outputs with
  | nil => simp [scopeOutputWires, outputs] at bodyOutputExact
  | cons output tail =>
      cases tail with
      | cons nextOutput rest => simp [scopeOutputWires, outputs] at bodyOutputExact
      | nil =>
          rcases output with ⟨name, wire⟩
          simp [scopeOutputWires, outputs] at bodyOutputExact
          subst wire
          simp [Mxx.Ir.collectOutputs, outputs, outputValue, next]

end MxxWe.Certificate
