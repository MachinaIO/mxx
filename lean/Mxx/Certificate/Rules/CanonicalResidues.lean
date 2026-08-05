import Mathlib.Data.List.GetD
import Mxx.Ir.ExecutionFacts
import Mxx.Toolkit.Norms
import Mxx.Certificate.CanonicalResidues

namespace Mxx.Certificate

theorem reduceCoefficient_canonical
    {modulus value : Int} (positive : 0 < modulus) :
    CanonicalResidue modulus (Mxx.reduceCoefficient modulus value) := by
  rw [Mxx.reduceCoefficient, if_neg (not_le.mpr positive)]
  exact ⟨Int.emod_nonneg _ positive.ne', Int.emod_lt_of_pos _ positive⟩

theorem reducedList_canonical
    {modulus : Int} (positive : 0 < modulus) (values : List Int) :
    ∀ value ∈ values.map (Mxx.reduceCoefficient modulus), CanonicalResidue modulus value := by
  intro value member
  obtain ⟨source, _, rfl⟩ := List.mem_map.mp member
  exact reduceCoefficient_canonical positive

theorem matrixAdd_hasCanonicalResidues
    {modulus : Int} (positive : 0 < modulus) (left right : Mxx.Matrix)
    (leftModulus : left.modulus = modulus) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixAdd left right) := by
  constructor
  · simpa [Mxx.matrixAdd] using leftModulus
  · simpa [Mxx.matrixAdd, leftModulus] using
      reducedList_canonical positive (Mxx.addCoefficients left.coefficients right.coefficients)

theorem matrixSubtract_hasCanonicalResidues
    {modulus : Int} (positive : 0 < modulus) (left right : Mxx.Matrix)
    (leftModulus : left.modulus = modulus) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixSubtract left right) := by
  constructor
  · simpa [Mxx.matrixSubtract] using leftModulus
  · simpa [Mxx.matrixSubtract, leftModulus] using
      reducedList_canonical positive (Mxx.subtractCoefficients left.coefficients right.coefficients)

theorem matrixNegate_hasCanonicalResidues
    {modulus : Int} (positive : 0 < modulus) (matrix : Mxx.Matrix)
    (matrixModulus : matrix.modulus = modulus) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixNegate matrix) := by
  constructor
  · simpa [Mxx.matrixNegate] using matrixModulus
  · simpa [Mxx.matrixNegate, matrixModulus, List.map_map, Function.comp_def] using
      reducedList_canonical positive (matrix.coefficients.map (-·))

theorem matrixScale_hasCanonicalResidues
    {modulus scalar : Int} (positive : 0 < modulus) (matrix : Mxx.Matrix)
    (matrixModulus : matrix.modulus = modulus) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixScale scalar matrix) := by
  constructor
  · simpa [Mxx.matrixScale] using matrixModulus
  · simpa [Mxx.matrixScale, matrixModulus, List.map_map, Function.comp_def] using
      reducedList_canonical positive (matrix.coefficients.map (scalar * ·))

theorem matrixMul_hasCanonicalResidues
    {modulus : Int} (positive : 0 < modulus) (left right : Mxx.Matrix)
    (leftModulus : left.modulus = modulus)
    (compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixMul left right) := by
  constructor
  · rw [Mxx.matrixMul, if_pos compatible]
    exact leftModulus
  · intro value member
    simp only [Mxx.matrixMul, if_pos compatible] at member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    simpa [leftModulus] using reduceCoefficient_canonical positive (value :=
      (List.range left.columns).foldl (fun total inner =>
        total + Mxx.negacyclicCoefficient left.ringDimension
          (left.coefficient
            (index.val / left.ringDimension / right.columns) inner)
          (right.coefficient inner
            (index.val / left.ringDimension % right.columns))
          (index.val % left.ringDimension)) 0)

theorem matrixPolynomialScale_hasCanonicalResidues
    {modulus : Int} (positive : 0 < modulus) (scalar matrix : Mxx.Matrix)
    (matrixModulus : matrix.modulus = modulus) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixPolynomialScale scalar matrix) := by
  constructor
  · simpa [Mxx.matrixPolynomialScale] using matrixModulus
  · intro value member
    simp only [Mxx.matrixPolynomialScale] at member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    simpa [matrixModulus] using reduceCoefficient_canonical positive (value :=
      Mxx.negacyclicCoefficient matrix.ringDimension
        (scalar.coefficient 0 0)
        (matrix.coefficient
          (index.val / matrix.ringDimension / matrix.columns)
          (index.val / matrix.ringDimension % matrix.columns))
        (index.val % matrix.ringDimension))

private theorem matrixMul_coefficients_canonical_of_outputModulus
    {modulus : Int} (positive : 0 < modulus) (left right : Mxx.Matrix)
    (outputModulus : (Mxx.matrixMul left right).modulus = modulus) :
    ∀ value ∈ (Mxx.matrixMul left right).coefficients, CanonicalResidue modulus value := by
  intro value member
  unfold Mxx.matrixMul at outputModulus member
  by_cases compatible : left.modulus = right.modulus ∧
      left.ringDimension = right.ringDimension ∧ left.columns = right.rows
  · rw [if_pos compatible] at outputModulus member
    obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
    have leftModulus : left.modulus = modulus := by simpa using outputModulus
    simpa [leftModulus] using reduceCoefficient_canonical positive (value :=
      (List.range left.columns).foldl (fun total inner =>
        total + Mxx.negacyclicCoefficient left.ringDimension
          (left.coefficient
            (index.val / left.ringDimension / right.columns) inner)
          (right.coefficient inner
            (index.val / left.ringDimension % right.columns))
          (index.val % left.ringDimension)) 0)
  · rw [if_neg compatible] at member
    simp at member

private theorem matrixPolynomialScale_coefficients_canonical_of_outputModulus
    {modulus : Int} (positive : 0 < modulus) (scalar matrix : Mxx.Matrix)
    (outputModulus : (Mxx.matrixPolynomialScale scalar matrix).modulus = modulus) :
    ∀ value ∈ (Mxx.matrixPolynomialScale scalar matrix).coefficients,
      CanonicalResidue modulus value := by
  intro value member
  have matrixModulus : matrix.modulus = modulus := by
    simpa [Mxx.matrixPolynomialScale] using outputModulus
  simp only [Mxx.matrixPolynomialScale] at member
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
  simpa [matrixModulus] using reduceCoefficient_canonical positive (value :=
    Mxx.negacyclicCoefficient matrix.ringDimension
      (scalar.coefficient 0 0)
      (matrix.coefficient
        (index.val / matrix.ringDimension / matrix.columns)
        (index.val / matrix.ringDimension % matrix.columns))
      (index.val % matrix.ringDimension))

/-- Every successful branch of the real scalar-broadcast dispatcher stores reduced residues. The
output-modulus premise is the exact condition excluding its invalid empty-matrix fallback. -/
theorem matrixMultiply_hasCanonicalResidues
    {modulus : Int} (positive : 0 < modulus) (left right : Mxx.Matrix)
    (outputModulus : (Mxx.matrixMultiply left right).modulus = modulus) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixMultiply left right) := by
  constructor
  · exact outputModulus
  · intro value member
    unfold Mxx.matrixMultiply at outputModulus member
    by_cases leftScalar : left.rows = 1 ∧ left.columns = 1
    · rw [if_pos leftScalar] at outputModulus member
      by_cases rightRow : right.rows = 1
      · rw [if_pos rightRow] at outputModulus member
        exact matrixMul_coefficients_canonical_of_outputModulus
          positive left right outputModulus value member
      · rw [if_neg rightRow] at outputModulus member
        exact matrixPolynomialScale_coefficients_canonical_of_outputModulus
          positive left right outputModulus value member
    · rw [if_neg leftScalar] at outputModulus member
      by_cases rightScalar : right.rows = 1 ∧ right.columns = 1
      · rw [if_pos rightScalar] at outputModulus member
        by_cases leftRow : left.rows = 1
        · rw [if_pos leftRow] at outputModulus member
          exact matrixMul_coefficients_canonical_of_outputModulus
            positive right left outputModulus value member
        · rw [if_neg leftRow] at outputModulus member
          exact matrixPolynomialScale_coefficients_canonical_of_outputModulus
            positive right left outputModulus value member
      · rw [if_neg rightScalar] at outputModulus member
        exact matrixMul_coefficients_canonical_of_outputModulus
          positive left right outputModulus value member

theorem MatrixHasCanonicalResidues.getD
    {modulus : Int} {matrix : Mxx.Matrix}
    (positive : 0 < modulus)
    (canonical : MatrixHasCanonicalResidues modulus matrix)
    (index : Nat) :
    CanonicalResidue modulus (matrix.coefficients.getD index 0) := by
  by_cases inRange : index < matrix.coefficients.length
  · rw [List.getD_eq_getElem matrix.coefficients 0 inRange]
    exact canonical.2 _ (List.getElem_mem inRange)
  · rw [List.getD_eq_default matrix.coefficients 0 (Nat.le_of_not_gt inRange)]
    exact ⟨le_rfl, positive⟩

/-- Canonical provenance yields the raw interval `[0,q-1]`; this statement intentionally does not
use the centered norm. -/
theorem MatrixHasCanonicalResidues.getD_rawInterval
    {modulus : Int} {matrix : Mxx.Matrix}
    (positive : 0 < modulus)
    (canonical : MatrixHasCanonicalResidues modulus matrix)
    (index : Nat) :
    0 ≤ matrix.coefficients.getD index 0 ∧
      matrix.coefficients.getD index 0 ≤ modulus - 1 := by
  rcases canonical.getD positive index with ⟨nonnegative, less⟩
  exact ⟨nonnegative, by omega⟩

/-- Centered control is separate from the raw interval and follows from the matrix norm itself. -/
theorem getD_centeredBound
    (matrix : Mxx.Matrix) (index bound : Nat)
    (matrixBound : Mxx.maxCenteredCoefficientNorm matrix ≤ bound) :
    (Mxx.centeredCoefficient matrix.modulus (matrix.coefficients.getD index 0)).natAbs ≤ bound :=
  le_trans (Mxx.Toolkit.centeredGetD_natAbs_le_norm matrix index) matrixBound

theorem MatrixHasCanonicalResidues.coefficient
    {modulus : Int} {matrix : Mxx.Matrix}
    (positive : 0 < modulus)
    (canonical : MatrixHasCanonicalResidues modulus matrix)
    (row column coefficient : Nat) :
    CanonicalResidue modulus (matrix.coefficient row column coefficient) := by
  unfold Mxx.Matrix.coefficient
  exact canonical.getD positive _

theorem matrixReshape_preservesCanonicalResidues
    {modulus : Int} {matrix : Mxx.Matrix}
    (canonical : MatrixHasCanonicalResidues modulus matrix) (rows columns : Nat) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixReshape matrix rows columns) := by
  simpa [Mxx.matrixReshape, MatrixHasCanonicalResidues] using canonical

theorem matrixSlice_preservesCanonicalResidues
    {modulus : Int} {matrix : Mxx.Matrix}
    (positive : 0 < modulus)
    (canonical : MatrixHasCanonicalResidues modulus matrix)
    (rowStart rowEnd columnStart columnEnd : Nat) :
    MatrixHasCanonicalResidues modulus
      (Mxx.matrixSlice matrix rowStart rowEnd columnStart columnEnd) := by
  constructor
  · simpa [Mxx.matrixSlice] using canonical.1
  · intro value member
    simp only [Mxx.matrixSlice] at member
    obtain ⟨row, _, rowMember⟩ := List.mem_flatMap.mp member
    obtain ⟨column, _, columnMember⟩ := List.mem_flatMap.mp rowMember
    obtain ⟨coefficient, _, rfl⟩ := List.mem_map.mp columnMember
    exact canonical.coefficient positive (rowStart + row) (columnStart + column) coefficient

theorem matrixConcatRows_two_preservesCanonicalResidues
    {modulus : Int} {top bottom : Mxx.Matrix}
    (topCanonical : MatrixHasCanonicalResidues modulus top)
    (bottomCanonical : MatrixHasCanonicalResidues modulus bottom) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixConcatRows [top, bottom]) := by
  constructor
  · simpa [Mxx.matrixConcatRows] using topCanonical.1
  · intro value member
    simp only [Mxx.matrixConcatRows, List.flatMap_cons, List.flatMap_nil,
      List.append_nil] at member
    rcases List.mem_append.mp member with member | member
    · exact topCanonical.2 value member
    · exact bottomCanonical.2 value member

theorem matrixConcatColumns_two_preservesCanonicalResidues
    {modulus : Int} {left right : Mxx.Matrix}
    (positive : 0 < modulus)
    (leftCanonical : MatrixHasCanonicalResidues modulus left)
    (rightCanonical : MatrixHasCanonicalResidues modulus right) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixConcatColumns [left, right]) := by
  constructor
  · simpa [Mxx.matrixConcatColumns] using leftCanonical.1
  · intro value member
    simp only [Mxx.matrixConcatColumns] at member
    obtain ⟨row, _, rowMember⟩ := List.mem_flatMap.mp member
    obtain ⟨matrix, matrixMember, matrixCoefficientMember⟩ := List.mem_flatMap.mp rowMember
    obtain ⟨column, _, columnMember⟩ := List.mem_flatMap.mp matrixCoefficientMember
    obtain ⟨coefficient, _, rfl⟩ := List.mem_map.mp columnMember
    simp only [List.mem_cons, List.not_mem_nil, or_false] at matrixMember
    rcases matrixMember with rfl | rfl
    · exact leftCanonical.coefficient positive row column coefficient
    · exact rightCanonical.coefficient positive row column coefficient

theorem diagonalCoefficient_two_canonical
    {modulus : Int} {top bottom : Mxx.Matrix}
    (positive : 0 < modulus)
    (topCanonical : MatrixHasCanonicalResidues modulus top)
    (bottomCanonical : MatrixHasCanonicalResidues modulus bottom)
    (row column coefficient rowOffset columnOffset : Nat) :
    CanonicalResidue modulus
      (Mxx.diagonalCoefficient [top, bottom] row column coefficient rowOffset columnOffset) := by
  simp only [Mxx.diagonalCoefficient]
  split
  · exact topCanonical.coefficient positive (row - rowOffset) (column - columnOffset) coefficient
  · split
    · exact bottomCanonical.coefficient positive
        (row - (rowOffset + top.rows)) (column - (columnOffset + top.columns)) coefficient
    · exact ⟨le_rfl, positive⟩

theorem matrixConcatDiagonal_two_preservesCanonicalResidues
    {modulus : Int} {top bottom : Mxx.Matrix}
    (positive : 0 < modulus)
    (topCanonical : MatrixHasCanonicalResidues modulus top)
    (bottomCanonical : MatrixHasCanonicalResidues modulus bottom) :
    MatrixHasCanonicalResidues modulus (Mxx.matrixConcatDiagonal [top, bottom]) := by
  constructor
  · simpa [Mxx.matrixConcatDiagonal] using topCanonical.1
  · intro value member
    simp only [Mxx.matrixConcatDiagonal] at member
    obtain ⟨row, _, rowMember⟩ := List.mem_flatMap.mp member
    obtain ⟨column, _, columnMember⟩ := List.mem_flatMap.mp rowMember
    obtain ⟨coefficient, _, rfl⟩ := List.mem_map.mp columnMember
    exact diagonalCoefficient_two_canonical positive topCanonical bottomCanonical
      row column coefficient 0 0

/-- Coefficient extraction observes the canonical residue, matching the Rust runtime backend. -/
theorem extractCoefficientNode_returnsCanonicalResidue
    (runChild : Mxx.Ir.ChildRunner)
    (samplers : Mxx.MxxSamplerFamily)
    (params : Mxx.Ir.ParamEnvironment)
    (inputs : Mxx.Ir.Environment)
    (wires : Mxx.Ir.WireEnvironment)
    (inputRef : Mxx.Ir.WireRef)
    (position : Mxx.Ir.IntExpr)
    (matrix : Mxx.Matrix)
    (evaluatedPosition : Int)
    (outputCount : Nat)
    (argumentsEvaluate : [inputRef].mapM (fun wire => Mxx.Ir.lookupWire wire wires) =
      some [.matrix matrix])
    (positionEvaluate : position.evaluate params = some evaluatedPosition)
    {values : List Mxx.Ir.Value}
    (member : values ∈ Mxx.Ir.evaluateNode runChild samplers params inputs wires {
      kind := .extractCoefficient position
      arguments := [inputRef]
      outputCount
    }) :
    values = [.integer (Mxx.reduceCoefficient matrix.modulus
      (matrix.coefficients.getD evaluatedPosition.toNat 0))] := by
  simpa [Mxx.Ir.evaluateNode, Mxx.Ir.arguments, argumentsEvaluate, positionEvaluate] using member

example : ¬MatrixHasCanonicalResidues 17 {
    coefficients := [18]
    modulus := 17
    ringDimension := 1
    rows := 1
    columns := 1
  } := by
  simp [MatrixHasCanonicalResidues, CanonicalResidue]

example : MatrixHasCanonicalResidues 17 {
    coefficients := [16]
    modulus := 17
    ringDimension := 1
    rows := 1
    columns := 1
  } := by
  norm_num [MatrixHasCanonicalResidues, CanonicalResidue]

example : Mxx.reduceCoefficient 17 18 = 1 := by
  norm_num [Mxx.reduceCoefficient]

example : ({
    coefficients := [18]
    modulus := 17
    ringDimension := 1
    rows := 1
    columns := 1
  } : Mxx.Matrix).coefficients.getD 0 0 = 18 := rfl

example : ({
    coefficients := [16]
    modulus := 17
    ringDimension := 1
    rows := 1
    columns := 1
  } : Mxx.Matrix).coefficients.getD 0 0 = 16 := rfl

end Mxx.Certificate
