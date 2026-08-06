import Mathlib.Data.ENNReal.Basic

namespace Mxx

structure Matrix where
  coefficients : List Int
  modulus : Int := 0
  ringDimension : Nat := 0
  rows : Nat := 0
  columns : Nat := 0
  deriving DecidableEq

def Matrix.coefficient (matrix : Matrix) (row column coefficient : Nat) : Int :=
  matrix.coefficients.getD (((row * matrix.columns + column) * matrix.ringDimension) + coefficient) 0

def reduceCoefficient (modulus value : Int) : Int :=
  if modulus ≤ 0 then value else value % modulus

def centeredCoefficient (modulus value : Int) : Int :=
  if modulus ≤ 0 then value
  else
    let residue := reduceCoefficient modulus value
    if 2 * residue > modulus then residue - modulus else residue

def negacyclicCoefficient (ringDimension : Nat)
    (left right : Nat → Int) (coefficient : Nat) : Int :=
  (List.range ringDimension).foldl (fun total leftCoefficient =>
    if leftCoefficient ≤ coefficient then
      total + left leftCoefficient * right (coefficient - leftCoefficient)
    else
      total - left leftCoefficient * right (ringDimension + coefficient - leftCoefficient)) 0

def matrixMul (left right : Matrix) : Matrix :=
  if left.modulus = right.modulus ∧ left.ringDimension = right.ringDimension ∧
      left.columns = right.rows then
    { coefficients :=
        List.ofFn fun linear : Fin (left.rows * right.columns * left.ringDimension) =>
          let coefficient := linear.val % left.ringDimension
          let entry := linear.val / left.ringDimension
          let column := entry % right.columns
          let row := entry / right.columns
          reduceCoefficient left.modulus <|
            (List.range left.columns).foldl (fun total inner =>
              total + negacyclicCoefficient left.ringDimension
                (left.coefficient row inner)
                (right.coefficient inner column)
                coefficient) 0
      modulus := left.modulus
      ringDimension := left.ringDimension
      rows := left.rows
      columns := right.columns }
  else
    { coefficients := [] }

/-- Entrywise multiplication by a polynomial scalar. This is the executable meaning of matrix
multiplication when either operand has shape `1 × 1`; the Rust backend applies the same scalar
broadcast rule before ordinary matrix multiplication. -/
def matrixPolynomialScale (scalar matrix : Matrix) : Matrix :=
  { coefficients :=
      List.ofFn fun linear : Fin (matrix.rows * matrix.columns * matrix.ringDimension) =>
        let coefficient := linear.val % matrix.ringDimension
        let entry := linear.val / matrix.ringDimension
        let column := entry % matrix.columns
        let row := entry / matrix.columns
        reduceCoefficient matrix.modulus <|
          negacyclicCoefficient matrix.ringDimension
            (scalar.coefficient 0 0) (matrix.coefficient row column) coefficient
    modulus := matrix.modulus
    ringDimension := matrix.ringDimension
    rows := matrix.rows
    columns := matrix.columns }

/-- Runtime matrix multiplication, including the DSL's `1 × 1` polynomial-scalar broadcast. -/
def matrixMultiply (left right : Matrix) : Matrix :=
  if left.rows = 1 ∧ left.columns = 1 then
    if right.rows = 1 then matrixMul left right else matrixPolynomialScale left right
  else if right.rows = 1 ∧ right.columns = 1 then
    if left.rows = 1 then matrixMul right left else matrixPolynomialScale right left
  else matrixMul left right

/-- Typed equality of executable matrices modulo their common coefficient modulus. The shape and
coefficient ring are part of the relation; coefficients are compared only at valid matrix and
polynomial indices. This is the cycle-free sampler-contract form of equality in `R_q`. -/
structure MatrixModEq (left right : Matrix) : Prop where
  modulus : left.modulus = right.modulus
  ringDimension : left.ringDimension = right.ringDimension
  rows : left.rows = right.rows
  columns : left.columns = right.columns
  coefficients : ∀ row column coefficient,
    row < left.rows → column < left.columns → coefficient < left.ringDimension →
    reduceCoefficient left.modulus (left.coefficient row column coefficient) =
      reduceCoefficient right.modulus (right.coefficient row column coefficient)

namespace MatrixModEq

/-- Typed equality modulo the coefficient modulus is reflexive, independently of the stored
coefficient representatives. -/
theorem refl (matrix : Matrix) : MatrixModEq matrix matrix := by
  exact ⟨rfl, rfl, rfl, rfl, fun _ _ _ _ _ _ ↦ rfl⟩

theorem symm {left right : Matrix} (relation : MatrixModEq left right) :
    MatrixModEq right left := by
  refine ⟨relation.modulus.symm, relation.ringDimension.symm, relation.rows.symm,
    relation.columns.symm, ?_⟩
  intro row column coefficient rowLt columnLt coefficientLt
  exact (relation.coefficients row column coefficient
    (relation.rows ▸ rowLt) (relation.columns ▸ columnLt)
    (relation.ringDimension ▸ coefficientLt)).symm

theorem trans {left middle right : Matrix}
    (leftMiddle : MatrixModEq left middle)
    (middleRight : MatrixModEq middle right) : MatrixModEq left right := by
  refine ⟨leftMiddle.modulus.trans middleRight.modulus,
    leftMiddle.ringDimension.trans middleRight.ringDimension,
    leftMiddle.rows.trans middleRight.rows,
    leftMiddle.columns.trans middleRight.columns, ?_⟩
  intro row column coefficient rowLt columnLt coefficientLt
  exact (leftMiddle.coefficients row column coefficient rowLt columnLt coefficientLt).trans <|
    middleRight.coefficients row column coefficient
      (leftMiddle.rows ▸ rowLt) (leftMiddle.columns ▸ columnLt)
      (leftMiddle.ringDimension ▸ coefficientLt)

end MatrixModEq

def addCoefficients : List Int → List Int → List Int
  | [], right => right
  | left, [] => left
  | left :: leftTail, right :: rightTail =>
      (left + right) :: addCoefficients leftTail rightTail

def subtractCoefficients : List Int → List Int → List Int
  | [], right => right.map (-·)
  | left, [] => left
  | left :: leftTail, right :: rightTail =>
      (left - right) :: subtractCoefficients leftTail rightTail

def matrixAdd (left right : Matrix) : Matrix :=
  { left with
    coefficients := (addCoefficients left.coefficients right.coefficients).map
      (reduceCoefficient left.modulus) }

def matrixSubtract (left right : Matrix) : Matrix :=
  { left with
    coefficients := (subtractCoefficients left.coefficients right.coefficients).map
      (reduceCoefficient left.modulus) }

def matrixNegate (matrix : Matrix) : Matrix :=
  { matrix with
    coefficients := matrix.coefficients.map fun coefficient =>
      reduceCoefficient matrix.modulus (-coefficient) }

def matrixScale (scalar : Int) (matrix : Matrix) : Matrix :=
  { matrix with
    coefficients := matrix.coefficients.map fun coefficient =>
      reduceCoefficient matrix.modulus (scalar * coefficient) }

def matrixConcatRows : List Matrix → Matrix
  | [] => { coefficients := [] }
  | first :: rest =>
      let matrices := first :: rest
      { coefficients := matrices.flatMap Matrix.coefficients
        modulus := first.modulus
        ringDimension := first.ringDimension
        rows := (matrices.map Matrix.rows).sum
        columns := first.columns }

def matrixConcatColumns : List Matrix → Matrix
  | [] => { coefficients := [] }
  | first :: rest =>
      let matrices := first :: rest
      { coefficients :=
          (List.range first.rows).flatMap fun row =>
            matrices.flatMap fun matrix =>
              (List.range matrix.columns).flatMap fun column =>
                (List.range first.ringDimension).map fun coefficient =>
                  matrix.coefficient row column coefficient
        modulus := first.modulus
        ringDimension := first.ringDimension
        rows := first.rows
        columns := (matrices.map Matrix.columns).sum }

def matrixSlice (matrix : Matrix) (rowStart rowEnd columnStart columnEnd : Nat) : Matrix :=
  { coefficients :=
      (List.range (rowEnd - rowStart)).flatMap fun row =>
        (List.range (columnEnd - columnStart)).flatMap fun column =>
          (List.range matrix.ringDimension).map fun coefficient =>
            matrix.coefficient (rowStart + row) (columnStart + column) coefficient
    modulus := matrix.modulus
    ringDimension := matrix.ringDimension
    rows := rowEnd - rowStart
    columns := columnEnd - columnStart }

def matrixReshape (matrix : Matrix) (rows columns : Nat) : Matrix :=
  { matrix with rows, columns }

def diagonalCoefficient (matrices : List Matrix)
    (row column coefficient rowOffset columnOffset : Nat) : Int :=
  match matrices with
  | [] => 0
  | matrix :: tail =>
      if rowOffset ≤ row ∧ row < rowOffset + matrix.rows ∧
          columnOffset ≤ column ∧ column < columnOffset + matrix.columns then
        matrix.coefficient (row - rowOffset) (column - columnOffset) coefficient
      else
        diagonalCoefficient tail row column coefficient
          (rowOffset + matrix.rows) (columnOffset + matrix.columns)

def matrixConcatDiagonal : List Matrix → Matrix
  | [] => { coefficients := [] }
  | first :: rest =>
      let matrices := first :: rest
      let rows := (matrices.map Matrix.rows).sum
      let columns := (matrices.map Matrix.columns).sum
      { coefficients :=
          (List.range rows).flatMap fun row =>
            (List.range columns).flatMap fun column =>
              (List.range first.ringDimension).map fun coefficient =>
                diagonalCoefficient matrices row column coefficient 0 0
        modulus := first.modulus
        ringDimension := first.ringDimension
        rows
        columns }

def coefficientNorm : List Int → Nat
  | [] => 0
  | coefficient :: coefficients => max coefficient.natAbs (coefficientNorm coefficients)

def maxCenteredCoefficientNorm (matrix : Matrix) : Nat :=
  coefficientNorm (matrix.coefficients.map (centeredCoefficient matrix.modulus))

theorem coefficient_natAbs_le_norm
    {coefficients : List Int} {coefficient : Int}
    (member : coefficient ∈ coefficients) :
    coefficient.natAbs ≤ coefficientNorm coefficients := by
  induction coefficients with
  | nil => simp at member
  | cons head tail induction =>
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · simp [coefficientNorm]
      · exact le_trans (induction member) (le_max_right _ _)

theorem headD_natAbs_le_norm (matrix : Matrix) :
    (centeredCoefficient matrix.modulus (matrix.coefficients.headD 0)).natAbs ≤
      maxCenteredCoefficientNorm matrix := by
  rcases matrix with ⟨coefficients, modulus, ringDimension, rows, columns⟩
  cases coefficients with
  | nil =>
      have zeroCentered : centeredCoefficient modulus 0 = 0 := by
        by_cases nonpositive : modulus ≤ 0
        · simp [centeredCoefficient, nonpositive]
        · have positive : 0 < modulus := lt_of_not_ge nonpositive
          simp [centeredCoefficient, reduceCoefficient, nonpositive, positive.ne', positive.le]
      simp [maxCenteredCoefficientNorm, zeroCentered]
  | cons head tail =>
      exact coefficient_natAbs_le_norm
        (coefficients := (head :: tail).map (centeredCoefficient modulus))
        (coefficient := centeredCoefficient modulus head) (by simp)

structure SamplerParams where
  maxCoefficientBound : Nat
  modulus : Int := 0
  ringDimension : Nat := 0
  rows : Nat := 0
  columns : Nat := 0
  deriving DecidableEq

inductive HashVariant where
  | plain
  | decomposed
  | smallDecomposed
  deriving DecidableEq

/-- Every input that determines a hash-to-matrix result. Hash sampling is deterministic: two
nodes with equal queries necessarily receive the same matrix. -/
structure HashQuery where
  params : SamplerParams
  key : ByteArray
  variant : HashVariant
  tagPrefix : List Nat
  tagValues : List Int
  tagDecimalValues : List Int
  tagU64LeValues : List Int
  base : Option Int
  digitCount : Option Int
  deriving DecidableEq

def Matrix.withSamplerParams (matrix : Matrix) (params : SamplerParams) : Matrix :=
  let count := params.rows * params.columns * params.ringDimension
  let coefficients := matrix.coefficients.take count ++
    List.replicate (count - matrix.coefficients.length) 0
  { matrix with
    coefficients
    modulus := params.modulus
    ringDimension := params.ringDimension
    rows := params.rows
    columns := params.columns }

def gadgetMatrix (params : SamplerParams) (base : Int) (digitCount : Nat) : Matrix :=
  let matrixParams := { params with rows := params.rows, columns := params.rows * digitCount }
  let coefficients :=
    (List.range matrixParams.rows).flatMap fun row =>
      (List.range matrixParams.columns).flatMap fun column =>
        (List.range matrixParams.ringDimension).map fun coefficient =>
          if digitCount ≠ 0 ∧ column / digitCount = row ∧ coefficient = 0 then
            reduceCoefficient matrixParams.modulus (base ^ (column % digitCount))
          else 0
  Matrix.withSamplerParams { coefficients } matrixParams

structure MxxSamplerFamily where
  gaussianSample : SamplerParams → List Matrix
  hashSample : HashQuery → Matrix
  gadgetDecompose : SamplerParams → Int → Nat → Matrix → List Matrix
  trapdoorSample : SamplerParams → List Matrix
  samplePreimage : SamplerParams → Matrix → Matrix → List Matrix

structure MxxBoundedSamplerContract (samplers : MxxSamplerFamily) : Prop where
  gaussianHardSupport :
    ∀ params sample, sample ∈ samplers.gaussianSample params →
      maxCenteredCoefficientNorm (sample.withSamplerParams params) ≤ params.maxCoefficientBound
  gadgetDecomposeContract :
    ∀ params base digitCount input output,
      output ∈ samplers.gadgetDecompose params base digitCount input →
      MatrixModEq (matrixMul
        (gadgetMatrix { params with rows := input.rows, columns := input.rows * digitCount }
          base digitCount)
        (output.withSamplerParams params)) input ∧
      maxCenteredCoefficientNorm (output.withSamplerParams params) ≤
        max (base.natAbs / 2) 1
  /-- Gadget decomposition first canonicalizes its input coefficients in `R_q` and is then
  deterministic.  Consequently, quotient-equal inputs produce the same normalized digit
  matrix, even when their stored integer representatives differ. -/
  gadgetDecomposeCongruent :
    ∀ params base digitCount leftInput rightInput left right,
      MatrixModEq leftInput rightInput →
      left ∈ samplers.gadgetDecompose params base digitCount leftInput →
      right ∈ samplers.gadgetDecompose params base digitCount rightInput →
      left.withSamplerParams params = right.withSamplerParams params
  preimageContract :
    ∀ params b p k, k ∈ samplers.samplePreimage params b p →
      MatrixModEq (matrixMul b (k.withSamplerParams params)) p ∧
      maxCenteredCoefficientNorm (k.withSamplerParams params) ≤ params.maxCoefficientBound

private def matrixModEqFixture (value : Int) : Matrix := {
  coefficients := [value]
  modulus := 5
  ringDimension := 1
  rows := 1
  columns := 1
}

/-- Distinct stored representatives of the same residue are related. -/
example : MatrixModEq (matrixModEqFixture 2) (matrixModEqFixture 7) := by
  refine ⟨rfl, rfl, rfl, rfl, ?_⟩
  intro row column coefficient rowLt columnLt coefficientLt
  have rowZero : row = 0 := Nat.lt_one_iff.mp (by simpa [matrixModEqFixture] using rowLt)
  have columnZero : column = 0 :=
    Nat.lt_one_iff.mp (by simpa [matrixModEqFixture] using columnLt)
  have coefficientZero : coefficient = 0 :=
    Nat.lt_one_iff.mp (by simpa [matrixModEqFixture] using coefficientLt)
  subst row
  subst column
  subst coefficient
  rfl

/-- Different residues are not related. -/
example : ¬ MatrixModEq (matrixModEqFixture 2) (matrixModEqFixture 3) := by
  intro relation
  have unequal := relation.coefficients 0 0 0 (by decide) (by decide) (by decide)
  norm_num [matrixModEqFixture, Matrix.coefficient, reduceCoefficient] at unequal

end Mxx
