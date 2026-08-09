import Mathlib.Data.ENNReal.Basic
import Mathlib.Data.List.GetD

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

/-- The unsigned canonical coefficient used by compact decomposition.  This deliberately differs
from `centeredCoefficient`: `-1` modulo a positive `q` has canonical value `q - 1`. -/
def canonicalCoefficient (modulus value : Int) : Nat :=
  (reduceCoefficient modulus value).toNat

def maxCanonicalCoefficient (matrix : Matrix) : Nat :=
  matrix.coefficients.foldl (fun maximum coefficient =>
    max maximum (canonicalCoefficient matrix.modulus coefficient)) 0

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
  deriving DecidableEq, Repr

abbrev SamplerParamsId := String

inductive TrapdoorOrigin where
  | sampled
  | gadget (paramsId : SamplerParamsId) (base : Int) (small : Bool) (digitCount : Nat)
  deriving DecidableEq

structure GadgetLayoutDescriptor where
  paramsId : SamplerParamsId
  ringDimension : Nat
  crtModuli : List Nat
  crtBits : Nat
  baseBits : Nat
  base : Int
  regularDigitCount : Nat
  smallDigitCount : Nat
  smallestCrtModulus : Nat
  deriving DecidableEq

private def ceilDivide (numerator denominator : Nat) : Option Nat :=
  if denominator = 0 then none else some ((numerator + denominator - 1) / denominator)

def GadgetLayoutDescriptor.valid (descriptor : GadgetLayoutDescriptor) : Bool :=
  let modulus := descriptor.crtModuli.foldl (· * ·) 1
  let smallest := descriptor.crtModuli.foldl min descriptor.crtModuli.head!
  match ceilDivide descriptor.crtBits descriptor.baseBits with
  | none => false
  | some digitsPerTower =>
      !descriptor.crtModuli.isEmpty && descriptor.baseBits > 0 && descriptor.base > 1 &&
        descriptor.base = 2 ^ descriptor.baseBits && descriptor.smallestCrtModulus = smallest &&
        descriptor.smallDigitCount = digitsPerTower &&
        descriptor.regularDigitCount = digitsPerTower * descriptor.crtModuli.length &&
        modulus > 0

def GadgetLayoutDescriptor.matches (descriptor : GadgetLayoutDescriptor) (params : SamplerParams) : Bool :=
  descriptor.valid && descriptor.ringDimension = params.ringDimension &&
    params.modulus = Int.ofNat (descriptor.crtModuli.foldl (fun product modulus => product * modulus) 1)

def gadgetDecompositionBound (base : Int) (small : Bool) : Nat :=
  if small then base.natAbs - 1 else max (base.natAbs / 2) 1

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
  trailingIntegerTagValues : List Int
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

/-- A matrix stores exactly one coefficient for every typed row, column, and ring position. -/
def Matrix.WellFormed (matrix : Matrix) : Prop :=
  matrix.coefficients.length = matrix.rows * matrix.columns * matrix.ringDimension

theorem Matrix.withSamplerParams_wellFormed (matrix : Matrix) (params : SamplerParams) :
    (matrix.withSamplerParams params).WellFormed := by
  simp only [Matrix.WellFormed, Matrix.withSamplerParams, List.length_append,
    List.length_take, List.length_replicate]
  omega

private theorem addCoefficients_length_of_eq
    (left right : List Int)
    (sameLength : left.length = right.length) :
    (addCoefficients left right).length = left.length := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail => simp at sameLength
  | cons leftHead leftTail induction =>
      cases right with
      | nil => simp at sameLength
      | cons rightHead rightTail =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp [addCoefficients, induction rightTail sameLength]

theorem matrixAdd_wellFormed
    (left right : Matrix)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (sameRingDimension : left.ringDimension = right.ringDimension) :
    (matrixAdd left right).WellFormed := by
  have sameLength : left.coefficients.length = right.coefficients.length := by
    rw [leftWellFormed, rightWellFormed, sameRows, sameColumns, sameRingDimension]
  simp only [Matrix.WellFormed, matrixAdd, List.length_map]
  rw [addCoefficients_length_of_eq left.coefficients right.coefficients sameLength]
  exact leftWellFormed

private theorem subtractCoefficients_length_of_eq
    (left right : List Int)
    (sameLength : left.length = right.length) :
    (subtractCoefficients left right).length = left.length := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail => simp at sameLength
  | cons leftHead leftTail induction =>
      cases right with
      | nil => simp at sameLength
      | cons rightHead rightTail =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp [subtractCoefficients, induction rightTail sameLength]

theorem matrixSubtract_wellFormed
    (left right : Matrix)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed)
    (sameRows : left.rows = right.rows)
    (sameColumns : left.columns = right.columns)
    (sameRingDimension : left.ringDimension = right.ringDimension) :
    (matrixSubtract left right).WellFormed := by
  have sameLength : left.coefficients.length = right.coefficients.length := by
    rw [leftWellFormed, rightWellFormed, sameRows, sameColumns, sameRingDimension]
  simp only [Matrix.WellFormed, matrixSubtract, List.length_map]
  rw [subtractCoefficients_length_of_eq left.coefficients right.coefficients sameLength]
  exact leftWellFormed

theorem matrixNegate_wellFormed
    (matrix : Matrix)
    (wellFormed : matrix.WellFormed) :
    (matrixNegate matrix).WellFormed := by
  simpa [Matrix.WellFormed, matrixNegate] using wellFormed

theorem matrixScale_wellFormed
    (scalar : Int)
    (matrix : Matrix)
    (wellFormed : matrix.WellFormed) :
    (matrixScale scalar matrix).WellFormed := by
  simpa [Matrix.WellFormed, matrixScale] using wellFormed

theorem reduced_coefficients_eq_of_matrixModEq
    {left right : Matrix}
    (relation : MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed) :
    left.coefficients.map (reduceCoefficient left.modulus) =
      right.coefficients.map (reduceCoefficient right.modulus) := by
  apply List.ext_getElem
  · simp only [List.length_map]
    calc
      left.coefficients.length = left.rows * left.columns * left.ringDimension := leftWellFormed
      _ = right.rows * right.columns * right.ringDimension := by
        rw [relation.rows, relation.columns, relation.ringDimension]
      _ = right.coefficients.length := rightWellFormed.symm
  · intro index leftLt rightLt
    have indexLt : index < left.rows * left.columns * left.ringDimension := by
      have coefficientLt : index < left.coefficients.length := by simpa using leftLt
      rw [leftWellFormed] at coefficientLt
      exact coefficientLt
    have ringPositive : 0 < left.ringDimension := by
      by_contra nonpositive
      have : left.ringDimension = 0 := Nat.eq_zero_of_not_pos nonpositive
      simp [this] at indexLt
    have columnsPositive : 0 < left.columns := by
      by_contra nonpositive
      have : left.columns = 0 := Nat.eq_zero_of_not_pos nonpositive
      simp [this] at indexLt
    let coefficient := index % left.ringDimension
    let entry := index / left.ringDimension
    let column := entry % left.columns
    let row := entry / left.columns
    have coefficientLt : coefficient < left.ringDimension := Nat.mod_lt _ ringPositive
    have columnLt : column < left.columns := Nat.mod_lt _ columnsPositive
    have rowLt : row < left.rows := by
      have entryLt : entry < left.rows * left.columns := by
        rw [Nat.div_lt_iff_lt_mul ringPositive]
        simpa [Nat.mul_assoc] using indexLt
      dsimp [row]
      rw [Nat.div_lt_iff_lt_mul columnsPositive]
      simpa [Nat.mul_comm] using entryLt
    have linearIndex : ((row * left.columns + column) * left.ringDimension) + coefficient =
        index := by
      dsimp [row, column, entry, coefficient]
      rw [Nat.mul_comm (index / left.ringDimension / left.columns) left.columns]
      rw [Nat.div_add_mod (index / left.ringDimension) left.columns]
      rw [Nat.mul_comm (index / left.ringDimension) left.ringDimension]
      rw [Nat.div_add_mod index left.ringDimension]
    have coefficientRelation :=
      relation.coefficients row column coefficient rowLt columnLt coefficientLt
    simp only [List.getElem_map]
    have leftCoefficientLt : index < left.coefficients.length := by simpa using leftLt
    have rightCoefficientLt : index < right.coefficients.length := by simpa using rightLt
    have rightLinearIndex :
        ((row * right.columns + column) * right.ringDimension) + coefficient = index := by
      rw [← relation.columns, ← relation.ringDimension]
      exact linearIndex
    unfold Matrix.coefficient at coefficientRelation
    rw [linearIndex, rightLinearIndex] at coefficientRelation
    rw [List.getD_eq_getElem _ _ leftCoefficientLt,
      List.getD_eq_getElem _ _ rightCoefficientLt] at coefficientRelation
    exact coefficientRelation

private def centeredReduced (modulus residue : Int) : Int :=
  if modulus ≤ 0 then residue
  else if 2 * residue > modulus then residue - modulus else residue

private theorem centeredCoefficient_eq_centeredReduced (modulus value : Int) :
    centeredCoefficient modulus value = centeredReduced modulus (reduceCoefficient modulus value) := by
  by_cases nonpositive : modulus ≤ 0
  · simp [centeredCoefficient, centeredReduced, reduceCoefficient, nonpositive]
  · simp [centeredCoefficient, centeredReduced, reduceCoefficient, nonpositive]

theorem centered_coefficients_eq_of_matrixModEq
    {left right : Matrix}
    (relation : MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed) :
    left.coefficients.map (centeredCoefficient left.modulus) =
      right.coefficients.map (centeredCoefficient right.modulus) := by
  have reduced := reduced_coefficients_eq_of_matrixModEq relation leftWellFormed rightWellFormed
  rw [← relation.modulus] at reduced ⊢
  have mapped := congrArg (List.map (centeredReduced left.modulus)) reduced
  simp only [List.map_map] at mapped
  have functionEq : centeredReduced left.modulus ∘ reduceCoefficient left.modulus =
      centeredCoefficient left.modulus := by
    funext value
    exact (centeredCoefficient_eq_centeredReduced left.modulus value).symm
  simpa only [functionEq] using mapped

theorem maxCenteredCoefficientNorm_eq_of_matrixModEq
    {left right : Matrix}
    (relation : MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed) :
    maxCenteredCoefficientNorm left = maxCenteredCoefficientNorm right := by
  unfold maxCenteredCoefficientNorm
  rw [centered_coefficients_eq_of_matrixModEq relation leftWellFormed rightWellFormed]

theorem canonical_coefficients_eq_of_matrixModEq
    {left right : Matrix}
    (relation : MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed) :
    left.coefficients.map (canonicalCoefficient left.modulus) =
      right.coefficients.map (canonicalCoefficient right.modulus) := by
  have reduced := reduced_coefficients_eq_of_matrixModEq relation leftWellFormed rightWellFormed
  rw [← relation.modulus] at reduced ⊢
  have mapped := congrArg (List.map Int.toNat) reduced
  simp only [List.map_map] at mapped
  have functionEq : Int.toNat ∘ reduceCoefficient left.modulus =
      canonicalCoefficient left.modulus := by
    funext value
    rfl
  simpa only [functionEq] using mapped

theorem maxCanonicalCoefficient_eq_of_matrixModEq
    {left right : Matrix}
    (relation : MatrixModEq left right)
    (leftWellFormed : left.WellFormed)
    (rightWellFormed : right.WellFormed) :
    maxCanonicalCoefficient left = maxCanonicalCoefficient right := by
  unfold maxCanonicalCoefficient
  rw [← List.foldl_map]
  rw [canonical_coefficients_eq_of_matrixModEq relation leftWellFormed rightWellFormed]
  rw [List.foldl_map]

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
  layoutId : SamplerParams → Option SamplerParamsId
  gadgetPublicMatrix :
    SamplerParamsId → SamplerParams → Nat → Int → Bool → Nat → Option Matrix
  gadgetDecompose :
    SamplerParamsId → SamplerParams → Int → Bool → Nat → Matrix → Option Matrix
  smallDecompositionInputLimit : SamplerParamsId → SamplerParams → Option Nat
  trapdoorSample : SamplerParams → List Matrix
  samplePreimage : SamplerParams → Matrix → Matrix → List Matrix

def HashQueriesMatchDecomposition
    (plain decomposed : HashQuery)
    (base : Int)
    (small : Bool)
    (digitCount : Nat) : Prop :=
  decomposed.params.rows = plain.params.rows * digitCount ∧
    decomposed.params.columns = plain.params.columns ∧
    plain.params.modulus = decomposed.params.modulus ∧
    plain.params.ringDimension = decomposed.params.ringDimension ∧
    plain.key = decomposed.key ∧
    plain.tagPrefix = decomposed.tagPrefix ∧
    plain.tagValues = decomposed.tagValues ∧
    plain.tagDecimalValues = decomposed.tagDecimalValues ∧
    plain.tagU64LeValues = decomposed.tagU64LeValues ∧
    plain.trailingIntegerTagValues = decomposed.trailingIntegerTagValues ∧
    plain.variant = .plain ∧
    plain.base = none ∧
    plain.digitCount = none ∧
    decomposed.variant = (if small then .smallDecomposed else .decomposed) ∧
    decomposed.base = some base ∧
    decomposed.digitCount = some (Int.ofNat digitCount)

structure GadgetLayoutAgrees
    (samplers : MxxSamplerFamily)
    (descriptor : GadgetLayoutDescriptor)
    (params : SamplerParams) : Prop where
  layoutId : samplers.layoutId params = some descriptor.paramsId
  smallInputLimit :
    samplers.smallDecompositionInputLimit descriptor.paramsId params =
      some descriptor.smallestCrtModulus
  publicDefined : ∀ inputRows small,
    let digits := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
    ∃ publicMatrix,
      samplers.gadgetPublicMatrix descriptor.paramsId params inputRows descriptor.base small digits =
        some publicMatrix ∧
      publicMatrix.modulus = params.modulus ∧
      publicMatrix.ringDimension = params.ringDimension ∧
      publicMatrix.rows = inputRows ∧
      publicMatrix.columns = inputRows * digits
  decompositionDefined : ∀ input small,
    let digits := if small then descriptor.smallDigitCount else descriptor.regularDigitCount
    input.modulus = params.modulus →
    input.ringDimension = params.ringDimension →
    params.rows = input.rows * digits →
    params.columns = input.columns →
    ∃ output,
      samplers.gadgetDecompose descriptor.paramsId params descriptor.base small digits input =
        some output

structure MxxBoundedSamplerContract (samplers : MxxSamplerFamily) : Prop where
  gaussianHardSupport :
    ∀ params sample, sample ∈ samplers.gaussianSample params →
      maxCenteredCoefficientNorm (sample.withSamplerParams params) ≤ params.maxCoefficientBound
  gadgetDecomposeRelation :
    ∀ paramsId params base small digitCount input publicMatrix output,
      samplers.gadgetPublicMatrix paramsId params input.rows base small digitCount = some publicMatrix →
      samplers.gadgetDecompose paramsId params base small digitCount input = some output →
      (small = false ∨ ∃ limit,
        samplers.smallDecompositionInputLimit paramsId params = some limit ∧
        maxCanonicalCoefficient input < limit) →
      MatrixModEq (matrixMul publicMatrix (output.withSamplerParams params)) input
  gadgetDecomposeHardBound :
    ∀ paramsId params base small digitCount input output,
      samplers.gadgetDecompose paramsId params base small digitCount input = some output →
      maxCenteredCoefficientNorm (output.withSamplerParams params) ≤
        gadgetDecompositionBound base small
  gadgetDecomposeSmallCanonicalRange :
    ∀ paramsId params base digitCount input output,
      samplers.gadgetDecompose paramsId params base true digitCount input = some output →
      maxCanonicalCoefficient (output.withSamplerParams params) < base.natAbs
  decomposedHashConsistency :
    ∀ paramsId plain decomposed base small digitCount,
      samplers.layoutId decomposed.params = some paramsId →
      HashQueriesMatchDecomposition plain decomposed base small digitCount →
      samplers.gadgetDecompose paramsId decomposed.params base small digitCount
        ((samplers.hashSample plain).withSamplerParams plain.params) =
          some (samplers.hashSample decomposed)
  gadgetLayoutAgreement :
    ∀ descriptor params,
      descriptor.valid = true →
      descriptor.matches params = true →
      samplers.layoutId params = some descriptor.paramsId →
      GadgetLayoutAgrees samplers descriptor params
  /-- Gadget decomposition first canonicalizes its input coefficients in `R_q` and is then
  deterministic.  Consequently, quotient-equal inputs produce the same normalized digit
  matrix, even when their stored integer representatives differ. -/
  gadgetDecomposeCongruent :
    ∀ paramsId params base small digitCount leftInput rightInput left right,
      MatrixModEq leftInput rightInput →
      samplers.gadgetDecompose paramsId params base small digitCount leftInput = some left →
      samplers.gadgetDecompose paramsId params base small digitCount rightInput = some right →
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
