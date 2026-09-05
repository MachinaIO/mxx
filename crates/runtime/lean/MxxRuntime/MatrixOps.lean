import MxxRuntime.Primitives

namespace MxxRuntime

open Mxx.Primitives

noncomputable def matrixAdd {q n r c : Nat} (left right : ExactMatrix q n r c) := left + right
noncomputable def matrixSub {q n r c : Nat} (left right : ExactMatrix q n r c) := left - right
noncomputable def matrixMul {q n r k c : Nat}
    (left : ExactMatrix q n r k) (right : ExactMatrix q n k c) := left * right
noncomputable def matrixMulScalarLeft {q n r c : Nat}
    (left : ExactMatrix q n 1 1) (right : ExactMatrix q n r c) : ExactMatrix q n r c :=
  fun i j ↦ left 0 0 * right i j
noncomputable def matrixMulScalarRight {q n r c : Nat}
    (left : ExactMatrix q n r c) (right : ExactMatrix q n 1 1) : ExactMatrix q n r c :=
  fun i j ↦ left i j * right 0 0

@[simp] theorem matrixMulScalarLeft_one {q n c : Nat}
    (left : ExactMatrix q n 1 1) (right : ExactMatrix q n 1 c) :
    matrixMulScalarLeft left right = left * right := by
  ext i j
  fin_cases i
  simp [matrixMulScalarLeft, Matrix.mul_apply]

@[simp] theorem matrixMulScalarRight_one {q n r : Nat}
    (left : ExactMatrix q n r 1) (right : ExactMatrix q n 1 1) :
    matrixMulScalarRight left right = left * right := by
  ext i j
  fin_cases j
  simp [matrixMulScalarRight, Matrix.mul_apply]
noncomputable def matrixNeg {q n r c : Nat} (value : ExactMatrix q n r c) := -value
noncomputable def matrixScale {q n r c : Nat} (scalar : Int) (value : ExactMatrix q n r c) :
    ExactMatrix q n r c := fun i j ↦ (scalar : ExactPoly q n) * value i j
def transpose {q n r c : Nat} (value : ExactMatrix q n r c) := value.transpose
def trapdoorPublic {Public Token : Type} (value : TrapdoorValue Public Token) := value.publicMatrix

def familyGetDynamic {α : Type} {count : Nat} (family : Fin count → α) (index : Int)
    (output : α) : Prop :=
  ∃ position : Fin count, (position.val : Int) = index ∧ output = family position

def familyGetStatic {α : Type} {count : Nat} := @familyGetDynamic α count

def select {α : Type} (index : Int) (choices : List α) (output : α) : Prop :=
  ∃ position : Fin choices.length, (position.val : Int) = index ∧ output = choices.get position

def familyPack {α : Type} {size : Nat} (count : Int) (values : List α)
    (output : Fin size → α) : Prop :=
  count = size ∧ values.length = size ∧
  ∀ index : Fin size, ∃ position : Fin values.length,
    position.val = index.val ∧ output index = values.get position

def uniformResidueSample {q n rows columns : Nat}
    (output : ExactMatrix q n rows columns) : Prop := output = output

def uniformIntervalSample {q n rows columns : Nat} (minimum maximum : Int)
    (output : ExactMatrix q n rows columns) : Prop :=
  minimum ≤ maximum ∧ ∃ witness : ErrorMatrix n rows columns,
    output = reduceMatrix q n rows columns witness ∧
    ∀ row column coefficient, minimum ≤ (witness row column).coeff coefficient ∧
      (witness row column).coeff coefficient ≤ maximum

def gaussianSample {q n rows columns : Nat} (sigma : Rat) (cutoff : Int)
    (output : ExactMatrix q n rows columns) : Prop :=
  0 ≤ sigma ∧ 0 ≤ cutoff ∧ PreimageWithin output cutoff.toNat ∧ (sigma = 0 → output = 0)

/-- The backend returns the canonical residue, not a centered integer. -/
noncomputable def extractCoefficient {q n : Nat} (position : Int)
    (input : ExactMatrix q n 1 1) (output : Int) : Prop :=
  ∃ index : Fin n, (index.val : Int) = position ∧
    output = ((input 0 0).coeff index).val

/-- A literal coefficient list in increasing degree order, reduced in the exact residue ring.
The exporter validates the scalar shape and the list length against the ring dimension. -/
noncomputable def matrixPolynomial {q n : Nat} (coefficients : List Int) :
    ExactMatrix q n 1 1 :=
  fun _ _ ↦ coefficients.foldr
    (fun coefficient tail ↦ (coefficient : ExactPoly q n) +
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) * tail) 0

/- Pure matrix relations used by generic frozen-IR extraction.  The dimensions are carried by the
   matrix types; the relations only describe the coefficient-wise operation and its range guards. -/
def sliceMatrix {q n rows columns outRows outColumns : Nat}
    (input : ExactMatrix q n rows columns)
    (rowStart rowEnd columnStart columnEnd : Int)
    (output : ExactMatrix q n outRows outColumns) : Prop :=
  0 ≤ rowStart ∧ rowStart ≤ rowEnd ∧ rowEnd ≤ Int.ofNat rows ∧
  0 ≤ columnStart ∧ columnStart ≤ columnEnd ∧ columnEnd ≤ Int.ofNat columns ∧
  rowEnd - rowStart = Int.ofNat outRows ∧
  columnEnd - columnStart = Int.ofNat outColumns ∧
  ∀ row column hRow hColumn,
    output row column =
      input
        ⟨rowStart.toNat + row.val, hRow⟩
        ⟨columnStart.toNat + column.val, hColumn⟩

def concatRows {q n leftRows rightRows columns : Nat}
    (left : ExactMatrix q n leftRows columns)
    (right : ExactMatrix q n rightRows columns)
    (output : ExactMatrix q n (leftRows + rightRows) columns) : Prop :=
  ∀ row column,
    output row column =
      if h : row.val < leftRows then
        left ⟨row.val, h⟩ column
      else
        right ⟨row.val - leftRows, by omega⟩ column

def concatColumns {q n rows leftColumns rightColumns : Nat}
    (left : ExactMatrix q n rows leftColumns)
    (right : ExactMatrix q n rows rightColumns)
    (output : ExactMatrix q n rows (leftColumns + rightColumns)) : Prop :=
  ∀ row column,
    output row column =
      if h : column.val < leftColumns then
        left row ⟨column.val, h⟩
      else
        right row ⟨column.val - leftColumns, by omega⟩

def concatDiagonal {q n leftRows leftColumns rightRows rightColumns : Nat}
    (left : ExactMatrix q n leftRows leftColumns)
    (right : ExactMatrix q n rightRows rightColumns)
    (output : ExactMatrix q n (leftRows + rightRows) (leftColumns + rightColumns)) : Prop :=
  ∀ row column,
    output row column =
      if hRow : row.val < leftRows then
        if hColumn : column.val < leftColumns then
          left ⟨row.val, hRow⟩ ⟨column.val, hColumn⟩
        else
          0
      else if hColumn : column.val < leftColumns then
        0
      else
        right ⟨row.val - leftRows, by omega⟩ ⟨column.val - leftColumns, by omega⟩

end MxxRuntime
