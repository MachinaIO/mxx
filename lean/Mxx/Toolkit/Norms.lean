import Mxx.Ir
import Mathlib.Data.List.GetD
import Mathlib.Tactic

namespace Mxx.Toolkit

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

/-- Every stored coefficient is bounded by the centered coefficient norm. The
statement also covers an out-of-range access, whose `getD` value is zero. -/
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
