import TopColumn

/-!
# The preimage targets

The preprocessing solves `A P = [r₀; r₁]` for the rows `r_i` of a decoder column: its combination
of keys (the mask carrier's and the one encoding's times the digits of `Q_q A'`, minus the target
component's times the action digits, plus the fresh column's) at the column, plus the smudge keys.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- A row of the preprocessing's combination at column `a`, plus the smudge keys. -/
theorem pre_row {kg k1 : ExactMatrix Q1 N 1 6} {gc : ExactMatrix Q1 N 1 (6 + 6)}
    (hgc : concatColumns kg k1 gc) {D : ExactMatrix Q1 N (6 + 6) 6} {kx : ExactMatrix Q1 N 1 4}
    {Y : ExactMatrix Q1 N 4 6} {kw R : ExactMatrix Q1 N 1 6}
    (hR : R = matrixAdd (matrixSub (matrixMul gc D) (matrixMul kx Y)) kw)
    {a b : Int} (ha : a.toNat < 6) {s : ExactMatrix Q1 N 1 1} (hs : sliceMatrix R 0 1 a b s)
    {sm r : ExactMatrix Q1 N 1 1} (hr : r = matrixAdd s sm) :
    r = (kg * topRows D + k1 * bottomRows D - kx * Y + kw) * unitCol (q := Q1) ⟨a.toNat, ha⟩ + sm := by
  rw [hr, slice_col hs ha, hR]
  simp only [matrixAdd, matrixSub, matrixMul, concatColumns_mul hgc]

/-- The rows of a preimage: row `i` of `A`, times the preimage, is target row `i`. -/
theorem target_rows {A : ExactMatrix Q1 N (1 + 1) 16} {P : ExactMatrix Q1 N 16 1}
    {r0 r1 : ExactMatrix Q1 N 1 1} {T : ExactMatrix Q1 N (1 + 1) 1} (hT : concatRows r0 r1 T)
    (hP : A * P = T) : ∀ i : Fin 2, rowOf A i * P = ![r0, r1] i := by
  intro i
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  have := congrFun (congrFun hP i) 0
  rw [hT i 0] at this
  fin_cases i <;> simpa [rowOf, Matrix.mul_apply] using this

end RefreshTopProof
