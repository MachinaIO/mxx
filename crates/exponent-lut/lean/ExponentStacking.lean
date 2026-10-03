import ExponentGadget

/-!
# Stacked products in generated relations

Evaluation multiplies the row and both key rows of an encoding by one action in a single product:
the three left blocks and the three right blocks are stacked (`concatRows`), placed side by side
(`concatColumns`), multiplied, and the result is sliced back into rows. These lemmas read such a
generated computation as `pairMul` of each row.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {q n : Nat}

/-- Row `k` of a product is row `k` of the left factor times the right factor. -/
theorem rowOf_mul {rows inner columns : Nat} (x : ExactMatrix q n rows inner)
    (y : ExactMatrix q n inner columns) (k : Fin rows) : rowOf (x * y) k = rowOf x k * y := by
  funext row column
  simp [rowOf, Matrix.mul_apply]

/-- The slice of one full row is that row. -/
theorem sliceMatrix_row {rows columns : Nat} {x : ExactMatrix q n rows columns}
    {out : ExactMatrix q n 1 columns} (k : Fin rows)
    (h : sliceMatrix x k.val (k.val + 1) 0 columns out) : out = rowOf x k := by
  obtain ⟨_, _, _, _, _, _, _, _, hout⟩ := h
  funext row column
  have hrow : row = 0 := Subsingleton.elim _ _
  subst hrow
  rw [hout 0 column (by simp) (by simp)]
  simp [rowOf]

theorem concatRows_rowOf_left {rows columns : Nat} {left : ExactMatrix q n rows columns}
    {right : ExactMatrix q n 1 columns} {out : ExactMatrix q n (rows + 1) columns}
    (h : concatRows left right out) (k : Fin rows) : rowOf out (Fin.castAdd 1 k) = rowOf left k := by
  funext row column
  simp [rowOf, h _ column]

theorem concatRows_rowOf_right {rows columns : Nat} {left : ExactMatrix q n rows columns}
    {right : ExactMatrix q n 1 columns} {out : ExactMatrix q n (rows + 1) columns}
    (h : concatRows left right out) : rowOf out (Fin.last rows) = right := by
  funext row column
  have hrow : row = 0 := Subsingleton.elim _ _
  subst hrow
  simp [rowOf, h _ column]

/-- The rows of a two-row `concatRows`. -/
theorem concatRows_two {columns : Nat} {x y : ExactMatrix q n 1 columns}
    {out : ExactMatrix q n (1 + 1) columns} (h : concatRows x y out) :
    rowOf out 0 = x ∧ rowOf out 1 = y := by
  constructor
  · funext row column
    have hrow : row = 0 := Subsingleton.elim _ _
    subst hrow
    simp [rowOf, h 0 column]
  · funext row column
    have hrow : row = 0 := Subsingleton.elim _ _
    subst hrow
    simp [rowOf, h 1 column]

/-- A side-by-side product splits into the products of its halves. -/
theorem concatColumns_mul {rows m m' : Nat} {left right : ExactMatrix q n rows m}
    {out : ExactMatrix q n rows (m + m)} (h : concatColumns left right out)
    (d : ExactMatrix q n (m + m) m') :
    out * d = left * topRows d + right * bottomRows d := by
  funext row column
  simp only [Matrix.add_apply, Matrix.mul_apply, topRows, bottomRows, Matrix.submatrix_apply, id]
  rw [Fin.sum_univ_add]
  congr 1
  · apply Finset.sum_congr rfl
    intro k _
    rw [h row (Fin.castAdd m k)]
    simp
  · apply Finset.sum_congr rfl
    intro k _
    rw [h row (Fin.natAdd m k)]
    simp

/-- Row `k` of a stacked product: `pairMul` of the `k`-th left and right blocks. -/
theorem stacked_row {m m' : Nat} {l0 l1 l2 r0 r1 r2 : ExactMatrix q n 1 m}
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows l0 l1 l01) (hl : concatRows l01 l2 l)
    (hr01 : concatRows r0 r1 r01) (hr : concatRows r01 r2 r)
    (hboth : concatColumns l r both) (d : ExactMatrix q n (m + m) m') :
    rowOf (both * d) 0 = pairMul ![l0, r0] d ∧ rowOf (both * d) 1 = pairMul ![l1, r1] d ∧
      rowOf (both * d) 2 = pairMul ![l2, r2] d := by
  have hl0 : rowOf l 0 = l0 := by
    have := concatRows_rowOf_left hl 0
    rw [show (Fin.castAdd 1 (0 : Fin (1 + 1))) = (0 : Fin (1 + 1 + 1)) from rfl] at this
    rw [this]; exact (concatRows_two hl01).1
  have hl1 : rowOf l 1 = l1 := by
    have := concatRows_rowOf_left hl 1
    rw [show (Fin.castAdd 1 (1 : Fin (1 + 1))) = (1 : Fin (1 + 1 + 1)) from rfl] at this
    rw [this]; exact (concatRows_two hl01).2
  have hl2 : rowOf l 2 = l2 := concatRows_rowOf_right hl
  have hr0 : rowOf r 0 = r0 := by
    have := concatRows_rowOf_left hr 0
    rw [show (Fin.castAdd 1 (0 : Fin (1 + 1))) = (0 : Fin (1 + 1 + 1)) from rfl] at this
    rw [this]; exact (concatRows_two hr01).1
  have hr1 : rowOf r 1 = r1 := by
    have := concatRows_rowOf_left hr 1
    rw [show (Fin.castAdd 1 (1 : Fin (1 + 1))) = (1 : Fin (1 + 1 + 1)) from rfl] at this
    rw [this]; exact (concatRows_two hr01).2
  have hr2 : rowOf r 2 = r2 := concatRows_rowOf_right hr
  have hsplit := concatColumns_mul hboth d
  refine ⟨?_, ?_, ?_⟩ <;>
  · rw [hsplit]
    funext row column
    simp only [rowOf, pairMul, Matrix.add_apply, Matrix.mul_apply, Matrix.cons_val_zero,
      Matrix.cons_val_one]
    first
    | rw [← hl0, ← hr0]; simp [rowOf]
    | rw [← hl1, ← hr1]; simp [rowOf]
    | rw [← hl2, ← hr2]; simp [rowOf]

end MxxExponentLut
