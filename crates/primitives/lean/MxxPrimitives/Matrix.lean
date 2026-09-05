import MxxPrimitives.Reduction
import Mathlib.Data.Matrix.Basic

namespace Mxx.Primitives

abbrev ExactMatrix (q n rows columns : Nat) := ModMatrix q n rows columns

abbrev ErrorMatrix (n rows columns : Nat) := IntMatrix n rows columns

theorem exactMatrix_ext {q n rows columns : Nat} {a b : ExactMatrix q n rows columns}
    (h : ∀ row column, a row column = b row column) : a = b := by
  funext row column
  exact h row column

theorem errorMatrix_ext {n rows columns : Nat} {a b : ErrorMatrix n rows columns}
    (h : ∀ row column, a row column = b row column) : a = b := by
  funext row column
  exact h row column

end Mxx.Primitives
