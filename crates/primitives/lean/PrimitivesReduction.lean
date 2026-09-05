import PrimitivesNegacyclic
import Mathlib.Data.Matrix.Basic

namespace Mxx.Primitives

open scoped BigOperators

/- Coefficientwise reduction is the quotient ring map induced by `Int → ZMod q`.
The polynomial relation is unchanged by this map, so `AdjoinRoot.map` supplies a
genuine `RingHom`, including multiplication preservation. -/
noncomputable def reducePoly (q n : Nat) : ErrorPoly n →+* ExactPoly q n := by
  refine AdjoinRoot.map (Int.castRingHom (ZMod q))
    (negacyclicModulus n Int) (negacyclicModulus n (ZMod q)) ?_
  refine ⟨1, ?_⟩
  simp [negacyclicModulus, Polynomial.map_add, Polynomial.map_pow, Polynomial.map_X]

@[simp] theorem reducePoly_zero (q n : Nat) : reducePoly q n (0 : ErrorPoly n) = 0 := by
  exact (reducePoly q n).map_zero

@[simp] theorem reducePoly_add (q n : Nat) (x y : ErrorPoly n) :
    reducePoly q n (x + y) = reducePoly q n x + reducePoly q n y := by
  exact (reducePoly q n).map_add x y

@[simp] theorem reducePoly_neg (q n : Nat) (x : ErrorPoly n) :
    reducePoly q n (-x) = -reducePoly q n x := by
  exact (reducePoly q n).map_neg x

@[simp] theorem reducePoly_mul (q n : Nat) (x y : ErrorPoly n) :
    reducePoly q n (x * y) = reducePoly q n x * reducePoly q n y := by
  exact (reducePoly q n).map_mul x y

theorem reducePoly_sum {α : Type u} [DecidableEq α] (q n : Nat)
    (values : Finset α) (f : α → ErrorPoly n) :
    reducePoly q n (Finset.sum values f) =
      Finset.sum values (fun value => reducePoly q n (f value)) := by
  exact map_sum (reducePoly q n) f values

noncomputable def reduceMatrix (q n rows columns : Nat) :
    Matrix (Fin rows) (Fin columns) (ErrorPoly n) →
      Matrix (Fin rows) (Fin columns) (ExactPoly q n) := fun matrix row column =>
  reducePoly q n (matrix row column)

theorem reduceMatrix_apply (q n rows columns : Nat)
    (matrix : Matrix (Fin rows) (Fin columns) (ErrorPoly n))
    (row : Fin rows) (column : Fin columns) :
    reduceMatrix q n rows columns matrix row column = reducePoly q n (matrix row column) := rfl

theorem reduceMatrix_add (q n rows columns : Nat)
    (a b : Matrix (Fin rows) (Fin columns) (ErrorPoly n)) :
    reduceMatrix q n rows columns (a + b) =
      reduceMatrix q n rows columns a + reduceMatrix q n rows columns b := by
  funext row column
  exact reducePoly_add q n _ _

theorem reduceMatrix_mul (q n rows inner columns : Nat)
    (a : Matrix (Fin rows) (Fin inner) (ErrorPoly n))
    (b : Matrix (Fin inner) (Fin columns) (ErrorPoly n)) :
    reduceMatrix q n rows columns (a * b) =
      reduceMatrix q n rows inner a * reduceMatrix q n inner columns b := by
  funext row column
  simp only [Matrix.mul_apply, reduceMatrix_apply]
  rw [reducePoly_sum]
  apply Finset.sum_congr rfl
  intro index hindex
  exact reducePoly_mul q n _ _

end Mxx.Primitives
