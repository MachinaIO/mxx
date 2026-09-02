import MxxPrimitives.Bounds

namespace Mxx.Gadgets

open Mxx.Primitives

namespace SelectorMagnitude

variable {q n rows columns inner : Nat}

/- A sampled matrix already represented by a bounded integer lift has exactly the magnitude
   witness required by selector propagation. No exact-expression reconstruction is needed. -/
noncomputable def ofBoundedLift
    {actual : ExactMatrix q n rows columns} {bound : Nat}
    (lift : BoundedLift actual bound) : MagnitudeFact actual :=
  { lift := lift.witness
    reduce_eq := lift.reduce_eq
    bound := bound
    norm_le := lift.norm_le
    support := .arbitrary
    support_valid := by
      intro h
      cases h }

/- The all-zero matrix is its own integer lift and has coefficient infinity norm zero. -/
noncomputable def zero : MagnitudeFact (0 : ExactMatrix q n rows columns) :=
  { lift := 0
    reduce_eq := by
      funext row column
      simp [reduceMatrix]
    bound := 0
    norm_le := matrixNorm_zero.le
    support := .constant
    support_valid := by simp }

/- Weakening changes only the reported upper bound. The represented exact matrix and integer
   lift remain identical. -/
noncomputable def weaken
    {actual : ExactMatrix q n rows columns} (fact : MagnitudeFact actual)
    {bound : Nat} (hbound : fact.bound ≤ bound) : MagnitudeFact actual :=
  { lift := fact.lift
    reduce_eq := fact.reduce_eq
    bound := bound
    norm_le := fact.norm_le.trans hbound
    support := fact.support
    support_valid := fact.support_valid }

/- A matrix select contributes no arithmetic growth: whichever branch is chosen has norm at
   most the maximum of the two branch bounds. -/
noncomputable def select
    {left right : ExactMatrix q n rows columns} (chooseRight : Bool)
    (leftFact : MagnitudeFact left) (rightFact : MagnitudeFact right) :
    MagnitudeFact (if chooseRight then right else left) := by
  classical
  by_cases h : chooseRight
  · simp [h]
    exact weaken rightFact (bound := Nat.max leftFact.bound rightFact.bound)
      (Nat.le_max_right _ _)
  · simp [Bool.eq_false_of_not_eq_true h]
    exact weaken leftFact (bound := Nat.max leftFact.bound rightFact.bound)
      (Nat.le_max_left _ _)

/- Concatenation only relocates entries and inserts zeros. This entrywise form is shared by row,
   column, and diagonal concatenation; their runtime bridges need only prove where each output
   entry came from. -/
theorem matrixNorm_le_max_of_entrywise
    (output : ErrorMatrix n rows columns) {leftRows leftColumns rightRows rightColumns : Nat}
    (left : ErrorMatrix n leftRows leftColumns)
    (right : ErrorMatrix n rightRows rightColumns)
    (entryBound : ∀ row column,
      polyNorm (output row column) ≤ Nat.max (matrixNorm left) (matrixNorm right)) :
    matrixNorm output ≤ Nat.max (matrixNorm left) (matrixNorm right) := by
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  exact entryBound row column

/- This is the common magnitude constructor for row, column, and diagonal concatenation. The
   exact equation comes from the concrete concat evaluator; the only norm obligation is the
   operation-independent entry relocation property above. -/
noncomputable def concat
    {leftRows leftColumns rightRows rightColumns : Nat}
    {actual : ExactMatrix q n rows columns}
    {left : ExactMatrix q n leftRows leftColumns}
    {right : ExactMatrix q n rightRows rightColumns}
    (leftFact : MagnitudeFact left) (rightFact : MagnitudeFact right)
    (lift : ErrorMatrix n rows columns)
    (reduceEq : reduceMatrix q n rows columns lift = actual)
    (entryBound : ∀ row column,
      polyNorm (lift row column) ≤
        Nat.max (matrixNorm leftFact.lift) (matrixNorm rightFact.lift)) :
    MagnitudeFact actual :=
  { lift
    reduce_eq := reduceEq
    bound := Nat.max leftFact.bound rightFact.bound
    norm_le := (matrixNorm_le_max_of_entrywise lift leftFact.lift rightFact.lift entryBound).trans
      (max_le_max leftFact.norm_le rightFact.norm_le)
    support := .arbitrary
    support_valid := by
      intro h
      cases h }

/- Multiplication by a constant-polynomial matrix pays only the matrix inner dimension. In
   particular the selector product `secret * bitValue` has inner dimension one, so two inputs
   bounded by one produce another value bounded by one, with no ring-dimension factor. -/
noncomputable def mulRightConstant
    {left : ExactMatrix q n rows inner} {right : ExactMatrix q n inner columns}
    (hn : 0 < n) (leftFact : MagnitudeFact left) (rightFact : MagnitudeFact right)
    (rightConstant : rightFact.support = .constant) : MagnitudeFact (left * right) :=
  { lift := leftFact.lift * rightFact.lift
    reduce_eq := by
      rw [reduceMatrix_mul, leftFact.reduce_eq, rightFact.reduce_eq]
    bound := inner * leftFact.bound * rightFact.bound
    norm_le := (matrixNorm_mul_right_constant_le hn
      (rightFact.isConstant_of_support_constant rightConstant)).trans (by
        exact Nat.mul_le_mul (Nat.mul_le_mul_left inner leftFact.norm_le) rightFact.norm_le)
    support := .arbitrary
    support_valid := by
      intro h
      cases h }

/- A sequential selector scan is a finite repetition of the same preservation rule. This lemma
   carries the exact value produced at every iteration; it does not replace the IR loop equation. -/
theorem sequentialInvariant
    {Value : Nat → ExactMatrix q n rows columns} {bound : Nat}
    (initial : MagnitudeFact (Value 0)) (initialBound : initial.bound ≤ bound)
    (step : ∀ index, MagnitudeFact (Value index) → MagnitudeFact (Value (index + 1)))
    (stepBound : ∀ index (fact : MagnitudeFact (Value index)),
      fact.bound ≤ bound → (step index fact).bound ≤ bound) :
    ∀ count, ∃ fact : MagnitudeFact (Value count), fact.bound ≤ bound := by
  intro count
  induction count with
  | zero => exact ⟨initial, initialBound⟩
  | succ index ih =>
      obtain ⟨fact, factBound⟩ := ih
      exact ⟨step index fact, stepBound index fact factBound⟩

end SelectorMagnitude

end Mxx.Gadgets
