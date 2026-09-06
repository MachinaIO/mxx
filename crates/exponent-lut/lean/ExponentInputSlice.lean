import ExponentSetup

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime
open scoped BigOperators

/-- Selecting the right encoding block preserves the sampled integer error and
its coefficient bound, using the actual runtime concat and tensor layout. -/
theorem initial_right_block {q n k columns bound : Nat}
    {secret : ExactMatrix q n 1 k}
    {leftPublic rightPublic : ExactMatrix q n k columns}
    {publicBatch : ExactMatrix q n k (columns + columns)}
    {leftMessage rightMessage : ExactMatrix q n 1 1}
    {messages : ExactMatrix q n 1 2}
    {payload : ExactMatrix q n 1 columns}
    {tensor batch : ExactMatrix q n 1 (columns + columns)}
    {actual : ExactMatrix q n 1 columns}
    (hpublic : concatColumns leftPublic rightPublic publicBatch)
    (hmessages : concatColumns leftMessage rightMessage messages)
    (htensor : tensorRuns messages payload
      (fun row column => tensor row ⟨column.val, by omega⟩))
    (hslice : sliceMatrix batch 0 1 (columns : Int) ((columns + columns : Nat) : Int) actual)
    (hbatch : ∃ error : ErrorMatrix n 1 (columns + columns),
      batch = secret * publicBatch - tensor + reduceMatrix q n 1 (columns + columns) error ∧
      CoeffBound error bound) :
    Approx actual (secret * rightPublic - (rightMessage 0 0) • payload) bound := by
  obtain ⟨error, heq, hbound⟩ := hbatch
  let columnIndex (column : Fin columns) : Fin (columns + columns) :=
    ⟨columns + column.val, by omega⟩
  let selected : ErrorMatrix n 1 columns := fun row column => error row (columnIndex column)
  refine ⟨selected, ?_, fun row column index => hbound row (columnIndex column) index⟩
  ext row column
  have hr : row = 0 := Subsingleton.elim _ _
  subst row
  have hp (i : Fin k) : publicBatch i (columnIndex column) = rightPublic i column := by
    simpa [concatColumns, columnIndex, Nat.not_lt.mpr (Nat.le_add_right _ _)] using
      hpublic i (columnIndex column)
  have hm : messages 0 1 = rightMessage 0 0 := by
    simpa [concatColumns] using hmessages 0 1
  have ht : tensor 0 (columnIndex column) = rightMessage 0 0 * payload 0 column := by
    simpa [columnIndex, hm] using htensor 0 0 1 column
  have hs : actual 0 column = batch 0 (columnIndex column) := by
    simpa [columnIndex] using hslice.2.2.2.2.2.2.2.2 0 column (by simp) (by simp)
  rw [hs, heq]
  simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.mul_apply, hp, ht,
    Matrix.smul_apply, smul_eq_mul]
  rfl

end MxxExponentLut
