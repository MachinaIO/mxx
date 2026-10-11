import TopDecompose
import ExponentStacking
import ExponentScale

/-!
# Key consistency

The preprocessing recomputes every public key of the online evaluation from two-row stacks of the
keys, while the online evaluation slices them out of three-row stacks of the encodings. These
lemmas relate the two.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The compact rebase acts entrywise, so it commutes with taking the top or bottom rows. -/
theorem rebase_topRows {q p m columns bound bound' : Nat} {D' : ExactMatrix q N (m + m) columns}
    {D : ExactMatrix p N (m + m) columns} (hD : compactCenteredRebaseRuns bound D' D)
    {X' : ExactMatrix q N m columns} {X : ExactMatrix p N m columns}
    (hX : compactCenteredRebaseRuns bound' X' X) (h : topRows D' = X') : topRows D = X := by
  rw [centeredRebaseRuns_reduce hD.2.2.1, centeredRebaseRuns_reduce hX.2.2.1, ← h]
  rfl

theorem rebase_bottomRows {q p m columns bound bound' : Nat} {D' : ExactMatrix q N (m + m) columns}
    {D : ExactMatrix p N (m + m) columns} (hD : compactCenteredRebaseRuns bound D' D)
    {X' : ExactMatrix q N m columns} {X : ExactMatrix p N m columns}
    (hX : compactCenteredRebaseRuns bound' X' X) (h : bottomRows D' = X') : bottomRows D = X := by
  rw [centeredRebaseRuns_reduce hD.2.2.1, centeredRebaseRuns_reduce hX.2.2.1, ← h]
  rfl

/-- A compact rebase within one modulus is the identity. -/
theorem rebase_self {q rows columns bound : Nat} (hq : 1 < q) {D' D : ExactMatrix q N rows columns}
    (hD : compactCenteredRebaseRuns bound D' D) : D = D' := by
  rw [centeredRebaseRuns_reduce hD.2.2.1]
  funext r c
  exact reducePoly_centeredIntLift hq hN _

/-- Row `k` of a two-row stack times `D`. -/
theorem stack2_mul_rows {q columns m' : Nat} {x y : ExactMatrix q N 1 columns}
    {z : ExactMatrix q N (1 + 1) columns} (hz : concatRows x y z) (D : ExactMatrix q N columns m') :
    rowOf (z * D) 0 = x * D ∧ rowOf (z * D) 1 = y * D := by
  obtain ⟨h0, h1⟩ := concatRows_two hz
  exact ⟨by rw [rowOf_mul, h0], by rw [rowOf_mul, h1]⟩

theorem rowOf_add {q rows columns : Nat} (x y : ExactMatrix q N rows columns) (k : Fin rows) :
    rowOf (x + y) k = rowOf x k + rowOf y k := by
  funext r c
  simp [rowOf]

/-- A one-row slice is the row. -/
theorem slice_rowOf {q rows columns : Nat} {x : ExactMatrix q N rows columns}
    {out : ExactMatrix q N 1 columns} {r : Nat} (hr : r < rows)
    (h : sliceMatrix x r (r + 1) 0 columns out) : out = rowOf x ⟨r, hr⟩ :=
  sliceMatrix_row (k := ⟨r, hr⟩) h

end RefreshTopProof
