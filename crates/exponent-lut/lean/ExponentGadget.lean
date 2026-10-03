import ExponentEncoding

/-!
# Gadget actions

A stacked right action is the gadget decomposition of a two-row matrix `X` at the next modulus
`q'`, re-encoded at `q` (`compactCenteredRebaseRuns`). Applied to a target `(p x_k) · g` whose gadget
`g` is the next modulus's gadget re-encoded at `q = p q'`, it returns `scaleUp` of `x₀ X₀ + x₁ X₁`:
the decomposition reconstructs `X` modulo `q'`, and the factor `p` cancels every multiple of `q'`.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- Row `k` of a matrix as a `1 × columns` block. -/
def rowOf {α : Type} {rows columns : Nat} (x : Matrix (Fin rows) (Fin columns) α) (k : Fin rows) :
    Matrix (Fin 1) (Fin columns) α :=
  fun _ column ↦ x k column

/-- A runtime gadget matrix is determined by its backend. -/
theorem gadgetMatrixRuns_unique {q rows columns : Nat} {backend : BackendContext}
    {base digits : Int} {g h : ExactMatrix q n rows columns}
    (hg : gadgetMatrixRuns backend base digits g) (hh : gadgetMatrixRuns backend base digits h) :
    g = h := by
  obtain ⟨layout, hl, _, _, hw, hg⟩ := hg
  obtain ⟨layout', hl', _, _, hw', hh⟩ := hh
  rw [hl] at hl'
  cases hl'
  rw [hg, hh]

theorem centeredRebaseRuns_unique {q p rows columns : Nat} {x : ExactMatrix q n rows columns}
    {y z : ExactMatrix p n rows columns} (hy : centeredRebaseRuns x y)
    (hz : centeredRebaseRuns x z) : y = z := hy.2.trans hz.2.symm

/-- Rebasing to the same modulus is the identity. -/
theorem centeredRebaseRuns_self {q rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (x : ExactMatrix q n rows columns) : centeredRebaseRuns x x := by
  refine ⟨hq, ?_⟩
  funext row column
  rw [centeredRebase_eq, reducePoly_centeredIntLift hq hn]

/-! ## Reconstruction at the decomposition modulus -/

/-- The one-row gadget times the digits of a two-row target, scaled per row. -/
theorem gadget_pairMul_decompose {q' m m' : Nat} (hq' : 1 < q') (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g : ExactMatrix q' n 1 m} (hg : gadgetMatrixRuns backend base digits g)
    {X : ExactMatrix q' n 2 m'} {D : ExactMatrix q' n (m + m) m'}
    (hD : gadgetDecomposeRuns backend base digits X D) (x : Fin 2 → ExactPoly q' n) :
    pairMul (fun k ↦ x k • g) D = x 0 • rowOf X 0 + x 1 • rowOf X 1 := by
  obtain ⟨layout₁, hl₁, _, _, hw₁, hg⟩ := hg
  obtain ⟨layout₂, hl₂, _, _, hw₂, hD, _⟩ := hD
  rw [hlayout] at hl₁ hl₂
  cases hl₁
  cases hl₂
  have hd : m = layout.digitCount := by simpa using hw₁
  subst hd
  have hrec := regularGadgetMatrix_reconstruct (rows := 2) layout X hq' hn hexact
  funext row column
  have hrow : row = 0 := Subsingleton.elim _ _
  subst hrow
  simp only [pairMul, Matrix.add_apply, Matrix.mul_apply, topRows, bottomRows,
    Matrix.submatrix_apply, id, Matrix.smul_apply, smul_eq_mul, rowOf, hg, hD,
    castMatrixColumns_apply, castMatrixRows_apply]
  rw [← congrFun (congrFun hrec 0) column, ← congrFun (congrFun hrec 1) column]
  simp only [Matrix.mul_apply, regularGadgetMatrix_two_rows layout hexact]
  rw [Finset.mul_sum, Finset.mul_sum]
  have hsplit (f : Fin (2 * layout.digitCount) → ExactPoly q' n) :
      ∑ c, f c = ∑ i : Fin layout.digitCount,
          f ⟨i.val, lt_of_lt_of_le i.isLt (Nat.le_mul_of_pos_left _ two_pos)⟩ +
        ∑ i : Fin layout.digitCount,
          f ⟨layout.digitCount + i.val, by rw [two_mul]; exact Nat.add_lt_add_left i.isLt _⟩ := by
    rw [← Fin.sum_congr' f
      (show layout.digitCount + layout.digitCount = 2 * layout.digitCount by omega),
      Fin.sum_univ_add]
    rfl
  have hlow (i : Fin layout.digitCount) : i.val / layout.digitCount = 0 :=
    Nat.div_eq_of_lt i.isLt
  have hhigh (i : Fin layout.digitCount) : (layout.digitCount + i.val) / layout.digitCount = 1 := by
    rw [Nat.add_div_left _ (by have := i.isLt; omega), hlow, zero_add]
  have hlowMod (i : Fin layout.digitCount) : i.val % layout.digitCount = i.val :=
    Nat.mod_eq_of_lt i.isLt
  have hhighMod (i : Fin layout.digitCount) :
      (layout.digitCount + i.val) % layout.digitCount = i.val := by
    rw [Nat.add_mod_left, hlowMod]
  rw [hsplit, hsplit]
  simp only [hlow, hhigh, hlowMod, hhighMod, Fin.val_zero, Fin.val_one, if_true,
    one_ne_zero, zero_ne_one, if_false, zero_mul, mul_zero, Finset.sum_const_zero, add_zero,
    zero_add]
  congr 1
  · apply Finset.sum_congr rfl
    intro i _
    rw [mul_assoc]
    congr 3
  · apply Finset.sum_congr rfl
    intro i _
    rw [mul_assoc]
    congr 3

theorem reduceMatrix_smul {q rows columns : Nat} (a : ErrorPoly n) (x : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (a • x) = reducePoly q n a • reduceMatrix q n rows columns x := by
  funext row column
  simp [reduceMatrix]

/-- A matrix is the reduction of the centered lifts of its entries. -/
theorem reduceMatrix_centeredIntLift {q rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (x : ExactMatrix q n rows columns) :
    reduceMatrix q n rows columns (fun row column ↦ centeredIntLift (x row column)) = x := by
  funext row column
  exact reducePoly_centeredIntLift hq hn _

/-! ## Scaled action at the evaluation modulus -/

/-- The next modulus's gadget re-encoded at `q = p q'`, scaled by `p x_k`, times the re-encoded
digits of `X`, is `scaleUp` of `x₀ X₀ + x₁ X₁`. -/
theorem gadget_action_scaleUp {q p q' m m' : Nat} (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {X : ExactMatrix q' n 2 m'} {D' : ExactMatrix q' n (m + m) m'}
    (hD' : gadgetDecomposeRuns backend base digits X D')
    {D : ExactMatrix q n (m + m) m'} {bound : Nat} (hD : compactCenteredRebaseRuns bound D' D)
    (x : Fin 2 → ErrorPoly n) :
    pairMul (fun k ↦ reducePoly q n (((p : Int) : ErrorPoly n) * x k) • g) D =
      scaleUpMatrix q p (reducePoly q' n (x 0) • rowOf X 0 + reducePoly q' n (x 1) • rowOf X 1) := by
  let G : ErrorMatrix n 1 m := fun row column ↦ centeredIntLift (g' row column)
  let E : ErrorMatrix n (m + m) m' := fun row column ↦ centeredIntLift (D' row column)
  have hgq : g = reduceMatrix q n 1 m G := centeredRebaseRuns_reduce hg
  have hDq : D = reduceMatrix q n (m + m) m' E := centeredRebaseRuns_reduce hD.2.2.1
  have hgq' : g' = reduceMatrix q' n 1 m G := (reduceMatrix_centeredIntLift hq' hn g').symm
  have hDq' : D' = reduceMatrix q' n (m + m) m' E := (reduceMatrix_centeredIntLift hq' hn D').symm
  have hrec := gadget_pairMul_decompose hq' hn layout hlayout hexact hg' hD'
    (fun k ↦ reducePoly q' n (x k))
  simp only at hrec
  rw [← hrec]
  have hleft : (fun k ↦ reducePoly q n (((p : Int) : ErrorPoly n) * x k) • g) =
      fun k ↦ reduceMatrix q n 1 m ((((p : Int) : ErrorPoly n) * x k) • G) := by
    funext k
    rw [hgq, reduceMatrix_smul]
  have hright : (fun k ↦ reducePoly q' n (x k) • g') =
      fun k ↦ reduceMatrix q' n 1 m (x k • G) := by
    funext k
    rw [hgq', reduceMatrix_smul]
  rw [hleft, hright, hDq, hDq', ← reduceMatrix_pairMul, ← reduceMatrix_pairMul]
  have hpair : (fun k ↦ (((p : Int) : ErrorPoly n) * x k) • G) =
      ((p : Int) : ErrorPoly n) • fun k ↦ x k • G := by
    funext k
    rw [Pi.smul_apply, smul_smul]
  rw [hpair, pairMul_smul]
  funext row column
  simp only [scaleUpMatrix, reduceMatrix, Matrix.smul_apply, smul_eq_mul]
  rw [scaleUp_reducePoly hq hq' hn]

end MxxExponentLut
