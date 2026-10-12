import TopLemmas

/-!
# Row locality of the gadget decomposition

The digits of row `r` of a decomposed matrix depend only on row `r` of the target: the
decomposition of two stacked rows is the stack of their decompositions.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem castExactMatrix_entry_congr {q q' rows rows' columns : Nat} (h : q = q')
    {M : ExactMatrix q N rows columns} {M' : ExactMatrix q N rows' columns} {i : Fin rows}
    {i' : Fin rows'} {j j' : Fin columns} (he : M i j = M' i' j') :
    castExactMatrix h M i j = castExactMatrix h M' i' j' := by
  subst h
  exact he

theorem regularColumnIndex_pair {q rows : Nat} (layout : RegularLayout q) (r : Fin rows)
    (d : Fin layout.digitCount) :
    regularColumnIndex layout (finProdFinEquiv (r, d)) =
      finProdFinEquiv (r, Fin.castLE (Nat.mul_le_mul_right layout.digitsPerTower
        (Nat.sub_le layout.crtModuli.length layout.droppedModuli)) d) := by
  simp [regularColumnIndex]

theorem regularIndexEquiv_symm_pair {q rows : Nat} (layout : RegularLayout q) (r : Fin rows)
    (x : Fin (layout.crtModuli.length * layout.digitsPerTower)) :
    (regularIndexEquiv layout rows).symm (finProdFinEquiv (r, x)) =
      (r, ((Equiv.sigmaEquivProd _ _).trans finProdFinEquiv).symm x) := by
  simp [regularIndexEquiv]

/-- The digits of a row depend only on the row. -/
theorem regularDecompose_row_local {q rows rows' columns : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) (T : ExactMatrix q N rows columns)
    (T' : ExactMatrix q N rows' columns) (r : Fin rows) (r' : Fin rows')
    (hrow : ∀ c, T r c = T' r' c) (d : Fin layout.digitCount) (c : Fin columns) :
    regularDecomposeMatrix layout T (finProdFinEquiv (r, d)) c =
      regularDecomposeMatrix layout T' (finProdFinEquiv (r', d)) c := by
  unfold regularDecomposeMatrix regularCorrectedTarget
  simp only [hexact, if_true]
  rw [regularColumnIndex_pair, regularColumnIndex_pair]
  unfold fullDecomposeMatrix
  apply castExactMatrix_entry_congr
  rw [regularIndexEquiv_symm_pair, regularIndexEquiv_symm_pair]
  simp only [regularDigitsUnflattened]
  congr 1
  exact castExactMatrix_entry_congr _ (hrow c)

/-- The decomposition of two stacked rows is the stack of their decompositions. -/
theorem decompose_stack {q m columns : Nat} {backend : BackendContext} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q N = some layout)
    (hexact : layout.droppedModuli = 0)
    {x y : ExactMatrix q N 1 columns} {z : ExactMatrix q N (1 + 1) columns} (hz : concatRows x y z)
    {Dz : ExactMatrix q N (m + m) columns} {Dx Dy : ExactMatrix q N m columns}
    (hDz : gadgetDecomposeRuns backend base digits z Dz)
    (hDx : gadgetDecomposeRuns backend base digits x Dx)
    (hDy : gadgetDecomposeRuns backend base digits y Dy) :
    topRows Dz = Dx ∧ bottomRows Dz = Dy := by
  obtain ⟨l₁, hl₁, _, _, hw₁, hD₁, _⟩ := hDz
  obtain ⟨l₂, hl₂, _, _, hw₂, hD₂, _⟩ := hDx
  obtain ⟨l₃, hl₃, _, _, hw₃, hD₃, _⟩ := hDy
  rw [hlayout] at hl₁ hl₂ hl₃
  cases hl₁
  cases hl₂
  cases hl₃
  have hm : m = layout.digitCount := by simpa using hw₂
  subst hD₁ hD₂ hD₃
  refine ⟨?_, ?_⟩
  · funext i c
    simp only [topRows, Matrix.submatrix_apply, id, castMatrixRows_apply]
    have hi : (Fin.cast hw₁ (Fin.castAdd m i) : Fin ((1 + 1) * layout.digitCount)) =
        finProdFinEquiv ((0 : Fin (1 + 1)), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
    have hi' : (Fin.cast hw₂ i : Fin (1 * layout.digitCount)) =
        finProdFinEquiv ((0 : Fin 1), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
    rw [hi, hi']
    exact regularDecompose_row_local layout hexact z x 0 0
      (fun c ↦ by rw [hz 0 c]; simp) _ c
  · funext i c
    simp only [bottomRows, Matrix.submatrix_apply, id, castMatrixRows_apply]
    have hi : (Fin.cast hw₁ (Fin.natAdd m i) : Fin ((1 + 1) * layout.digitCount)) =
        finProdFinEquiv ((1 : Fin (1 + 1)), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
      omega
    have hi' : (Fin.cast hw₃ i : Fin (1 * layout.digitCount)) =
        finProdFinEquiv ((0 : Fin 1), (Fin.cast hm i)) := by
      ext
      simp [finProdFinEquiv]
    rw [hi, hi']
    exact regularDecompose_row_local layout hexact z y 1 0
      (fun c ↦ by rw [hz 1 c]; simp) _ c

end RefreshTopProof
