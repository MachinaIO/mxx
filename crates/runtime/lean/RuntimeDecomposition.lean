import RuntimePrimitives

namespace MxxRuntime
open Mxx.Primitives
open scoped BigOperators

/-- Actual integer digits, retaining row/tower/digit order from the runtime. -/
noncomputable def regularIntegerDigits {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns) :
    ErrorMatrix n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns :=
  fun row column => CrtDecomposition.regularDigitPoly layout.ordered
    ((castExactMatrix layout.product_eq target) ((regularIndexEquiv layout rows).symm row).1 column)
    ((regularIndexEquiv layout rows).symm row).2

/-- The exact dense L1 expression: add sharp digit bounds in tower order, then
multiply by the number of source rows and ring coefficients. -/
def fullDenseRegularGain {q : Nat} (layout : RegularLayout q) (n rows : Nat) : Nat :=
  (∑ tower : Fin layout.crtModuli.length, ∑ digit : Fin layout.digitsPerTower,
    denseDigitBound layout.base (layout.crtModuli.get tower / 2) digit.val) * (rows * n)

theorem regular_digit_sharp_bound {q n : Nat} (layout : RegularLayout q)
    (hn : 0 < n)
    (value : ExactPoly layout.ordered.modulusProduct n)
    (limb : MxxRuntime.RegularLimb layout) (coefficient : Fin n) :
    ((CrtDecomposition.regularDigitPoly layout.ordered value limb).coeff coefficient).natAbs ≤
      denseDigitBound layout.base (layout.crtModuli.get limb.1 / 2) limb.2.val := by
  rw [CrtDecomposition.regularDigitPoly_coeff layout.ordered value limb hn]
  unfold CrtDecomposition.regularDigitCoefficient
  change (balancedDigitAt layout.base
    (centeredLift (layout.crtModuli.get limb.1)
      (layout.ordered.crtEquiv (value.coeff coefficient) limb.1)) limb.2.val).natAbs ≤ _
  exact balancedDigitAt_sharp_bound layout.base_gt_one layout.base_even
    (centeredLift_natAbs_le_half _ (layout.modulus_pos limb.1) _) _

theorem regularIntegerDigits_action_gain {q n rows columns : Nat}
    (layout : RegularLayout q) (hn : 0 < n)
    (target : ExactMatrix q n rows columns) :
    ActionGain (regularIntegerDigits layout target) (fullDenseRegularGain layout n rows) := by
  intro column
  rw [← (regularIndexEquiv layout rows).sum_comp]
  simp only [regularIntegerDigits, Equiv.symm_apply_apply]
  rw [Fintype.sum_prod_type]
  calc
    _ ≤ ∑ row : Fin rows, ∑ limb : MxxRuntime.RegularLimb layout,
        n * denseDigitBound layout.base (layout.crtModuli.get limb.1 / 2) limb.2.val := by
      apply Finset.sum_le_sum
      intro row hr
      apply Finset.sum_le_sum
      intro limb hl
      unfold polynomialL1
      calc
        _ ≤ ∑ coefficient : Fin n,
            denseDigitBound layout.base (layout.crtModuli.get limb.1 / 2) limb.2.val := by
          apply Finset.sum_le_sum
          intro coefficient hc
          exact regular_digit_sharp_bound layout hn _ limb coefficient
        _ = _ := by simp
    _ = fullDenseRegularGain layout n rows := by
      simp only [Fintype.sum_sigma, Finset.mul_sum, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, smul_eq_mul, fullDenseRegularGain]
      simp_rw [← Finset.mul_sum]
      ring

private theorem castExactMatrix_reduce {q q' n rows columns : Nat} (h : q = q')
    (error : ErrorMatrix n rows columns) :
    castExactMatrix h (reduceMatrix q n rows columns error) =
      reduceMatrix q' n rows columns error := by
  cases h
  rfl

/-- The bounded witness reduces to the actual runtime decomposition. -/
theorem regularIntegerDigits_reduce {q n rows columns : Nat}
    (layout : RegularLayout q) (hq : 1 < q) (hn : 0 < n)
    (target : ExactMatrix q n rows columns) :
    fullDecomposeMatrix layout target =
      reduceMatrix q n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns
        (regularIntegerDigits layout target) := by
  have he : (fun row column => regularDigitsUnflattened layout
      (castExactMatrix layout.product_eq target) ((regularIndexEquiv layout rows).symm row) column) =
      reduceMatrix layout.ordered.modulusProduct n
        (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns
        (regularIntegerDigits layout target) := by
    funext row column
    exact CrtDecomposition.regularDigitExactPoly_eq_reduce layout.ordered _ _
      (by rw [layout.ordered.modulusProduct_eq_q]; exact hq) hn
  unfold fullDecomposeMatrix
  rw [he]
  exact castExactMatrix_reduce layout.product_eq.symm _

/-- With no dropped towers, the only difference from the full gadget is a dimension cast. -/
theorem castMatrixColumns_apply {q n rows columns columns' : Nat} (h : columns = columns')
    (value : ExactMatrix q n rows columns) (r : Fin rows) (c : Fin columns') :
    castMatrixColumns h value r c = value r (Fin.cast h.symm c) := by
  cases h
  rfl

theorem castMatrixRows_apply {q n rows rows' columns : Nat} (h : rows = rows')
    (value : ExactMatrix q n rows columns) (r : Fin rows') (c : Fin columns) :
    castMatrixRows h value r c = value (Fin.cast h.symm r) c := by
  cases h
  rfl

@[simp] theorem castMatrixColumns_comp {q n rows a b c : Nat} (h : a = b) (h' : b = c)
    (value : ExactMatrix q n rows a) :
    castMatrixColumns h' (castMatrixColumns h value) = castMatrixColumns (h.trans h') value := by
  cases h
  cases h'
  rfl

@[simp] theorem castMatrixRows_comp {q n columns a b c : Nat} (h : a = b) (h' : b = c)
    (value : ExactMatrix q n a columns) :
    castMatrixRows h' (castMatrixRows h value) = castMatrixRows (h.trans h') value := by
  cases h
  cases h'
  rfl

theorem regularGadgetMatrix_eq_full {q n rows : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) :
    regularGadgetMatrix (n := n) (rows := rows) layout =
      castMatrixColumns (by
        simp [RegularLayout.digitCount, RegularLayout.retainedTowers, hexact])
        (fullGadgetMatrix layout) := by
  funext row column
  rw [castMatrixColumns_apply]
  apply congrArg (fullGadgetMatrix layout row)
  apply Fin.ext
  simp [regularColumnIndex, RegularLayout.digitCount,
    RegularLayout.retainedTowers, hexact, finProdFinEquiv, Nat.mod_add_div]

theorem regularDecomposeMatrix_eq_full {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hexact : layout.droppedModuli = 0) :
    regularDecomposeMatrix layout target =
      castMatrixRows (by
        simp [RegularLayout.digitCount, RegularLayout.retainedTowers, hexact])
        (fullDecomposeMatrix layout target) := by
  funext row column
  rw [castMatrixRows_apply]
  simp only [regularDecomposeMatrix, regularCorrectedTarget, hexact, ↓reduceIte]
  apply congrArg (fun index ↦ fullDecomposeMatrix layout target index column)
  apply Fin.ext
  simp [regularColumnIndex, RegularLayout.digitCount,
    RegularLayout.retainedTowers, hexact, finProdFinEquiv, Nat.mod_add_div]

/-- Integer witness of the retained decomposition of the corrected target. -/
noncomputable def retainedIntegerDigits {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns) :
    ErrorMatrix n (rows * layout.digitCount) columns :=
  fun row column ↦ regularIntegerDigits layout (regularCorrectedTarget layout target)
    (regularColumnIndex layout row) column

/-- Sharp digit maxima summed over the actual retained rows. -/
def denseRegularGain {q : Nat} (layout : RegularLayout q) (n rows : Nat) : Nat :=
  ∑ row : Fin (rows * layout.digitCount),
    n * denseDigitBound layout.base
      (layout.crtModuli.get
        ((regularIndexEquiv layout rows).symm (regularColumnIndex layout row)).2.1 / 2)
      ((regularIndexEquiv layout rows).symm (regularColumnIndex layout row)).2.2.val

theorem retainedIntegerDigits_reduce {q n rows columns : Nat}
    (layout : RegularLayout q) (hq : 1 < q) (hn : 0 < n)
    (target : ExactMatrix q n rows columns) :
    regularDecomposeMatrix layout target =
      reduceMatrix q n (rows * layout.digitCount) columns (retainedIntegerDigits layout target) := by
  funext row column
  exact congrFun (congrFun
    (regularIntegerDigits_reduce layout hq hn (regularCorrectedTarget layout target))
    (regularColumnIndex layout row)) column

theorem retainedIntegerDigits_action_gain {q n rows columns : Nat}
    (layout : RegularLayout q) (hn : 0 < n) (target : ExactMatrix q n rows columns) :
    ActionGain (retainedIntegerDigits layout target) (denseRegularGain layout n rows) := by
  intro column
  apply Finset.sum_le_sum
  intro row hr
  apply polynomialL1_le
  intro coefficient
  exact regular_digit_sharp_bound layout hn _ _ coefficient

theorem regularColumnIndex_injective {q rows : Nat} (layout : RegularLayout q) :
    Function.Injective (regularColumnIndex (rows := rows) layout) := by
  intro a b h
  apply finProdFinEquiv.symm.injective
  have hp := finProdFinEquiv.injective h
  have hfst := congrArg
    (fun x : Fin rows × Fin (layout.crtModuli.length * layout.digitsPerTower) ↦ x.1) hp
  have hsnd := congrArg
    (fun x : Fin rows × Fin (layout.crtModuli.length * layout.digitsPerTower) ↦ x.2) hp
  exact Prod.ext hfst ((Fin.castLE_injective
    (Nat.mul_le_mul_right layout.digitsPerTower
      (Nat.sub_le layout.crtModuli.length layout.droppedModuli))) hsnd)

/-- Restricting a nonnegative column sum cannot increase its action gain. -/
theorem retainedIntegerDigits_action_gain_of_full {q n rows columns gain : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns)
    (hgain : ActionGain (regularIntegerDigits layout
      (regularCorrectedTarget layout target)) gain) :
    ActionGain (retainedIntegerDigits layout target) gain := by
  classical
  intro column
  let f := fun row ↦ polynomialL1 (regularIntegerDigits layout
    (regularCorrectedTarget layout target) row column)
  change (∑ row, f (regularColumnIndex layout row)) ≤ gain
  calc
    _ = ∑ row ∈ Finset.univ.image (regularColumnIndex layout), f row := by
      rw [Finset.sum_image]
      intro a ha b hb h
      exact regularColumnIndex_injective layout h
    _ ≤ ∑ row, f row := Finset.sum_le_sum_of_subset_of_nonneg
      (Finset.subset_univ _) (by intros; exact Nat.zero_le _)
    _ ≤ gain := hgain column

/-- Consume the actual runtime relation without discarding its retained layout. -/
theorem gadgetDecomposeRuns_dense_gain {q n rows columns inner : Nat}
    {backend : BackendContext} (layout : RegularLayout q)
    (hlookup : backend.regularLayout q n = some layout)
    (hq : 1 < q) (hn : 0 < n) {base digits : Int}
    {target : ExactMatrix q n rows columns} {decomposition : ExactMatrix q n inner columns}
    (hrun : gadgetDecomposeRuns backend base digits target decomposition) :
    ∃ witness : ErrorMatrix n inner columns,
      decomposition = reduceMatrix q n inner columns witness ∧
      ActionGain witness (denseRegularGain layout n rows) := by
  obtain ⟨other, hother, _, _, hwidth, hd, _⟩ := hrun
  have hl : other = layout := Option.some.inj (hother.symm.trans hlookup)
  subst other
  subst inner
  simp only [castMatrixRows] at hd
  rw [hd]
  exact ⟨retainedIntegerDigits layout target, retainedIntegerDigits_reduce layout hq hn target,
    retainedIntegerDigits_action_gain layout hn target⟩

/-- Repeated gadget construction in producer and consumer selects one matrix
from their shared backend, independently of the surrounding sampler runs. -/
theorem gadget_runs_deterministic {q n rows columns : Nat}
    {backend : BackendContext} {base digits : Int}
    {first second : ExactMatrix q n rows columns}
    (hfirst : gadgetMatrixRuns backend base digits first)
    (hsecond : gadgetMatrixRuns backend base digits second) : first = second := by
  obtain ⟨layout, hlookup, _, _, hwidth, hf⟩ := hfirst
  obtain ⟨other, hother, _, _, hwidth', hs⟩ := hsecond
  have hl : other = layout := Option.some.inj (hother.symm.trans hlookup)
  subst other
  exact hf.trans hs.symm

/-- Reconstruction is derived from the two actual runtime gadget relations,
which resolve the same backend layout. -/
theorem gadget_runs_reconstruct {q n rows columns inner : Nat}
    (hq : 1 < q) (hn : 0 < n) {backend : BackendContext} {base digits : Int}
    {gadget : ExactMatrix q n rows inner} {target : ExactMatrix q n rows columns}
    {decomposition : ExactMatrix q n inner columns}
    (hgadget : gadgetMatrixRuns backend base digits gadget)
    (hdecomposition : gadgetDecomposeRuns backend base digits target decomposition)
    (hexact : (backend.regularLayout q n).map RegularLayout.droppedModuli = some 0) :
    gadget * decomposition = target := by
  obtain ⟨layout, hlookup, _, _, hwidth, hg⟩ := hgadget
  obtain ⟨other, hother, _, _, hwidth', hd, _⟩ := hdecomposition
  have hl : other = layout := Option.some.inj (hother.symm.trans hlookup)
  subst other
  subst inner
  simp only [castMatrixColumns, castMatrixRows] at hg hd
  rw [hg, hd]
  apply regularGadgetMatrix_reconstruct layout target hq hn
  simpa only [hlookup, Option.map_some, Option.some.injEq] using hexact

end MxxRuntime
