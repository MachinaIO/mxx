import ExponentSparseDecomposition

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime
open scoped BigOperators

private theorem cast_smul {q q' n rows columns : Nat} (h : q = q')
    (mu : Int) (matrix : ExactMatrix q n rows columns) :
    castExactMatrix h (mu • matrix) = mu • castExactMatrix h matrix := by
  cases h
  rfl

private theorem digit_zero (base digit : Nat) : balancedDigitAt base 0 digit = 0 := by
  have hb : balancedDigit base 0 = 0 := by
    unfold balancedDigit
    simp only [Int.zero_emod, Int.zero_ediv]
    split_ifs <;> omega
  have hr : ∀ d, balancedResidual base 0 d = 0 := by
    intro d
    induction d with
    | zero => rfl
    | succ d ih => simpa [balancedResidual, hb] using ih
  simp [balancedDigitAt, hr, hb]

/-- The scaled runtime gadget has one source row, coefficient and CRT tower. -/
theorem refresh_crt_coefficient {q n rows : Nat} (layout : RegularLayout q)
    (hq : 1 < q) (hn : 0 < n) (mu : Int) (r : Fin rows)
    (c : Fin (rows * (layout.crtModuli.length * layout.digitsPerTower)))
    (tower : Fin layout.crtModuli.length) (k : Fin n) :
    layout.ordered.crtEquiv
      (((castExactMatrix layout.product_eq (mu • regularGadgetMatrix layout)) r c).coeff k)
      tower =
    if r = ((regularIndexEquiv layout rows).symm c).1 ∧ k = ⟨0, hn⟩ ∧
        tower = ((regularIndexEquiv layout rows).symm c).2.1 then
      ((mu * (layout.base : Int) ^ ((regularIndexEquiv layout rows).symm c).2.2.val : Int) :
        ZMod (layout.crtModuli.get tower)) else 0 := by
  classical
  have hp : 1 < layout.ordered.modulusProduct := by
    rw [layout.ordered.modulusProduct_eq_q]; exact hq
  letI : Fact (1 < layout.ordered.modulusProduct) := ⟨hp⟩
  rw [cast_smul]
  unfold regularGadgetMatrix
  rw [castExactMatrix_cancel]
  change layout.ordered.crtEquiv
    ((mu • regularGadgetUnflattened layout r ((regularIndexEquiv layout rows).symm c)).coeff k)
    tower = _
  unfold regularGadgetUnflattened
  by_cases hr : r = ((regularIndexEquiv layout rows).symm c).1
  · rw [if_pos hr]
    rw [zsmul_eq_mul, ← map_intCast (algebraMap (ZMod layout.ordered.modulusProduct)
      (ExactPoly layout.ordered.modulusProduct n)), ← map_mul]
    have hc (a : ZMod layout.ordered.modulusProduct) :
        (algebraMap _ (ExactPoly layout.ordered.modulusProduct n) a).coeff k =
          if k = ⟨0, hn⟩ then a else 0 := by
      have hh := Negacyclic.coeff_smul a
        ((AdjoinRoot.root (negacyclicModulus n (ZMod layout.ordered.modulusProduct))) ^
          (⟨0, hn⟩ : Fin n).val) k
      rw [Negacyclic.coeff_root_pow hn] at hh
      simpa only [Fin.val_zero, pow_zero, mul_one, mul_ite, mul_zero, eq_comm] using hh
    rw [hc]
    by_cases hk : k = ⟨0, hn⟩
    · rw [if_pos hk]
      simp only [CrtDecomposition.regularGadgetWeight]
      have he : (mu : ZMod layout.ordered.modulusProduct) *
          (layout.ordered.crtWeight ((regularIndexEquiv layout rows).symm c).2.1 *
            (layout.ordered.base : ZMod layout.ordered.modulusProduct) ^
              ((regularIndexEquiv layout rows).symm c).2.2.val) =
          layout.ordered.crtWeight ((regularIndexEquiv layout rows).symm c).2.1 *
            ((mu * (layout.base : Int) ^ ((regularIndexEquiv layout rows).symm c).2.2.val : Int) :
              ZMod layout.ordered.modulusProduct) := by
        simp only [Int.cast_mul, Int.cast_pow, Int.cast_natCast]
        change (mu : ZMod layout.ordered.modulusProduct) * (_ * _) = _ * ((mu : ZMod layout.ordered.modulusProduct) * _)
        change (mu : ZMod layout.ordered.modulusProduct) * (_ * (layout.base : ZMod layout.ordered.modulusProduct) ^ _) = _
        ring
      rw [he, layout.ordered.crtWeight_intCast_apply]
      simp [hr, hk]
    · simp [hk]
  · simp [hr]

/-- Summing the actual digits costs neither a source-row nor a ring-degree
factor. The selected CRT tower is bounded by the full scalar CRT profile. -/
theorem regularIntegerDigits_refresh_gain {q n rows : Nat} (layout : RegularLayout q)
    (hq : 1 < q) (hn : 0 < n) (mu : Int) :
    ActionGain (regularIntegerDigits layout (mu • regularGadgetMatrix (n := n) (rows := rows) layout))
      (Finset.univ.sup (fun d : Fin layout.digitsPerTower =>
        sparseColumnGain layout (mu * (layout.base : Int) ^ d.val))) := by
  classical
  intro column
  let index := (regularIndexEquiv layout rows).symm column
  have hb (r : Fin rows) (limb : MxxRuntime.RegularLimb layout) (k : Fin n) :
      ((CrtDecomposition.regularDigitPoly layout.ordered
        ((castExactMatrix layout.product_eq (mu • regularGadgetMatrix layout)) r column)
        limb).coeff k).natAbs ≤
      if r = index.1 ∧ k = ⟨0, hn⟩ then
        (balancedDigitAt layout.base
          (centeredLift (layout.crtModuli.get limb.1)
            ((mu * (layout.base : Int) ^ index.2.2.val : Int) :
              ZMod (layout.crtModuli.get limb.1))) limb.2.val).natAbs else 0 := by
    rw [CrtDecomposition.regularDigitPoly_coeff layout.ordered _ limb hn]
    unfold CrtDecomposition.regularDigitCoefficient
    rw [refresh_crt_coefficient layout hq hn]
    change (balancedDigitAt layout.base
      (centeredLift (layout.crtModuli.get limb.1)
        (if r = index.1 ∧ k = ⟨0, hn⟩ ∧ limb.1 = index.2.1 then _ else 0))
      limb.2.val).natAbs ≤ _
    split_ifs <;> simp_all [centeredLift, digit_zero, index]
  rw [← (regularIndexEquiv layout rows).sum_comp]
  simp only [regularIntegerDigits, Equiv.symm_apply_apply, polynomialL1]
  rw [Fintype.sum_prod_type]
  calc
    _ ≤ ∑ r : Fin rows, ∑ limb : MxxRuntime.RegularLimb layout, ∑ k : Fin n,
        if r = index.1 ∧ k = ⟨0, hn⟩ then
          (balancedDigitAt layout.base
            (centeredLift (layout.crtModuli.get limb.1)
              ((mu * (layout.base : Int) ^ index.2.2.val : Int) :
                ZMod (layout.crtModuli.get limb.1))) limb.2.val).natAbs else 0 := by
      apply Finset.sum_le_sum
      intro r hr
      apply Finset.sum_le_sum
      intro limb hl
      apply Finset.sum_le_sum
      intro k hk
      exact hb r limb k
    _ = sparseColumnGain layout (mu * (layout.base : Int) ^ index.2.2.val) := by
      simp [ite_and, sparseColumnGain, Fintype.sum_sigma]
    _ ≤ _ := Finset.le_sup (f := fun d : Fin layout.digitsPerTower =>
        sparseColumnGain layout (mu * (layout.base : Int) ^ d.val))
      (Finset.mem_univ index.2.2)

/-- Exact GadgetScalar gain from both actual registered runtime executions. -/
theorem gadgetRuns_refresh_gain {q n rows columns inner : Nat}
    {backend : BackendContext} (layout : RegularLayout q)
    (hlookup : backend.regularLayout q n = some layout) (hq : 1 < q) (hn : 0 < n)
    {base digits : Int} (mu : Int) {G : ExactMatrix q n rows columns}
    {D : ExactMatrix q n inner columns}
    (hg : gadgetMatrixRuns backend base digits G)
    (hd : gadgetDecomposeRuns backend base digits (mu • G) D) :
    ∃ action : ErrorMatrix n inner columns,
      D = reduceMatrix q n inner columns action ∧
      ActionGain action (Finset.univ.sup (fun d : Fin layout.digitsPerTower =>
        sparseColumnGain layout (mu * (layout.base : Int) ^ d.val))) := by
  obtain ⟨gl, hgl, _, _, hcolumns, hg⟩ := hg
  have he : gl = layout := Option.some.inj (hgl.symm.trans hlookup)
  subst gl
  subst columns
  simp only [castMatrixColumns] at hg
  subst G
  obtain ⟨dl, hdl, _, _, hinner, hd⟩ := hd
  have he : dl = layout := Option.some.inj (hdl.symm.trans hlookup)
  subst dl
  subst inner
  simp only [castMatrixRows] at hd
  rw [hd]
  exact ⟨regularIntegerDigits layout (mu • regularGadgetMatrix layout),
    regularIntegerDigits_reduce layout hq hn _,
    regularIntegerDigits_refresh_gain layout hq hn mu⟩

#print axioms refresh_crt_coefficient
#print axioms regularIntegerDigits_refresh_gain
#print axioms gadgetRuns_refresh_gain

end MxxExponentLut
