import ExponentDecomposition

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime
open scoped BigOperators

/-- Exact sparse target contract: at most one source row and polynomial
coefficient per output column, with zero values permitted. -/
def SparseTarget {q n rows columns : Nat} (target : ExactMatrix q n rows columns)
    (row : Fin columns → Fin rows) (position : Fin columns → Fin n)
    (value : Fin columns → Int) : Prop :=
  ∀ r c k, (target r c).coeff k =
    if r = row c ∧ k = position c then (value c : ZMod q) else 0

private theorem cast_sparse {q q' n rows columns : Nat} (h : q = q')
    {target : ExactMatrix q n rows columns} {row : Fin columns → Fin rows}
    {position : Fin columns → Fin n} {value : Fin columns → Int}
    (hsparse : SparseTarget target row position value) :
    SparseTarget (castExactMatrix h target) row position value := by
  cases h
  exact hsparse

private theorem balancedDigit_zero (base : Nat) : balancedDigit base 0 = 0 := by
  dsimp only [balancedDigit]
  simp only [Int.zero_emod, Int.zero_ediv]
  split_ifs <;> omega

private theorem balancedResidual_zero (base digit : Nat) : balancedResidual base 0 digit = 0 := by
  induction digit with
  | zero => rfl
  | succ digit ih => simpa only [balancedResidual, balancedDigit_zero, sub_self, Int.zero_ediv] using ih

private theorem balancedDigitAt_zero (base digit : Nat) : balancedDigitAt base 0 digit = 0 := by
  simp [balancedDigitAt, balancedResidual_zero, balancedDigit_zero]

/-- Actual tower profile, including centered reduction and balanced carry. -/
def sparseColumnGain {q : Nat} (layout : RegularLayout q) (value : Int) : Nat :=
  ∑ tower : Fin layout.crtModuli.length, ∑ digit : Fin layout.digitsPerTower,
    (balancedDigitAt layout.base
      (centeredLift (layout.crtModuli.get tower) (value : ZMod (layout.crtModuli.get tower)))
      digit.val).natAbs

def sparseRegularGain {q columns : Nat} (layout : RegularLayout q)
    (value : Fin columns → Int) : Nat :=
  Finset.univ.sup (fun column => sparseColumnGain layout (value column))

private theorem sparse_digit_coefficient {q n rows columns : Nat}
    (layout : RegularLayout q) (hn : 0 < n)
    {target : ExactMatrix q n rows columns} {row : Fin columns → Fin rows}
    {position : Fin columns → Fin n} {value : Fin columns → Int}
    (hsparse : SparseTarget target row position value)
    (r : Fin rows) (c : Fin columns) (limb : MxxRuntime.RegularLimb layout) (k : Fin n) :
    (CrtDecomposition.regularDigitPoly layout.ordered
      ((castExactMatrix layout.product_eq target) r c) limb).coeff k =
    if r = row c ∧ k = position c then
      balancedDigitAt layout.base
        (centeredLift (layout.crtModuli.get limb.1)
          (value c : ZMod (layout.crtModuli.get limb.1))) limb.2.val else 0 := by
  rw [CrtDecomposition.regularDigitPoly_coeff layout.ordered _ limb hn]
  unfold CrtDecomposition.regularDigitCoefficient
  rw [cast_sparse layout.product_eq hsparse r c k]
  split_ifs with h
  · simp only [map_intCast, Pi.intCast_apply]
    rfl
  · simp [centeredLift, balancedDigitAt_zero]

/-- The full column sum of the actual integer witness equals the sparse CRT
profile; no extra source-row or ring-dimension factor is introduced. -/
theorem regularIntegerDigits_sparse_column {q n rows columns : Nat}
    (layout : RegularLayout q) (hn : 0 < n)
    {target : ExactMatrix q n rows columns} {row : Fin columns → Fin rows}
    {position : Fin columns → Fin n} {value : Fin columns → Int}
    (hsparse : SparseTarget target row position value) (column : Fin columns) :
    (∑ r, polynomialL1 (regularIntegerDigits layout target r column)) =
      sparseColumnGain layout (value column) := by
  rw [← (regularIndexEquiv layout rows).sum_comp]
  simp only [regularIntegerDigits, Equiv.symm_apply_apply]
  rw [Fintype.sum_prod_type]
  rw [Finset.sum_eq_single (row column)]
  · simp only [polynomialL1]
    simp_rw [sparse_digit_coefficient layout hn hsparse]
    simp only [true_and]
    simp only [apply_ite Int.natAbs, Int.natAbs_zero]
    simp [sparseColumnGain, Fintype.sum_sigma]
  · intro r hr hne
    simp only [polynomialL1]
    simp_rw [sparse_digit_coefficient layout hn hsparse]
    simp [hne]
  · simp

/-- Sharp gain derived from the sparse coefficient equations and the actual
registered decomposition, covering monomials, units and base powers. -/
theorem regularIntegerDigits_sparse_gain {q n rows columns : Nat}
    (layout : RegularLayout q) (hn : 0 < n)
    {target : ExactMatrix q n rows columns} {row : Fin columns → Fin rows}
    {position : Fin columns → Fin n} {value : Fin columns → Int}
    (hsparse : SparseTarget target row position value) :
    ActionGain (regularIntegerDigits layout target) (sparseRegularGain layout value) := by
  intro column
  rw [regularIntegerDigits_sparse_column layout hn hsparse]
  exact Finset.le_sup (s := Finset.univ)
    (f := fun c => sparseColumnGain layout (value c)) (Finset.mem_univ column)

theorem regularDecomposeMatrix_sparse_gain {q n rows columns : Nat}
    (layout : RegularLayout q) (hq : 1 < q) (hn : 0 < n)
    {target : ExactMatrix q n rows columns} {row : Fin columns → Fin rows}
    {position : Fin columns → Fin n} {value : Fin columns → Int}
    (hsparse : SparseTarget target row position value) :
    ∃ witness : ErrorMatrix n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns,
      regularDecomposeMatrix layout target =
        reduceMatrix q n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns witness ∧
      ActionGain witness (sparseRegularGain layout value) :=
  ⟨regularIntegerDigits layout target, regularIntegerDigits_reduce layout hq hn target,
    regularIntegerDigits_sparse_gain layout hn hsparse⟩

/-- Direct bridge from an exported regular-decomposition execution to its
sparse integer action witness and exact maximum-column gain. -/
theorem gadgetDecomposeRuns_sparse_gain {q n rows columns inner : Nat}
    {backend : BackendContext} (layout : RegularLayout q)
    (hlookup : backend.regularLayout q n = some layout) (hq : 1 < q) (hn : 0 < n)
    {base digits : Int} {target : ExactMatrix q n rows columns}
    {decomposition : ExactMatrix q n inner columns} {row : Fin columns → Fin rows}
    {position : Fin columns → Fin n} {value : Fin columns → Int}
    (hrun : gadgetDecomposeRuns backend base digits target decomposition)
    (hsparse : SparseTarget target row position value) :
    ∃ witness : ErrorMatrix n inner columns,
      decomposition = reduceMatrix q n inner columns witness ∧
      ActionGain witness (sparseRegularGain layout value) := by
  obtain ⟨other, hother, _, _, hwidth, hd⟩ := hrun
  have hl : other = layout := Option.some.inj (hother.symm.trans hlookup)
  subst other
  subst inner
  simp only [castMatrixRows] at hd
  rw [hd]
  exact regularDecomposeMatrix_sparse_gain layout hq hn hsparse

end MxxExponentLut
