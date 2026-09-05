import ExponentBounds
import RuntimePrimitives

namespace MxxExponentLut
open Mxx.Primitives MxxRuntime
open scoped BigOperators

/-- The balanced carry used by the concrete runtime radix operation. -/
def balancedNext (base : Nat) (value : Int) : Int :=
  (value - balancedDigit base value) / base

/-- Exact endpoint recurrence, including round-to-even at a half digit. -/
def balancedIntervalNext (base bound : Nat) : Nat :=
  max (balancedNext base (bound : Int)).natAbs (balancedNext base (-(bound : Int))).natAbs

theorem balancedNext_eq (base : Nat) (hbase : 1 < base) (value : Int) :
    balancedNext base value = value / base +
      if value % base < (base / 2 : Nat) then 0
      else if value % base > (base / 2 : Nat) then 1
      else if (value / base) % 2 = 0 then 0 else 1 := by
  have hstep := balancedStep_eq base hbase value
  have hdiv := Int.mul_ediv_add_emod value (base : Int)
  change value = balancedDigit base value + (base : Int) * balancedNext base value at hstep
  dsimp only [balancedDigit] at hstep
  split_ifs at hstep ⊢ <;> nlinarith

theorem balancedNext_monotone (base : Nat) (hbase : 1 < base) :
    Monotone (balancedNext base) := by
  intro a b hab
  have hq := Int.ediv_le_ediv (show (0 : Int) < base by omega) hab
  have ha := Int.mul_ediv_add_emod a (base : Int)
  have hb := Int.mul_ediv_add_emod b (base : Int)
  rw [balancedNext_eq base hbase, balancedNext_eq base hbase]
  by_cases heq : a / (base : Int) = b / (base : Int)
  · have hrem : a % (base : Int) ≤ b % (base : Int) := by nlinarith
    split_ifs <;> omega
  · split_ifs <;> omega

theorem balancedNext_interval {base bound : Nat} (hbase : 1 < base) {value : Int}
    (hvalue : value.natAbs ≤ bound) :
    (balancedNext base value).natAbs ≤ balancedIntervalNext base bound := by
  have hlo : -(bound : Int) ≤ value := by omega
  have hhi : value ≤ (bound : Int) := by omega
  have hl := balancedNext_monotone base hbase hlo
  have hh := balancedNext_monotone base hbase hhi
  unfold balancedIntervalNext
  omega

theorem balancedDigit_le_value (base : Nat) (hbase : 1 < base) (heven : Even base)
    (value : Int) : (balancedDigit base value).natAbs ≤ value.natAbs := by
  by_cases hlarge : base / 2 ≤ value.natAbs
  · exact (balancedDigit_abs_le base hbase heven value).trans hlarge
  · have hsmall : value.natAbs < base / 2 := by omega
    have hdiv := Int.mul_ediv_add_emod value (base : Int)
    by_cases hnonneg : 0 ≤ value
    · have hrem := Int.emod_eq_of_lt hnonneg (show value < (base : Int) by omega)
      dsimp only [balancedDigit]
      rw [hrem, if_pos (show value < (base / 2 : Nat) by omega)]
    · have hquot := Int.ediv_eq_neg_one_of_neg_of_le (show value < 0 by omega)
        (show -value ≤ (base : Int) by omega)
      have hrem : value % (base : Int) = value + base := by rw [hquot] at hdiv; omega
      dsimp only [balancedDigit]
      rw [hrem, if_neg (by omega), if_pos (by omega)]
      simp

/-- State bounds and sharp per-digit maxima mirror `dense_coefficient_gain`. -/
def denseStateBound (base bound : Nat) : Nat → Nat
  | 0 => bound
  | digit + 1 => denseStateBound base (balancedIntervalNext base bound) digit

def denseDigitBound (base bound digit : Nat) : Nat :=
  min (denseStateBound base bound digit) (base / 2)

theorem balancedResidual_bound {base bound : Nat} (hbase : 1 < base) {value : Int}
    (hvalue : value.natAbs ≤ bound) (digit : Nat) :
    (balancedResidual base value digit).natAbs ≤ denseStateBound base bound digit := by
  induction digit generalizing value bound with
  | zero => exact hvalue
  | succ digit ih => exact ih (balancedNext_interval hbase hvalue)

theorem balancedDigitAt_sharp_bound {base bound : Nat}
    (hbase : 1 < base) (heven : Even base) {value : Int}
    (hvalue : value.natAbs ≤ bound) (digit : Nat) :
    (balancedDigitAt base value digit).natAbs ≤ denseDigitBound base bound digit := by
  apply Nat.le_min.mpr
  constructor
  · exact (balancedDigit_le_value base hbase heven _).trans
      (balancedResidual_bound hbase hvalue digit)
  · exact balancedDigit_abs_le base hbase heven _

/-- Actual integer digits, retaining row/tower/digit order from the runtime. -/
noncomputable def regularIntegerDigits {q n rows columns : Nat}
    (layout : RegularLayout q) (target : ExactMatrix q n rows columns) :
    ErrorMatrix n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns :=
  fun row column => CrtDecomposition.regularDigitPoly layout.ordered
    ((castExactMatrix layout.product_eq target) ((regularIndexEquiv layout rows).symm row).1 column)
    ((regularIndexEquiv layout rows).symm row).2

/-- The exact dense L1 expression: add sharp digit bounds in tower order, then
multiply by the number of source rows and ring coefficients. -/
def denseRegularGain {q : Nat} (layout : RegularLayout q) (n rows : Nat) : Nat :=
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
    ActionGain (regularIntegerDigits layout target) (denseRegularGain layout n rows) := by
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
    _ = denseRegularGain layout n rows := by
      simp only [Fintype.sum_sigma, Finset.mul_sum, Finset.sum_const, Finset.card_univ,
        Fintype.card_fin, smul_eq_mul, denseRegularGain]
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
    regularDecomposeMatrix layout target =
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
  unfold regularDecomposeMatrix
  rw [he]
  exact castExactMatrix_reduce layout.product_eq.symm _

/-- Sharp dense action gain for the actual digit matrix, with no assumed
`ActionGain` premise. This is directly consumable by private gate transfers. -/
theorem regularDecomposeMatrix_dense_gain {q n rows columns : Nat}
    (layout : RegularLayout q) (hq : 1 < q) (hn : 0 < n)
    (target : ExactMatrix q n rows columns) :
    ∃ witness : ErrorMatrix n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns,
      regularDecomposeMatrix layout target =
        reduceMatrix q n (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns witness ∧
      ActionGain witness (denseRegularGain layout n rows) :=
  ⟨regularIntegerDigits layout target, regularIntegerDigits_reduce layout hq hn target,
    regularIntegerDigits_action_gain layout hn target⟩

/-- Consume the actual exported decomposition execution and the registered
backend layout; the integer action and its sharp gain are conclusions. -/
theorem gadgetDecomposeRuns_dense_gain {q n rows columns inner : Nat}
    {backend : BackendContext} (layout : RegularLayout q)
    (hlookup : backend.regularLayout q n = some layout)
    (hq : 1 < q) (hn : 0 < n) {base digits : Int}
    {target : ExactMatrix q n rows columns} {decomposition : ExactMatrix q n inner columns}
    (hrun : gadgetDecomposeRuns backend base digits target decomposition) :
    ∃ witness : ErrorMatrix n inner columns,
      decomposition = reduceMatrix q n inner columns witness ∧
      ActionGain witness (denseRegularGain layout n rows) := by
  obtain ⟨other, hother, _, _, hwidth, hd⟩ := hrun
  have hl : other = layout := Option.some.inj (hother.symm.trans hlookup)
  subst other
  subst inner
  simp only [castMatrixRows] at hd
  rw [hd]
  exact regularDecomposeMatrix_dense_gain layout hq hn target

end MxxExponentLut
