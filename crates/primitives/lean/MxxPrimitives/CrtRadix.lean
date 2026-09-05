import MxxPrimitives.Bounds
import Mathlib.Data.ZMod.QuotientRing

namespace Mxx.Primitives

open scoped BigOperators

/- Centered lift and the actual balanced tie-to-even digit recurrence used by
   the regular (small=false) DCRT decomposition. -/
def centeredLift (modulus : Nat) (value : ZMod modulus) : Int :=
  if 2 * value.val ≤ modulus then value.val else value.val - modulus

theorem centeredLift_cast (modulus : Nat) (hmod : 0 < modulus) (value : ZMod modulus) :
    (centeredLift modulus value : ZMod modulus) = value := by
  letI : NeZero modulus := ⟨Nat.ne_of_gt hmod⟩
  unfold centeredLift
  split
  · simp
  · rw [Int.cast_sub, Int.cast_natCast, Int.cast_natCast, ZMod.natCast_zmod_val,
      ZMod.natCast_self, sub_zero]

theorem centeredLift_natAbs_le_half (modulus : Nat) (hmod : 0 < modulus)
    (value : ZMod modulus) :
    (centeredLift modulus value).natAbs ≤ modulus / 2 := by
  letI : NeZero modulus := ⟨Nat.ne_of_gt hmod⟩
  unfold centeredLift
  split
  · have hval_nonneg : 0 ≤ (value.val : Int) := by omega
    have hval_lt : (value.val : Int) < modulus := by exact_mod_cast ZMod.val_lt value
    have hbound : ((value.val : Int).natAbs : Int) ≤ (modulus / 2 : Nat) := by
      rw [Int.natAbs_of_nonneg hval_nonneg]
      omega
    exact_mod_cast hbound
  · have hval_nonneg : 0 ≤ (value.val : Int) := by omega
    have hval_lt : (value.val : Int) < modulus := by exact_mod_cast ZMod.val_lt value
    have hexpr : (value.val : Int) - modulus = -(modulus - value.val) := by ring
    rw [hexpr, Int.natAbs_neg]
    have hnonneg : 0 ≤ (modulus : Int) - value.val := by omega
    have hbound : (((modulus : Int) - value.val).natAbs : Int) ≤ (modulus / 2 : Nat) := by
      rw [Int.natAbs_of_nonneg hnonneg]
      omega
    exact_mod_cast hbound

def balancedDigit (base : Nat) (value : Int) : Int :=
  let remainder := value % base
  let half := base / 2
  if remainder < half then remainder
  else if remainder > half then remainder - base
  else if (value / base) % 2 = 0 then remainder else remainder - base

def balancedResidual (base : Nat) (value : Int) : Nat → Int
  | 0 => value
  | digits + 1 =>
      balancedResidual base ((value - balancedDigit base value) / base) digits

def balancedDigitAt (base : Nat) (value : Int) (digit : Nat) : Int :=
  balancedDigit base (balancedResidual base value digit)

theorem balancedResidual_succ (base : Nat) (value : Int) (digits : Nat) :
    balancedResidual base value (digits + 1) =
      balancedResidual base ((value - balancedDigit base value) / base) digits := rfl

theorem balancedDigit_congruent (base : Nat) (value : Int) :
    balancedDigit base value ≡ value [ZMOD base] := by
  dsimp [balancedDigit]
  have hmod : value % (base : Int) ≡ value [ZMOD base] := by
    simp [Int.ModEq]
  by_cases h₁ : value % (base : Int) < base / 2
  · simp [h₁, hmod]
  by_cases h₂ : value % (base : Int) > base / 2
  · simp [h₁, h₂, hmod]
  by_cases h₃ : value / (base : Int) % 2 = 0
  · simp [h₁, h₂, h₃, hmod]
  · simp [h₁, h₂, h₃, hmod]

theorem balancedStep_eq (base : Nat) (hbase : 1 < base) (value : Int) :
    value = balancedDigit base value + base *
      ((value - balancedDigit base value) / base) := by
  have hcongr := balancedDigit_congruent base value
  apply (Int.modEq_iff_dvd).mp at hcongr
  have hbase0 : (base : Int) ≠ 0 := by omega
  have hdiv : (base : Int) ∣ value - balancedDigit base value := hcongr
  obtain ⟨quotient, hquotient⟩ := hdiv
  have hquotient_eq : (value - balancedDigit base value) / base = quotient := by
    rw [hquotient, Int.mul_ediv_cancel_left _ hbase0]
  rw [hquotient_eq]
  omega

theorem balancedDigit_abs_le (base : Nat) (hbase : 1 < base) (heven : Even base)
    (value : Int) :
    (balancedDigit base value).natAbs ≤ base / 2 := by
  dsimp [balancedDigit]
  obtain ⟨half, hhalf⟩ := heven
  have hmod_nonneg : 0 ≤ value % (base : Int) := Int.emod_nonneg _ (by omega)
  have hmod_lt : value % (base : Int) < base := by
    exact Int.emod_lt_of_pos _ (by omega)
  by_cases h₁ : value % (base : Int) < base / 2
  · simp only [if_pos h₁]
    have hbound : ((value % (base : Int)).natAbs : Int) ≤ (base / 2 : Nat) := by
      rw [Int.natAbs_of_nonneg hmod_nonneg]
      omega
    exact_mod_cast hbound
  by_cases h₂ : value % (base : Int) > base / 2
  · simp only [if_neg h₁, if_pos h₂]
    have hexpr : value % (base : Int) - base =
        -(base - value % (base : Int)) := by ring
    rw [hexpr, Int.natAbs_neg]
    have hnonneg : 0 ≤ base - value % (base : Int) := by omega
    have hbound : ((base - value % (base : Int)).natAbs : Int) ≤ (base / 2 : Nat) := by
      rw [Int.natAbs_of_nonneg hnonneg]
      omega
    exact_mod_cast hbound
  by_cases h₃ : value / (base : Int) % 2 = 0
  · simp only [if_neg h₁, if_neg h₂, if_pos h₃]
    have hbound : ((value % (base : Int)).natAbs : Int) ≤ (base / 2 : Nat) := by
      rw [Int.natAbs_of_nonneg hmod_nonneg]
      omega
    exact_mod_cast hbound
  · simp only [if_neg h₁, if_neg h₂, if_neg h₃]
    have hexpr : value % (base : Int) - base =
        -(base - value % (base : Int)) := by ring
    rw [hexpr, Int.natAbs_neg]
    have hnonneg : 0 ≤ base - value % (base : Int) := by omega
    have hbound : ((base - value % (base : Int)).natAbs : Int) ≤ (base / 2 : Nat) := by
      rw [Int.natAbs_of_nonneg hnonneg]
      omega
    exact_mod_cast hbound

private theorem balancedResidual_step_bound
    (base : Nat) (hbase : 1 < base) (heven : Even base) (bound : Nat)
    (value : Int) (hvalue : value.natAbs ≤ base * bound) :
    ((value - balancedDigit base value) / base).natAbs ≤ bound := by
  have hstep := balancedStep_eq base hbase value
  have hdigit := balancedDigit_abs_le base hbase heven value
  have hvalue_bounds :
      -(base * bound : Int) ≤ value ∧ value ≤ (base * bound : Int) := by
    have hvalue_int : (value.natAbs : Int) ≤ (base * bound : Nat) := by
      exact_mod_cast hvalue
    by_cases hnonneg : 0 ≤ value
    · rw [Int.natAbs_of_nonneg hnonneg] at hvalue_int
      constructor <;> omega
    · have hnonpos : value ≤ 0 := le_of_not_ge hnonneg
      have hneg_nonneg : 0 ≤ -value := neg_nonneg.mpr hnonpos
      have hvalue_neg : ((-value).natAbs : Int) ≤ (base * bound : Nat) := by
        simpa only [Int.natAbs_neg] using hvalue_int
      rw [Int.natAbs_of_nonneg hneg_nonneg] at hvalue_neg
      constructor <;> omega
  have hdigit_bounds :
      -(base / 2 : Int) ≤ balancedDigit base value ∧
        balancedDigit base value ≤ (base / 2 : Int) := by
    have hdigit_int : ((balancedDigit base value).natAbs : Int) ≤ (base / 2 : Nat) := by
      exact_mod_cast hdigit
    by_cases hnonneg : 0 ≤ balancedDigit base value
    · rw [Int.natAbs_of_nonneg hnonneg] at hdigit_int
      constructor <;> omega
    · have hnonpos : balancedDigit base value ≤ 0 := le_of_not_ge hnonneg
      have hneg_nonneg : 0 ≤ -balancedDigit base value := neg_nonneg.mpr hnonpos
      have hdigit_neg : ((-balancedDigit base value).natAbs : Int) ≤ (base / 2 : Nat) := by
        simpa only [Int.natAbs_neg] using hdigit_int
      rw [Int.natAbs_of_nonneg hneg_nonneg] at hdigit_neg
      constructor <;> omega
  let next : Int := (value - balancedDigit base value) / base
  have hstep' : value = balancedDigit base value + base * next := by
    exact hstep
  dsimp [next] at *
  by_cases hnext : 0 ≤ next
  · have hnext_int : (next.natAbs : Int) ≤ (bound : Nat) := by
      rw [Int.natAbs_of_nonneg hnext]
      by_contra hbound
      have hlarge : bound + 1 ≤ next := by omega
      have hprod : (base : Int) * (bound + 1) ≤ base * next :=
        mul_le_mul_of_nonneg_left hlarge (by omega)
      have hprod' : (base : Int) * bound + base ≤ base * next := by
        convert hprod using 1
      dsimp [next] at hprod'
      omega
    exact_mod_cast hnext_int
  · have hnext_neg : next ≤ -1 := by omega
    have hnext_abs : ((-next).natAbs : Int) ≤ (bound : Nat) := by
      rw [Int.natAbs_of_nonneg (by omega)]
      by_contra hbound
      have hlarge : bound + 1 ≤ -next := by omega
      have hprod : (base : Int) * (bound + 1) ≤ base * (-next) :=
        mul_le_mul_of_nonneg_left hlarge (by omega)
      have hprod' : base * next ≤ -((base : Int) * bound + base) := by
        have hneg := neg_le_neg hprod
        convert hneg using 1; ring
      dsimp [next] at hprod'
      omega
    have hnext_int : (next.natAbs : Int) ≤ (bound : Nat) := by
      simpa only [Int.natAbs_neg] using hnext_abs
    exact_mod_cast hnext_int

private theorem balancedResidual_one_zero
    (base : Nat) (hbase : 1 < base) (heven : Even base) (value : Int)
    (hvalue : value.natAbs ≤ base / 2) :
    balancedResidual base value 1 = 0 := by
  have hbase_int : (1 : Int) < base := by exact_mod_cast hbase
  obtain ⟨halfNat, hbase_even⟩ := heven
  have hbase_nat : base = 2 * halfNat := by omega
  have hnat_half : base / 2 = halfNat := by omega
  have hbase_half : (base : Int) / 2 = halfNat := by
    apply Int.ediv_eq_of_eq_mul_right (by norm_num)
    exact_mod_cast hbase_nat
  have hbase_int_eq : (base : Int) = 2 * halfNat := by exact_mod_cast hbase_nat
  have hvalue_int : (value.natAbs : Int) ≤ (base / 2 : Nat) := by exact_mod_cast hvalue
  rw [hnat_half] at hvalue_int
  by_cases hnonneg : 0 ≤ value
  · have hval : value ≤ (base / 2 : Int) := by
      rw [← Int.natAbs_of_nonneg hnonneg, hbase_half]
      exact hvalue_int
    have hbase_abs : |(base : Int)| = base := abs_of_pos (by omega)
    have hhalf_le : (base : Int) / 2 ≤ base - 1 :=
      Int.ediv_le_of_le_mul (by norm_num) (by omega)
    have hhalf_lt : (base : Int) / 2 < base := lt_of_le_of_lt hhalf_le (by omega)
    rw [hbase_half] at hval
    have hhalf_lt' : (halfNat : Int) < base := by omega
    have hmod : value % (base : Int) = value :=
      Int.emod_eq_of_lt (by omega) (by omega)
    have hval_lt : value < (base : Int) := lt_of_le_of_lt hval hhalf_lt'
    have hquot : value / (base : Int) = 0 :=
      Int.ediv_eq_zero_of_lt_abs (by omega) (by simpa [hbase_abs] using hval_lt)
    have hdigit : balancedDigit base value = value := by
      dsimp [balancedDigit]
      rw [hmod, hquot, hbase_half]
      by_cases hlt : value < (halfNat : Int)
      · simp [hlt]
      · have heq : value = (halfNat : Int) := by omega
        simp [heq]
    simp [balancedResidual, hdigit]
  · have hnonpos : value ≤ 0 := le_of_not_ge hnonneg
    have hneg_nonneg : 0 ≤ -value := neg_nonneg.mpr hnonpos
    have hneg_bound : -value ≤ (base / 2 : Int) := by
      rw [← Int.natAbs_of_nonneg hneg_nonneg, hbase_half]
      simpa only [Int.natAbs_neg] using hvalue_int
    have hneg_bound' : -value ≤ (halfNat : Int) := by
      simpa [hbase_half] using hneg_bound
    have hhalf_le : (base : Int) / 2 ≤ base - 1 :=
      Int.ediv_le_of_le_mul (by norm_num) (by omega)
    have hhalf_lt : (base : Int) / 2 < base := lt_of_le_of_lt hhalf_le (by omega)
    have hneg_lt : -value - 1 < (base : Int) := by omega
    have hbase_abs : |(base : Int)| = base := abs_of_pos (by omega)
    have hquot0 : (-value - 1) / (base : Int) = 0 :=
      Int.ediv_eq_zero_of_lt_abs (by omega) (by simpa [hbase_abs] using hneg_lt)
    have hquot : value / (base : Int) = -1 := by
      have hneg : value < 0 := by omega
      have hpos : (0 : Int) < base := by omega
      calc
        value / (base : Int) = -((-value - 1) / (base : Int) + 1) :=
          Int.ediv_of_neg_of_pos hneg hpos
        _ = -1 := by rw [hquot0]; norm_num
    have hmod : value % (base : Int) = base + value := by
      have hrec := Int.emod_add_mul_ediv value (base : Int)
      rw [hquot] at hrec
      omega
    have hdigit : balancedDigit base value = value := by
      dsimp [balancedDigit]
      rw [hmod, hquot, hbase_half]
      have hmod_nonneg : 0 ≤ base + value := by omega
      clear hmod hquot hmod_nonneg hbase_abs hneg_bound
      by_cases hzero : value = 0
      · omega
      · have hneg : value < 0 := by omega
        by_cases hlt : base + value < (halfNat : Int)
        · omega
        · by_cases hgt : base + value > (halfNat : Int)
          · simp [hgt]
            omega
          · have heq : base + value = (halfNat : Int) := by omega
            have heqval : value = halfNat - base := by omega
            simp [heqval]
    simp [balancedResidual, hdigit]

theorem balancedResidual_zero_of_capacity
    (base : Nat) (hbase : 1 < base) (heven : Even base) (digits : Nat) (value : Int)
    (hvalue : value.natAbs ≤ base ^ digits / 2) :
    balancedResidual base value digits = 0 := by
  induction digits generalizing value with
  | zero =>
      have hzero : value.natAbs = 0 := by simpa using hvalue
      exact Int.natAbs_eq_zero.mp hzero
  | succ digits ih =>
      cases digits with
      | zero =>
          apply balancedResidual_one_zero base hbase heven value
          simpa [Nat.pow_succ] using hvalue
      | succ digits =>
          have hpow_even : Even (base ^ (digits + 1)) := by
            rw [Nat.even_pow]
            exact ⟨heven, by omega⟩
          obtain ⟨halfPow, hhalfPow⟩ := hpow_even
          have hdiv : 2 ∣ base ^ (digits + 1) := by
            exact ⟨halfPow, by omega⟩
          have hcap : value.natAbs ≤ base * (base ^ (digits + 1) / 2) := by
            rw [← Nat.mul_div_assoc base hdiv]
            simpa [Nat.pow_succ, Nat.mul_comm] using hvalue
          have hstep :=
            balancedResidual_step_bound base hbase heven (base ^ (digits + 1) / 2) value hcap
          rw [balancedResidual_succ]
          apply ih
          exact hstep

theorem centeredLift_residual_zero_of_capacity
    (modulus base digits : Nat) (hmod : 0 < modulus) (hbase : 1 < base)
    (heven : Even base) (hcapacity : modulus ≤ base ^ digits) (value : ZMod modulus) :
    balancedResidual base (centeredLift modulus value) digits = 0 := by
  apply balancedResidual_zero_of_capacity base hbase heven digits (centeredLift modulus value)
  exact (centeredLift_natAbs_le_half modulus hmod value).trans (Nat.div_le_div_right hcapacity)

/-
  The runtime layout is ordered by its registered CRT towers.  The residual
  condition is deliberately an explicit admissibility obligation: it is
  discharged by the adapter from the registered modulus/base/digit data, not
  invented by the exporter.
-/
structure OrderedCrtLayout where
  q : Nat
  Tower : Type
  instFintypeTower : Fintype Tower
  modulus : Tower → Nat
  modulus_pos : ∀ tower, 0 < modulus tower
  pairwise_coprime : Pairwise (fun left right => Nat.Coprime (modulus left) (modulus right))
  product_eq : q = ∏ tower, modulus tower
  instDecidableEqTower : DecidableEq Tower
  baseBits : Nat
  base : Nat := 2 ^ baseBits
  base_eq : base = 2 ^ baseBits
  base_gt_one : 1 < base
  base_even : Even base
  digits : Tower → Nat
  digits_pos : ∀ tower, 0 < digits tower
  residual_zero : ∀ tower (value : ZMod (modulus tower)),
    balancedResidual base (centeredLift (modulus tower) value) (digits tower) = 0

attribute [instance] OrderedCrtLayout.instFintypeTower
attribute [instance] OrderedCrtLayout.instDecidableEqTower

noncomputable def OrderedCrtLayout.ofCapacity
    {q : Nat} {Tower : Type} [Fintype Tower] [DecidableEq Tower]
    (modulus : Tower → Nat) (modulus_pos : ∀ tower, 0 < modulus tower)
    (pairwise_coprime : Pairwise (fun left right => Nat.Coprime (modulus left) (modulus right)))
    (product_eq : q = ∏ tower, modulus tower) (baseBits base : Nat)
    (base_eq : base = 2 ^ baseBits) (base_gt_one : 1 < base) (base_even : Even base)
    (digits : Tower → Nat) (digits_pos : ∀ tower, 0 < digits tower)
    (capacity : ∀ tower, modulus tower ≤ base ^ digits tower) : OrderedCrtLayout :=
  { q := q
    Tower := Tower
    instFintypeTower := inferInstance
    modulus := modulus
    modulus_pos := modulus_pos
    pairwise_coprime := pairwise_coprime
    product_eq := product_eq
    instDecidableEqTower := inferInstance
    baseBits := baseBits
    base := base
    base_eq := base_eq
    base_gt_one := base_gt_one
    base_even := base_even
    digits := digits
    digits_pos := digits_pos
    residual_zero := fun tower value =>
      centeredLift_residual_zero_of_capacity (modulus tower) base (digits tower)
        (modulus_pos tower) base_gt_one base_even (capacity tower) value }

namespace OrderedCrtLayout

variable (layout : OrderedCrtLayout)

def modulusProduct : Nat := ∏ tower, layout.modulus tower

theorem modulusProduct_eq_q : layout.modulusProduct = layout.q := by
  exact layout.product_eq.symm

theorem modulusProduct_pos : 0 < layout.modulusProduct := by
  dsimp [modulusProduct]
  exact Finset.prod_pos (fun tower _ => layout.modulus_pos tower)

/- The CRT equivalence is the one used by mathlib, transported along the
   registered product equality. -/
noncomputable def crtEquiv : ZMod layout.modulusProduct ≃+* (∀ tower, ZMod (layout.modulus tower)) :=
  ZMod.prodEquivPi layout.modulus layout.pairwise_coprime

noncomputable def crtWeight (tower : layout.Tower) : ZMod layout.modulusProduct :=
  layout.crtEquiv.symm (fun other => if other = tower then 1 else 0)

theorem crtWeight_apply (tower other : layout.Tower) :
    layout.crtEquiv (layout.crtWeight tower) other = if other = tower then 1 else 0 := by
  classical
  simp [crtWeight]

theorem crtWeight_idempotent (tower : layout.Tower) :
    layout.crtWeight tower * layout.crtWeight tower = layout.crtWeight tower := by
  apply layout.crtEquiv.injective
  funext other
  rw [map_mul]
  change layout.crtEquiv (layout.crtWeight tower) other *
      layout.crtEquiv (layout.crtWeight tower) other =
    layout.crtEquiv (layout.crtWeight tower) other
  rw [crtWeight_apply]
  by_cases h : other = tower <;> simp [h]

theorem crtWeight_orthogonal {left right : layout.Tower} (hne : left ≠ right) :
    layout.crtWeight left * layout.crtWeight right = 0 := by
  apply layout.crtEquiv.injective
  funext other
  rw [map_mul]
  change layout.crtEquiv (layout.crtWeight left) other *
      layout.crtEquiv (layout.crtWeight right) other =
    layout.crtEquiv 0 other
  rw [crtWeight_apply, crtWeight_apply]
  simp only [map_zero]
  have hne' : right ≠ left := Ne.symm hne
  by_cases hl : other = left <;> by_cases hr : other = right <;> simp [hl, hr, hne, hne']

theorem crtWeight_intCast_apply (tower other : layout.Tower) (value : Int) :
    layout.crtEquiv
        (layout.crtWeight tower * (value : ZMod layout.modulusProduct)) other =
      if other = tower then (value : ZMod (layout.modulus other)) else 0 := by
  rw [map_mul]
  change layout.crtEquiv (layout.crtWeight tower) other *
      layout.crtEquiv (value : ZMod layout.modulusProduct) other = _
  rw [crtWeight_apply]
  by_cases h : other = tower
  · subst other
    simp [crtEquiv]
  · simp [h, crtEquiv]

theorem crtExpand_eq (values : ∀ tower, ZMod (layout.modulus tower)) :
    layout.crtEquiv (layout.crtEquiv.symm values) = values := by
  exact layout.crtEquiv.apply_symm_apply values

end OrderedCrtLayout

/- A recursive expansion avoids hiding the recurrence behind a caller-supplied
   reconstruction field.  The adapter may enumerate these digits in runtime's
   tower-major order. -/
def balancedExpansion (base : Nat) (value : Int) : Nat → Int
  | 0 => 0
  | digits + 1 =>
      balancedDigit base value +
        base * balancedExpansion base ((value - balancedDigit base value) / base) digits

theorem balancedExpansion_spec (base : Nat) (hbase : 1 < base) (value : Int) (digits : Nat) :
    value = balancedExpansion base value digits +
      (base : Int) ^ digits * balancedResidual base value digits := by
  induction digits generalizing value with
  | zero => simp [balancedExpansion, balancedResidual]
  | succ digits ih =>
      have hi := ih ((value - balancedDigit base value) / base)
      calc
        value = balancedDigit base value + base *
            ((value - balancedDigit base value) / base) := balancedStep_eq base hbase value
        _ = balancedDigit base value + base *
            (balancedExpansion base ((value - balancedDigit base value) / base) digits +
              (base : Int) ^ digits *
                balancedResidual base ((value - balancedDigit base value) / base) digits) := by
          congr 1
          exact congrArg (fun value : Int => (base : Int) * value) hi
        _ = balancedExpansion base value (digits + 1) +
            (base : Int) ^ (digits + 1) * balancedResidual base value (digits + 1) := by
          simp [balancedExpansion, balancedResidual_succ]
          ring

theorem balancedExpansion_cast_of_residual_zero
    (modulus base : Nat) (hbase : 1 < base) (value : Int) (digits : Nat)
    (hzero : balancedResidual base value digits = 0) :
    (value : ZMod modulus) = (balancedExpansion base value digits : ZMod modulus) := by
  have hspec := balancedExpansion_spec base hbase value digits
  rw [hzero, mul_zero, add_zero] at hspec
  exact congrArg (fun integer : Int => (integer : ZMod modulus)) hspec

theorem balancedExpansion_eq_sum (base : Nat) (value : Int)
    (digits : Nat) :
    balancedExpansion base value digits =
      ∑ digit : Fin digits,
        (base : Int) ^ digit.val * balancedDigitAt base value digit.val := by
  induction digits generalizing value with
  | zero => simp [balancedExpansion]
  | succ digits ih =>
      rw [balancedExpansion, Fin.sum_univ_succ]
      simp only [Fin.val_zero, pow_zero, one_mul]
      have hsum :
          (∑ digit : Fin digits,
            (base : Int) ^ (Fin.succ digit).val *
              balancedDigitAt base value (Fin.succ digit).val) =
            (base : Int) *
              ∑ digit : Fin digits,
              (base : Int) ^ digit.val *
                  balancedDigitAt base
                    ((value - balancedDigit base value) / base) digit.val := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro digit hmem
        simp only [Fin.val_succ, balancedDigitAt, balancedResidual_succ]
        rw [pow_succ]
        ring
      rw [hsum, ih]
      simp [balancedDigitAt, balancedResidual]

end Mxx.Primitives
