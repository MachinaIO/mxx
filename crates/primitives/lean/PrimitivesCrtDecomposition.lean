import PrimitivesRadix
import PrimitivesCrtRadix

namespace Mxx.Primitives.CrtDecomposition

open Mxx.Primitives
variable {q n rows columns : Nat}

/- The regular runtime decomposition is indexed in the registered,
   tower-major order.  A limb carries its tower and its digit within that
   tower; no factorisation or ordering is inferred from q by Lean. -/
abbrev RegularLimb (layout : OrderedCrtLayout) :=
  Sigma (fun tower => Fin (layout.digits tower))

noncomputable def regularDigitCoefficient
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) (coefficient : Fin n) : Int :=
  balancedDigitAt layout.base
    (centeredLift (layout.modulus limb.1)
      (layout.crtEquiv (value.coeff coefficient) limb.1)) limb.2.val

theorem regularDigitCoefficient_bound
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) (coefficient : Fin n) :
    (regularDigitCoefficient layout value limb coefficient).natAbs ≤ layout.base / 2 := by
  exact balancedDigit_abs_le layout.base layout.base_gt_one layout.base_even _

theorem regularTowerExpansion_reconstruct
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (tower : layout.Tower) (coefficient : Fin n) :
    centeredLift (layout.modulus tower)
        (layout.crtEquiv (value.coeff coefficient) tower) =
      balancedExpansion layout.base
        (centeredLift (layout.modulus tower)
          (layout.crtEquiv (value.coeff coefficient) tower))
        (layout.digits tower) := by
  have hspec := balancedExpansion_spec layout.base layout.base_gt_one
    (centeredLift (layout.modulus tower)
      (layout.crtEquiv (value.coeff coefficient) tower)) (layout.digits tower)
  calc
    _ = balancedExpansion layout.base
        (centeredLift (layout.modulus tower)
          (layout.crtEquiv (value.coeff coefficient) tower)) (layout.digits tower) +
        (layout.base : Int) ^ layout.digits tower *
          balancedResidual layout.base
            (centeredLift (layout.modulus tower)
              (layout.crtEquiv (value.coeff coefficient) tower)) (layout.digits tower) := hspec
    _ = _ := by
      rw [layout.residual_zero tower (layout.crtEquiv (value.coeff coefficient) tower)]
      simp

noncomputable def regularTowerCrtContribution
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (tower : layout.Tower) (coefficient : Fin n) :
    ZMod layout.modulusProduct :=
  layout.crtWeight tower *
    (balancedExpansion layout.base
      (centeredLift (layout.modulus tower)
        (layout.crtEquiv (value.coeff coefficient) tower))
      (layout.digits tower) : ZMod layout.modulusProduct)

noncomputable def regularCrtReconstruction
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n) (coefficient : Fin n) :
    ZMod layout.modulusProduct :=
  ∑ tower : layout.Tower, regularTowerCrtContribution layout value tower coefficient

theorem regularCrtReconstruction_apply
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n) (coefficient : Fin n) :
    layout.crtEquiv (regularCrtReconstruction layout value coefficient) =
      layout.crtEquiv (value.coeff coefficient) := by
  funext other
  rw [regularCrtReconstruction, map_sum]
  rw [Finset.sum_apply]
  rw [Finset.sum_eq_single other]
  · rw [regularTowerCrtContribution, layout.crtWeight_intCast_apply]
    have hexpand := regularTowerExpansion_reconstruct layout value other coefficient
    have hcast := congrArg (fun integer : Int =>
      (integer : ZMod (layout.modulus other))) hexpand
    change (centeredLift (layout.modulus other)
      (layout.crtEquiv (value.coeff coefficient) other) :
        ZMod (layout.modulus other)) =
      (balancedExpansion layout.base
        (centeredLift (layout.modulus other)
          (layout.crtEquiv (value.coeff coefficient) other))
        (layout.digits other) : ZMod (layout.modulus other)) at hcast
    rw [← hcast, centeredLift_cast (layout.modulus other)
      (layout.modulus_pos other)]
    simp
  · intro tower hmem hne
    rw [regularTowerCrtContribution]
    change layout.crtEquiv (layout.crtWeight tower *
      ((balancedExpansion layout.base
        (centeredLift (layout.modulus tower)
          (layout.crtEquiv (value.coeff coefficient) tower))
        (layout.digits tower) : Int) : ZMod layout.modulusProduct)) other = 0
    rw [OrderedCrtLayout.crtWeight_intCast_apply]
    have hne' : other ≠ tower := Ne.symm hne
    simp [hne']
  · simp

theorem regularCrtReconstruction_eq
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n) (coefficient : Fin n) :
    regularCrtReconstruction layout value coefficient = value.coeff coefficient := by
  apply layout.crtEquiv.injective
  exact regularCrtReconstruction_apply layout value coefficient

/- The regular gadget column is the CRT idempotent of the source tower
   multiplied by its base power. -/
noncomputable def regularGadgetWeight
    (layout : OrderedCrtLayout) (limb : RegularLimb layout) :
    ZMod layout.modulusProduct :=
  layout.crtWeight limb.1 * (layout.base : ZMod layout.modulusProduct) ^ limb.2.val

/- Each regular digit is paired with the idempotent-weighted base column that
   carries it.  This is the scalar core of the tower-major gadget matrix; the
   runtime adapter is responsible only for placing these columns in its
   registered matrix layout. -/
noncomputable def regularCrtDigitContribution
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (tower : layout.Tower) (digit : Fin (layout.digits tower))
    (coefficient : Fin n) : ZMod layout.modulusProduct :=
  layout.crtWeight tower *
    ((layout.base : ZMod layout.modulusProduct) ^ digit.val *
      (regularDigitCoefficient layout value ⟨tower, digit⟩ coefficient :
        ZMod layout.modulusProduct))

noncomputable def regularDigitExactPoly
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) : ExactPoly layout.modulusProduct n :=
  ∑ coefficient : Fin n,
    algebraMap (ZMod layout.modulusProduct)
      (ExactPoly layout.modulusProduct n)
      (regularDigitCoefficient layout value limb coefficient :
        ZMod layout.modulusProduct) *
      AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
        (coefficient : Nat)

private theorem coeff_sum_exact_local {R : Type u} [CommRing R]
    {α : Type v} [DecidableEq α] {n : Nat} (s : Finset α)
    (f : α → RingPoly n R) (k : Fin n) :
    (∑ i ∈ s, f i).coeff k = ∑ i ∈ s, (f i).coeff k := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    simp only [Finset.sum_insert ha, Negacyclic.coeff_add, ih]

private theorem coeff_sum_univ_exact_local {R : Type u} [CommRing R]
    {α : Type v} [Fintype α] [DecidableEq α] {n : Nat}
    (f : α → RingPoly n R) (k : Fin n) :
    (∑ i : α, f i).coeff k = ∑ i : α, (f i).coeff k := by
  induction (Finset.univ : Finset α) using Finset.induction_on with
  | empty => simp
  | @insert i s hi ih =>
    simp only [Finset.sum_insert hi, Negacyclic.coeff_add, ih]

theorem regularDigitExactPoly_coeff
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) (hq : 1 < layout.modulusProduct)
    (hn : 0 < n) (coefficient : Fin n) :
    (regularDigitExactPoly layout value limb).coeff coefficient =
      (regularDigitCoefficient layout value limb coefficient :
        ZMod layout.modulusProduct) := by
  letI : Fact (1 < layout.modulusProduct) := ⟨hq⟩
  rw [regularDigitExactPoly, coeff_sum_exact_local]
  rw [Finset.sum_eq_single coefficient]
  · change (algebraMap (ZMod layout.modulusProduct)
      (ExactPoly layout.modulusProduct n)
      (regularDigitCoefficient layout value limb coefficient :
        ZMod layout.modulusProduct) *
        AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
          (coefficient : Nat)).coeff coefficient = _
    rw [Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
    simp
  · intro index hindex hne
    change (algebraMap (ZMod layout.modulusProduct)
      (ExactPoly layout.modulusProduct n)
      (regularDigitCoefficient layout value limb index :
        ZMod layout.modulusProduct) *
        AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
          (index : Nat)).coeff coefficient = _
    rw [Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
    simp [hne]
  · simp

noncomputable def regularCrtDigitPolyContribution
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (tower : layout.Tower) (digit : Fin (layout.digits tower)) :
    ExactPoly layout.modulusProduct n :=
  algebraMap (ZMod layout.modulusProduct)
      (ExactPoly layout.modulusProduct n)
      (regularGadgetWeight layout ⟨tower, digit⟩) *
    regularDigitExactPoly layout value ⟨tower, digit⟩

theorem regularCrtDigitPolyContribution_coeff
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (tower : layout.Tower) (digit : Fin (layout.digits tower))
    (hq : 1 < layout.modulusProduct) (hn : 0 < n) (coefficient : Fin n) :
    (regularCrtDigitPolyContribution layout value tower digit).coeff coefficient =
      regularCrtDigitContribution layout value tower digit coefficient := by
  rw [regularCrtDigitPolyContribution, Negacyclic.coeff_smul,
    regularDigitExactPoly_coeff layout value ⟨tower, digit⟩ hq hn coefficient]
  simp [regularCrtDigitContribution, regularGadgetWeight]
  ring

theorem regularTowerCrtContribution_eq_digit_sum
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (tower : layout.Tower) (coefficient : Fin n) :
    regularTowerCrtContribution layout value tower coefficient =
      ∑ digit : Fin (layout.digits tower),
        regularCrtDigitContribution layout value tower digit coefficient := by
  rw [regularTowerCrtContribution,
    balancedExpansion_eq_sum layout.base
      (centeredLift (layout.modulus tower)
        (layout.crtEquiv (value.coeff coefficient) tower))
      (layout.digits tower)]
  simp only [Int.cast_sum, Int.cast_mul, Int.cast_pow]
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro digit hmem
  simp [regularCrtDigitContribution, regularDigitCoefficient]

theorem regularCrtReconstruction_eq_digit_sum
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n) (coefficient : Fin n) :
    regularCrtReconstruction layout value coefficient =
      ∑ tower : layout.Tower, ∑ digit : Fin (layout.digits tower),
        regularCrtDigitContribution layout value tower digit coefficient := by
  rw [regularCrtReconstruction]
  apply Finset.sum_congr rfl
  intro tower hmem
  exact regularTowerCrtContribution_eq_digit_sum layout value tower coefficient

theorem regularCrtDigitPoly_reconstruct
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (hq : 1 < layout.modulusProduct) (hn : 0 < n) :
    (∑ tower : layout.Tower, ∑ digit : Fin (layout.digits tower),
      regularCrtDigitPolyContribution layout value tower digit) = value := by
  classical
  letI : Fact (1 < layout.modulusProduct) := ⟨hq⟩
  let reconstructed : ExactPoly layout.modulusProduct n :=
    ∑ tower : layout.Tower, ∑ digit : Fin (layout.digits tower),
      regularCrtDigitPolyContribution layout value tower digit
  have hcoeff (coefficient : Fin n) :
      reconstructed.coeff coefficient = value.coeff coefficient := by
    rw [show reconstructed =
        ∑ tower : layout.Tower, ∑ digit : Fin (layout.digits tower),
          regularCrtDigitPolyContribution layout value tower digit by rfl]
    rw [coeff_sum_univ_exact_local]
    simp_rw [coeff_sum_univ_exact_local]
    calc
      (∑ tower : layout.Tower, ∑ digit : Fin (layout.digits tower),
          (regularCrtDigitPolyContribution layout value tower digit).coeff coefficient) =
          ∑ tower : layout.Tower, ∑ digit : Fin (layout.digits tower),
            regularCrtDigitContribution layout value tower digit coefficient := by
        apply Finset.sum_congr rfl
        intro tower htower
        apply Finset.sum_congr rfl
        intro digit hdigit
        exact regularCrtDigitPolyContribution_coeff layout value tower digit hq hn coefficient
      _ = regularCrtReconstruction layout value coefficient :=
        (regularCrtReconstruction_eq_digit_sum layout value coefficient).symm
      _ = value.coeff coefficient := regularCrtReconstruction_eq layout value coefficient
  change reconstructed = value
  calc
    reconstructed = ∑ coefficient : Fin n,
        algebraMap (ZMod layout.modulusProduct)
          (ExactPoly layout.modulusProduct n) (reconstructed.coeff coefficient) *
          AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
            (coefficient : Nat) := Negacyclic.expansion hn reconstructed
    _ = ∑ coefficient : Fin n,
        algebraMap (ZMod layout.modulusProduct)
          (ExactPoly layout.modulusProduct n) (value.coeff coefficient) *
          AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
            (coefficient : Nat) := by
      apply Finset.sum_congr rfl
      intro coefficient hcoefficient
      rw [hcoeff]
    _ = value := (Negacyclic.expansion hn value).symm

/- A one-row/one-column matrix presentation makes the equation explicit while
   retaining the registered tower-major sigma index.  A runtime adapter may
   transport this finite index to its concrete `Fin` column numbering. -/
noncomputable def regularGadgetRow
    (layout : OrderedCrtLayout) {n : Nat} :
    Matrix Unit (RegularLimb layout) (ExactPoly layout.modulusProduct n) :=
  fun _ limb =>
    algebraMap (ZMod layout.modulusProduct)
      (ExactPoly layout.modulusProduct n)
      (regularGadgetWeight layout limb)

noncomputable def regularDecompositionColumn
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n) :
    Matrix (RegularLimb layout) Unit (ExactPoly layout.modulusProduct n) :=
  fun limb _ => regularDigitExactPoly layout value limb

theorem regularGadgetRow_mul_regularDecompositionColumn
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (hq : 1 < layout.modulusProduct) (hn : 0 < n) :
    regularGadgetRow layout * regularDecompositionColumn layout value =
      (fun _ _ => value) := by
  classical
  funext row column
  cases row
  cases column
  change (∑ limb : RegularLimb layout,
      algebraMap (ZMod layout.modulusProduct)
        (ExactPoly layout.modulusProduct n) (regularGadgetWeight layout limb) *
        regularDigitExactPoly layout value limb) = value
  simpa [regularCrtDigitPolyContribution, Finset.sum_sigma'] using
    (regularCrtDigitPoly_reconstruct layout value hq hn)

noncomputable def regularDigitPoly
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) : ErrorPoly n :=
  ∑ coefficient : Fin n,
    scaledBasis (regularDigitCoefficient layout value limb coefficient) coefficient

theorem coeff_scaledBasis (hn : 0 < n) (a : Int) (i k : Fin n) :
    (scaledBasis a i).coeff k = if i = k then a else 0 := by
  unfold scaledBasis
  rw [Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  split
  · simp_all
  · simp_all

private theorem coeff_sum_univ_local {n : Nat} (f : Fin n → ErrorPoly n) (k : Fin n) :
    (∑ i : Fin n, f i).coeff k = ∑ i : Fin n, (f i).coeff k := by
  induction (Finset.univ : Finset (Fin n)) using Finset.induction_on with
  | empty => simp
  | @insert i s hi ih =>
      simp only [Finset.sum_insert hi, Negacyclic.coeff_add, ih]

theorem regularDigitPoly_coeff
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) (hn : 0 < n) (coefficient : Fin n) :
    (regularDigitPoly layout value limb).coeff coefficient =
      regularDigitCoefficient layout value limb coefficient := by
  rw [regularDigitPoly, coeff_sum_univ_local]
  rw [Finset.sum_eq_single coefficient]
  · simp [coeff_scaledBasis hn]
  · intro index hindex hne
    simp [coeff_scaledBasis hn, hne]
  · simp

theorem regularDigitPoly_bound
    (layout : OrderedCrtLayout) {n : Nat} (hn : 0 < n)
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) :
    polyNorm (regularDigitPoly layout value limb) ≤ layout.base / 2 := by
  apply Finset.sup_le
  intro coefficient hcoefficient
  rw [show regularDigitPoly layout value limb =
      ∑ index : Fin n, scaledBasis
        (regularDigitCoefficient layout value limb index) index by rfl]
  rw [coeff_sum_univ_local]
  rw [Finset.sum_eq_single coefficient]
  · simp [coeff_scaledBasis hn]
    exact regularDigitCoefficient_bound layout value limb coefficient
  · intro index hindex hne
    simp [coeff_scaledBasis hn, hne]
  · simp

theorem regularDigitExactPoly_eq_reduce
    (layout : OrderedCrtLayout) {n : Nat}
    (value : ExactPoly layout.modulusProduct n)
    (limb : RegularLimb layout) (hq : 1 < layout.modulusProduct)
    (hn : 0 < n) :
    regularDigitExactPoly layout value limb =
      reducePoly layout.modulusProduct n
        (regularDigitPoly layout value limb) := by
  letI : Fact (1 < layout.modulusProduct) := ⟨hq⟩
  calc
    regularDigitExactPoly layout value limb =
        ∑ coefficient : Fin n,
          algebraMap (ZMod layout.modulusProduct)
            (ExactPoly layout.modulusProduct n)
            ((regularDigitExactPoly layout value limb).coeff coefficient) *
            AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
              (coefficient : Nat) := Negacyclic.expansion hn _
    _ = ∑ coefficient : Fin n,
          algebraMap (ZMod layout.modulusProduct)
            (ExactPoly layout.modulusProduct n)
            ((reducePoly layout.modulusProduct n
              (regularDigitPoly layout value limb)).coeff coefficient) *
            AdjoinRoot.root (negacyclicModulus n (ZMod layout.modulusProduct)) ^
              (coefficient : Nat) := by
      apply Finset.sum_congr rfl
      intro coefficient hcoefficient
      rw [regularDigitExactPoly_coeff layout value limb hq hn coefficient,
        reducePoly_coeff hq hn (regularDigitPoly layout value limb) coefficient,
        regularDigitPoly_coeff layout value limb hn coefficient]
    _ = reducePoly layout.modulusProduct n
        (regularDigitPoly layout value limb) :=
      (Negacyclic.expansion hn _).symm

theorem regularGadgetWeight_bound_indexed
    (layout : OrderedCrtLayout) (limb : RegularLimb layout) :
    regularGadgetWeight layout limb =
      layout.crtWeight limb.1 * (layout.base : ZMod layout.modulusProduct) ^ limb.2.val := rfl


end Mxx.Primitives.CrtDecomposition
