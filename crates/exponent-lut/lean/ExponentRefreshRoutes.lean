import RuntimeMatrixOps
import ExponentSparseDecomposition
import ExponentTransfers

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

theorem route_family_error_bound {n rows inner columns bound gain : Nat}
    {ι : Type} [Fintype ι] (hn : 0 < n)
    (error : ι → ErrorMatrix n rows inner) (action : ι → ErrorMatrix n inner columns)
    (bounded : ∀ i, CoeffBound (error i) bound)
    (columnGain : ∀ column, ∑ i, ∑ row, polynomialL1 (action i row column) ≤ gain) :
    CoeffBound (∑ i, error i * action i) (gain * bound) := by
  classical
  intro row column coefficient
  apply (coeff_natAbs_le_polyNorm _ _).trans
  simp only [Matrix.sum_apply, Matrix.mul_apply]
  calc
    _ ≤ ∑ i, ∑ j, polyNorm (error i row j * action i j column) := by
      apply (polyNorm_sum_le Finset.univ _).trans
      exact Finset.sum_le_sum (fun i _ => polyNorm_sum_le Finset.univ _)
    _ ≤ ∑ i, ∑ j, polynomialL1 (action i j column) * bound := by
      apply Finset.sum_le_sum
      intro i hi
      apply Finset.sum_le_sum
      intro j hj
      rw [mul_comm]
      apply (polynomial_action_bound hn _ _).trans
      apply Nat.mul_le_mul_left
      apply Finset.sup_le
      intro k hk
      exact bounded i row j k
    _ = (∑ i, ∑ j, polynomialL1 (action i j column)) * bound := by
      simp only [Finset.sum_mul]
    _ ≤ gain * bound := Nat.mul_le_mul_right bound (columnGain column)

theorem route_family_approx {q n rows inner columns bound gain : Nat}
    {ι : Type} [Fintype ι] (hn : 0 < n)
    (actual ideal : ι → ExactMatrix q n rows inner)
    (decomposition : ι → ExactMatrix q n inner columns)
    (action : ι → ErrorMatrix n inner columns)
    (close : ∀ i, Approx (actual i) (ideal i) bound)
    (reduction : ∀ i, decomposition i = reduceMatrix q n inner columns (action i))
    (columnGain : ∀ column, ∑ i, ∑ row, polynomialL1 (action i row column) ≤ gain) :
    Approx (∑ i, actual i * decomposition i) (∑ i, ideal i * decomposition i)
      (gain * bound) := by
  classical
  choose error equation bounded using close
  refine ⟨∑ i, error i * action i, ?_, route_family_error_bound hn error action bounded columnGain⟩
  have reduce_sum : reduceMatrix q n rows columns (∑ i, error i * action i) =
      ∑ i, reduceMatrix q n rows columns (error i * action i) := by
    ext row column
    simp [reduceMatrix, Matrix.sum_apply, map_sum]
  rw [reduce_sum]
  simp_rw [equation, Matrix.add_mul, reduceMatrix_mul, ← reduction]
  rw [Finset.sum_add_distrib]

noncomputable def routeTarget {q n columns : Nat} (value : Int)
    (position : Fin n) (component : Fin columns) : ExactMatrix q n 2 columns :=
  fun row column => if row = 1 ∧ column = component then
    (value : ExactPoly q n) * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ position.val else 0

theorem route_target_sparse {q n columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (value : Int) (position : Fin n) (component : Fin columns) :
    SparseTarget (routeTarget (q := q) value position component) (fun _ => 1) (fun _ => position)
      (fun column => if column = component then value else 0) := by
  letI : Fact (1 < q) := ⟨hq⟩
  intro row column k
  simp only [routeTarget]
  by_cases hr : row = 1
  · subst row
    by_cases hc : column = component
    · subst column
      simp only [and_self, ↓reduceIte, true_and]
      change (algebraMap (ZMod q) (ExactPoly q n) (value : ZMod q) *
        AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ position.val).coeff k = _
      rw [Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn position]
      by_cases hk : position = k
      · subst k; simp
      · simp [hk, Ne.symm hk]
    · simp [hc]
  · simp [hr]

private theorem route_balancedDigit_zero (base : Nat) : balancedDigit base 0 = 0 := by
  dsimp only [balancedDigit]
  simp only [Int.zero_emod, Int.zero_ediv]
  split_ifs <;> omega

private theorem route_balancedResidual_zero (base digit : Nat) : balancedResidual base 0 digit = 0 := by
  induction digit with
  | zero => rfl
  | succ digit ih => simpa only [balancedResidual, route_balancedDigit_zero, sub_self, Int.zero_ediv] using ih

private theorem route_sparseColumnGain_zero {q : Nat} (layout : MxxRuntime.RegularLayout q) :
    sparseColumnGain layout 0 = 0 := by
  simp [sparseColumnGain, centeredLift, balancedDigitAt, route_balancedResidual_zero,
    route_balancedDigit_zero]

theorem route_joint_column_gain {q n columns digits : Nat}
    (layout : MxxRuntime.RegularLayout q) (hq : 1 < q) (hn : 0 < n)
    (value : Fin digits → Int) (column : Fin columns) :
    (∑ component : Fin columns, ∑ position : Fin n, ∑ digit : Fin digits,
      ∑ row, polynomialL1
        (regularIntegerDigits layout (routeTarget (value digit) position component) row column)) =
      n * ∑ digit, sparseColumnGain layout (value digit) := by
  classical
  simp_rw [regularIntegerDigits_sparse_column layout hn
    (route_target_sparse hq hn _ _ _)]
  rw [Finset.sum_eq_single column]
  · simp
  · intro component hcomponent hne
    simp [Ne.symm hne, route_sparseColumnGain_zero]
  · simp

private theorem sum_pairs {A : Type} [AddCommMonoid A] (f : Nat → A) (m : Nat) :
    (∑ i ∈ Finset.range m, (f (2*i) + f (2*i+1))) = ∑ i ∈ Finset.range (2*m), f i := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [Finset.sum_range_succ, ih]
    simp only [Nat.mul_succ]
    rw [show 2*m+2 = (2*m+1)+1 by omega, Finset.sum_range_succ, Finset.sum_range_succ]
    exact (add_assoc _ _ _).symm

private theorem sum_padded {A : Type} [AddCommMonoid A] {n m : Nat}
    (f : Fin n → A) (hle : n ≤ m) :
    (∑ i ∈ Finset.range m, if h : i < n then f ⟨i,h⟩ else 0) = ∑ i, f i := by
  classical
  let padded : Nat → A := fun i => if h : i < n then f ⟨i,h⟩ else 0
  have truncate : (∑ i ∈ Finset.range n, padded i) = ∑ i ∈ Finset.range m, padded i := by
    apply Finset.sum_subset (Finset.range_mono hle)
    intro i hi hnot
    simp only [Finset.mem_range, not_lt] at hnot
    simp [padded, show ¬i<n by omega]
  rw [← truncate, ← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro i hi
  simp [padded, i.isLt]

theorem pairwise_sum {A : Type} [AddCommMonoid A] {n : Nat}
    (f : Fin n → A) (g : Fin ((n+1)/2) → A)
    (equation : ∀ i, g i =
      (if h : 2*i.val < n then f ⟨2*i.val,h⟩ else 0) +
      (if h : 2*i.val+1 < n then f ⟨2*i.val+1,h⟩ else 0)) :
    ∑ i, g i = ∑ i, f i := by
  classical
  let padded : Nat → A := fun i => if h : i<n then f ⟨i,h⟩ else 0
  calc
    _ = ∑ i : Fin ((n+1)/2), (padded (2*i.val) + padded (2*i.val+1)) := by
      exact Finset.sum_congr rfl (fun i _ => equation i)
    _ = ∑ i ∈ Finset.range ((n+1)/2), (padded (2*i) + padded (2*i+1)) :=
      Fin.sum_univ_eq_sum_range (fun i => padded (2*i) + padded (2*i+1)) _
    _ = ∑ i ∈ Finset.range (2*((n+1)/2)), padded i := sum_pairs padded _
    _ = _ := sum_padded f (by omega)

theorem route_lookup_eq {α : Type} {count : Nat} {family : Fin count → α}
    {index : Int} {output : α} (position : Fin count)
    (hindex : index = (position.val : Int))
    (h : MxxRuntime.familyGetDynamic family index output) : output = family position := by
  obtain ⟨actual, ha, hv⟩ := h
  have same : actual = position := by apply Fin.ext; omega
  simpa only [same] using hv

theorem route_select_zero {α : Type} {first second output : α}
    (h : MxxRuntime.select 0 [first, second] output) : output = first := by
  obtain ⟨position, hp, hv⟩ := h
  have same : position = (0 : Fin 2) := by apply Fin.ext; change position.val = 0; omega
  subst position
  exact hv

theorem route_select_one {α : Type} {first second output : α}
    (h : MxxRuntime.select 1 [first, second] output) : output = second := by
  obtain ⟨position, hp, hv⟩ := h
  have same : position = (1 : Fin 2) := by apply Fin.ext; change position.val = 1; omega
  subst position
  exact hv

theorem route_family_ideal {q n secret columns : Nat} {ι : Type} [Fintype ι]
    (mask : ExactMatrix q n 1 secret) (payload : ExactMatrix q n 1 2)
    (gadget : ExactMatrix q n 2 columns)
    (publicKey : ι → ExactMatrix q n secret columns)
    (message : ι → ExactPoly q n)
    (action : ι → ExactMatrix q n columns columns)
    (target : ι → ExactMatrix q n 2 columns)
    (reconstruction : ∀ i, gadget * action i = target i) :
    (∑ i, (mask * publicKey i - message i • (payload * gadget)) * action i) =
      mask * (∑ i, publicKey i * action i) - ∑ i, message i • (payload * target i) := by
  classical
  simp_rw [Matrix.sub_mul, Matrix.smul_mul, Matrix.mul_assoc, reconstruction]
  rw [Finset.sum_sub_distrib, Matrix.mul_sum]

theorem route_target_unit {q n columns : Nat}
    (secret : ExactMatrix q n 1 2) (unit : secret 0 1 = 1)
    (value : Int) (position : Fin n) (component : Fin columns) :
    secret * routeTarget (q := q) value position component =
      fun _ column => if column = component then
        (value : ExactPoly q n) * AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ position.val
        else 0 := by
  ext row column
  have hr : row = 0 := Subsingleton.elim _ _
  subst row
  simp [Matrix.mul_apply, Fin.sum_univ_two, routeTarget, unit]

theorem route_sum_flat {A : Type} [AddCommMonoid A] {a b : Nat} (f : Fin (a*b) → A) :
    ∑ i, f i = ∑ x : Fin a, ∑ y : Fin b,
      f ⟨x.val*b+y.val, by have hx := x.isLt; have hy := y.isLt; nlinarith⟩ := by
  calc
    _ = ∑ z : Fin a × Fin b, f (finProdFinEquiv z) := (Equiv.sum_comp finProdFinEquiv f).symm
    _ = _ := by simp only [Fintype.sum_prod_type, finProdFinEquiv, Equiv.coe_fn_mk, Nat.add_comm, Nat.mul_comm]

theorem route_sparse_family_approx {q n columns digits bound : Nat}
    (layout : MxxRuntime.RegularLayout q) (hq : 1 < q) (hn : 0 < n)
    (value : Fin digits → Int)
    (actual ideal : Fin columns → Fin n → Fin digits → ExactMatrix q n 1 (2 * (layout.crtModuli.length * layout.digitsPerTower)))
    (decomposition : Fin columns → Fin n → Fin digits → ExactMatrix q n (2 * (layout.crtModuli.length * layout.digitsPerTower)) columns)
    (close : ∀ c k d, Approx (actual c k d) (ideal c k d) bound)
    (reduction : ∀ c k d, decomposition c k d =
      reduceMatrix q n (2 * (layout.crtModuli.length * layout.digitsPerTower)) columns
        (regularIntegerDigits layout (routeTarget (value d) k c))) :
    Approx (∑ c, ∑ k, ∑ d, actual c k d * decomposition c k d)
      (∑ c, ∑ k, ∑ d, ideal c k d * decomposition c k d)
      ((n * ∑ d, sparseColumnGain layout (value d)) * bound) := by
  classical
  have h := route_family_approx (ι := Fin columns × Fin n × Fin digits) hn
    (fun i => actual i.1 i.2.1 i.2.2) (fun i => ideal i.1 i.2.1 i.2.2)
    (fun i => decomposition i.1 i.2.1 i.2.2)
    (fun i => regularIntegerDigits layout (routeTarget (value i.2.2) i.2.1 i.1))
    (fun i => close i.1 i.2.1 i.2.2) (fun i => reduction i.1 i.2.1 i.2.2)
    (gain := n * ∑ d, sparseColumnGain layout (value d)) ?_
  · simpa only [Fintype.sum_prod_type] using h
  · intro column
    simp only [Fintype.sum_prod_type]
    exact (route_joint_column_gain layout hq hn value column).le

theorem route_native_target_sum {q n columns width digits : Nat}
    (payload : ExactMatrix q n 1 2) (unit : payload 0 1 = 1)
    (scale : Int) (base : Nat) (native : Fin (2*width) → ExactPoly q n)
    (selection : Fin 1 → Fin columns → Fin n → Fin digits → Fin (2*width)) :
    (∑ component : Fin columns, ∑ coefficient : Fin n, ∑ digit : Fin digits,
      native (selection 0 component coefficient digit) •
        (payload * routeTarget (-(scale : Int) * (base : Int)^digit.val) coefficient component)) =
    fun _ column => -(scale : ExactPoly q n) *
      (∑ coefficient : Fin n, (∑ digit : Fin digits,
        (base : ExactPoly q n)^digit.val * native (selection 0 column coefficient digit)) *
        AdjoinRoot.root (negacyclicModulus n (ZMod q))^coefficient.val) := by
  classical
  simp_rw [route_target_unit payload unit]
  ext row column
  simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul]
  simp_rw [mul_ite, mul_zero, Finset.sum_ite_irrel]
  simp only [Finset.sum_const_zero, Finset.sum_ite_eq, Finset.mem_univ, if_true]
  simp only [Int.cast_mul, Int.cast_neg, Int.cast_pow, Int.cast_natCast,
    Finset.mul_sum, Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro k hk
  apply Finset.sum_congr rfl
  intro d hd
  ring

theorem route_native_ideal {q n columns width digits : Nat}
    (mask payload : ExactMatrix q n 1 2) (unit : payload 0 1 = 1)
    (gadget : ExactMatrix q n 2 columns)
    (publicKey : Fin columns → Fin n → Fin digits → ExactMatrix q n 2 columns)
    (action : Fin columns → Fin n → Fin digits → ExactMatrix q n columns columns)
    (scale : Int) (base : Nat) (native : Fin (2*width) → ExactPoly q n)
    (selection : Fin 1 → Fin columns → Fin n → Fin digits → Fin (2*width))
    (reconstruction : ∀ c k d, gadget * action c k d =
      routeTarget (-scale * (base : Int)^d.val) k c) :
    (∑ c, ∑ k, ∑ d, (mask * publicKey c k d - native (selection 0 c k d) •
        (payload * gadget)) * action c k d) =
      mask * (∑ c, ∑ k, ∑ d, publicKey c k d * action c k d) +
        (show ExactMatrix q n 1 columns from fun (_ : Fin 1) column => (scale : ExactPoly q n) *
          (∑ k : Fin n, (∑ d : Fin digits,
            (base : ExactPoly q n)^d.val * native (selection 0 column k d)) *
            AdjoinRoot.root (negacyclicModulus n (ZMod q))^k.val)) := by
  classical
  have ideal := route_family_ideal (ι := Fin columns × Fin n × Fin digits)
    mask payload gadget (fun i => publicKey i.1 i.2.1 i.2.2)
    (fun i => native (selection 0 i.1 i.2.1 i.2.2))
    (fun i => action i.1 i.2.1 i.2.2)
    (fun i => routeTarget (-scale * (base : Int)^i.2.2.val) i.2.1 i.1)
    (fun i => reconstruction i.1 i.2.1 i.2.2)
  simp only [Fintype.sum_prod_type] at ideal
  rw [ideal, route_native_target_sum payload unit scale base native selection]
  ext row column
  simp only [Matrix.sub_apply, Matrix.add_apply, neg_mul, sub_neg_eq_add]

theorem route_sum_three {A : Type} [AddCommMonoid A] {columns n digits : Nat}
    (f : Fin (columns*n*digits) → A) :
    ∑ i, f i = ∑ c : Fin columns, ∑ k : Fin n, ∑ d : Fin digits,
      f ⟨(c.val*n+k.val)*digits+d.val, by
        have hc := c.isLt; have hk := k.isLt; have hd := d.isLt
        have hck : c.val*n+k.val < columns*n := by nlinarith
        have hm := Nat.mul_le_mul_right digits (Nat.succ_le_of_lt hck)
        nlinarith⟩ := by
  rw [route_sum_flat (a := columns*n) (b := digits), route_sum_flat (a := columns) (b := n)]

end MxxExponentLut
