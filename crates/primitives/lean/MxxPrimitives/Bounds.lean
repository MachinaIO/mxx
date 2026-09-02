import MxxPrimitives.Matrix
import Mathlib.Data.Finset.Max
import Mathlib.Data.ZMod.ValMinAbs

namespace Mxx.Primitives

open scoped BigOperators

private theorem coeff_sum_generic {R : Type u} [CommRing R]
    {α : Type v} [DecidableEq α] {n : Nat} (s : Finset α)
    (f : α → Negacyclic n R) (k : Fin n) :
    (∑ i ∈ s, f i).coeff k = ∑ i ∈ s, (f i).coeff k := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    simp only [Finset.sum_insert ha, Negacyclic.coeff_add, ih]

theorem reducePoly_coeff {q n : Nat} (hq : 1 < q) (hn : 0 < n) (x : ErrorPoly n) (i : Fin n) :
    (reducePoly q n x).coeff i = (x.coeff i : ZMod q) := by
  letI : Fact (1 < q) := ⟨hq⟩
  calc
    (reducePoly q n x).coeff i =
        (reducePoly q n (∑ j : Fin n,
          algebraMap Int (ErrorPoly n) (x.coeff j) *
            AdjoinRoot.root (negacyclicModulus n Int) ^ (j : Nat))).coeff i := by
      exact congrArg (fun z : ErrorPoly n => (reducePoly q n z).coeff i)
        (Negacyclic.expansion hn x)
    _ = (∑ j : Fin n,
          reducePoly q n (algebraMap Int (ErrorPoly n) (x.coeff j) *
            AdjoinRoot.root (negacyclicModulus n Int) ^ (j : Nat))).coeff i := by
      rw [map_sum]
    _ = (x.coeff i : ZMod q) := by
      rw [coeff_sum_generic]
      rw [Finset.sum_eq_single i]
      · have hmap_int (a : Int) :
            reducePoly q n (algebraMap Int (ErrorPoly n) a) =
              algebraMap (ZMod q) (ExactPoly q n) (a : ZMod q) := by
          simp [reducePoly]
        have hmap_root :
            reducePoly q n (AdjoinRoot.root (negacyclicModulus n Int)) =
              AdjoinRoot.root (negacyclicModulus n (ZMod q)) := by
          simp [reducePoly]
        rw [map_mul, hmap_int, map_pow, hmap_root,
          Negacyclic.coeff_smul,
          Negacyclic.coeff_root_pow (R := ZMod q) hn]
        simp
      · intro j hj hji
        have hmap_int (a : Int) :
            reducePoly q n (algebraMap Int (ErrorPoly n) a) =
              algebraMap (ZMod q) (ExactPoly q n) (a : ZMod q) := by
          simp [reducePoly]
        have hmap_root :
            reducePoly q n (AdjoinRoot.root (negacyclicModulus n Int)) =
              AdjoinRoot.root (negacyclicModulus n (ZMod q)) := by
          simp [reducePoly]
        rw [map_mul, hmap_int, map_pow, hmap_root,
          Negacyclic.coeff_smul,
          Negacyclic.coeff_root_pow (R := ZMod q) hn]
        simp [hji]
      · simp

theorem valMinAbs_intCast_eq_of_natAbs_lt_half {q : Nat} (hq : 1 < q)
    (z : Int) (hz : z.natAbs < q / 2) :
    (z : ZMod q).valMinAbs = z := by
  letI : Fact (1 < q) := ⟨hq⟩
  apply (ZMod.valMinAbs_spec (z : ZMod q) z).2
  constructor
  · rfl
  · change -(q : Int) < z * 2 ∧ z * 2 ≤ (q : Int)
    cases z with
    | ofNat value =>
        change -(q : Int) < (value : Int) * 2 ∧ (value : Int) * 2 ≤ (q : Int)
        change value < q / 2 at hz
        have hnat : 2 * value ≤ q := by omega
        have hq' : (1 : Int) < q := by exact_mod_cast hq
        have hupper : (value : Int) * 2 ≤ q := by
          exact_mod_cast (show value * 2 ≤ q by simpa [Nat.mul_comm] using hnat)
        constructor
        · omega
        · exact hupper
    | negSucc value =>
        simp only [Int.negSucc_eq]
        change -(q : Int) < -((value : Int) + 1) * 2 ∧ -((value : Int) + 1) * 2 ≤ (q : Int)
        change value + 1 < q / 2 at hz
        have htwo : 2 * (value + 1) < q := by omega
        have htwo' : (2 : Int) * (value + 1) < q := by exact_mod_cast htwo
        constructor
        · omega
        · omega

variable {q n rows columns : Nat} {Parameter : Type u}

inductive BoundExpr (Parameter : Type u)
  | literal (value : Nat)
  | parameter (name : Parameter)
  | add (left right : BoundExpr Parameter)
  | mul (left right : BoundExpr Parameter)
  | max (left right : BoundExpr Parameter)

def BoundExpr.eval (environment : Parameter → Nat) : BoundExpr Parameter → Nat
  | .literal value => value
  | .parameter name => environment name
  | .add left right => eval environment left + eval environment right
  | .mul left right => eval environment left * eval environment right
  | .max left right => Nat.max (eval environment left) (eval environment right)

@[simp] theorem BoundExpr.eval_literal (environment : Parameter → Nat) (value : Nat) :
    BoundExpr.eval environment (.literal value) = value := rfl

@[simp] theorem BoundExpr.eval_parameter (environment : Parameter → Nat) (name : Parameter) :
    BoundExpr.eval environment (.parameter name) = environment name := rfl

noncomputable def polyNorm (x : ErrorPoly n) : Nat :=
  Finset.univ.sup (fun index => (x.coeff index).natAbs)

noncomputable def matrixNorm (x : ErrorMatrix n rows columns) : Nat :=
  Finset.univ.sup (fun row =>
    Finset.univ.sup (fun column => polyNorm (x row column)))

theorem coeff_natAbs_le_polyNorm (x : ErrorPoly n) (i : Fin n) :
    (x.coeff i).natAbs ≤ polyNorm x :=
  Finset.le_sup (s := Finset.univ) (f := fun j => (x.coeff j).natAbs)
    (Finset.mem_univ i)

@[simp] theorem polyNorm_zero : polyNorm (0 : ErrorPoly n) = 0 := by
  apply Nat.le_antisymm
  · apply Finset.sup_le
    intro index hindex
    simp
  · exact Nat.zero_le _

theorem polyNorm_neg (x : ErrorPoly n) : polyNorm (-x) = polyNorm x := by
  simp only [polyNorm, Negacyclic.coeff_neg, Int.natAbs_neg]

theorem polyNorm_add_le (x y : ErrorPoly n) :
    polyNorm (x + y) ≤ polyNorm x + polyNorm y := by
  apply Finset.sup_le
  intro index hindex
  rw [Negacyclic.coeff_add]
  exact (Int.natAbs_add_le _ _).trans (Nat.add_le_add
    (Finset.le_sup (s := Finset.univ) (f := fun j => (x.coeff j).natAbs) hindex)
    (Finset.le_sup (s := Finset.univ) (f := fun j => (y.coeff j).natAbs) hindex))

theorem polyNorm_sum_le {α : Type u} [DecidableEq α] (values : Finset α)
    (f : α → ErrorPoly n) :
    polyNorm (∑ value ∈ values, f value) ≤ ∑ value ∈ values, polyNorm (f value) := by
  induction values using Finset.induction_on with
  | empty => simp
  | @insert value values hnotmem ih =>
    simp only [Finset.sum_insert hnotmem]
    exact (polyNorm_add_le _ _).trans (Nat.add_le_add_left ih _)

theorem polyNorm_int_smul_root_le (hn : 0 < n) (a : Int) (i : Fin n) :
    polyNorm (algebraMap Int (ErrorPoly n) a *
      AdjoinRoot.root (negacyclicModulus n Int) ^ (i : Nat)) ≤ a.natAbs := by
  apply Finset.sup_le
  intro k hk
  rw [Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  split <;> simp

theorem polyNorm_scaled_root_mul_le (hn : 0 < n) (a b : Int) (i j : Fin n) :
    polyNorm ((algebraMap Int (ErrorPoly n) a *
      AdjoinRoot.root (negacyclicModulus n Int) ^ (i : Nat)) *
      (algebraMap Int (ErrorPoly n) b *
        AdjoinRoot.root (negacyclicModulus n Int) ^ (j : Nat))) ≤
      a.natAbs * b.natAbs := by
  have hscalar :
      (algebraMap Int (ErrorPoly n) a) * (algebraMap Int (ErrorPoly n) b) =
        algebraMap Int (ErrorPoly n) (a * b) :=
    ((algebraMap Int (ErrorPoly n)).map_mul a b).symm
  calc
    polyNorm ((algebraMap Int (ErrorPoly n) a *
        AdjoinRoot.root (negacyclicModulus n Int) ^ (i : Nat)) *
        (algebraMap Int (ErrorPoly n) b *
          AdjoinRoot.root (negacyclicModulus n Int) ^ (j : Nat))) =
        polyNorm (algebraMap Int (ErrorPoly n) (a * b) *
          (AdjoinRoot.root (negacyclicModulus n Int) ^ (i : Nat) *
            AdjoinRoot.root (negacyclicModulus n Int) ^ (j : Nat))) := by
      congr 1
      rw [← hscalar]
      ring
    _ = polyNorm (algebraMap Int (ErrorPoly n) (a * b) *
          (if (i : Nat) + (j : Nat) < n then
            AdjoinRoot.root (negacyclicModulus n Int) ^ ((i : Nat) + (j : Nat))
          else -AdjoinRoot.root (negacyclicModulus n Int) ^
            ((i : Nat) + (j : Nat) - n))) := by
      rw [Negacyclic.root_pow_mul]
  split
  · calc
      _ ≤ (a * b).natAbs := polyNorm_int_smul_root_le hn (a * b) ⟨_, by omega⟩
      _ = a.natAbs * b.natAbs := Int.natAbs_mul a b
  · have hneg :
        algebraMap Int (ErrorPoly n) (a * b) *
            -AdjoinRoot.root (negacyclicModulus n Int) ^
              ((i : Nat) + (j : Nat) - n) =
          -(algebraMap Int (ErrorPoly n) (a * b) *
            AdjoinRoot.root (negacyclicModulus n Int) ^
              ((i : Nat) + (j : Nat) - n)) := by ring
    rw [hneg, polyNorm_neg]
    calc
      _ ≤ (a * b).natAbs := polyNorm_int_smul_root_le hn (a * b) ⟨_, by omega⟩
      _ = a.natAbs * b.natAbs := Int.natAbs_mul a b

/- For a fixed output coefficient `k`, exactly one input coefficient of the
   right factor can contribute for each coefficient `i` of the left factor. -/
noncomputable def matchingIndex (i k : Fin n) : Fin n :=
  if h : i.val ≤ k.val then ⟨k.val - i.val, by omega⟩
  else ⟨n + k.val - i.val, by
    have hi := i.isLt
    have hk := k.isLt
    have hki : k.val < i.val := by omega
    have hpos : 0 < i.val - k.val := Nat.sub_pos_of_lt hki
    have hlt : n + k.val - i.val < n := by omega
    exact hlt⟩

private theorem mod_two_mul_cases {s n k : Nat} (hn : 0 < n) (hs : s < n + n)
    (hmod : s % n = k) :
    (s < n ∧ s = k) ∨ (n ≤ s ∧ s = n + k) := by
  by_cases hlt : s < n
  · left
    exact ⟨hlt, by rw [Nat.mod_eq_of_lt hlt] at hmod; exact hmod⟩
  · right
    have hge : n ≤ s := Nat.le_of_not_gt hlt
    have hsub : s - n < n := by omega
    rw [Nat.mod_eq_sub_mod hge, Nat.mod_eq_of_lt hsub] at hmod
    exact ⟨hge, by omega⟩

theorem matchingIndex_unique (i j k : Fin n) :
    (i.val + j.val) % n = k.val ↔ j = matchingIndex i k := by
  have hi := i.isLt
  have hj := j.isLt
  have hk := k.isLt
  have hn : 0 < n := by omega
  have hs : i.val + j.val < n + n := by omega
  unfold matchingIndex
  by_cases hik : i.val ≤ k.val
  · rw [dif_pos hik]
    constructor
    · intro h
      obtain ⟨hno, hsum⟩ | ⟨hwrap, hsum⟩ := mod_two_mul_cases hn hs h
      · apply Fin.ext
        change j.val = k.val - i.val
        omega
      · exfalso
        omega
    · intro h
      have hsum : i.val + (k.val - i.val) < n := by omega
      subst j
      change (i.val + (k.val - i.val)) % n = k.val
      rw [Nat.mod_eq_of_lt hsum]
      omega
  · rw [dif_neg hik]
    constructor
    · intro h
      obtain ⟨hno, hsum⟩ | ⟨hwrap, hsum⟩ := mod_two_mul_cases hn hs h
      · exfalso
        omega
      · apply Fin.ext
        change j.val = n + k.val - i.val
        omega
    · intro h
      have hsum : i.val + (n + k.val - i.val) = n + k.val := by omega
      subst j
      change (i.val + (n + k.val - i.val)) % n = k.val
      rw [hsum, Nat.mod_eq_sub_mod (by omega)]
      have hnk : n + k.val - n = k.val := by omega
      rw [hnk, Nat.mod_eq_of_lt hk]

private theorem coeff_sum {α : Type u} [DecidableEq α] (s : Finset α)
    (f : α → ErrorPoly n) (k : Fin n) :
    (∑ i ∈ s, f i).coeff k = ∑ i ∈ s, (f i).coeff k := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    simp only [Finset.sum_insert ha, Negacyclic.coeff_add, ih]

private theorem coeff_sum_univ (f : Fin n → ErrorPoly n) (k : Fin n) :
    (∑ i : Fin n, f i).coeff k = ∑ i : Fin n, (f i).coeff k := by
  simpa using coeff_sum (n := n) (s := (Finset.univ : Finset (Fin n))) f k

private theorem natAbs_sum_le {α : Type u} [DecidableEq α] (s : Finset α)
    (f : α → Int) :
    (∑ i ∈ s, f i).natAbs ≤ ∑ i ∈ s, (f i).natAbs := by
  induction s using Finset.induction_on with
  | empty => simp
  | @insert a s ha ih =>
    calc
      (∑ i ∈ insert a s, f i).natAbs ≤
          (f a).natAbs + (∑ i ∈ s, f i).natAbs := by
        rw [Finset.sum_insert ha]
        exact Int.natAbs_add_le _ _
      _ ≤ (f a).natAbs + ∑ i ∈ s, (f i).natAbs := Nat.add_le_add_left ih _
      _ = ∑ i ∈ insert a s, (f i).natAbs := by rw [Finset.sum_insert ha]

noncomputable def scaledBasis (a : Int) (i : Fin n) : ErrorPoly n :=
  algebraMap Int (ErrorPoly n) a * AdjoinRoot.root (negacyclicModulus n Int) ^ i.val

theorem coeff_scaled_basis_mul (hn : 0 < n) (a b : Int) (i j k : Fin n) :
    (scaledBasis a i * scaledBasis b j).coeff k =
      if (i.val + j.val) % n = k.val then
        if i.val + j.val < n then a * b else -(a * b)
      else 0 := by
  unfold scaledBasis
  have hscalar :
      (algebraMap Int (ErrorPoly n) a) * (algebraMap Int (ErrorPoly n) b) =
        algebraMap Int (ErrorPoly n) (a * b) :=
    ((algebraMap Int (ErrorPoly n)).map_mul a b).symm
  rw [show
      (algebraMap Int (ErrorPoly n) a * AdjoinRoot.root (negacyclicModulus n Int) ^ i.val) *
          (algebraMap Int (ErrorPoly n) b * AdjoinRoot.root (negacyclicModulus n Int) ^ j.val) =
        algebraMap Int (ErrorPoly n) (a * b) *
          (AdjoinRoot.root (negacyclicModulus n Int) ^ i.val *
            AdjoinRoot.root (negacyclicModulus n Int) ^ j.val) by
        rw [← hscalar]
        ring]
  rw [Negacyclic.root_pow_mul]
  split
  · have hsum : i.val + j.val < n := by omega
    have hc := Negacyclic.coeff_root_pow (R := Int) hn
      (⟨i.val + j.val, hsum⟩ : Fin n) k
    rw [Negacyclic.coeff_smul, hc]
    have hiff : (⟨i.val + j.val, hsum⟩ : Fin n) = k ↔
        (i.val + j.val) % n = k.val := by
      rw [Nat.mod_eq_of_lt hsum]
      constructor
      · intro h
        exact congrArg Fin.val h
      · intro h
        exact Fin.ext h
    by_cases hfin : (⟨i.val + j.val, hsum⟩ : Fin n) = k
    · have hmod : (i.val + j.val) % n = k.val := hiff.mp hfin
      simp [hfin, hmod]
    · have hmod : ¬(i.val + j.val) % n = k.val := fun h => hfin (hiff.mpr h)
      simp [hfin, hmod]
  · have hsum : n ≤ i.val + j.val := by omega
    have hsub : i.val + j.val - n < n := by omega
    have hc := Negacyclic.coeff_root_pow (R := Int) hn
      (⟨i.val + j.val - n, hsub⟩ : Fin n) k
    rw [Negacyclic.coeff_smul, Negacyclic.coeff_neg, hc]
    have hmodbase : (i.val + j.val) % n = i.val + j.val - n := by
      rw [Nat.mod_eq_sub_mod (by omega), Nat.mod_eq_of_lt hsub]
    have hiff : (⟨i.val + j.val - n, hsub⟩ : Fin n) = k ↔
        (i.val + j.val) % n = k.val := by
      rw [hmodbase]
      constructor
      · intro h
        exact congrArg Fin.val h
      · intro h
        exact Fin.ext h
    by_cases hfin : (⟨i.val + j.val - n, hsub⟩ : Fin n) = k
    · have hmod : (i.val + j.val) % n = k.val := hiff.mp hfin
      simp [hfin, hmod]
    · have hmod : ¬(i.val + j.val) % n = k.val := fun h => hfin (hiff.mpr h)
      simp [hfin, hmod]

theorem polyNorm_mul_le_tight (hn : 0 < n) (x y : ErrorPoly n) :
    polyNorm (x * y) ≤ n * polyNorm x * polyNorm y := by
  let term : Fin n → Fin n → ErrorPoly n := fun i j =>
    scaledBasis (x.coeff i) i * scaledBasis (y.coeff j) j
  have hsum : x * y = ∑ i : Fin n, ∑ j : Fin n, term i j := by
    calc
      x * y =
          (∑ i : Fin n, scaledBasis (x.coeff i) i) *
          (∑ j : Fin n, scaledBasis (y.coeff j) j) := by
        conv_lhs => rw [Negacyclic.expansion hn x, Negacyclic.expansion hn y]
        simp only [scaledBasis]
      _ = ∑ i : Fin n, ∑ j : Fin n, term i j := by
        rw [Finset.sum_mul_sum]
  rw [hsum]
  apply Finset.sup_le
  intro k hk
  rw [coeff_sum_univ]
  calc
    (∑ i : Fin n, (∑ j : Fin n, term i j).coeff k).natAbs ≤
        ∑ i : Fin n, ((∑ j : Fin n, term i j).coeff k).natAbs := by
      exact natAbs_sum_le (Finset.univ : Finset (Fin n))
        (fun i => (∑ j : Fin n, term i j).coeff k)
    _ ≤ ∑ i : Fin n, polyNorm x * polyNorm y := by
      apply Finset.sum_le_sum
      intro i hi
      rw [coeff_sum_univ]
      rw [Finset.sum_eq_single (matchingIndex i k)]
      · exact (coeff_natAbs_le_polyNorm _ _).trans (by
          dsimp [term]
          exact (polyNorm_scaled_root_mul_le hn _ _ _ _).trans
            (Nat.mul_le_mul
              (Finset.le_sup (s := Finset.univ) (f := fun r => (x.coeff r).natAbs)
                (Finset.mem_univ i))
              (Finset.le_sup (s := Finset.univ) (f := fun r => (y.coeff r).natAbs)
                (Finset.mem_univ (matchingIndex i k)))))
      · intro j hj hne
        dsimp [term]
        rw [coeff_scaled_basis_mul hn]
        have hnon : ¬(i.val + j.val) % n = k.val := by
          intro h
          exact hne ((matchingIndex_unique i j k).mp h)
        simp [hnon]
      · simp
    _ = n * polyNorm x * polyNorm y := by
      simp [Finset.sum_const, Fintype.card_fin]
      ring

example : matchingIndex (⟨0, by omega⟩ : Fin 2) (⟨1, by omega⟩ : Fin 2) =
    (⟨1, by omega⟩ : Fin 2) := by
  rfl

example : matchingIndex (⟨1, by omega⟩ : Fin 2) (⟨0, by omega⟩ : Fin 2) =
    (⟨1, by omega⟩ : Fin 2) := by
  rfl

example : matchingIndex (⟨2, by omega⟩ : Fin 3) (⟨0, by omega⟩ : Fin 3) =
    (⟨1, by omega⟩ : Fin 3) := by
  rfl

theorem polyNorm_mul_le (hn : 0 < n) (x y : ErrorPoly n) :
    polyNorm (x * y) ≤ n * polyNorm x * polyNorm y :=
  polyNorm_mul_le_tight hn x y

theorem polyNorm_int_smul_le (a : Int) (x : ErrorPoly n) :
    polyNorm (algebraMap Int (ErrorPoly n) a * x) ≤ a.natAbs * polyNorm x := by
  apply Finset.sup_le
  intro i hi
  rw [Negacyclic.coeff_smul, Int.natAbs_mul]
  exact Nat.mul_le_mul_left _
    (Finset.le_sup (s := Finset.univ) (f := fun k => (x.coeff k).natAbs)
      (Finset.mem_univ i))

@[simp] theorem matrixNorm_zero : matrixNorm (0 : ErrorMatrix n rows columns) = 0 := by
  apply Nat.le_antisymm
  · apply Finset.sup_le
    intro row hrow
    apply Finset.sup_le
    intro column hcolumn
    simp
  · exact Nat.zero_le _

theorem matrixNorm_add_le (x y : ErrorMatrix n rows columns) :
    matrixNorm (x + y) ≤ matrixNorm x + matrixNorm y := by
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  apply (polyNorm_add_le _ _).trans
  apply Nat.add_le_add
  · exact (Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (x row c)) hcolumn).trans
      (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.univ.sup
        (fun c => polyNorm (x r c))) hrow)
  · exact (Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (y row c)) hcolumn).trans
      (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.univ.sup
        (fun c => polyNorm (y r c))) hrow)

theorem matrixNorm_two_smul_le
    {rows columns : Nat} (x : ErrorMatrix n rows columns) :
    matrixNorm ((2 : ErrorPoly n) • x) ≤ 2 * matrixNorm x := by
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  change polyNorm ((2 : ErrorPoly n) * x row column) ≤ _
  have htwo : (2 : ErrorPoly n) = algebraMap Int (ErrorPoly n) (2 : Int) := by
    norm_num
  rw [htwo]
  exact (polyNorm_int_smul_le 2 _).trans (Nat.mul_le_mul_left 2
    ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (x row c))
      (Finset.mem_univ column)).trans
      (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
        (fun c => polyNorm (x r c))) hrow)))

theorem matrixNorm_mul_le
    (hn : 0 < n) {inner : Nat} {a : ErrorMatrix n rows inner}
    {b : ErrorMatrix n inner columns} :
    matrixNorm (a * b) ≤ inner * n * matrixNorm a * matrixNorm b := by
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  change polyNorm (∑ index : Fin inner, a row index * b index column) ≤ _
  calc
    polyNorm (∑ index : Fin inner, a row index * b index column) ≤
        ∑ index : Fin inner, polyNorm (a row index * b index column) :=
      polyNorm_sum_le (Finset.univ : Finset (Fin inner)) _
    _ ≤ ∑ index : Fin inner, n * polyNorm (a row index) * polyNorm (b index column) := by
      apply Finset.sum_le_sum
      intro index hindex
      exact polyNorm_mul_le_tight hn (a row index) (b index column)
    _ ≤ ∑ index : Fin inner, n * matrixNorm a * matrixNorm b := by
      apply Finset.sum_le_sum
      intro index hindex
      calc
        n * polyNorm (a row index) * polyNorm (b index column) ≤
            n * (matrixNorm a * matrixNorm b) := by
          have hprod : polyNorm (a row index) * polyNorm (b index column) ≤
              matrixNorm a * matrixNorm b := Nat.mul_le_mul
            ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (a row c))
              (Finset.mem_univ index)).trans
              (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
                (fun c => polyNorm (a r c))) hrow))
            ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (b index c))
              hcolumn).trans
              (by simpa [matrixNorm] using
                (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
                  (fun c => polyNorm (b r c))) (Finset.mem_univ index))))
          simpa only [Nat.mul_assoc] using Nat.mul_le_mul_left n hprod
        _ = n * matrixNorm a * matrixNorm b := by ring
    _ = inner * n * matrixNorm a * matrixNorm b := by
      simp [Finset.sum_const, Fintype.card_fin]
      ring

structure BoundedLift (actual : ExactMatrix q n rows columns) (bound : Nat) where
  witness : ErrorMatrix n rows columns
  reduce_eq : reduceMatrix q n rows columns witness = actual
  norm_le : matrixNorm witness ≤ bound

structure Approx (actual ideal : ExactMatrix q n rows columns) where
  error : ErrorMatrix n rows columns
  equation : actual = ideal + reduceMatrix q n rows columns error

structure ApproxWithin (actual ideal : ExactMatrix q n rows columns) (bound : Nat) where
  toApprox : Approx actual ideal
  norm_le : matrixNorm toApprox.error ≤ bound

namespace ApproxWithin

variable {q n rows columns bound : Nat}
variable {actual ideal : ExactMatrix q n rows columns}

def error (h : ApproxWithin actual ideal bound) : ErrorMatrix n rows columns :=
  h.toApprox.error

def equation (h : ApproxWithin actual ideal bound) :
    actual = ideal + reduceMatrix q n rows columns h.toApprox.error :=
  h.toApprox.equation

end ApproxWithin

noncomputable def centeredMatrixDistance
    (q n rows columns : Nat)
    (x y : ExactMatrix q n rows columns) : Nat :=
  Finset.univ.sup (fun row =>
    Finset.univ.sup (fun column =>
      Finset.univ.sup (fun coefficient =>
        (ZMod.valMinAbs
          ((x row column).coeff coefficient -
           (y row column).coeff coefficient)).natAbs)))

theorem centeredMatrixDistance_le_of_approx
    {q n rows columns bound : Nat} (hq : 1 < q) (hn : 0 < n)
    {actual ideal : ExactMatrix q n rows columns}
    (approximation : ApproxWithin actual ideal bound)
    (hbound : bound < q / 2) :
    centeredMatrixDistance q n rows columns actual ideal ≤ bound := by
  letI : Fact (1 < q) := ⟨hq⟩
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  apply Finset.sup_le
  intro coefficient hcoefficient
  have hcoeff :
      ((approximation.error row column).coeff coefficient).natAbs ≤ bound := by
    exact (coeff_natAbs_le_polyNorm _ _).trans
      ((Finset.le_sup (s := Finset.univ)
        (f := fun c => polyNorm (approximation.error row c))
        (Finset.mem_univ column)).trans
        ((Finset.le_sup (s := Finset.univ)
          (f := fun r => Finset.sup Finset.univ
            (fun c => polyNorm (approximation.error r c)))
          (Finset.mem_univ row)).trans approximation.norm_le))
  have hcoeff_lt :
      ((approximation.error row column).coeff coefficient).natAbs < q / 2 :=
    hcoeff.trans_lt hbound
  have heq := congrArg (fun value : ExactMatrix q n rows columns =>
      (value row column).coeff coefficient) (approximation.equation)
  have hdiff :
      (actual row column).coeff coefficient - (ideal row column).coeff coefficient =
        ((approximation.error row column).coeff coefficient : ZMod q) := by
    have heq' : (actual row column).coeff coefficient =
        ((ideal + reduceMatrix q n rows columns approximation.toApprox.error)
          row column).coeff coefficient := heq
    rw [heq']
    rw [Matrix.add_apply, Negacyclic.coeff_add, reduceMatrix_apply,
      reducePoly_coeff hq hn]
    simp [ApproxWithin.error]
  rw [hdiff, valMinAbs_intCast_eq_of_natAbs_lt_half hq _ hcoeff_lt]
  exact hcoeff

inductive SupportClass
  | constant
  | arbitrary

/- A constant polynomial has support only at coefficient zero.  This predicate
is deliberately independent of the representation used by the quotient ring. -/
def IsConstantPoly (x : ErrorPoly n) : Prop :=
  ∀ (coefficient : Fin n), coefficient.val ≠ 0 → x.coeff coefficient = 0

def IsConstantMatrix (x : ErrorMatrix n rows columns) : Prop :=
  ∀ (row : Fin rows) (column : Fin columns), IsConstantPoly (x row column)

theorem constant_eq_int_smul (hn : 0 < n) (x : ErrorPoly n)
    (hconstant : IsConstantPoly x) :
    x = algebraMap Int (ErrorPoly n) (x.coeff ⟨0, hn⟩) := by
  conv_lhs => rw [Negacyclic.expansion hn x]
  rw [Finset.sum_eq_single ⟨0, hn⟩]
  · simp
  · intro i hi hne
    have hnezero : (i : Nat) ≠ 0 := by
      intro hz
      apply hne
      exact Fin.ext hz
    simp [hconstant i hnezero]
  · intro hzero
    exact False.elim (hzero (Finset.mem_univ (⟨0, hn⟩ : Fin n)))

theorem polyNorm_constant_eq (hn : 0 < n) (x : ErrorPoly n)
    (hconstant : IsConstantPoly x) :
    polyNorm x = (x.coeff ⟨0, hn⟩).natAbs := by
  apply Nat.le_antisymm
  · apply Finset.sup_le
    intro i hi
    by_cases hzero : (i : Nat) = 0
    · have hi_eq : i = (⟨0, hn⟩ : Fin n) := by
        apply Fin.ext
        simpa using hzero
      simpa [hi_eq]
    · exact (hconstant i hzero).symm ▸ Nat.zero_le _
  · exact Finset.le_sup (s := Finset.univ)
      (f := fun i => (x.coeff i).natAbs) (Finset.mem_univ (⟨0, hn⟩ : Fin n))

theorem polyNorm_constant_mul_le (hn : 0 < n) (x y : ErrorPoly n)
    (hconstant : IsConstantPoly x) :
    polyNorm (x * y) ≤ polyNorm x * polyNorm y := by
  calc
    polyNorm (x * y) =
        polyNorm (algebraMap Int (ErrorPoly n) (x.coeff ⟨0, hn⟩) * y) := by
      exact congrArg (fun z => polyNorm (z * y)) (constant_eq_int_smul hn x hconstant)
    _ ≤ (x.coeff ⟨0, hn⟩).natAbs * polyNorm y := polyNorm_int_smul_le _ _
    _ = polyNorm x * polyNorm y :=
      (congrArg (fun z => z * polyNorm y) (polyNorm_constant_eq hn x hconstant)).symm

theorem polyNorm_mul_constant_le (hn : 0 < n) (x y : ErrorPoly n)
    (hconstant : IsConstantPoly y) :
    polyNorm (x * y) ≤ polyNorm x * polyNorm y := by
  rw [mul_comm]
  exact (polyNorm_constant_mul_le hn y x hconstant).trans_eq (Nat.mul_comm _ _)

theorem matrixNorm_mul_left_constant_le
    (hn : 0 < n) {inner : Nat} {a : ErrorMatrix n rows inner}
    {b : ErrorMatrix n inner columns}
    (hconstant : IsConstantMatrix a) :
    matrixNorm (a * b) ≤ inner * matrixNorm a * matrixNorm b := by
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  change polyNorm (∑ index : Fin inner, a row index * b index column) ≤ _
  calc
    polyNorm (∑ index : Fin inner, a row index * b index column) ≤
        ∑ index : Fin inner, polyNorm (a row index * b index column) :=
      polyNorm_sum_le (Finset.univ : Finset (Fin inner)) _
    _ ≤ ∑ index : Fin inner, polyNorm (a row index) * polyNorm (b index column) := by
      apply Finset.sum_le_sum
      intro index hindex
      exact polyNorm_constant_mul_le hn _ _ (hconstant row index)
    _ ≤ ∑ index : Fin inner, matrixNorm a * matrixNorm b := by
      apply Finset.sum_le_sum
      intro index hindex
      exact Nat.mul_le_mul
        ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (a row c))
          (Finset.mem_univ index)).trans
          (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
            (fun c => polyNorm (a r c))) hrow))
        ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (b index c))
          hcolumn).trans
          (by simpa [matrixNorm] using
            (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
              (fun c => polyNorm (b r c))) (Finset.mem_univ index))))
    _ = inner * matrixNorm a * matrixNorm b := by
      simp [Finset.sum_const, Fintype.card_fin]
      ring

theorem matrixNorm_mul_right_constant_le
    (hn : 0 < n) {inner : Nat} {a : ErrorMatrix n rows inner}
    {b : ErrorMatrix n inner columns}
    (hconstant : IsConstantMatrix b) :
    matrixNorm (a * b) ≤ inner * matrixNorm a * matrixNorm b := by
  apply Finset.sup_le
  intro row hrow
  apply Finset.sup_le
  intro column hcolumn
  change polyNorm (∑ index : Fin inner, a row index * b index column) ≤ _
  calc
    polyNorm (∑ index : Fin inner, a row index * b index column) ≤
        ∑ index : Fin inner, polyNorm (a row index * b index column) :=
      polyNorm_sum_le (Finset.univ : Finset (Fin inner)) _
    _ ≤ ∑ index : Fin inner, polyNorm (a row index) * polyNorm (b index column) := by
      apply Finset.sum_le_sum
      intro index hindex
      exact polyNorm_mul_constant_le hn _ _ (hconstant index column)
    _ ≤ ∑ index : Fin inner, matrixNorm a * matrixNorm b := by
      apply Finset.sum_le_sum
      intro index hindex
      exact Nat.mul_le_mul
        ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (a row c))
          (Finset.mem_univ index)).trans
          (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
            (fun c => polyNorm (a r c))) hrow))
        ((Finset.le_sup (s := Finset.univ) (f := fun c => polyNorm (b index c))
          hcolumn).trans
          (by simpa [matrixNorm] using
            (Finset.le_sup (s := Finset.univ) (f := fun r => Finset.sup Finset.univ
              (fun c => polyNorm (b r c))) (Finset.mem_univ index))))
    _ = inner * matrixNorm a * matrixNorm b := by
      simp [Finset.sum_const, Fintype.card_fin]
      ring

theorem matrixNorm_constant_mul_le
    (hn : 0 < n) {inner : Nat} {a : ErrorMatrix n rows inner}
    {b : ErrorMatrix n inner columns}
    (hconstant : IsConstantMatrix a) :
    matrixNorm (a * b) ≤ inner * matrixNorm a * matrixNorm b :=
  matrixNorm_mul_left_constant_le hn hconstant

theorem matrixNorm_mul_constant_le
    (hn : 0 < n) {inner : Nat} {a : ErrorMatrix n rows inner}
    {b : ErrorMatrix n inner columns}
    (hconstant : IsConstantMatrix b) :
    matrixNorm (a * b) ≤ inner * matrixNorm a * matrixNorm b :=
  matrixNorm_mul_right_constant_le hn hconstant

structure MagnitudeFact (actual : ExactMatrix q n rows columns) where
  lift : ErrorMatrix n rows columns
  reduce_eq : reduceMatrix q n rows columns lift = actual
  bound : Nat
  norm_le : matrixNorm lift ≤ bound
  support : SupportClass
  support_valid : support = .constant →
    ∀ (row : Fin rows) (column : Fin columns) (coefficient : Fin n),
      coefficient.val ≠ 0 → (lift row column).coeff coefficient = 0

/- Multiplying two magnitude facts retains only the one inner-dimension factor required by the
   tight negacyclic matrix-product estimate.  The product is intentionally classified as
   arbitrary: callers may use the constant-side lemmas separately when that structure is known. -/
noncomputable def MagnitudeFact.mul
    {inner : Nat}
    {a : ExactMatrix q n rows inner}
    {b : ExactMatrix q n inner columns}
    (hn : 0 < n)
    (aFact : MagnitudeFact a)
    (bFact : MagnitudeFact b) :
    MagnitudeFact (a * b) :=
  { lift := aFact.lift * bFact.lift
    reduce_eq := by
      rw [reduceMatrix_mul, aFact.reduce_eq, bFact.reduce_eq]
    bound := inner * n * aFact.bound * bFact.bound
    norm_le := by
      exact (matrixNorm_mul_le hn).trans (by
        calc
          inner * n * matrixNorm aFact.lift * matrixNorm bFact.lift ≤
              inner * n * (aFact.bound * bFact.bound) := by
            simpa [Nat.mul_assoc] using Nat.mul_le_mul_left (inner * n)
              (Nat.mul_le_mul aFact.norm_le bFact.norm_le)
          _ = inner * n * aFact.bound * bFact.bound := by ring)
    support := .arbitrary
    support_valid := by
      intro hsupport
      cases hsupport }

theorem MagnitudeFact.isConstant_of_support_constant
    {x : ExactMatrix q n rows columns} (fact : MagnitudeFact x)
    (hsupport : fact.support = SupportClass.constant) : IsConstantMatrix fact.lift := by
  intro row column coefficient hcoefficient
  exact fact.support_valid hsupport row column coefficient hcoefficient

def ApproxWithin.approx {actual ideal : ExactMatrix q n rows columns} {bound : Nat}
    (h : ApproxWithin actual ideal bound) : Approx actual ideal := by
  exact h.toApprox

def BoundedLift.approx {actual : ExactMatrix q n rows columns} {bound : Nat}
    (lift : BoundedLift actual bound) :
    ApproxWithin (reduceMatrix q n rows columns lift.witness)
      (0 : ExactMatrix q n rows columns) bound := by
  refine { toApprox := { error := lift.witness, equation := ?_ }, norm_le := lift.norm_le }
  simp only [zero_add]

end Mxx.Primitives
