import ExponentMaskForm

/-!
# Weights on errors

The terminal error of a decoded coefficient is a linear function of every error and residue of
the run. A real weight `W` on the coefficients of an integer matrix pairs with it
(`pairing W e`); an operation that maps errors linearly pulls the weight back, so the terminal
error is the sum, over the sources, of their pairings with pulled-back weights. The norms of a
pulled-back weight are bounded by the row norms of the operation: `l1` gains the row `l1` of the
action, and the energy (squared `l2`) is bounded by the squared `l1` of the weight times the
largest row energy of the action.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- A real weight on the coefficients of an integer matrix. -/
abbrev Weight (n rows columns : Nat) := Fin rows → Fin columns → Fin n → ℝ

/-- The pairing of a weight with an integer matrix. -/
noncomputable def pairing {rows columns : Nat} (W : Weight n rows columns)
    (e : ErrorMatrix n rows columns) : ℝ :=
  ∑ r, ∑ c, ∑ k, W r c k * (((e r c).coeff k : ℤ) : ℝ)

section Pairing

variable {rows columns : Nat} (W : Weight n rows columns)

theorem pairing_add (e f : ErrorMatrix n rows columns) :
    pairing W (e + f) = pairing W e + pairing W f := by
  simp only [pairing, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦
    Finset.sum_congr rfl fun k _ ↦ ?_
  simp only [Matrix.add_apply, Negacyclic.coeff_add, Int.cast_add]
  ring

theorem pairing_sub (e f : ErrorMatrix n rows columns) :
    pairing W (e - f) = pairing W e - pairing W f := by
  simp only [pairing, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦
    Finset.sum_congr rfl fun k _ ↦ ?_
  simp only [Matrix.sub_apply, Negacyclic.coeff_sub, Int.cast_sub]
  ring

theorem pairing_zero : pairing W (0 : ErrorMatrix n rows columns) = 0 := by
  simp [pairing]

theorem pairing_sum {ι : Type} [DecidableEq ι] (s : Finset ι) (e : ι → ErrorMatrix n rows columns) :
    pairing W (∑ i ∈ s, e i) = ∑ i ∈ s, pairing W (e i) := by
  induction s using Finset.induction_on with
  | empty => simp [pairing_zero]
  | @insert i s hi ih => rw [Finset.sum_insert hi, Finset.sum_insert hi, pairing_add, ih]

/-- An exact division by `p`. -/
theorem pairing_div {p : ℤ} (hp : p ≠ 0) {v w : ErrorMatrix n rows columns}
    (h : ∀ r c, w r c = (p : ErrorPoly n) * v r c) : pairing W v = pairing W w / p := by
  have hp' : (p : ℝ) ≠ 0 := Int.cast_ne_zero.mpr hp
  rw [eq_div_iff hp', pairing, pairing, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ ↦ ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun c _ ↦ ?_
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [h r c, intCast_mul_coeff, Int.cast_mul]
  ring

/-- The `l1` norm of a weight. -/
noncomputable def weightL1 : ℝ := ∑ r, ∑ c, ∑ k, |W r c k|

/-- The energy (squared `l2` norm) of a weight. -/
noncomputable def weightEnergy : ℝ := ∑ r, ∑ c, ∑ k, W r c k ^ 2

theorem weightL1_nonneg : 0 ≤ weightL1 W :=
  Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ abs_nonneg _

/-- A pairing is at most the `l1` norm of the weight times a coefficient bound. -/
theorem abs_pairing_le {e : ErrorMatrix n rows columns} {B : Nat} (he : CoeffBound e B) :
    |pairing W e| ≤ weightL1 W * B := by
  unfold pairing weightL1
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun r _ ↦ (Finset.abs_sum_le_sum_abs _ _).trans ?_
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun c _ ↦ (Finset.abs_sum_le_sum_abs _ _).trans ?_
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun k _ ↦ ?_
  rw [abs_mul]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  have := he r c k
  rw [← Int.cast_abs, Int.abs_eq_natAbs]
  exact_mod_cast this

end Pairing

/-! ## Right multiplication -/

/-- The negacyclic signs of a fixed `j` and `k` have at most one nonzero `i`. -/
theorem sum_abs_negSign_le (j k : Fin n) : ∑ i : Fin n, |(negSign n i j k : ℝ)| ≤ 1 := by
  classical
  by_cases hn : n = 0
  · subst hn; exact j.elim0
  have hle : ∀ i, |(negSign n i j k : ℝ)| ≤ if (i.val + j.val) % n = k.val then 1 else 0 := by
    intro i
    unfold negSign
    split_ifs <;> simp
  refine (Finset.sum_le_sum fun i _ ↦ hle i).trans ?_
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
  have hcard : (Finset.univ.filter fun i : Fin n ↦ (i.val + j.val) % n = k.val).card ≤ 1 := by
    refine Finset.card_le_one.mpr fun a ha b hb ↦ ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    have ha' := a.isLt
    have hb' := b.isLt
    have hj := j.isLt
    apply Fin.ext
    have : (a.val + j.val) % n = (b.val + j.val) % n := ha.trans hb.symm
    have h2 := Nat.ModEq.add_right_cancel' j.val this
    exact Nat.ModEq.eq_of_lt_of_lt h2 ha' hb'
  exact_mod_cast hcard

/-- A weight pulled back through the right multiplication by `d`. -/
noncomputable def pullMul {rows inner columns : Nat} (W : Weight n rows columns)
    (d : ErrorMatrix n inner columns) : Weight n rows inner :=
  fun r l i ↦ ∑ c, ∑ k, W r c k * ∑ j, (((d l c).coeff j : ℤ) : ℝ) * (negSign n i j k : ℝ)

theorem pairing_mul (hn : 0 < n) {rows inner columns : Nat} (W : Weight n rows columns)
    (e : ErrorMatrix n rows inner) (d : ErrorMatrix n inner columns) :
    pairing W (e * d) = pairing (pullMul W d) e := by
  simp only [pairing, pullMul, Matrix.mul_apply, Negacyclic.coeff_sum, coeff_mul_eq hn,
    Int.cast_sum, Int.cast_mul, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ ↦ ?_
  conv_lhs => enter [2, c]; rw [Finset.sum_comm]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun l _ ↦ ?_
  conv_lhs => enter [2, c]; rw [Finset.sum_comm]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun c _ ↦
    Finset.sum_congr rfl fun k _ ↦ Finset.sum_congr rfl fun j _ ↦ ?_
  ring

theorem negSign_comm (i j k : Fin n) : negSign n i j k = negSign n j i k := by
  unfold negSign
  rw [Nat.add_comm i.val j.val]

theorem sum_abs_negSign_le' (i k : Fin n) : ∑ j : Fin n, |(negSign n i j k : ℝ)| ≤ 1 := by
  simpa only [negSign_comm i] using sum_abs_negSign_le i k

/-- Weighted Cauchy–Schwarz: `(Σ a b)² ≤ (Σ |a|) (Σ |a| b²)`. -/
theorem sq_sum_mul_le {ι : Type} (s : Finset ι) (a b : ι → ℝ) :
    (∑ x ∈ s, a x * b x) ^ 2 ≤ (∑ x ∈ s, |a x|) * ∑ x ∈ s, |a x| * b x ^ 2 := by
  have h1 : |∑ x ∈ s, a x * b x| ≤ ∑ x ∈ s, Real.sqrt |a x| * (Real.sqrt |a x| * |b x|) := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (le_of_eq (Finset.sum_congr rfl fun x _ ↦ ?_))
    rw [abs_mul, ← mul_assoc, Real.mul_self_sqrt (abs_nonneg _)]
  have h2 := Finset.sum_mul_sq_le_sq_mul_sq s (fun x ↦ Real.sqrt |a x|)
    (fun x ↦ Real.sqrt |a x| * |b x|)
  have h3 : (∑ x ∈ s, a x * b x) ^ 2 ≤
      (∑ x ∈ s, Real.sqrt |a x| * (Real.sqrt |a x| * |b x|)) ^ 2 := by
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) h1 2
  refine h3.trans (h2.trans (le_of_eq ?_))
  congr 1
  · exact Finset.sum_congr rfl fun x _ ↦ Real.sq_sqrt (abs_nonneg _)
  · refine Finset.sum_congr rfl fun x _ ↦ ?_
    rw [mul_pow, Real.sq_sqrt (abs_nonneg _), sq_abs]

/-- The kernel of `pullMul`: column `c`, coefficient `k`, read at row `l`, coefficient `i`. -/
theorem abs_mulKernel_le {inner columns : Nat} {d : ErrorMatrix n inner columns} {β : Nat}
    (hd : CoeffBound d β) (l : Fin inner) (c : Fin columns) (i k : Fin n) :
    |∑ j, (((d l c).coeff j : ℤ) : ℝ) * (negSign n i j k : ℝ)| ≤ β := by
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  have hb : ∀ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ β := fun j ↦ by
    have := hd l c j
    rw [← Int.cast_abs, Int.abs_eq_natAbs]
    exact_mod_cast this
  calc ∑ j, |(((d l c).coeff j : ℤ) : ℝ) * (negSign n i j k : ℝ)|
      ≤ ∑ j, (β : ℝ) * |(negSign n i j k : ℝ)| := Finset.sum_le_sum fun j _ ↦ by
        rw [abs_mul]; exact mul_le_mul_of_nonneg_right (hb j) (abs_nonneg _)
    _ = β * ∑ j, |(negSign n i j k : ℝ)| := by rw [Finset.mul_sum]
    _ ≤ β * 1 := mul_le_mul_of_nonneg_left (sum_abs_negSign_le' i k) (Nat.cast_nonneg _)
    _ = β := mul_one _

/-- The `l1` norm of a weight pulled back through `d` gains the largest column `l1` norm `S` of
`d`. -/
theorem weightL1_pullMul_le_column {rows inner columns : Nat} (W : Weight n rows columns)
    {d : ErrorMatrix n inner columns} {S : ℝ}
    (hd : ∀ c, ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ S) :
    weightL1 (pullMul W d) ≤ weightL1 W * S := by
  have hpt : ∀ r l i, |pullMul W d r l i| ≤ ∑ c, ∑ k, |W r c k| *
      ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| * |(negSign n i j k : ℝ)| := by
    intro r l i
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun c _ ↦ ?_)
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun k _ ↦ ?_)
    rw [abs_mul]
    refine mul_le_mul_of_nonneg_left ((Finset.abs_sum_le_sum_abs _ _).trans
      (le_of_eq (Finset.sum_congr rfl fun j _ ↦ abs_mul _ _))) (abs_nonneg _)
  unfold weightL1
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun r _ ↦ ?_
  calc ∑ l, ∑ i, |pullMul W d r l i|
      ≤ ∑ l, ∑ i, ∑ c, ∑ k, |W r c k| *
          ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| * |(negSign n i j k : ℝ)| :=
        Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun i _ ↦ hpt r l i
    _ = ∑ c, ∑ k, |W r c k| * ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| *
          ∑ i, |(negSign n i j k : ℝ)| := by
        set F : Fin inner → Fin n → Fin columns → Fin n → Fin n → ℝ := fun l i c k j ↦
          |W r c k| * (|(((d l c).coeff j : ℤ) : ℝ)| * |(negSign n i j k : ℝ)|) with hF
        have hL : ∀ l i, (∑ c, ∑ k, |W r c k| *
            ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| * |(negSign n i j k : ℝ)|) =
            ∑ c, ∑ k, ∑ j, F l i c k j := fun l i ↦ by simp only [hF, Finset.mul_sum]
        have hR : ∀ c k, |W r c k| * ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| *
            ∑ i, |(negSign n i j k : ℝ)| = ∑ l, ∑ i, ∑ j, F l i c k j := fun c k ↦ by
          simp only [hF, Finset.mul_sum]
          refine Finset.sum_congr rfl fun l _ ↦ ?_
          rw [Finset.sum_comm]
        simp only [hL, hR]
        calc ∑ l, ∑ i, ∑ c, ∑ k, ∑ j, F l i c k j
            = ∑ l, ∑ c, ∑ i, ∑ k, ∑ j, F l i c k j :=
              Finset.sum_congr rfl fun l _ ↦ Finset.sum_comm
          _ = ∑ l, ∑ c, ∑ k, ∑ i, ∑ j, F l i c k j :=
              Finset.sum_congr rfl fun l _ ↦ Finset.sum_congr rfl fun c _ ↦ Finset.sum_comm
          _ = ∑ c, ∑ l, ∑ k, ∑ i, ∑ j, F l i c k j := Finset.sum_comm
          _ = ∑ c, ∑ k, ∑ l, ∑ i, ∑ j, F l i c k j :=
              Finset.sum_congr rfl fun c _ ↦ Finset.sum_comm
    _ ≤ ∑ c, ∑ k, |W r c k| * S := by
        refine Finset.sum_le_sum fun c _ ↦ Finset.sum_le_sum fun k _ ↦
          mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
        calc ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| * ∑ i, |(negSign n i j k : ℝ)|
            ≤ ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| * 1 :=
              Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun j _ ↦
                mul_le_mul_of_nonneg_left (sum_abs_negSign_le j k) (abs_nonneg _)
          _ ≤ S := by simpa using hd c
    _ = (∑ c, ∑ k, |W r c k|) * S := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun c _ ↦ by rw [Finset.sum_mul]

/-- The `l1` norm of a weight pulled back through digits of size `β` gains `inner · n · β`. -/
theorem weightL1_pullMul_le {rows inner columns : Nat} (W : Weight n rows columns)
    {d : ErrorMatrix n inner columns} {β : Nat} (hd : CoeffBound d β) :
    weightL1 (pullMul W d) ≤ weightL1 W * (inner * n * β) := by
  refine weightL1_pullMul_le_column W fun c ↦ ?_
  calc ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ ∑ _l : Fin inner, ∑ _j : Fin n, (β : ℝ) :=
        Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun j _ ↦ by
          have := hd l c j
          rw [← Int.cast_abs, Int.abs_eq_natAbs]
          exact_mod_cast this
    _ = inner * n * β := by simp [Finset.card_univ]; ring

/-- The energy of a weight pulled back through digits of size `β` is at most the squared `l1`
norm of the weight times the row energy `inner · n · β²`. -/
theorem weightEnergy_pullMul_le {rows inner columns : Nat} (W : Weight n rows columns)
    {d : ErrorMatrix n inner columns} {β : Nat} (hd : CoeffBound d β) :
    weightEnergy (pullMul W d) ≤ weightL1 W ^ 2 * (inner * n * β ^ 2) := by
  set K : Fin inner → Fin columns → Fin n → Fin n → ℝ := fun l c i k ↦
    ∑ j, (((d l c).coeff j : ℤ) : ℝ) * (negSign n i j k : ℝ) with hK
  have hKb : ∀ l c i k, K l c i k ^ 2 ≤ (β : ℝ) ^ 2 := fun l c i k ↦ by
    have := abs_mulKernel_le hd l c i k
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) this 2
  set L1 : Fin rows → ℝ := fun r ↦ ∑ c, ∑ k, |W r c k| with hL1
  have hrow : ∀ r l i, pullMul W d r l i ^ 2 ≤ L1 r * ∑ c, ∑ k, |W r c k| * K l c i k ^ 2 := by
    intro r l i
    have h := sq_sum_mul_le (Finset.univ ×ˢ Finset.univ)
      (fun x : Fin columns × Fin n ↦ W r x.1 x.2) (fun x ↦ K l x.1 i x.2)
    simp only [Finset.sum_product] at h
    exact h
  have hrowsum : ∀ r, ∑ l, ∑ i, pullMul W d r l i ^ 2 ≤ L1 r * (L1 r * (inner * n * β ^ 2)) := by
    intro r
    calc ∑ l, ∑ i, pullMul W d r l i ^ 2
        ≤ ∑ l, ∑ i, L1 r * ∑ c, ∑ k, |W r c k| * K l c i k ^ 2 :=
          Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun i _ ↦ hrow r l i
      _ = L1 r * ∑ c, ∑ k, |W r c k| * ∑ l, ∑ i, K l c i k ^ 2 := by
          simp only [← Finset.mul_sum]
          congr 1
          calc ∑ l, ∑ i, ∑ c, ∑ k, |W r c k| * K l c i k ^ 2
              = ∑ l, ∑ c, ∑ i, ∑ k, |W r c k| * K l c i k ^ 2 :=
                Finset.sum_congr rfl fun l _ ↦ Finset.sum_comm
            _ = ∑ c, ∑ l, ∑ i, ∑ k, |W r c k| * K l c i k ^ 2 := Finset.sum_comm
            _ = ∑ c, ∑ l, ∑ k, ∑ i, |W r c k| * K l c i k ^ 2 :=
                Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun l _ ↦ Finset.sum_comm
            _ = ∑ c, ∑ k, ∑ l, ∑ i, |W r c k| * K l c i k ^ 2 :=
                Finset.sum_congr rfl fun c _ ↦ Finset.sum_comm
            _ = ∑ c, ∑ k, |W r c k| * ∑ l, ∑ i, K l c i k ^ 2 := by simp only [Finset.mul_sum]
      _ ≤ L1 r * ∑ c, ∑ k, |W r c k| * (inner * n * β ^ 2) := by
          refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun c _ ↦ Finset.sum_le_sum
            fun k _ ↦ mul_le_mul_of_nonneg_left ?_ (abs_nonneg _))
            (Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ abs_nonneg _)
          calc ∑ l, ∑ i, K l c i k ^ 2 ≤ ∑ _l : Fin inner, ∑ _i : Fin n, (β : ℝ) ^ 2 :=
                Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun i _ ↦ hKb l c i k
            _ = inner * n * β ^ 2 := by simp [Finset.card_univ]; ring
      _ = L1 r * (L1 r * (inner * n * β ^ 2)) := by
          congr 1
          simp only [hL1, Finset.sum_mul]
  have hL1nn : ∀ r, 0 ≤ L1 r := fun r ↦
    Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ abs_nonneg _
  have hsq : ∑ r, L1 r ^ 2 ≤ (∑ r, L1 r) ^ 2 := by
    rw [sq (∑ r, L1 r), Finset.sum_mul_sum]
    refine Finset.sum_le_sum fun r _ ↦ ?_
    rw [sq]
    exact Finset.single_le_sum (f := fun r' ↦ L1 r * L1 r') (fun r' _ ↦ mul_nonneg (hL1nn r) (hL1nn r'))
      (Finset.mem_univ r)
  unfold weightEnergy
  calc ∑ r, ∑ l, ∑ i, pullMul W d r l i ^ 2
      ≤ ∑ r, L1 r * (L1 r * (inner * n * β ^ 2)) := Finset.sum_le_sum fun r _ ↦ hrowsum r
    _ = (∑ r, L1 r ^ 2) * (inner * n * β ^ 2) := by
        rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun r _ ↦ by ring
    _ ≤ (∑ r, L1 r) ^ 2 * (inner * n * β ^ 2) :=
        mul_le_mul_of_nonneg_right hsq (by positivity)
    _ = weightL1 W ^ 2 * (inner * n * β ^ 2) := rfl

/-! ## Families of right multiplications -/

theorem pairing_weight_sum {ι : Type} [Fintype ι] {rows columns : Nat}
    (W : ι → Weight n rows columns) (e : ErrorMatrix n rows columns) :
    pairing (∑ x, W x) e = ∑ x, pairing (W x) e := by
  set f : ι → Fin rows → Fin columns → Fin n → ℝ := fun x r c k ↦
    W x r c k * (((e r c).coeff k : ℤ) : ℝ) with hf
  have hL : pairing (∑ x, W x) e = ∑ r, ∑ c, ∑ k, ∑ x, f x r c k := by
    simp only [pairing, Finset.sum_apply, Finset.sum_mul, hf]
  have hR : ∑ x, pairing (W x) e = ∑ x, ∑ r, ∑ c, ∑ k, f x r c k := rfl
  rw [hL, hR]
  calc ∑ r, ∑ c, ∑ k, ∑ x, f x r c k
      = ∑ r, ∑ c, ∑ x, ∑ k, f x r c k :=
        Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦ Finset.sum_comm
    _ = ∑ r, ∑ x, ∑ c, ∑ k, f x r c k := Finset.sum_congr rfl fun r _ ↦ Finset.sum_comm
    _ = ∑ x, ∑ r, ∑ c, ∑ k, f x r c k := Finset.sum_comm

/-- The sum of the weights `W x` pulled back through `d x`. -/
noncomputable def pullMulFam {ι : Type} [Fintype ι] {rows inner columns : Nat}
    (W : ι → Weight n rows columns) (d : ι → ErrorMatrix n inner columns) : Weight n rows inner :=
  ∑ x, pullMul (W x) (d x)

theorem pairing_mulFam (hn : 0 < n) {ι : Type} [Fintype ι] {rows inner columns : Nat}
    (W : ι → Weight n rows columns) (e : ErrorMatrix n rows inner)
    (d : ι → ErrorMatrix n inner columns) :
    ∑ x, pairing (W x) (e * d x) = pairing (pullMulFam W d) e := by
  rw [pullMulFam, pairing_weight_sum]
  exact Finset.sum_congr rfl fun x _ ↦ pairing_mul hn _ _ _

theorem weightL1_add {rows columns : Nat} (W V : Weight n rows columns) :
    weightL1 (W + V) ≤ weightL1 W + weightL1 V := by
  unfold weightL1
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun r _ ↦ ?_
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun c _ ↦ ?_
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun k _ ↦ abs_add_le _ _

theorem weightL1_finsetSum {ι : Type} [DecidableEq ι] {rows columns : Nat} (s : Finset ι)
    (W : ι → Weight n rows columns) : weightL1 (∑ x ∈ s, W x) ≤ ∑ x ∈ s, weightL1 (W x) := by
  induction s using Finset.induction_on with
  | empty =>
    simp only [Finset.sum_empty]
    simp [weightL1]
  | @insert x s hx ih =>
    rw [Finset.sum_insert hx, Finset.sum_insert hx]
    exact (weightL1_add _ _).trans (by linarith)

/-- The energy of a family pulled back through digits of size `β` is at most the squared total
`l1` norm of the weights times the row energy `inner · n · β²`. -/
theorem weightEnergy_pullMulFam_le {ι : Type} [Fintype ι] {rows inner columns : Nat}
    (W : ι → Weight n rows columns) {d : ι → ErrorMatrix n inner columns} {β : Nat}
    (hd : ∀ x, CoeffBound (d x) β) :
    weightEnergy (pullMulFam W d) ≤ (∑ x, weightL1 (W x)) ^ 2 * (inner * n * β ^ 2) := by
  classical
  set K : ι → Fin inner → Fin columns → Fin n → Fin n → ℝ := fun x l c i k ↦
    ∑ j, (((d x l c).coeff j : ℤ) : ℝ) * (negSign n i j k : ℝ) with hK
  have hKb : ∀ x l c i k, K x l c i k ^ 2 ≤ (β : ℝ) ^ 2 := fun x l c i k ↦ by
    have := abs_mulKernel_le (hd x) l c i k
    rw [← sq_abs]
    exact pow_le_pow_left₀ (abs_nonneg _) this 2
  set L1 : Fin rows → ℝ := fun r ↦ ∑ x, ∑ c, ∑ k, |W x r c k| with hL1
  have hpull : ∀ r l i, pullMulFam W d r l i = ∑ x, ∑ c, ∑ k, W x r c k * K x l c i k := by
    intro r l i
    simp only [pullMulFam, pullMul, Finset.sum_apply, hK]
  have hrow : ∀ r l i, pullMulFam W d r l i ^ 2 ≤
      L1 r * ∑ x, ∑ c, ∑ k, |W x r c k| * K x l c i k ^ 2 := by
    intro r l i
    have h := sq_sum_mul_le (Finset.univ ×ˢ (Finset.univ ×ˢ Finset.univ))
      (fun y : ι × (Fin columns × Fin n) ↦ W y.1 r y.2.1 y.2.2) (fun y ↦ K y.1 l y.2.1 i y.2.2)
    simp only [Finset.sum_product] at h
    rw [hpull]
    exact h
  have hrowsum : ∀ r, ∑ l, ∑ i, pullMulFam W d r l i ^ 2 ≤
      L1 r * (L1 r * (inner * n * β ^ 2)) := by
    intro r
    have hK2 : ∀ x c k, ∑ l, ∑ i, K x l c i k ^ 2 ≤ inner * n * β ^ 2 := fun x c k ↦ by
      calc ∑ l, ∑ i, K x l c i k ^ 2 ≤ ∑ _l : Fin inner, ∑ _i : Fin n, (β : ℝ) ^ 2 :=
            Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun i _ ↦ hKb x l c i k
        _ = inner * n * β ^ 2 := by simp [Finset.card_univ]; ring
    set g : ι → Fin columns → Fin n → Fin inner → Fin n → ℝ := fun x c k l i ↦
      |W x r c k| * K x l c i k ^ 2 with hg
    calc ∑ l, ∑ i, pullMulFam W d r l i ^ 2
        ≤ ∑ l, ∑ i, L1 r * ∑ x, ∑ c, ∑ k, g x c k l i :=
          Finset.sum_le_sum fun l _ ↦ Finset.sum_le_sum fun i _ ↦ hrow r l i
      _ = L1 r * ∑ x, ∑ c, ∑ k, ∑ l, ∑ i, g x c k l i := by
          simp only [← Finset.mul_sum]
          congr 1
          calc ∑ l, ∑ i, ∑ x, ∑ c, ∑ k, g x c k l i
              = ∑ l, ∑ x, ∑ c, ∑ k, ∑ i, g x c k l i := by
                refine Finset.sum_congr rfl fun l _ ↦ ?_
                rw [Finset.sum_comm]
                refine Finset.sum_congr rfl fun x _ ↦ ?_
                rw [Finset.sum_comm]
                refine Finset.sum_congr rfl fun c _ ↦ ?_
                rw [Finset.sum_comm]
            _ = ∑ x, ∑ c, ∑ k, ∑ l, ∑ i, g x c k l i := by
                rw [Finset.sum_comm]
                refine Finset.sum_congr rfl fun x _ ↦ ?_
                rw [Finset.sum_comm]
                refine Finset.sum_congr rfl fun c _ ↦ ?_
                rw [Finset.sum_comm]
      _ ≤ L1 r * ∑ x, ∑ c, ∑ k, |W x r c k| * (inner * n * β ^ 2) := by
          refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun x _ ↦ Finset.sum_le_sum
            fun c _ ↦ Finset.sum_le_sum fun k _ ↦ ?_)
            (Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦
              Finset.sum_nonneg fun _ _ ↦ abs_nonneg _)
          simp only [hg, ← Finset.mul_sum]
          exact mul_le_mul_of_nonneg_left (hK2 x c k) (abs_nonneg _)
      _ = L1 r * (L1 r * (inner * n * β ^ 2)) := by
          congr 1
          simp only [hL1, Finset.sum_mul]
  have hL1nn : ∀ r, 0 ≤ L1 r := fun r ↦ Finset.sum_nonneg fun _ _ ↦
    Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ abs_nonneg _
  have hsq : ∑ r, L1 r ^ 2 ≤ (∑ r, L1 r) ^ 2 := by
    rw [sq (∑ r, L1 r), Finset.sum_mul_sum]
    refine Finset.sum_le_sum fun r _ ↦ ?_
    rw [sq]
    exact Finset.single_le_sum (f := fun r' ↦ L1 r * L1 r')
      (fun r' _ ↦ mul_nonneg (hL1nn r) (hL1nn r')) (Finset.mem_univ r)
  have htot : ∑ r, L1 r = ∑ x, weightL1 (W x) := by
    simp only [hL1, weightL1]
    exact Finset.sum_comm
  unfold weightEnergy
  calc ∑ r, ∑ l, ∑ i, pullMulFam W d r l i ^ 2
      ≤ ∑ r, L1 r * (L1 r * (inner * n * β ^ 2)) := Finset.sum_le_sum fun r _ ↦ hrowsum r
    _ = (∑ r, L1 r ^ 2) * (inner * n * β ^ 2) := by
        rw [Finset.sum_mul]; exact Finset.sum_congr rfl fun r _ ↦ by ring
    _ ≤ (∑ r, L1 r) ^ 2 * (inner * n * β ^ 2) :=
        mul_le_mul_of_nonneg_right hsq (by positivity)
    _ = (∑ x, weightL1 (W x)) ^ 2 * (inner * n * β ^ 2) := by rw [htot]

theorem weightL1_pullMulFam_le {ι : Type} [Fintype ι] {rows inner columns : Nat}
    (W : ι → Weight n rows columns) {d : ι → ErrorMatrix n inner columns} {β : Nat}
    (hd : ∀ x, CoeffBound (d x) β) :
    weightL1 (pullMulFam W d) ≤ (∑ x, weightL1 (W x)) * (inner * n * β) := by
  classical
  refine (weightL1_finsetSum Finset.univ _).trans ?_
  rw [Finset.sum_mul]
  exact Finset.sum_le_sum fun x _ ↦ weightL1_pullMul_le (W x) (hd x)

/-! ## Stacked actions on pairs of rows -/

/-- The pairing of a pair of row weights with a pair of rows. -/
noncomputable def pairing2 {m : Nat} (W : Fin 2 → Weight n 1 m) (e : Fin 2 → ErrorMatrix n 1 m) : ℝ :=
  ∑ j, pairing (W j) (e j)

/-- A pair of row weights pulled back through the stacked action `[e₀ | e₁] · d_j`. -/
noncomputable def pullPairMul {m m' : Nat} (W : Fin 2 → Weight n 1 m')
    (d : Fin 2 → ErrorMatrix n (m + m) m') : Fin 2 → Weight n 1 m :=
  ![pullMulFam W fun j ↦ topRows (d j), pullMulFam W fun j ↦ bottomRows (d j)]

theorem pairing2_pairMul (hn : 0 < n) {m m' : Nat} (W : Fin 2 → Weight n 1 m')
    (e : Fin 2 → ErrorMatrix n 1 m) (d : Fin 2 → ErrorMatrix n (m + m) m') :
    pairing2 W (fun j ↦ pairMul e (d j)) = pairing2 (pullPairMul W d) e := by
  simp only [pairing2, pairMul, pairing_add, Finset.sum_add_distrib]
  rw [pairing_mulFam hn W (e 0) (fun j ↦ topRows (d j)),
    pairing_mulFam hn W (e 1) (fun j ↦ bottomRows (d j)), Fin.sum_univ_two]
  rfl

theorem coeffBound_topRows {m m' : Nat} {d : ErrorMatrix n (m + m) m'} {β : Nat}
    (hd : CoeffBound d β) : CoeffBound (topRows d) β := fun r c k ↦ hd _ c k

theorem coeffBound_bottomRows {m m' : Nat} {d : ErrorMatrix n (m + m) m'} {β : Nat}
    (hd : CoeffBound d β) : CoeffBound (bottomRows d) β := fun r c k ↦ hd _ c k

/-- The total `l1` norm of a pair weight gains `(m + m) n β` through a stacked action. -/
theorem weightL1_pullPairMul_le {m m' : Nat} (W : Fin 2 → Weight n 1 m')
    {d : Fin 2 → ErrorMatrix n (m + m) m'} {β : Nat} (hd : ∀ j, CoeffBound (d j) β) :
    ∑ j, weightL1 (pullPairMul W d j) ≤ (∑ j, weightL1 (W j)) * ((m + m) * n * β) := by
  rw [Fin.sum_univ_two]
  have h0 := weightL1_pullMulFam_le W (d := fun j ↦ topRows (d j)) (β := β)
    (fun j ↦ coeffBound_topRows (hd j)) (inner := m)
  have h1 := weightL1_pullMulFam_le W (d := fun j ↦ bottomRows (d j)) (β := β)
    (fun j ↦ coeffBound_bottomRows (hd j)) (inner := m)
  simp only [pullPairMul, Matrix.cons_val_zero, Matrix.cons_val_one]
  push_cast at h0 h1 ⊢
  nlinarith [h0, h1]

/-- Each component of a pair weight pulled back through a stacked action has the energy of the
row action. -/
theorem weightEnergy_pullPairMul_le {m m' : Nat} (W : Fin 2 → Weight n 1 m')
    {d : Fin 2 → ErrorMatrix n (m + m) m'} {β : Nat} (hd : ∀ j, CoeffBound (d j) β) (j' : Fin 2) :
    weightEnergy (pullPairMul W d j') ≤ (∑ j, weightL1 (W j)) ^ 2 * (m * n * β ^ 2) := by
  have h0 := weightEnergy_pullMulFam_le W (fun j ↦ coeffBound_topRows (hd j))
  have h1 := weightEnergy_pullMulFam_le W (fun j ↦ coeffBound_bottomRows (hd j))
  rcases Fin.exists_fin_two.mp ⟨j', rfl⟩ with h | h <;> subst h
  · simpa only [pullPairMul, Matrix.cons_val_zero] using h0
  · simpa only [pullPairMul, Matrix.cons_val_one, Matrix.cons_val_fin_one] using h1

/-- The pairing with a stacked action splits over the two rows. -/
theorem pairing_pairMul (hn : 0 < n) {m m' : Nat} (V : Weight n 1 m')
    (y : Fin 2 → ErrorMatrix n 1 m) (d : ErrorMatrix n (m + m) m') :
    pairing V (pairMul y d) =
      pairing (pullMul V (topRows d)) (y 0) + pairing (pullMul V (bottomRows d)) (y 1) := by
  rw [pairMul, pairing_add, pairing_mul hn, pairing_mul hn]

theorem pairing_neg {rows columns : Nat} (W : Weight n rows columns) (e : ErrorMatrix n rows columns) :
    pairing W (-e) = -pairing W e := by
  have h := pairing_add W e (-e)
  rw [add_neg_cancel, pairing_zero] at h
  linarith

theorem pairing_smul_weight {rows columns : Nat} (a : ℝ) (W : Weight n rows columns)
    (e : ErrorMatrix n rows columns) : pairing (a • W) e = a * pairing W e := by
  simp only [pairing, Pi.smul_apply, smul_eq_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦
    Finset.sum_congr rfl fun k _ ↦ by ring

theorem weightL1_smul {rows columns : Nat} (a : ℝ) (W : Weight n rows columns) :
    weightL1 (a • W) = |a| * weightL1 W := by
  simp only [weightL1, Pi.smul_apply, smul_eq_mul, abs_mul, Finset.mul_sum]

theorem weightEnergy_smul {rows columns : Nat} (a : ℝ) (W : Weight n rows columns) :
    weightEnergy (a • W) = a ^ 2 * weightEnergy W := by
  simp only [weightEnergy, Pi.smul_apply, smul_eq_mul, mul_pow, Finset.mul_sum]

/-! ## Multiplication by a polynomial -/

/-- A weight pulled back through the multiplication of every entry by `a`. -/
noncomputable def pullSmul {rows columns : Nat} (W : Weight n rows columns) (a : ErrorPoly n) :
    Weight n rows columns :=
  fun r c j ↦ ∑ k, W r c k * ∑ i, ((a.coeff i : ℤ) : ℝ) * (negSign n i j k : ℝ)

theorem pairing_smul (hn : 0 < n) {rows columns : Nat} (W : Weight n rows columns)
    (a : ErrorPoly n) (e : ErrorMatrix n rows columns) :
    pairing W (a • e) = pairing (pullSmul W a) e := by
  simp only [pairing, pullSmul, Matrix.smul_apply, smul_eq_mul, coeff_mul_eq hn, Int.cast_sum,
    Int.cast_mul, Finset.mul_sum, Finset.sum_mul]
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦ ?_
  calc ∑ k, ∑ i, ∑ j, W r c k * (((a.coeff i : ℤ) : ℝ) * (((e r c).coeff j : ℤ) : ℝ) *
          (negSign n i j k : ℝ))
      = ∑ k, ∑ j, ∑ i, W r c k * (((a.coeff i : ℤ) : ℝ) * (((e r c).coeff j : ℤ) : ℝ) *
          (negSign n i j k : ℝ)) := Finset.sum_congr rfl fun k _ ↦ Finset.sum_comm
    _ = ∑ j, ∑ k, ∑ i, W r c k * (((a.coeff i : ℤ) : ℝ) * (((e r c).coeff j : ℤ) : ℝ) *
          (negSign n i j k : ℝ)) := Finset.sum_comm
    _ = _ := Finset.sum_congr rfl fun j _ ↦ Finset.sum_congr rfl fun k _ ↦
          Finset.sum_congr rfl fun i _ ↦ by ring

/-- The `l1` norm of a weight pulled back through `a` gains the `l1` norm of `a`. -/
theorem weightL1_pullSmul_le {rows columns : Nat} (W : Weight n rows columns) (a : ErrorPoly n) :
    weightL1 (pullSmul W a) ≤ weightL1 W * ∑ i, |((a.coeff i : ℤ) : ℝ)| := by
  unfold weightL1
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun r _ ↦ ?_
  rw [Finset.sum_mul]
  refine Finset.sum_le_sum fun c _ ↦ ?_
  set A : Fin n → ℝ := fun i ↦ |((a.coeff i : ℤ) : ℝ)|
  calc ∑ j, |pullSmul W a r c j|
      ≤ ∑ j, ∑ k, ∑ i, |W r c k| * (A i * |(negSign n i j k : ℝ)|) :=
        Finset.sum_le_sum fun j _ ↦ (Finset.abs_sum_le_sum_abs _ _).trans
          (Finset.sum_le_sum fun k _ ↦ by
            rw [abs_mul, ← Finset.mul_sum]
            refine mul_le_mul_of_nonneg_left ((Finset.abs_sum_le_sum_abs _ _).trans
              (le_of_eq (Finset.sum_congr rfl fun i _ ↦ abs_mul _ _))) (abs_nonneg _))
    _ = ∑ k, |W r c k| * ∑ i, A i * ∑ j, |(negSign n i j k : ℝ)| := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun k _ ↦ ?_
        rw [Finset.sum_comm, Finset.mul_sum]
        refine Finset.sum_congr rfl fun i _ ↦ ?_
        rw [Finset.mul_sum, Finset.mul_sum]
    _ ≤ ∑ k, |W r c k| * ∑ i, A i := Finset.sum_le_sum fun k _ ↦
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ ↦ by
          simpa using mul_le_mul_of_nonneg_left (sum_abs_negSign_le' i k) (abs_nonneg _))
          (abs_nonneg _)
    _ = (∑ k, |W r c k|) * ∑ i, A i := by rw [Finset.sum_mul]

/-! ## Substitutions -/

/-- The position of coefficient `i` of `x` in `x(X^index)`. -/
def substIndex (hn : 0 < n) (index : Nat) (i : Fin n) : Fin n := ⟨(i.val * index) % n, Nat.mod_lt _ hn⟩

/-- A weight pulled back through the substitution `X ↦ X^index`. -/
noncomputable def pullSubst (hn : 0 < n) {rows columns : Nat} (index : Nat)
    (W : Weight n rows columns) : Weight n rows columns :=
  fun r c i ↦ W r c (substIndex hn index i) * ((-1 : ℤ) ^ ((i.val * index) / n) : ℤ)

theorem substitution_coeff (hn : 0 < n) {index : Nat} (hodd : Odd index) (x : ErrorPoly n)
    (k : Fin n) : (substitution index hodd x).coeff k =
      ∑ i : Fin n, if (i.val * index) % n = k.val then
        x.coeff i * (-1 : ℤ) ^ ((i.val * index) / n) else 0 := by
  rw [substitution_coeff_sum index hodd hn, Negacyclic.coeff_sum]
  exact Finset.sum_congr rfl fun i _ ↦ substituted_term_coeff hn _ _ k

theorem pairing_substitute (hn : 0 < n) {rows columns : Nat} {index : Nat} (hodd : Odd index)
    (W : Weight n rows columns) (e : ErrorMatrix n rows columns) :
    pairing W (substituteMatrix index hodd e) = pairing (pullSubst hn index W) e := by
  simp only [pairing, pullSubst, substituteMatrix, substitution_coeff hn hodd]
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦ ?_
  simp only [Int.cast_sum, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [Finset.sum_eq_single (substIndex hn index i)]
  · simp only [substIndex, if_true, Int.cast_mul, Int.cast_pow, Int.cast_neg, Int.cast_one]
    ring
  · intro k _ hk
    rw [if_neg]
    · simp
    · intro h
      exact hk (Fin.ext h.symm)
  · simp

/-- `i ↦ i · index mod n` is a permutation for an index coprime to `n`. -/
noncomputable def substPerm (hn : 0 < n) {index : Nat} (hcoprime : n.Coprime index) :
    Fin n ≃ Fin n :=
  Equiv.ofBijective (substIndex hn index)
    ((Finite.injective_iff_bijective).mp fun i j h ↦
      multiply_index_injective hcoprime (congrArg Fin.val h))

theorem weightL1_pullSubst (hn : 0 < n) {rows columns : Nat} {index : Nat}
    (hcoprime : n.Coprime index) (W : Weight n rows columns) :
    weightL1 (pullSubst hn index W) = weightL1 W := by
  unfold weightL1 pullSubst
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦ ?_
  have hsign : ∀ i : Fin n, |(((-1 : ℤ) ^ ((i.val * index) / n) : ℤ) : ℝ)| = 1 := fun i ↦ by
    push_cast; simp
  simp only [abs_mul, hsign, mul_one]
  exact Equiv.sum_comp (substPerm hn hcoprime) (fun k ↦ |W r c k|)

theorem weightEnergy_pullSubst (hn : 0 < n) {rows columns : Nat} {index : Nat}
    (hcoprime : n.Coprime index) (W : Weight n rows columns) :
    weightEnergy (pullSubst hn index W) = weightEnergy W := by
  unfold weightEnergy pullSubst
  refine Finset.sum_congr rfl fun r _ ↦ Finset.sum_congr rfl fun c _ ↦ ?_
  have hsign : ∀ i : Fin n, ((((-1 : ℤ) ^ ((i.val * index) / n) : ℤ) : ℝ)) ^ 2 = 1 := fun i ↦ by
    push_cast; rw [← pow_mul, mul_comm, pow_mul]; simp
  simp only [mul_pow, hsign, mul_one]
  exact Equiv.sum_comp (substPerm hn hcoprime) (fun k ↦ W r c k ^ 2)

end MxxExponentLut
