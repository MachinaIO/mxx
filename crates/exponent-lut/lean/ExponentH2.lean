import ExponentCellStep
import RuntimeHeuristics
import ExponentRuntime
import ExponentFunctional
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# The tail of a noise form under H2

The terminal error of a decoded coefficient is `L + Σ_j g_j ρ_j`: `L` is bounded on every tape,
and the sites `ρ_j` are the residues that H2 replaces by independent uniforms. The sites split
into the row residues, whose coefficients have a deterministic energy, and the mask-weighted key
residues, whose coefficient times the residue bound is a linear form `Σ_l u_l s_l` in the
coefficients `s_l` of the mask. With probability `1 - 2 e^{-τ²/2}` per site the form is below
`τ ‖u‖` (Hoeffding), so the energy of the coefficients is bounded except on a union of these
events, and the replaced form has a Gaussian tail.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime MeasureTheory
open scoped BigOperators

variable {n : Nat}

/-! ## Site selection

`Sum.elim_inl` and `Sum.elim_inr` as rewrites with proofs. Their `rfl` proofs make `simp` leave the
step to the kernel's definitional unfolding, which unfolds the selected function before
`Sum.elim` when that function is defined later, as a residue read from a polynomial is: at
`N = 65536` the unfolding of such reads does not finish. -/

theorem elim_inl_eq {α β γ : Type*} (f : α → γ) (g : β → γ) (x : α) :
    Sum.elim f g (.inl x) = f x :=
  (Sum.elim_inl f g x).trans rfl

theorem elim_inr_eq {α β γ : Type*} (f : α → γ) (g : β → γ) (x : β) :
    Sum.elim f g (.inr x) = g x :=
  (Sum.elim_inr f g x).trans rfl

/-! ## Energies with tape-dependent bounds -/

/-- The energy of `c` exceeds `E` only where some coordinate exceeds its bound `b`, when the bounds
have energy at most `E` on every tape. -/
theorem tapeMeasure_energy_gt_le {α : Type} [Fintype α] (c b : SampleTape → α → ℝ) {E : ℝ}
    (hb : ∀ t, ∑ x, b t x ^ 2 ≤ E) :
    tapeMeasure {t | E < ∑ x, c t x ^ 2} ≤ ∑ x, tapeMeasure {t | |b t x| < |c t x|} := by
  classical
  have hsub : {t | E < ∑ x, c t x ^ 2} ⊆ ⋃ x ∈ (Finset.univ : Finset α), {t | |b t x| < |c t x|} := by
    intro t ht
    by_contra hnot
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, Finset.mem_univ, exists_true_left, not_exists,
      not_lt] at hnot
    have hle : ∑ x, c t x ^ 2 ≤ ∑ x, b t x ^ 2 := Finset.sum_le_sum fun x _ ↦ by
      rw [← sq_abs (c t x), ← sq_abs (b t x)]
      exact pow_le_pow_left₀ (abs_nonneg _) (hnot x) 2
    exact absurd (ht.trans_le (hle.trans (hb t))) (lt_irrefl _)
  exact (measure_mono hsub).trans (measure_biUnion_finset_le _ _)

/-! ## Linear forms in the mask -/

/-- A mask coefficient clamped to `[-1, 1]`, as the model reads it. -/
def maskClamp (site : List Nat) (t : SampleTape) (l : Nat) : Int :=
  max (-1) (min 1 (t (maskKey site l)))

/-- Every mask coefficient lies in `[-1, 1]`, except on a null set of tapes. -/
theorem tapeMeasure_mask_out (site : List Nat) (l : Nat) :
    tapeMeasure {t | t (maskKey site l) < -1 ∨ 1 < t (maskKey site l)} = 0 := by
  have hpre : {t : SampleTape | t (maskKey site l) < -1 ∨ 1 < t (maskKey site l)} =
      Function.eval (maskKey site l) ⁻¹' {y : Int | y < -1 ∨ 1 < y} := rfl
  rw [hpre, tapeMeasure, (measurePreserving_eval_infinitePi _ (maskKey site l)).measure_preimage
    (MeasurableSet.of_discrete).nullMeasurableSet]
  simp only [maskKey, SamplerLaw.measure, SamplerLaw.pmf, show (-1 : Int) ≤ 1 by norm_num,
    dite_true]
  rw [PMF.toMeasure_apply_eq_zero_iff _ MeasurableSet.of_discrete, PMF.support_uniformOfFinset]
  rw [Set.disjoint_left]
  intro y hy hout
  simp only [Finset.coe_Icc, Set.mem_Icc] at hy
  simp only [Set.mem_setOf_eq] at hout
  omega

/-- A linear form in the clamped mask coefficients whose coefficients read only `T`, which holds no
mask key, exceeds `τ` times the norm of its coefficients with probability at most
`2 exp (-τ² / 2)`. -/
theorem maskForm_tail (site : List Nat) (T : Finset SampleKey)
    (hT : ∀ k : Fin n, maskKey site k.val ∉ T) {c : SampleTape → Fin n → ℝ} (hc : DependsOn c T)
    {τ : ℝ} (hτ : 0 < τ) :
    tapeMeasure {t | τ * Real.sqrt (∑ l, c t l ^ 2) <
        |∑ l, c t l * (maskClamp site t l.val : ℝ)|} ≤
      ENNReal.ofReal (2 * Real.exp (-τ ^ 2 / 2)) := by
  classical
  set key : Fin n → SampleKey := fun l ↦ maskKey site l.val with hkey
  have hinj : Function.Injective key := fun a b h ↦ Fin.ext (maskKey_injective site h)
  set c' : SampleTape → Unit → Fin n → ℝ := fun t _ l ↦ c t l / Real.sqrt (∑ l, c t l ^ 2) with hc'
  have hc'dep : DependsOn c' T := by
    intro t t' h
    have := hc h
    funext u l
    simp only [hc', this]
  have henergy : ∀ t (u : Unit), ∑ l, c' t u l ^ 2 ≤ 1 := by
    intro t u
    simp only [hc', div_pow]
    rw [← Finset.sum_div]
    rcases eq_or_lt_of_le (Finset.sum_nonneg fun l _ ↦ sq_nonneg (c t l) :
        (0 : ℝ) ≤ ∑ l, c t l ^ 2) with h0 | hpos
    · rw [← h0]; simp
    · rw [Real.sq_sqrt hpos.le, div_self hpos.ne']
  have hsub := CSubG.linear (μ := fun key : SampleKey ↦ key.law.measure) key hinj zero_le_one
    (fun l a ↦ maskKey_law site l.val a) T (fun l ↦ hT l) hc'dep henergy
  have hdep : DependsOn (fun t (u : Unit) ↦ ∑ l, c' t u l * (t (key l) : ℝ))
      ((T ∪ Finset.univ.image key : Finset SampleKey) : Set SampleKey) := by
    intro t t' h
    have h1 := hc'dep (fun y hy ↦ h y (by simp [Finset.mem_coe.mp hy]))
    funext u
    refine Finset.sum_congr rfl fun l _ ↦ ?_
    rw [show c' t = c' t' from h1, h (key l) (by simp)]
  have htail := CSubG.tail (μ := fun key : SampleKey ↦ key.law.measure) hsub _ hdep () (by norm_num)
    hτ.le
  simp only [mul_one] at htail
  -- Off the null set where a mask coefficient leaves `[-1, 1]`, the event lies in the tail event.
  set bad : Set SampleTape := ⋃ l ∈ (Finset.univ : Finset (Fin n)),
    {t | t (key l) < -1 ∨ 1 < t (key l)} with hbad
  have hnull : tapeMeasure bad = 0 :=
    measure_biUnion_null_iff (Finset.univ : Finset (Fin n)).countable_toSet |>.mpr
      fun l _ ↦ tapeMeasure_mask_out site l.val
  have hincl : {t | τ * Real.sqrt (∑ l, c t l ^ 2) < |∑ l, c t l * (maskClamp site t l.val : ℝ)|} ⊆
      {t | τ ≤ |∑ l, c' t () l * (t (key l) : ℝ)|} ∪ bad := by
    intro t ht
    by_cases hb : t ∈ bad
    · exact Or.inr hb
    left
    simp only [hbad, Set.mem_iUnion, Set.mem_setOf_eq, Finset.mem_univ, exists_true_left,
      not_exists, not_or, not_lt] at hb
    have hclamp : ∀ l : Fin n, (maskClamp site t l.val : ℝ) = (t (key l) : ℝ) := fun l ↦ by
      have h1 := hb l
      show ((max (-1) (min 1 (t (key l))) : ℤ) : ℝ) = (t (key l) : ℝ)
      rw [show max (-1) (min 1 (t (key l))) = t (key l) by omega]
    simp only [Set.mem_setOf_eq, hclamp] at ht ⊢
    set S := Real.sqrt (∑ l, c t l ^ 2) with hS
    rcases eq_or_lt_of_le (Real.sqrt_nonneg (∑ l, c t l ^ 2)) with h0 | hpos
    · -- A zero norm makes every coefficient zero, so the event is empty.
      have hz : ∀ l, c t l = 0 := by
        have hsum : ∑ l, c t l ^ 2 = 0 := by
          have := Real.sqrt_eq_zero'.mp h0.symm
          linarith [Finset.sum_nonneg fun l (_ : l ∈ Finset.univ) ↦ sq_nonneg (c t l)]
        intro l
        have := (Finset.sum_eq_zero_iff_of_nonneg fun l _ ↦ sq_nonneg (c t l)).mp hsum l
          (Finset.mem_univ l)
        exact pow_eq_zero_iff (n := 2) (by norm_num) |>.mp this
      simp only [hz, zero_mul, Finset.sum_const_zero, abs_zero] at ht
      have hS0 : S = 0 := by rw [hS]; exact h0.symm
      rw [hS0, mul_zero] at ht
      exact absurd ht (lt_irrefl 0)
    · have hform : ∑ l, c' t () l * (t (key l) : ℝ) = (∑ l, c t l * (t (key l) : ℝ)) / S := by
        simp only [hc', Finset.sum_div]
        refine Finset.sum_congr rfl fun l _ ↦ ?_
        ring
      rw [hform, abs_div, abs_of_pos hpos, le_div_iff₀ hpos]
      exact ht.le
  calc tapeMeasure {t | τ * Real.sqrt (∑ l, c t l ^ 2) <
          |∑ l, c t l * (maskClamp site t l.val : ℝ)|}
      ≤ tapeMeasure ({t | τ ≤ |∑ l, c' t () l * (t (key l) : ℝ)|} ∪ bad) := measure_mono hincl
    _ ≤ tapeMeasure {t | τ ≤ |∑ l, c' t () l * (t (key l) : ℝ)|} + tapeMeasure bad :=
        measure_union_le _ _
    _ ≤ ENNReal.ofReal (2 * Real.exp (-τ ^ 2 / 2)) := by
        rw [hnull, add_zero]
        exact htail


/-! ## Coefficient norms -/

/-- For fixed `i` and `j`, the negacyclic signs have at most one nonzero `k`. -/
theorem sum_abs_negSign_le_k (i j : Fin n) : ∑ k : Fin n, |(negSign n i j k : ℝ)| ≤ 1 := by
  classical
  have hle : ∀ k, |(negSign n i j k : ℝ)| ≤ if (i.val + j.val) % n = k.val then 1 else 0 := by
    intro k
    unfold negSign
    split_ifs <;> simp
  refine (Finset.sum_le_sum fun k _ ↦ hle k).trans ?_
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
  have hcard : (Finset.univ.filter fun k : Fin n ↦ (i.val + j.val) % n = k.val).card ≤ 1 := by
    refine Finset.card_le_one.mpr fun a ha b hb ↦ ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha hb
    exact Fin.ext (ha.symm.trans hb)
  exact_mod_cast hcard

/-- Coefficient `j` of `a w`, for a weight row `w`, as a linear form in the coefficients of `a`. -/
theorem pullSmul_eq_form {rows columns : Nat} (W : Weight n rows columns) (a : ErrorPoly n)
    (r : Fin rows) (c : Fin columns) (j : Fin n) :
    pullSmul W a r c j = ∑ i, (((a.coeff i : ℤ) : ℝ)) * ∑ k, W r c k * (negSign n i j k : ℝ) := by
  simp only [pullSmul, Finset.mul_sum]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun k _ ↦ by ring

/-- The coefficients of the linear form of `pullSmul_eq_form` have at most the energy of the
weight row. -/
theorem sum_sq_negSign_form_le (w : Fin n → ℝ) (j : Fin n) :
    ∑ i, (∑ k, w k * (negSign n i j k : ℝ)) ^ 2 ≤ ∑ k, w k ^ 2 := by
  calc ∑ i, (∑ k, w k * (negSign n i j k : ℝ)) ^ 2
      ≤ ∑ i, (∑ k, |(negSign n i j k : ℝ)|) * ∑ k, |(negSign n i j k : ℝ)| * w k ^ 2 :=
        Finset.sum_le_sum fun i _ ↦ by
          have := sq_sum_mul_le Finset.univ (fun k ↦ (negSign n i j k : ℝ)) w
          simpa only [mul_comm (w _)] using this
    _ ≤ ∑ i, ∑ k, |(negSign n i j k : ℝ)| * w k ^ 2 :=
        Finset.sum_le_sum fun i _ ↦ by
          have h1 := sum_abs_negSign_le_k i j
          have h2 : 0 ≤ ∑ k, |(negSign n i j k : ℝ)| * w k ^ 2 :=
            Finset.sum_nonneg fun _ _ ↦ mul_nonneg (abs_nonneg _) (sq_nonneg _)
          nlinarith
    _ = ∑ k, w k ^ 2 * ∑ i, |(negSign n i j k : ℝ)| := by
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun k _ ↦ by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun i _ ↦ by ring
    _ ≤ ∑ k, w k ^ 2 * 1 :=
        Finset.sum_le_sum fun k _ ↦ mul_le_mul_of_nonneg_left (sum_abs_negSign_le j k) (sq_nonneg _)
    _ = ∑ k, w k ^ 2 := by simp

/-- A substitution permutes the coefficients up to sign. -/
theorem sum_abs_substitution_coeff_le (hn : 0 < n) {index : Nat} (hodd : Odd index)
    (x : ErrorPoly n) :
    ∑ k, |(((substitution index hodd x).coeff k : ℤ) : ℝ)| ≤ ∑ i, |((x.coeff i : ℤ) : ℝ)| := by
  classical
  have hpt : ∀ k : Fin n, |(((substitution index hodd x).coeff k : ℤ) : ℝ)| ≤
      ∑ i : Fin n, if (i.val * index) % n = k.val then |((x.coeff i : ℤ) : ℝ)| else 0 := by
    intro k
    rw [substitution_coeff hn hodd, Int.cast_sum]
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ ↦ ?_)
    split_ifs
    · push_cast
      rw [abs_mul, abs_pow, abs_neg, abs_one, one_pow, mul_one]
    · simp
  refine (Finset.sum_le_sum fun k _ ↦ hpt k).trans (le_of_eq ?_)
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  have h : ∀ k : Fin n, ((i.val * index) % n = k.val) ↔ (⟨(i.val * index) % n, Nat.mod_lt _ hn⟩ : Fin n) = k :=
    fun k ↦ ⟨fun h ↦ Fin.ext h, fun h ↦ congrArg Fin.val h⟩
  simp only [h, Finset.sum_ite_eq, Finset.mem_univ, if_true]


/-- A monomial `X^k` has coefficient `l1` norm `1`. -/
theorem sum_abs_Xe_coeff_le (hn : 0 < n) (k : Nat) :
    ∑ c, |(((Xe (n := n) k).coeff c : ℤ) : ℝ)| ≤ 1 := by
  classical
  have hsplit : Xe (n := n) k = (((-1 : ℤ) ^ (k / n) : ℤ) : ErrorPoly n) *
      AdjoinRoot.root (negacyclicModulus n Int) ^ ((⟨k % n, Nat.mod_lt _ hn⟩ : Fin n) : Nat) := by
    rw [Xe]
    conv_lhs => rw [← Nat.div_add_mod k n]
    rw [pow_add, pow_mul, Negacyclic.root_pow_n]
    push_cast
    ring
  rw [hsplit]
  simp only [intCast_mul_coeff]
  simp only [Negacyclic.coeff_root_pow hn ⟨k % n, Nat.mod_lt _ hn⟩]
  refine le_of_eq ?_
  calc ∑ c : Fin n, |((((-1 : ℤ) ^ (k / n) * if (⟨k % n, Nat.mod_lt _ hn⟩ : Fin n) = c then 1 else 0 : ℤ)) : ℝ)|
      = ∑ c : Fin n, if (⟨k % n, Nat.mod_lt _ hn⟩ : Fin n) = c then (1 : ℝ) else 0 :=
        Finset.sum_congr rfl fun c _ ↦ by split_ifs <;> simp
    _ = 1 := by simp


/-! ## Coefficient bounds of the model pieces -/

theorem coeffBound_downResidue (hn : 0 < n) {q rows columns p : Nat} (hp : 0 < p)
    (x : ExactMatrix q n rows columns) : CoeffBound (downResidue p x) (p / 2) := by
  intro r c k
  rw [downResidue, intPoly_coeff hn]
  exact rnsCentered_natAbs_le hp _

/-- The lift of a compact rebase is a preimage within its bound. -/
theorem coeffBound_lift_compact {q p rows columns bound : Nat} (hp : 1 < p) (hn : 0 < n)
    (hb : 2 * bound < p) (x : ExactMatrix q n rows columns) :
    CoeffBound (liftMatrix (theOutput (compactCenteredRebaseRuns bound) x :
      ExactMatrix p n rows columns)) bound := by
  classical
  unfold theOutput
  split
  · rename_i h
    obtain ⟨-, -, -, w, hw, hwb⟩ := h.choose_spec
    rw [hw, liftMatrix_reduce hp hn hwb hb]
    exact hwb
  · have h0 : (0 : ExactMatrix p n rows columns) = reduceMatrix p n rows columns 0 := by
      funext r c
      simp [reduceMatrix, reducePoly]
    rw [h0, liftMatrix_reduce hp hn (fun _ _ _ ↦ by simp) (by omega)]
    intro r c k
    simp

/-- The lift of a gadget decomposition is within half the base. -/
theorem coeffBound_lift_decompose {q rows columns inner : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (hb : 2 * (base.toNat / 2) < q)
    (x : ExactMatrix q n rows columns) :
    CoeffBound (liftMatrix (theOutput (gadgetDecomposeRuns backend base digits) x :
      ExactMatrix q n inner columns)) (base.toNat / 2) := by
  classical
  unfold theOutput
  split
  · rename_i h
    obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq hn h.choose_spec
    rw [hd, liftMatrix_reduce hq hn hdb hb]
    exact hdb
  · have h0 : (0 : ExactMatrix q n inner columns) = reduceMatrix q n inner columns 0 := by
      funext r c
      simp [reduceMatrix, reducePoly]
    rw [h0, liftMatrix_reduce hq hn (fun _ _ _ ↦ by simp) (by omega)]
    intro r c k
    simp

/-- A polynomial with coefficients at most `B` has coefficient `l1` norm at most `n B`. -/
theorem sum_abs_coeff_le {x : ErrorPoly n} {B : Nat} (h : ∀ k, (x.coeff k).natAbs ≤ B) :
    ∑ k, |((x.coeff k : ℤ) : ℝ)| ≤ n * B := by
  calc ∑ k, |((x.coeff k : ℤ) : ℝ)| ≤ ∑ _k : Fin n, (B : ℝ) :=
        Finset.sum_le_sum fun k _ ↦ by
          rw [← Int.cast_abs, Int.abs_eq_natAbs]
          exact_mod_cast h k
    _ = n * B := by simp

/-- The column `l1` norm of a matrix of constant polynomials within `B`. -/
theorem column_l1_constant_le {rows columns : Nat} {d : ErrorMatrix n rows columns} {B : Nat}
    (hd : CoeffBound d B) (hc : ∀ r c (i : Fin n), i.val ≠ 0 → (d r c).coeff i = 0) (c : Fin columns) :
    ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ rows * B := by
  calc ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ ∑ _l : Fin rows, (B : ℝ) := by
        refine Finset.sum_le_sum fun l _ ↦ ?_
        rcases Nat.eq_zero_or_pos n with h0 | hpos
        · subst h0; simp
        rw [Finset.sum_eq_single ⟨0, hpos⟩]
        · rw [← Int.cast_abs, Int.abs_eq_natAbs]
          exact_mod_cast hd l c _
        · intro j _ hj
          rw [hc l c j (fun h ↦ hj (Fin.ext h)), Int.cast_zero, abs_zero]
        · simp
    _ = rows * B := by simp

/-- A column `l1` norm from a coefficient bound. -/
theorem column_l1_le {rows columns : Nat} {d : ErrorMatrix n rows columns} {B : Nat}
    (hd : CoeffBound d B) (c : Fin columns) :
    ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ rows * n * B := by
  calc ∑ l, ∑ j, |(((d l c).coeff j : ℤ) : ℝ)| ≤ ∑ _l : Fin rows, (n * B : ℝ) :=
        Finset.sum_le_sum fun l _ ↦ sum_abs_coeff_le fun k ↦ hd l c k
    _ = rows * n * B := by simp; ring

/-! ## Site energies -/

section Sites

variable {C m : Nat}

/-- The energy of the subtracted residue sites. -/
theorem sum_sq_subCoef (U : Fin C → Fin 2 → Weight n 1 m) (D h : ℝ) :
    ∑ x, h ^ 2 * subCoef U D x ^ 2 = (h / D) ^ 2 * ∑ i, ∑ j, weightEnergy (U i j) := by
  simp only [subCoef, weightEnergy, Fin.sum_univ_one, Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦
    Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
  ring

/-- The coefficients of the mask form of a mask-weighted site, times the residue bound `h`. -/
noncomputable def maskU (U : Fin C → Fin 2 → Weight n 1 m) (D h : ℝ)
    (x : Fin C × Fin 2 × Fin m × Fin n) (l : Fin n) : ℝ :=
  h / D * ∑ k, U x.1 x.2.1 0 x.2.2.1 k * (negSign n l x.2.2.2 k : ℝ)

theorem maskCoef_eq_form (U : Fin C → Fin 2 → Weight n 1 m) (sm : ErrorPoly n) (D h : ℝ)
    (x : Fin C × Fin 2 × Fin m × Fin n) :
    h * maskCoef U sm D x = ∑ l, maskU U D h x l * (((sm.coeff l : ℤ) : ℝ)) := by
  simp only [maskCoef, maskU, pullSmul_eq_form, Finset.mul_sum, Finset.sum_div]
  refine Finset.sum_congr rfl fun l _ ↦ ?_
  rw [Finset.sum_mul]
  exact Finset.sum_congr rfl fun k _ ↦ by ring

theorem sum_sq_maskU_le (U : Fin C → Fin 2 → Weight n 1 m) (D h : ℝ) :
    ∑ x, ∑ l, maskU U D h x l ^ 2 ≤ (h / D) ^ 2 * (n * ∑ i, ∑ j, weightEnergy (U i j)) := by
  have hpt : ∀ x : Fin C × Fin 2 × Fin m × Fin n, ∑ l, maskU U D h x l ^ 2 ≤
      (h / D) ^ 2 * ∑ k, U x.1 x.2.1 0 x.2.2.1 k ^ 2 := by
    intro x
    simp only [maskU, mul_pow, ← Finset.mul_sum]
    exact mul_le_mul_of_nonneg_left (sum_sq_negSign_form_le _ _) (sq_nonneg _)
  refine (Finset.sum_le_sum fun x _ ↦ hpt x).trans (le_of_eq ?_)
  rw [← Finset.mul_sum]
  congr 1
  simp only [weightEnergy, Fin.sum_univ_one, Fintype.sum_prod_type, Finset.mul_sum, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end Sites


/-! ## Bounds of a block in terms of the `l1` norm of its weight -/

section BlockBounds

variable {C mA ms mo : Nat}

theorem sum_weightL1_smul {k m : Nat} (W : Fin k → Weight n 1 m) {a : ℝ} (ha : 0 ≤ a) :
    ∑ j, weightL1 (a • W j) = a * ∑ j, weightL1 (W j) := by
  simp only [weightL1_smul, abs_of_nonneg ha, Finset.mul_sum]

/-- The weight on the state error of a block. -/
theorem lam_state_le (V : Fin 2 → Weight n 1 mo) {lam : ℝ} (hlam : ∑ j, weightL1 (V j) ≤ lam)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (α : Fin C → ErrorPoly n) {βr : Nat}
    (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hα : ∀ i, ∑ k, |(((α i).coeff k : ℤ) : ℝ)| ≤ 1)
    {P : ℝ} (hP : 0 < P) :
    ∑ j', weightL1 (P⁻¹ • stateWeight V Dr α j') ≤ lam * (2 * (ms * n * βr)) * C / P := by
  rw [sum_weightL1_smul _ (inv_nonneg.mpr hP.le)]
  have h := weightL1_stateWeight_le V Dr α hDr zero_le_one hα
  rw [mul_one] at h
  rw [inv_mul_eq_div]
  refine div_le_div_of_nonneg_right (h.trans ?_) hP.le
  have : (0 : ℝ) ≤ 2 * (ms * n * βr) * C := by positivity
  nlinarith

theorem lam_align_le (V : Fin 2 → Weight n 1 mo) {lam : ℝ} (hlam : ∑ j, weightL1 (V j) ≤ lam)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    {βr βs : Nat} (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hDs : ∀ j', CoeffBound (Ds j') βs) :
    ∑ i, ∑ k, weightL1 (alignWeight V Dr Ds i k) ≤
      2 * C * (lam * (2 * (ms * n * βr)) * (mA * n * βs)) := by
  have hpt : ∀ i k, weightL1 (alignWeight V Dr Ds i k) ≤ lam * (2 * (ms * n * βr)) * (mA * n * βs) :=
    fun i k ↦ (weightL1_alignWeight_le V Dr Ds hDr hDs i k).trans (by
      have : (0 : ℝ) ≤ 2 * (ms * n * βr) * (mA * n * βs) := by positivity
      nlinarith)
  calc ∑ i, ∑ k, weightL1 (alignWeight V Dr Ds i k)
      ≤ ∑ _i : Fin C, ∑ _k : Fin 2, lam * (2 * (ms * n * βr)) * (mA * n * βs) :=
        Finset.sum_le_sum fun i _ ↦ Finset.sum_le_sum fun k _ ↦ hpt i k
    _ = _ := by simp; ring

theorem energy_align_le (V : Fin 2 → Weight n 1 mo) {lam : ℝ} (hlam : ∑ j, weightL1 (V j) ≤ lam)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    {βr βs : Nat} (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hDs : ∀ j', CoeffBound (Ds j') βs) :
    ∑ i, ∑ k, weightEnergy (alignWeight V Dr Ds i k) ≤
      2 * C * ((lam * (2 * (ms * n * βr))) ^ 2 * (mA * n * βs ^ 2)) := by
  have hnn := sum_weightL1_nonneg V
  have hpt : ∀ i k, weightEnergy (alignWeight V Dr Ds i k) ≤
      (lam * (2 * (ms * n * βr))) ^ 2 * (mA * n * βs ^ 2) := by
    intro i k
    refine (weightEnergy_alignWeight_le V Dr Ds hDr hDs i k).trans
      (mul_le_mul_of_nonneg_right (pow_le_pow_left₀ (by positivity) ?_ 2) (by positivity))
    exact mul_le_mul_of_nonneg_right hlam (by positivity)
  calc ∑ i, ∑ k, weightEnergy (alignWeight V Dr Ds i k)
      ≤ ∑ _i : Fin C, ∑ _k : Fin 2, (lam * (2 * (ms * n * βr))) ^ 2 * (mA * n * βs ^ 2) :=
        Finset.sum_le_sum fun i _ ↦ Finset.sum_le_sum fun k _ ↦ hpt i k
    _ = _ := by simp; ring

theorem energy_rows_le (V : Fin 2 → Weight n 1 mo) {lam : ℝ} (hlam : ∑ j, weightL1 (V j) ≤ lam) :
    ∑ _i : Fin C, ∑ j, weightEnergy (V j) ≤ C * lam ^ 2 := by
  have hnn := sum_weightL1_nonneg V
  have h1 : ∑ j, weightEnergy (V j) ≤ (∑ j, weightL1 (V j)) ^ 2 :=
    (Finset.sum_le_sum fun j _ ↦ weightEnergy_le_sq (V j)).trans
      (sum_sq_le_sq_sum _ _ fun j _ ↦ weightL1_nonneg _)
  have h2 : (∑ j, weightL1 (V j)) ^ 2 ≤ lam ^ 2 := pow_le_pow_left₀ hnn hlam 2
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  exact mul_le_mul_of_nonneg_left (h1.trans h2) (Nat.cast_nonneg _)

theorem abs_cellRem_le' (V : Fin 2 → Weight n 1 mo) {lam : ℝ} (hlam : ∑ j, weightL1 (V j) ≤ lam)
    {P : ℝ} (hP : 0 < P) (sm : ErrorPoly n) (M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo)
    (ce : Fin C → ErrorPoly n) (eB : Fin C → Fin 2 → ErrorMatrix n 1 mo) {Ssm Sce : ℝ}
    {B0 B1 BB : Nat} (hSsm : 0 ≤ Ssm) (hSce : 0 ≤ Sce)
    (hsm : ∑ k, |((sm.coeff k : ℤ) : ℝ)| ≤ Ssm) (hce : ∀ i, ∑ k, |(((ce i).coeff k : ℤ) : ℝ)| ≤ Sce)
    (hM0 : ∀ i j, CoeffBound (M0 i j) B0) (hM1 : ∀ i j, CoeffBound (M1 i j) B1)
    (heB : ∀ i j, CoeffBound (eB i j) BB) :
    |cellRem V P sm M0 M1 ce eB| ≤ lam * C * ((Ssm * B0 + B1) / P + Sce * BB) := by
  refine (abs_cellRem_le V hP sm M0 M1 ce eB hsm hce hM0 hM1 heB).trans ?_
  have : (0 : ℝ) ≤ C * ((Ssm * B0 + B1) / P + Sce * BB) := by positivity
  nlinarith

theorem abs_alignRem_le' {mT : Nat} (V : Fin 2 → Weight n 1 mo) {lam : ℝ}
    (hlam : ∑ j, weightL1 (V j) ≤ lam) (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo)
    (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms) {βr βs : Nat} (hDr : ∀ i j, CoeffBound (Dr i j) βr)
    (hDs : ∀ j', CoeffBound (Ds j') βs) {D : ℝ} (hD : 0 < D) (H : ErrorMatrix n mT mA)
    (ea : Fin C → Fin 2 → ErrorMatrix n 1 mT) (K1 : Fin C → Fin 2 → ErrorMatrix n 1 mA) {SH : ℝ}
    {Bea B1 : Nat} (hSH : 0 ≤ SH) (hH : ∀ c, ∑ l, ∑ j, |(((H l c).coeff j : ℤ) : ℝ)| ≤ SH)
    (hea : ∀ i k, CoeffBound (ea i k) Bea) (hK1 : ∀ i k, CoeffBound (K1 i k) B1) :
    |alignRem (alignWeight V Dr Ds) D H ea K1| ≤
      2 * C * (lam * (2 * (ms * n * βr)) * (mA * n * βs)) * (SH * Bea + B1) / D := by
  refine (abs_alignRem_le _ hD H ea K1 hH hea hK1).trans ?_
  refine div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right
    (lam_align_le V hlam Dr Ds hDr hDs) (by positivity)) hD.le

theorem abs_pairing2_le {m : Nat} (W : Fin 2 → Weight n 1 m) (e : Fin 2 → ErrorMatrix n 1 m) {B : Nat}
    (he : ∀ j, CoeffBound (e j) B) : |pairing2 W e| ≤ (∑ j, weightL1 (W j)) * B :=
  abs_sum_pairing_le _ W e fun j _ ↦ he j

theorem abs_memberRem_le (V : Fin 2 → Weight n 1 mo) {lam : ℝ} (hlam : ∑ j, weightL1 (V j) ≤ lam)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    {βr βs : Nat} (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hDs : ∀ j', CoeffBound (Ds j') βs)
    {P : ℝ} (hP : 0 < P) (ea : Fin C → Fin 2 → ErrorMatrix n 1 mA) {Bea : Nat}
    (hea : ∀ i k, CoeffBound (ea i k) Bea) :
    |(∑ i, ∑ k, pairing (alignWeight V Dr Ds i k) (ea i k)) / P| ≤
      2 * C * (lam * (2 * (ms * n * βr)) * (mA * n * βs)) * Bea / P := by
  rw [abs_div, abs_of_pos hP]
  refine div_le_div_of_nonneg_right ?_ hP.le
  calc |∑ i, ∑ k, pairing (alignWeight V Dr Ds i k) (ea i k)|
      ≤ ∑ i, ∑ k, |pairing (alignWeight V Dr Ds i k) (ea i k)| :=
        (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ ↦ Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ i, ∑ k, weightL1 (alignWeight V Dr Ds i k) * Bea :=
        Finset.sum_le_sum fun i _ ↦ Finset.sum_le_sum fun k _ ↦ abs_pairing_le _ (hea i k)
    _ = (∑ i, ∑ k, weightL1 (alignWeight V Dr Ds i k)) * Bea := by simp only [Finset.sum_mul]
    _ ≤ _ := mul_le_mul_of_nonneg_right (lam_align_le V hlam Dr Ds hDr hDs) (Nat.cast_nonneg _)

theorem sum_sq_lutCoef {m : Nat} (W : Weight n 1 m) (P h : ℝ) :
    ∑ x, h ^ 2 * lutCoef W P x ^ 2 = (h / P) ^ 2 * weightEnergy W := by
  simp only [lutCoef, weightEnergy, Fin.sum_univ_one, Fintype.sum_prod_type, Finset.mul_sum]
  refine Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
  ring

/-- The weight of coefficient `0` of a `1 × 1` matrix. -/
theorem weightL1_coeffZero (hn : 0 < n) :
    weightL1 (fun (_ : Fin 1) (_ : Fin 1) (k : Fin n) ↦ if k.val = 0 then (1 : ℝ) else 0) = 1 := by
  simp only [weightL1, Fin.sum_univ_one]
  rw [Finset.sum_eq_single ⟨0, hn⟩]
  · simp
  · intro k _ hk
    rw [if_neg (fun h ↦ hk (Fin.ext h)), abs_zero]
  · simp

theorem pairing_coeffZero (hn : 0 < n) (e : ErrorMatrix n 1 1) :
    pairing (fun (_ : Fin 1) (_ : Fin 1) (k : Fin n) ↦ if k.val = 0 then (1 : ℝ) else 0) e =
      (((e 0 0).coeff ⟨0, hn⟩ : ℤ) : ℝ) := by
  simp only [pairing, Fin.sum_univ_one]
  rw [Finset.sum_eq_single ⟨0, hn⟩]
  · simp
  · intro k _ hk
    rw [if_neg (fun h ↦ hk (Fin.ext h)), zero_mul]
  · simp


end BlockBounds


/-! ## Reading only some keys -/

/-- Tapes that agree on `S` give the same clamped samples at a site whose keys lie in `S`. -/
theorem clampedWitness_eq_of_agree {S : Finset SampleKey} {t t' : SampleTape}
    (h : ∀ k ∈ (S : Set SampleKey), t k = t' k) {site : List Nat} {law : SamplerLaw} {rows columns : Nat}
    (hS : SiteKeysIn S site law rows columns n) (lo hi : Int) :
    (clampedWitness t site law lo hi : ErrorMatrix n rows columns) = clampedWitness t' site law lo hi := by
  funext r c
  unfold clampedWitness
  congr 1
  funext k
  rw [h _ (hS r c k)]


/-! ## Numeric tails -/

theorem exp_neg_nat_le (a : ℕ) : Real.exp (-(a : ℝ)) ≤ (10 / 27 : ℝ) ^ a := by
  have he : (27 / 10 : ℝ) ≤ Real.exp 1 := by
    have := Real.exp_one_gt_d9
    linarith
  have h1 : Real.exp (-1) ≤ 10 / 27 := by
    rw [Real.exp_neg]
    calc (Real.exp 1)⁻¹ ≤ (27 / 10 : ℝ)⁻¹ := inv_anti₀ (by norm_num) he
      _ = 10 / 27 := by norm_num
  calc Real.exp (-(a : ℝ)) = Real.exp (-1) ^ a := by
        rw [← Real.exp_nat_mul]; ring_nf
    _ ≤ (10 / 27 : ℝ) ^ a := pow_le_pow_left₀ (Real.exp_pos _).le h1 a

theorem exp_le_of_le_neg_nat {x : ℝ} (a : ℕ) (hx : x ≤ -(a : ℝ)) : Real.exp x ≤ (10 / 27 : ℝ) ^ a :=
  (Real.exp_le_exp.mpr hx).trans (exp_neg_nat_le a)

/-- The sum of a union bound and a Gaussian tail below `2^-100`. -/
theorem ennreal_tail_le {c : ℕ} {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (h : c * a + b ≤ (1 / 2 : ℝ) ^ 100) :
    (c : ENNReal) * ENNReal.ofReal a + ENNReal.ofReal b ≤ (2 : ENNReal)⁻¹ ^ 100 := by
  have h1 : (c : ENNReal) * ENNReal.ofReal a + ENNReal.ofReal b = ENNReal.ofReal (c * a + b) := by
    rw [ENNReal.ofReal_add (mul_nonneg (Nat.cast_nonneg _) ha) hb, ENNReal.ofReal_mul (Nat.cast_nonneg _),
      ENNReal.ofReal_natCast]
  have h2 : (2 : ENNReal)⁻¹ ^ 100 = ENNReal.ofReal ((1 / 2 : ℝ) ^ 100) := by
    rw [ENNReal.ofReal_pow (by norm_num), one_div, ENNReal.ofReal_inv_of_pos (by norm_num)]
    norm_num
  rw [h1, h2]
  exact ENNReal.ofReal_le_ofReal h


/-! ## Gadgets of one CRT tower -/

private theorem castExactMatrix_algebraMap {a b r c : Nat} (h : a = b) (x : Nat) (i : Fin r) (j : Fin c) :
    castExactMatrix h (fun _ _ ↦ algebraMap (ZMod a) (ExactPoly a n) (x : ZMod a)) i j =
      algebraMap (ZMod b) (ExactPoly b n) (x : ZMod b) := by
  subst h
  rfl

/-- The CRT weight of the only tower is `1`. -/
theorem crtWeight_single (layout : OrderedCrtLayout) [Subsingleton layout.Tower] (tower : layout.Tower) :
    layout.crtWeight tower = 1 := by
  apply layout.crtEquiv.injective
  funext other
  rw [OrderedCrtLayout.crtWeight_apply, map_one, if_pos (Subsingleton.elim _ _)]
  rfl

/-- Without dropped limbs and with one CRT tower, entry `j` of the one-row gadget is `base^j`. -/
theorem gadgetMatrixRuns_single_tower {q m : Nat} {backend : BackendContext} {base digits : Int}
    {g : ExactMatrix q n 1 m} (hg : gadgetMatrixRuns backend base digits g) {layout : RegularLayout q}
    (hl : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (hone : layout.crtModuli.length = 1) (j : Fin m) :
    g 0 j = algebraMap (ZMod q) (ExactPoly q n) ((layout.base ^ j.val : Nat) : ZMod q) := by
  obtain ⟨layout', hl', -, -, hw, hg⟩ := hg
  rw [hl] at hl'
  cases hl'
  rw [hg, castMatrixColumns_apply]
  have hcount := digitCount_of_exact layout hexact
  rw [hone, one_mul] at hcount
  set c : Fin (1 * layout.digitCount) := Fin.cast hw j with hc
  have hcv : c.val = j.val := rfl
  have hjlt : j.val < layout.digitsPerTower := by
    have h1 := c.isLt
    omega
  have hidx := regularColumnIndex_val_of_exact layout hexact c
  obtain ⟨-, hlimb⟩ := regularIndexEquiv_symm (rows := 1) layout (regularColumnIndex layout c)
  set p := (regularIndexEquiv layout 1).symm (regularColumnIndex layout c) with hp
  have hdigit : p.2.2.val = j.val := by
    have htower : p.2.1.val = 0 := by
      have h1 := p.2.1.isLt
      omega
    rw [htower, zero_mul, zero_add, hidx, hcv] at hlimb
    rw [hlimb]
    apply Nat.mod_eq_of_lt
    rw [hone, one_mul]
    exact hjlt
  unfold regularGadgetMatrix fullGadgetMatrix
  have hentry : regularGadgetUnflattened (n := n) layout 0
      ((regularIndexEquiv layout 1).symm (regularColumnIndex layout c)) =
      algebraMap (ZMod layout.ordered.modulusProduct) (ExactPoly _ n)
        ((layout.base ^ j.val : Nat) : ZMod layout.ordered.modulusProduct) := by
    simp only [regularGadgetUnflattened]
    rw [if_pos (Subsingleton.elim _ _)]
    congr 1
    haveI : Subsingleton layout.ordered.Tower := by
      change Subsingleton (Fin layout.crtModuli.length)
      rw [hone]
      infer_instance
    rw [CrtDecomposition.regularGadgetWeight, crtWeight_single, one_mul, ← hp, hdigit]
    push_cast
    rfl
  refine (castExactMatrix_entry _ (fun row column ↦ regularGadgetUnflattened layout row
    ((regularIndexEquiv layout 1).symm column))
    (fun (_ : Fin 1) (_ : Fin 1) ↦ algebraMap (ZMod layout.ordered.modulusProduct)
      (ExactPoly _ n) ((layout.base ^ j.val : Nat) : ZMod layout.ordered.modulusProduct))
    (i' := 0) (j' := 0) hentry).trans ?_
  exact castExactMatrix_algebraMap _ _ _ _

theorem intCast_coeff_zero_poly (hn : 0 < n) (x : Int) : ((x : ErrorPoly n)).coeff ⟨0, hn⟩ = x := by
  have h := intCast_mul_coeff x (1 : ErrorPoly n) ⟨0, hn⟩
  rw [mul_one] at h
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := n) hn ⟨0, hn⟩ ⟨0, hn⟩
  rw [show ((⟨0, hn⟩ : Fin n) : Nat) = 0 from rfl, pow_zero, if_pos rfl] at h1
  rw [h, h1, mul_one]

/-- The coefficient `0` of a recomposition by a one-tower gadget: the digits of a constant
matrix `d` recompose `k` modulo `q`. -/
theorem recompose_coeff_zero {q m : Nat} (hq : 1 < q) (hn : 0 < n) {b : Nat}
    {g : ExactMatrix q n 1 m} (hg : ∀ j, g 0 j = algebraMap (ZMod q) (ExactPoly q n) ((b ^ j.val : Nat) : ZMod q))
    {d : ErrorMatrix n m 1} (k : Int)
    (h : g * reduceMatrix q n m 1 d = (matrixPolynomial [k] : ExactMatrix q n 1 1)) :
    (((∑ j : Fin m, ((b : Int) ^ j.val) * (d j 0).coeff ⟨0, hn⟩) - k : Int) : ZMod q) = 0 := by
  have h00 := congrArg (fun M : ExactMatrix q n 1 1 ↦ (M 0 0).coeff ⟨0, hn⟩) h
  simp only [Matrix.mul_apply, matrixPolynomial_single, hg, reduceMatrix] at h00
  rw [Negacyclic.coeff_sum] at h00
  simp only [Negacyclic.coeff_smul, reducePoly_coeff_eq hq hn] at h00
  rw [← reducePoly_intCast, reducePoly_coeff_eq hq hn, intCast_coeff_zero_poly hn] at h00
  push_cast
  rw [sub_eq_zero, ← h00]
  push_cast
  rfl

/-! ## The tail under H2 -/

/-- The tail of the noise form `L + Σ_j g_j ρ_j` under H2, for row sites `ιR` with a deterministic
energy bound and mask-weighted key sites `α` whose coefficient times the residue bound is the mask
form `Σ_l u_l s_l`: at most the union of the Hoeffding events of the key sites plus the Gaussian
tail at `m - Lmax` of the total energy. -/
theorem h2_tail {ιR α : Type} [Fintype ιR] [Fintype α] [DecidableEq ιR] [DecidableEq α]
    {idx : ιR ⊕ α → Nat} (hidx : Function.Injective idx) {ρ : ιR ⊕ α → SampleTape → Int}
    {h : ιR ⊕ α → Nat} {L : SampleTape → ℝ} {g : ιR ⊕ α → SampleTape → ℝ} {m : ℝ}
    (hH2 : ResidueReplacement idx ρ h L g m) {Lmax : ℝ} (hL : ∀ t, |L t| ≤ Lmax) (hm : Lmax ≤ m)
    (T : Finset SampleKey) (hT : ∀ j, residueKey (h j) (idx j) ∉ T)
    (hg : DependsOn (fun t j ↦ g j t) T)
    {ER : ℝ} (hR : ∀ t, ∑ x : ιR, (h (.inl x) : ℝ) ^ 2 * g (.inl x) t ^ 2 ≤ ER)
    (site : List Nat) {u : α → SampleTape → Fin n → ℝ} (Tm : Finset SampleKey)
    (hTm : ∀ k : Fin n, maskKey site k.val ∉ Tm) (hu : ∀ x, DependsOn (u x) Tm)
    (hK : ∀ x t, (h (.inr x) : ℝ) * g (.inr x) t = ∑ l, u x t l * (maskClamp site t l.val : ℝ))
    {τ EK : ℝ} (hτ : 0 < τ) (hEK : ∀ t, ∑ x, τ ^ 2 * ∑ l, u x t l ^ 2 ≤ EK) (hE : 0 < ER + EK) :
    tapeMeasure {t | m ≤ |L t + ∑ j, g j t * (ρ j t : ℝ)|} ≤
      Fintype.card α * ENNReal.ofReal (2 * Real.exp (-τ ^ 2 / 2)) +
        ENNReal.ofReal (2 * Real.exp (-(m - Lmax) ^ 2 / (2 * (ER + EK)))) := by
  refine (tail_of_residueReplacement hidx hH2 hL hm T hT hg hE).trans (add_le_add ?_ le_rfl)
  set c : SampleTape → α → ℝ := fun t x ↦ (h (.inr x) : ℝ) * g (.inr x) t with hcdef
  have hsub : {t | ER + EK < ∑ j, (h j : ℝ) ^ 2 * g j t ^ 2} ⊆ {t | EK < ∑ x, c t x ^ 2} := by
    intro t ht
    simp only [Set.mem_setOf_eq, Fintype.sum_sum_type] at ht ⊢
    have h1 := hR t
    have h2 : ∑ x, (h (.inr x) : ℝ) ^ 2 * g (.inr x) t ^ 2 = ∑ x, c t x ^ 2 :=
      Finset.sum_congr rfl fun x _ ↦ by simp only [hcdef, mul_pow]
    linarith
  set b : SampleTape → α → ℝ := fun t x ↦ τ * Real.sqrt (∑ l, u x t l ^ 2) with hbdef
  have hb : ∀ t, ∑ x, b t x ^ 2 ≤ EK := by
    intro t
    refine le_of_eq_of_le (Finset.sum_congr rfl fun x _ ↦ ?_) (hEK t)
    rw [hbdef, mul_pow, Real.sq_sqrt (Finset.sum_nonneg fun l _ ↦ sq_nonneg _)]
  refine (measure_mono hsub).trans ((tapeMeasure_energy_gt_le c b hb).trans ?_)
  have hone : ∀ x, tapeMeasure {t | |b t x| < |c t x|} ≤ ENNReal.ofReal (2 * Real.exp (-τ ^ 2 / 2)) := by
    intro x
    have hset : {t | |b t x| < |c t x|} = {t | τ * Real.sqrt (∑ l, u x t l ^ 2) <
        |∑ l, u x t l * (maskClamp site t l.val : ℝ)|} := by
      ext t
      simp only [Set.mem_setOf_eq, hbdef, hcdef, hK x t]
      rw [abs_of_nonneg (mul_nonneg hτ.le (Real.sqrt_nonneg _))]
    rw [hset]
    exact maskForm_tail site Tm hTm (hu x) hτ
  calc ∑ x, tapeMeasure {t | |b t x| < |c t x|}
      ≤ ∑ _x : α, ENNReal.ofReal (2 * Real.exp (-τ ^ 2 / 2)) := Finset.sum_le_sum fun x _ ↦ hone x
    _ = Fintype.card α * ENNReal.ofReal (2 * Real.exp (-τ ^ 2 / 2)) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

end MxxExponentLut
