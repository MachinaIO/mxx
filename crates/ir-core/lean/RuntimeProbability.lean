import RuntimeSampling
import Mathlib.Logic.Function.DependsOn
import Mathlib.MeasureTheory.Integral.Lebesgue.Markov

/-!
# Tail bounds over independent integer coordinates

Protocol-independent tools for bounding the failure probability of a claim over sampling tapes:
the product measure splits off any one coordinate, a sum `∑ j ∈ B, c j ω * ω j` whose
coefficients do not read the block `B` has its exponential moment bounded block by block, and a
moment bound gives a two-sided tail bound.
-/

namespace MxxRuntime

open MeasureTheory Function

variable {ι : Type} [DecidableEq ι]

/-- A function of finitely many integer coordinates is measurable. -/
theorem measurable_of_dependsOn {β : Type} [MeasurableSpace β] {f : (ι → ℤ) → β} (S : Finset ι)
    (hf : DependsOn f S) : Measurable f := by
  let extend : (S → ℤ) → ι → ℤ := fun r i ↦ if h : i ∈ S then r ⟨i, h⟩ else 0
  have hfac : f = (f ∘ extend) ∘ (fun ω i ↦ ω i.val) := by
    funext ω
    exact hf fun i hi ↦ by simp [extend, Finset.mem_coe.mp hi]
  rw [hfac]
  exact (measurable_of_countable _).comp (measurable_pi_lambda _ fun i ↦ measurable_pi_apply _)

theorem dependsOn_update {β : Type} {f : (ι → ℤ) → β} {S : Set ι} (hf : DependsOn f S) {k : ι}
    (hk : k ∉ S) (ω : ι → ℤ) (x : ℤ) : f (update ω k x) = f ω :=
  hf fun i hi ↦ update_of_ne (by rintro rfl; exact hk hi) _ _

variable (μ : ι → Measure ℤ) [∀ i, IsProbabilityMeasure (μ i)]

theorem measurable_update_pair (k : ι) :
    Measurable fun p : ℤ × (ι → ℤ) ↦ update p.2 k p.1 := by
  refine measurable_pi_lambda _ fun i ↦ ?_
  by_cases h : i = k
  · subst h
    simpa using measurable_fst
  · simpa [update_of_ne h] using (measurable_pi_apply i).comp measurable_snd

/-- Resampling one coordinate independently leaves the product measure unchanged. -/
theorem infinitePi_eq_map_update (k : ι) :
    Measure.infinitePi μ =
      ((μ k).prod (Measure.infinitePi μ)).map fun p ↦ update p.2 k p.1 := by
  symm
  apply Measure.eq_infinitePi
  intro s t ht
  rw [Measure.map_apply (measurable_update_pair k)
    (MeasurableSet.pi s.countable_toSet fun i _ ↦ ht i)]
  have hpre : (fun p : ℤ × (ι → ℤ) ↦ update p.2 k p.1) ⁻¹' Set.pi (↑s) t =
      (if k ∈ s then t k else Set.univ) ×ˢ Set.pi (↑(s.erase k)) t := by
    ext ⟨x, p⟩
    simp only [Set.mem_preimage, Set.mem_pi, Finset.mem_coe, Set.mem_prod, Finset.mem_erase]
    constructor
    · intro h
      refine ⟨?_, fun i ⟨hik, hi⟩ ↦ by simpa [update_of_ne hik] using h i hi⟩
      split_ifs with hk
      · simpa using h k hk
      · trivial
    · rintro ⟨hx, hp⟩ i hi
      by_cases hik : i = k
      · subst hik
        simpa [if_pos hi] using hx
      · simpa [update_of_ne hik] using hp i ⟨hik, hi⟩
  rw [hpre, Measure.prod_prod, Measure.infinitePi_pi _ fun i _ ↦ ht i]
  split_ifs with hk
  · rw [← Finset.mul_prod_erase s _ hk]
  · rw [Finset.erase_eq_of_notMem hk, measure_univ, one_mul]

/-- Integrate one coordinate first. -/
theorem lintegral_infinitePi_update (k : ι) {f : (ι → ℤ) → ENNReal} (hf : Measurable f) :
    ∫⁻ ω, f ω ∂Measure.infinitePi μ =
      ∫⁻ ω, ∫⁻ x, f (update ω k x) ∂μ k ∂Measure.infinitePi μ := by
  conv_lhs => rw [infinitePi_eq_map_update μ k]
  rw [lintegral_map hf (measurable_update_pair k),
    lintegral_prod (fun p : ℤ × (ι → ℤ) ↦ f (update p.2 k p.1))
      (hf.comp (measurable_update_pair k)).aemeasurable]
  exact lintegral_lintegral_swap (hf.comp (measurable_update_pair k)).aemeasurable

/-- One coordinate whose coefficient does not read it contributes at most its conditional
exponential moment. -/
theorem lintegral_mul_exp_coordinate_le (S : Finset ι) {k : ι} (hk : k ∉ S)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {c : (ι → ℤ) → ℝ} (hc : DependsOn c S)
    {K : ℝ} (hK : ∀ ω, ∫⁻ x, ENNReal.ofReal (Real.exp (c ω * x)) ∂μ k ≤ ENNReal.ofReal (Real.exp K)) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (c ω * ω k)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp K) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
  have hk' : k ∉ (S : Set ι) := by simpa using hk
  have hmeas : Measurable fun ω ↦ F ω * ENNReal.ofReal (Real.exp (c ω * ω k)) := by
    refine measurable_of_dependsOn (insert k S) fun ω ω' h ↦ ?_
    have hS : ∀ i ∈ (S : Set ι), ω i = ω' i := fun i hi ↦ h i (by simp [Finset.mem_coe.mp hi])
    rw [hF hS, hc hS, h k (by simp)]
  rw [lintegral_infinitePi_update μ k hmeas, ← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  refine lintegral_mono fun ω ↦ ?_
  simp only [dependsOn_update hF hk', dependsOn_update hc hk', update_self]
  rw [lintegral_const_mul _ (measurable_of_countable _), mul_comm (ENNReal.ofReal _)]
  gcongr
  exact hK ω


/-- A block of coordinates whose coefficients do not read the block contributes at most the
product of the coordinates' conditional exponential moments. -/
theorem lintegral_mul_exp_sum_le (S B : Finset ι) (hdisjoint : Disjoint S B)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {c : ι → (ι → ℤ) → ℝ}
    (hc : ∀ j ∈ B, DependsOn (c j) S) {K : ι → ℝ}
    (hK : ∀ j ∈ B, ∀ ω, ∫⁻ x, ENNReal.ofReal (Real.exp (c j ω * x)) ∂μ j ≤
      ENNReal.ofReal (Real.exp (K j))) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ j ∈ B, c j ω * ω j)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (∑ j ∈ B, K j)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
  induction B using Finset.induction_on with
  | empty => simp
  | @insert k B hkB ih =>
    have hdisjoint' : Disjoint S B := hdisjoint.mono_right (Finset.subset_insert _ _)
    have hkS : k ∉ S := fun h ↦ Finset.disjoint_left.mp hdisjoint h (Finset.mem_insert_self _ _)
    have hcB : ∀ j ∈ B, DependsOn (c j) S := fun j hj ↦ hc j (Finset.mem_insert_of_mem hj)
    -- The rest of the block, with its coefficients, reads only `S ∪ B`.
    have hrest : DependsOn (fun ω ↦ F ω * ENNReal.ofReal (Real.exp (∑ j ∈ B, c j ω * ω j)))
        ↑(S ∪ B) := by
      intro ω ω' h
      have hS : ∀ i ∈ (S : Set ι), ω i = ω' i := fun i hi ↦ h i (by simp_all)
      dsimp only
      rw [hF hS]
      congr 3
      refine Finset.sum_congr rfl fun j hj ↦ ?_
      rw [hcB j hj hS, h j (by simp [hj])]
    have hstep := lintegral_mul_exp_coordinate_le μ (S ∪ B) (by simp [hkS, hkB]) hrest
      ((hc k (Finset.mem_insert_self _ _)).mono (by simp)) (hK k (Finset.mem_insert_self _ _))
    calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ j ∈ insert k B, c j ω * ω j))
          ∂Measure.infinitePi μ
        = ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (∑ j ∈ B, c j ω * ω j))) *
            ENNReal.ofReal (Real.exp (c k ω * ω k)) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [Finset.sum_insert hkB, Real.exp_add,
            ENNReal.ofReal_mul (Real.exp_pos _).le]
          ring
      _ ≤ ENNReal.ofReal (Real.exp (K k)) *
            ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ j ∈ B, c j ω * ω j)) ∂Measure.infinitePi μ :=
          hstep
      _ ≤ ENNReal.ofReal (Real.exp (K k)) *
            (ENNReal.ofReal (Real.exp (∑ j ∈ B, K j)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ) := by
          gcongr
          exact ih hdisjoint' (fun j hj ↦ hc j (Finset.mem_insert_of_mem hj))
            (fun j hj ↦ hK j (Finset.mem_insert_of_mem hj))
      _ = ENNReal.ofReal (Real.exp (∑ j ∈ insert k B, K j)) *
            ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
          rw [Finset.sum_insert hkB, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le,
            mul_assoc]

/-- `lintegral_mul_exp_sum_le` for a block indexed through an injective key map. -/
theorem lintegral_mul_exp_sum_key_le {α : Type} [DecidableEq α] (key : α → ι) (S : Finset ι)
    (B : Finset α) (hinj : Set.InjOn key B) (hdisjoint : ∀ x ∈ B, key x ∉ S)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {c : α → (ι → ℤ) → ℝ}
    (hc : ∀ x ∈ B, DependsOn (c x) S) {K : α → ℝ}
    (hK : ∀ x ∈ B, ∀ ω, ∫⁻ y, ENNReal.ofReal (Real.exp (c x ω * y)) ∂μ (key x) ≤
      ENNReal.ofReal (Real.exp (K x))) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ x ∈ B, c x ω * ω (key x))) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (∑ x ∈ B, K x)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
  induction B using Finset.induction_on with
  | empty => simp
  | @insert x B hxB ih =>
    have hinj' : Set.InjOn key B := hinj.mono (by simp)
    have hxS : key x ∉ S := hdisjoint x (Finset.mem_insert_self _ _)
    have hxB' : key x ∉ B.image key := by
      simp only [Finset.mem_image, not_exists, not_and]
      intro y hy hyx
      exact hxB (hinj (by simp [hy]) (by simp) hyx ▸ hy)
    have hcB : ∀ y ∈ B, DependsOn (c y) S := fun y hy ↦ hc y (Finset.mem_insert_of_mem hy)
    have hrest : DependsOn (fun ω ↦ F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y))))
        ↑(S ∪ B.image key) := by
      intro ω ω' h
      have hS : ∀ i ∈ (S : Set ι), ω i = ω' i := fun i hi ↦ h i (by simp_all)
      dsimp only
      rw [hF hS]
      congr 3
      refine Finset.sum_congr rfl fun y hy ↦ ?_
      rw [hcB y hy hS, h (key y) (by simp only [Finset.coe_union, Finset.coe_image, Set.mem_union, Set.mem_image, Finset.mem_coe]; exact Or.inr ⟨y, hy, rfl⟩)]
    have hstep := lintegral_mul_exp_coordinate_le μ (S ∪ B.image key)
      (by simp only [Finset.mem_union, not_or]; exact ⟨hxS, hxB'⟩) hrest
      ((hc x (Finset.mem_insert_self _ _)).mono (by simp)) (hK x (Finset.mem_insert_self _ _))
    calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ y ∈ insert x B, c y ω * ω (key y)))
          ∂Measure.infinitePi μ
        = ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y)))) *
            ENNReal.ofReal (Real.exp (c x ω * ω (key x))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [Finset.sum_insert hxB, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
          ring
      _ ≤ ENNReal.ofReal (Real.exp (K x)) *
            ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y)))
              ∂Measure.infinitePi μ := hstep
      _ ≤ ENNReal.ofReal (Real.exp (K x)) *
            (ENNReal.ofReal (Real.exp (∑ y ∈ B, K y)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ) := by
          gcongr
          exact ih hinj' (fun y hy ↦ hdisjoint y (Finset.mem_insert_of_mem hy)) hcB
            (fun y hy ↦ hK y (Finset.mem_insert_of_mem hy))
      _ = ENNReal.ofReal (Real.exp (∑ y ∈ insert x B, K y)) *
            ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
          rw [Finset.sum_insert hxB, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le,
            mul_assoc]

/-- A sub-Gaussian exponential moment bound gives the two-sided Gaussian tail bound. -/
theorem measure_le_abs_le {Ω : Type} [MeasurableSpace Ω] (ν : Measure Ω) {Y : Ω → ℝ}
    (hY : Measurable Y) {V : ℝ} (hV : 0 < V)
    (hmgf : ∀ a : ℝ, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * Y ω)) ∂ν ≤
      ENNReal.ofReal (Real.exp (V * a ^ 2 / 2))) {t : ℝ} (ht : 0 ≤ t) :
    ν {ω | t ≤ |Y ω|} ≤ ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * V))) := by
  -- One tail of `Z`, at the optimal exponent `t / V`.
  have tail : ∀ Z : Ω → ℝ, Measurable Z →
      (∀ a : ℝ, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * Z ω)) ∂ν ≤
        ENNReal.ofReal (Real.exp (V * a ^ 2 / 2))) →
      ν {ω | t ≤ Z ω} ≤ ENNReal.ofReal (Real.exp (-t ^ 2 / (2 * V))) := by
    intro Z hZ hmgfZ
    set a := t / V
    have ha : 0 ≤ a := div_nonneg ht hV.le
    have hmarkov := mul_meas_ge_le_lintegral₀ (μ := ν)
      (f := fun ω ↦ ENNReal.ofReal (Real.exp (a * Z ω)))
      (ENNReal.measurable_ofReal.comp (Real.measurable_exp.comp
        (hZ.const_mul a))).aemeasurable (ENNReal.ofReal (Real.exp (a * t)))
    have hsub : {ω | t ≤ Z ω} ⊆
        {ω | ENNReal.ofReal (Real.exp (a * t)) ≤ ENNReal.ofReal (Real.exp (a * Z ω))} :=
      fun ω hω ↦ ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr
        (mul_le_mul_of_nonneg_left hω ha))
    have hbound : ENNReal.ofReal (Real.exp (a * t)) * ν {ω | t ≤ Z ω} ≤
        ENNReal.ofReal (Real.exp (V * a ^ 2 / 2)) :=
      ((mul_le_mul_right (measure_mono hsub) _).trans hmarkov).trans (hmgfZ a)
    have hexp : Real.exp (-t ^ 2 / (2 * V)) * Real.exp (a * t) = Real.exp (V * a ^ 2 / 2) := by
      rw [← Real.exp_add]
      congr 1
      simp only [a]
      field_simp
      ring
    have hpos : ENNReal.ofReal (Real.exp (a * t)) ≠ 0 := by
      simpa using Real.exp_pos (a * t)
    calc ν {ω | t ≤ Z ω}
        = (ENNReal.ofReal (Real.exp (a * t)))⁻¹ * (ENNReal.ofReal (Real.exp (a * t)) *
            ν {ω | t ≤ Z ω}) := by
          rw [← mul_assoc, ENNReal.inv_mul_cancel hpos ENNReal.ofReal_ne_top, one_mul]
      _ ≤ (ENNReal.ofReal (Real.exp (a * t)))⁻¹ * ENNReal.ofReal (Real.exp (V * a ^ 2 / 2)) := by
          gcongr
      _ = ENNReal.ofReal (Real.exp (-t ^ 2 / (2 * V))) := by
          rw [← hexp, ENNReal.ofReal_mul (Real.exp_pos _).le, mul_comm, mul_assoc,
            ENNReal.mul_inv_cancel hpos ENNReal.ofReal_ne_top, mul_one]
  have hupper := tail Y hY hmgf
  have hlower := tail (fun ω ↦ -Y ω) hY.neg fun a ↦ by
    have h := hmgf (-a)
    simp only [neg_mul, mul_neg, neg_sq] at h ⊢
    exact h
  calc ν {ω | t ≤ |Y ω|} ≤ ν ({ω | t ≤ Y ω} ∪ {ω | t ≤ -Y ω}) := by
        refine measure_mono fun ω hω ↦ ?_
        simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
        rcases le_abs'.mp hω with h | h
        · right; linarith
        · left; exact h
    _ ≤ ν {ω | t ≤ Y ω} + ν {ω | t ≤ -Y ω} := measure_union_le _ _
    _ ≤ ENNReal.ofReal (Real.exp (-t ^ 2 / (2 * V))) +
          ENNReal.ofReal (Real.exp (-t ^ 2 / (2 * V))) := add_le_add hupper hlower
    _ = ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * V))) := by
        rw [← ENNReal.ofReal_add (Real.exp_pos _).le (Real.exp_pos _).le]
        congr 1
        ring

end MxxRuntime
