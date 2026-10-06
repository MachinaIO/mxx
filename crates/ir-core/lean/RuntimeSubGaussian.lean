import RuntimeGaussian
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Series

/-!
# Sub-Gaussian vectors over independent integer coordinates

A real vector `X` of the coordinates is sub-Gaussian with variance proxy `V` over a block `B` of
coordinates when its exponential moment along any direction `λ` that, like the weight `F`, reads
only coordinates outside `B`, is at most `exp (V ‖λ‖² / 2)` times the integral of `F`. Fresh
coordinates with sub-Gaussian laws form such a vector; actions whose coefficients read no
coordinate of `B` scale the proxy by their squared operator norm; vectors over disjoint blocks, the
second reading no coordinate of the first, add their proxies; and a coordinate of such a vector
has a Gaussian tail.
-/

namespace MxxRuntime

open MeasureTheory Function

variable {ι : Type} [DecidableEq ι] (μ : ι → Measure ℤ) [∀ i, IsProbabilityMeasure (μ i)]

/-- One coordinate whose coefficient does not read it contributes at most its conditional
exponential moment `exp (G ω)`, where `G` may vary with the other coordinates. -/
theorem lintegral_mul_exp_coordinate_le_fun (S : Finset ι) {k : ι} (hk : k ∉ S)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {c G : (ι → ℤ) → ℝ} (hc : DependsOn c S)
    (hK : ∀ ω, ∫⁻ x, ENNReal.ofReal (Real.exp (c ω * x)) ∂μ k ≤
      ENNReal.ofReal (Real.exp (G ω))) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (c ω * ω k)) ∂Measure.infinitePi μ ≤
      ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (G ω)) ∂Measure.infinitePi μ := by
  have hk' : k ∉ (S : Set ι) := by simpa using hk
  have hmeas : Measurable fun ω ↦ F ω * ENNReal.ofReal (Real.exp (c ω * ω k)) := by
    refine measurable_of_dependsOn (insert k S) fun ω ω' h ↦ ?_
    have hS : ∀ i ∈ (S : Set ι), ω i = ω' i := fun i hi ↦ h i (by simp [Finset.mem_coe.mp hi])
    rw [hF hS, hc hS, h k (by simp)]
  rw [lintegral_infinitePi_update μ k hmeas]
  refine lintegral_mono fun ω ↦ ?_
  simp only [dependsOn_update hF hk', dependsOn_update hc hk', update_self]
  rw [lintegral_const_mul _ (measurable_of_countable _)]
  gcongr
  exact hK ω

/-- A block of fresh coordinates whose coefficients do not read the block: the exponential moment
is bounded by the weight times `exp (V ∑ c² / 2)`, coordinate by coordinate. -/
theorem lintegral_mul_exp_sum_key_le_fun {α : Type} [DecidableEq α] (key : α → ι) (S : Finset ι)
    (B : Finset α) (hinj : Set.InjOn key B) (hdisjoint : ∀ x ∈ B, key x ∉ S)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {c : α → (ι → ℤ) → ℝ}
    (hc : ∀ x ∈ B, DependsOn (c x) S) {V : ℝ}
    (hlaw : ∀ x ∈ B, ∀ a : ℝ, ∫⁻ y, ENNReal.ofReal (Real.exp (a * y)) ∂μ (key x) ≤
      ENNReal.ofReal (Real.exp (V * a ^ 2 / 2))) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ x ∈ B, c x ω * ω (key x))) ∂Measure.infinitePi μ ≤
      ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (V * ∑ x ∈ B, c x ω ^ 2 / 2))
        ∂Measure.infinitePi μ := by
  induction B using Finset.induction_on generalizing F with
  | empty => simp
  | @insert x B hxB ih =>
    have hinj' : Set.InjOn key B := hinj.mono (by simp)
    have hxS : key x ∉ S := hdisjoint x (Finset.mem_insert_self _ _)
    have hxB' : key x ∉ B.image key := by
      simp only [Finset.mem_image, not_exists, not_and]
      intro y hy hyx
      exact hxB (hinj (by simp [hy]) (by simp) hyx ▸ hy)
    have hcB : ∀ y ∈ B, DependsOn (c y) S := fun y hy ↦ hc y (Finset.mem_insert_of_mem hy)
    have hcx : DependsOn (c x) S := hc x (Finset.mem_insert_self _ _)
    have hrest : DependsOn (fun ω ↦ F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y))))
        ↑(S ∪ B.image key) := by
      intro ω ω' h
      have hS : ∀ i ∈ (S : Set ι), ω i = ω' i := fun i hi ↦ h i (by simp_all)
      dsimp only
      rw [hF hS]
      congr 3
      refine Finset.sum_congr rfl fun y hy ↦ ?_
      rw [hcB y hy hS, h (key y) (by
        simp only [Finset.coe_union, Finset.coe_image, Set.mem_union, Set.mem_image,
          Finset.mem_coe]
        exact Or.inr ⟨y, hy, rfl⟩)]
    have hstep := lintegral_mul_exp_coordinate_le_fun μ (S ∪ B.image key)
      (by simp only [Finset.mem_union, not_or]; exact ⟨hxS, hxB'⟩) hrest
      (hcx.mono (by simp)) (G := fun ω ↦ V * c x ω ^ 2 / 2)
      (fun ω ↦ hlaw x (Finset.mem_insert_self _ _) (c x ω))
    have hF' : DependsOn (fun ω ↦ F ω * ENNReal.ofReal (Real.exp (V * c x ω ^ 2 / 2))) S := by
      intro ω ω' h
      simp only [hF h, hcx h]
    calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ y ∈ insert x B, c y ω * ω (key y)))
          ∂Measure.infinitePi μ
        = ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y)))) *
            ENNReal.ofReal (Real.exp (c x ω * ω (key x))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [Finset.sum_insert hxB, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
          ring
      _ ≤ ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y)))) *
            ENNReal.ofReal (Real.exp (V * c x ω ^ 2 / 2)) ∂Measure.infinitePi μ := hstep
      _ = ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (V * c x ω ^ 2 / 2))) *
            ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          ring
      _ ≤ ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (V * c x ω ^ 2 / 2))) *
            ENNReal.ofReal (Real.exp (V * ∑ y ∈ B, c y ω ^ 2 / 2)) ∂Measure.infinitePi μ :=
          ih hinj' (fun y hy ↦ hdisjoint y (Finset.mem_insert_of_mem hy)) hF' hcB
            (fun y hy ↦ hlaw y (Finset.mem_insert_of_mem hy))
      _ = ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (V * ∑ y ∈ insert x B, c y ω ^ 2 / 2))
            ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [Finset.sum_insert hxB, mul_assoc, ← ENNReal.ofReal_mul (Real.exp_pos _).le,
            ← Real.exp_add]
          congr 3
          ring

/-- `lintegral_mul_exp_sum_key_le_fun` with a proxy `V x` for every coordinate. -/
theorem lintegral_mul_exp_sum_key_le_fun' {α : Type} [DecidableEq α] (key : α → ι) (S : Finset ι)
    (B : Finset α) (hinj : Set.InjOn key B) (hdisjoint : ∀ x ∈ B, key x ∉ S)
    {F : (ι → ℤ) → ENNReal} (hF : DependsOn F S) {c : α → (ι → ℤ) → ℝ}
    (hc : ∀ x ∈ B, DependsOn (c x) S) {V : α → ℝ}
    (hlaw : ∀ x ∈ B, ∀ a : ℝ, ∫⁻ y, ENNReal.ofReal (Real.exp (a * y)) ∂μ (key x) ≤
      ENNReal.ofReal (Real.exp (V x * a ^ 2 / 2))) :
    ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ x ∈ B, c x ω * ω (key x))) ∂Measure.infinitePi μ ≤
      ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ x ∈ B, V x * c x ω ^ 2 / 2))
        ∂Measure.infinitePi μ := by
  induction B using Finset.induction_on generalizing F with
  | empty => simp
  | @insert x B hxB ih =>
    have hinj' : Set.InjOn key B := hinj.mono (by simp)
    have hxS : key x ∉ S := hdisjoint x (Finset.mem_insert_self _ _)
    have hxB' : key x ∉ B.image key := by
      simp only [Finset.mem_image, not_exists, not_and]
      intro y hy hyx
      exact hxB (hinj (by simp [hy]) (by simp) hyx ▸ hy)
    have hcB : ∀ y ∈ B, DependsOn (c y) S := fun y hy ↦ hc y (Finset.mem_insert_of_mem hy)
    have hcx : DependsOn (c x) S := hc x (Finset.mem_insert_self _ _)
    have hrest : DependsOn (fun ω ↦ F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y))))
        ↑(S ∪ B.image key) := by
      intro ω ω' h
      have hS : ∀ i ∈ (S : Set ι), ω i = ω' i := fun i hi ↦ h i (by simp_all)
      dsimp only
      rw [hF hS]
      congr 3
      refine Finset.sum_congr rfl fun y hy ↦ ?_
      rw [hcB y hy hS, h (key y) (by
        simp only [Finset.coe_union, Finset.coe_image, Set.mem_union, Set.mem_image,
          Finset.mem_coe]
        exact Or.inr ⟨y, hy, rfl⟩)]
    have hstep := lintegral_mul_exp_coordinate_le_fun μ (S ∪ B.image key)
      (by simp only [Finset.mem_union, not_or]; exact ⟨hxS, hxB'⟩) hrest
      (hcx.mono (by simp)) (G := fun ω ↦ V x * c x ω ^ 2 / 2)
      (fun ω ↦ hlaw x (Finset.mem_insert_self _ _) (c x ω))
    have hF' : DependsOn (fun ω ↦ F ω * ENNReal.ofReal (Real.exp (V x * c x ω ^ 2 / 2))) S := by
      intro ω ω' h
      simp only [hF h, hcx h]
    calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ y ∈ insert x B, c y ω * ω (key y)))
          ∂Measure.infinitePi μ
        = ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y)))) *
            ENNReal.ofReal (Real.exp (c x ω * ω (key x))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [Finset.sum_insert hxB, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
          ring
      _ ≤ ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y)))) *
            ENNReal.ofReal (Real.exp (V x * c x ω ^ 2 / 2)) ∂Measure.infinitePi μ := hstep
      _ = ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (V x * c x ω ^ 2 / 2))) *
            ENNReal.ofReal (Real.exp (∑ y ∈ B, c y ω * ω (key y))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          ring
      _ ≤ ∫⁻ ω, (F ω * ENNReal.ofReal (Real.exp (V x * c x ω ^ 2 / 2))) *
            ENNReal.ofReal (Real.exp (∑ y ∈ B, V y * c y ω ^ 2 / 2)) ∂Measure.infinitePi μ :=
          ih hinj' (fun y hy ↦ hdisjoint y (Finset.mem_insert_of_mem hy)) hF' hcB
            (fun y hy ↦ hlaw y (Finset.mem_insert_of_mem hy))
      _ = ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ y ∈ insert x B, V y * c y ω ^ 2 / 2))
            ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [Finset.sum_insert hxB, mul_assoc, ← ENNReal.ofReal_mul (Real.exp_pos _).le,
            ← Real.exp_add]

/-- `X` is sub-Gaussian with variance proxy `V` over the block `B`. -/
def SubG {κ : Type} [Fintype κ] (X : (ι → ℤ) → κ → ℝ) (B : Finset ι) (V : ℝ) : Prop :=
  ∀ (S : Finset ι), Disjoint S B → ∀ (lam : (ι → ℤ) → κ → ℝ) (F : (ι → ℤ) → ENNReal) (L : ℝ),
    DependsOn F S → DependsOn lam S → (∀ ω, ∑ i, lam ω i ^ 2 ≤ L) →
      ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ i, lam ω i * X ω i)) ∂Measure.infinitePi μ ≤
        ENNReal.ofReal (Real.exp (V * L / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ

variable {μ}

/-- Fresh coordinates `ω (key i)` whose laws are sub-Gaussian with proxy `V` form a sub-Gaussian
vector over their keys. -/
theorem SubG.fresh {κ : Type} [Fintype κ] [DecidableEq κ] (key : κ → ι)
    (hinj : Function.Injective key) {V : ℝ} (hV : 0 ≤ V)
    (hlaw : ∀ i (a : ℝ), ∫⁻ x, ENNReal.ofReal (Real.exp (a * x)) ∂μ (key i) ≤
      ENNReal.ofReal (Real.exp (V * a ^ 2 / 2))) :
    SubG μ (fun ω i ↦ (ω (key i) : ℝ)) (Finset.univ.image key) V := by
  intro S hS lam F L hF hlam hL
  have hkey : ∀ i ∈ (Finset.univ : Finset κ), key i ∉ S := fun i _ h ↦
    Finset.disjoint_left.mp hS h (Finset.mem_image_of_mem key (Finset.mem_univ i))
  have h := lintegral_mul_exp_sum_key_le_fun μ key S Finset.univ hinj.injOn hkey hF
    (c := fun i ω ↦ lam ω i) (fun i _ ω ω' h ↦ by simp only [hlam h]) (V := V)
    (fun i _ a ↦ hlaw i a)
  refine h.trans ?_
  rw [← lintegral_const_mul' _ _ ENNReal.ofReal_ne_top]
  refine lintegral_mono fun ω ↦ ?_
  rw [mul_comm (ENNReal.ofReal _)]
  refine mul_le_mul_right (ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr ?_)) _
  have := mul_le_mul_of_nonneg_left (hL ω) hV
  simp only
  rw [← Finset.sum_div]
  linarith

omit [∀ i, IsProbabilityMeasure (μ i)] in
/-- An action whose coefficients read no coordinate of the block scales the proxy by the square
`ρ2` of its operator norm. -/
theorem SubG.action {κ κ' : Type} [Fintype κ] [Fintype κ'] {X : (ι → ℤ) → κ → ℝ}
    {B : Finset ι} {V : ℝ} (hX : SubG μ X B V) {M : (ι → ℤ) → κ' → κ → ℝ}
    (T : Finset ι) (hT : Disjoint T B) (hM : DependsOn M T) {ρ2 : ℝ} (hρ2 : 0 ≤ ρ2)
    (hnorm : ∀ ω (v : κ' → ℝ), ∑ i, (∑ j, v j * M ω j i) ^ 2 ≤ ρ2 * ∑ j, v j ^ 2) :
    SubG μ (fun ω j ↦ ∑ i, M ω j i * X ω i) B (ρ2 * V) := by
  intro S hS lam F L hF hlam hL
  have h := hX (S ∪ T) (Finset.disjoint_union_left.mpr ⟨hS, hT⟩)
    (fun ω i ↦ ∑ j, lam ω j * M ω j i) F (ρ2 * L) (hF.mono (by simp))
    (fun ω ω' hω ↦ by
      have h1 := hlam (fun x hx ↦ hω x (by simp only [Finset.coe_union, Set.mem_union]; exact Or.inl hx))
      have h2 := hM (fun x hx ↦ hω x (by simp only [Finset.coe_union, Set.mem_union]; exact Or.inr hx))
      simp only [h1, h2])
    (fun ω ↦ (hnorm ω (lam ω)).trans (mul_le_mul_of_nonneg_left (hL ω) hρ2))
  calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ j, lam ω j * ∑ i, M ω j i * X ω i))
        ∂Measure.infinitePi μ
      = ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ i, (∑ j, lam ω j * M ω j i) * X ω i))
        ∂Measure.infinitePi μ := by
        refine lintegral_congr fun ω ↦ ?_
        congr 3
        simp only [Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        exact Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦ by ring
    _ ≤ ENNReal.ofReal (Real.exp (V * (ρ2 * L) / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := h
    _ = ENNReal.ofReal (Real.exp (ρ2 * V * L / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
        congr 3
        ring

omit [∀ i, IsProbabilityMeasure (μ i)] in
/-- Two sub-Gaussian vectors over disjoint blocks, the second reading no coordinate of the first,
add their proxies. -/
theorem SubG.add {κ : Type} [Fintype κ] {X Y : (ι → ℤ) → κ → ℝ} {B₁ B₂ : Finset ι}
    {V₁ V₂ : ℝ} (hX : SubG μ X B₁ V₁) (hY : SubG μ Y B₂ V₂)
    (T : Finset ι) (hT : Disjoint T B₁) (hYdep : DependsOn Y T) :
    SubG μ (fun ω i ↦ X ω i + Y ω i) (B₁ ∪ B₂) (V₁ + V₂) := by
  intro S hS lam F L hF hlam hL
  have hS₁ : Disjoint S B₁ := (Finset.disjoint_union_right.mp hS).1
  have hS₂ : Disjoint S B₂ := (Finset.disjoint_union_right.mp hS).2
  set G : (ι → ℤ) → ENNReal := fun ω ↦ F ω * ENNReal.ofReal (Real.exp (∑ i, lam ω i * Y ω i))
  have hG : DependsOn G ((S ∪ T : Finset ι) : Set ι) := by
    intro ω ω' hω
    have h1 := hF (fun x hx ↦ hω x (by simp [Finset.mem_coe.mp hx]))
    have h2 := hlam (fun x hx ↦ hω x (by simp [Finset.mem_coe.mp hx]))
    have h3 := hYdep (fun x hx ↦ hω x (by simp [Finset.mem_coe.mp hx]))
    simp only [G, h1, h2, h3]
  have h1 := hX (S ∪ T) (Finset.disjoint_union_left.mpr ⟨hS₁, hT⟩) lam G L hG
    (hlam.mono (by simp)) hL
  have h2 := hY S hS₂ lam F L hF hlam hL
  calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ i, lam ω i * (X ω i + Y ω i)))
        ∂Measure.infinitePi μ
      = ∫⁻ ω, G ω * ENNReal.ofReal (Real.exp (∑ i, lam ω i * X ω i)) ∂Measure.infinitePi μ := by
        refine lintegral_congr fun ω ↦ ?_
        simp only [G, mul_add, Finset.sum_add_distrib, Real.exp_add,
          ENNReal.ofReal_mul (Real.exp_pos _).le]
        ring
    _ ≤ ENNReal.ofReal (Real.exp (V₁ * L / 2)) * ∫⁻ ω, G ω ∂Measure.infinitePi μ := h1
    _ ≤ ENNReal.ofReal (Real.exp (V₁ * L / 2)) *
          (ENNReal.ofReal (Real.exp (V₂ * L / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ) := by
        gcongr
    _ = ENNReal.ofReal (Real.exp ((V₁ + V₂) * L / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
        rw [← mul_assoc, ← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 3
        ring

/-- A coordinate of a sub-Gaussian vector reading finitely many coordinates has a two-sided
Gaussian tail. -/
theorem SubG.tail {κ : Type} [Fintype κ] [DecidableEq κ] {X : (ι → ℤ) → κ → ℝ} {B : Finset ι}
    {V : ℝ} (hX : SubG μ X B V) (hV : 0 < V) (R : Finset ι) (hXdep : DependsOn X R) (i : κ)
    {t : ℝ} (ht : 0 ≤ t) :
    Measure.infinitePi μ {ω | t ≤ |X ω i|} ≤ ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * V))) := by
  have hmeas : Measurable fun ω ↦ X ω i :=
    measurable_of_dependsOn R fun ω ω' h ↦ by rw [hXdep h]
  refine measure_le_abs_le _ hmeas hV (fun a ↦ ?_) ht
  have h := hX ∅ (Finset.disjoint_empty_left _) (fun _ j ↦ if j = i then a else 0) (fun _ ↦ 1)
    (a ^ 2) (fun _ _ _ ↦ rfl) (fun _ _ _ ↦ rfl) (fun _ ↦ by simp)
  simp only [ite_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, if_true, one_mul,
    lintegral_const, measure_univ, mul_one] at h
  calc ∫⁻ ω, ENNReal.ofReal (Real.exp (a * X ω i)) ∂Measure.infinitePi μ
      ≤ ENNReal.ofReal (Real.exp (V * a ^ 2 / 2)) := h

/-- A symmetric uniform law on `[-h, h]` is sub-Gaussian with variance proxy `h²`
(Hoeffding): its moment is the mean of `cosh (a x)` over `|x| ≤ h`. -/
theorem SamplerLaw.lintegral_exp_interval_symmetric_le (h : ℕ) (a : ℝ) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (a * x)) ∂(SamplerLaw.interval (-(h : ℤ)) h).measure ≤
      ENNReal.ofReal (Real.exp ((h : ℝ) ^ 2 * a ^ 2 / 2)) := by
  set I : Finset ℤ := Finset.Icc (-(h : ℤ)) h
  have hle : -(h : ℤ) ≤ h := by omega
  simp only [SamplerLaw.measure, SamplerLaw.pmf, hle, dite_true]
  rw [lintegral_countable']
  simp_rw [PMF.toMeasure_apply_singleton _ _ (measurableSet_singleton _),
    PMF.uniformOfFinset_apply]
  rw [tsum_eq_sum (s := I) fun x hx ↦ by rw [if_neg hx, mul_zero]]
  rw [Finset.sum_congr rfl fun x hx ↦ by rw [if_pos hx], ← Finset.sum_mul]
  -- The real sum of the moments over the symmetric interval.
  have hsymm : ∑ x ∈ I, Real.exp (a * x) = ∑ x ∈ I, Real.cosh (a * x) := by
    have hneg : ∑ x ∈ I, Real.exp (a * x) = ∑ x ∈ I, Real.exp (-(a * x)) := by
      refine Finset.sum_nbij' (fun x ↦ -x) (fun x ↦ -x) ?_ ?_ (fun _ _ ↦ neg_neg _)
        (fun _ _ ↦ neg_neg _) fun x _ ↦ ?_
      · intro x hx; simp only [I, Finset.mem_Icc] at hx ⊢; omega
      · intro x hx; simp only [I, Finset.mem_Icc] at hx ⊢; omega
      · push_cast; ring_nf
    have : 2 * ∑ x ∈ I, Real.exp (a * x) = 2 * ∑ x ∈ I, Real.cosh (a * x) := by
      rw [two_mul]
      nth_rewrite 2 [hneg]
      rw [← Finset.sum_add_distrib, Finset.mul_sum]
      exact Finset.sum_congr rfl fun x _ ↦ by rw [Real.cosh_eq]; ring
    linarith
  have hbound : ∑ x ∈ I, Real.exp (a * x) ≤ I.card * Real.exp ((h : ℝ) ^ 2 * a ^ 2 / 2) := by
    rw [hsymm]
    calc ∑ x ∈ I, Real.cosh (a * x) ≤ ∑ _x ∈ I, Real.exp ((h : ℝ) ^ 2 * a ^ 2 / 2) := by
          refine Finset.sum_le_sum fun x hx ↦ ?_
          simp only [I, Finset.mem_Icc] at hx
          have hx' : |(x : ℝ)| ≤ h := by
            rw [abs_le]; constructor <;> exact_mod_cast (by omega : _)
          calc Real.cosh (a * x) ≤ Real.cosh (a * h) := by
                rw [Real.cosh_le_cosh, abs_mul, abs_mul, Nat.abs_cast]
                exact mul_le_mul_of_nonneg_left hx' (abs_nonneg a)
            _ ≤ Real.exp ((a * h) ^ 2 / 2) := Real.cosh_le_exp_half_sq _
            _ = Real.exp ((h : ℝ) ^ 2 * a ^ 2 / 2) := by ring_nf
      _ = I.card * Real.exp ((h : ℝ) ^ 2 * a ^ 2 / 2) := by
          rw [Finset.sum_const, nsmul_eq_mul]
  have hcard : (0 : ℝ) < I.card := by
    exact_mod_cast Finset.card_pos.mpr ⟨0, by simp [I]⟩
  rw [← ENNReal.ofReal_sum_of_nonneg fun _ _ ↦ (Real.exp_pos _).le]
  rw [show ((I.card : ENNReal))⁻¹ = ENNReal.ofReal ((I.card : ℝ)⁻¹) by
    rw [ENNReal.ofReal_inv_of_pos hcard]; simp, ← ENNReal.ofReal_mul (by positivity)]
  apply ENNReal.ofReal_le_ofReal
  rw [mul_inv_le_iff₀ hcard]
  linarith

/-! ## Coordinatewise sub-Gaussian vectors

After a general action the coordinates of a vector are no longer independent, but each stays
sub-Gaussian. `CSubG X B v` bounds the moment of each coordinate `i` separately with proxy `v i`;
a linear form in fresh coordinates has the proxy of its coefficient energy, an action composes
by Hölder's inequality with the squared `l1` weight `(Σ |m| √v)²`, and two vectors over disjoint
blocks, the second reading none of the first, add their proxies. -/

variable (μ) in
/-- Every coordinate `i` of `X` is sub-Gaussian with proxy `v i` over the block `B`. -/
def CSubG {κ : Type} (X : (ι → ℤ) → κ → ℝ) (B : Finset ι) (v : κ → ℝ) : Prop :=
  ∀ (i : κ) (S : Finset ι), Disjoint S B → ∀ (a : (ι → ℤ) → ℝ) (F : (ι → ℤ) → ENNReal) (A : ℝ),
    DependsOn F S → DependsOn a S → (∀ ω, a ω ^ 2 ≤ A) →
      ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (a ω * X ω i)) ∂Measure.infinitePi μ ≤
        ENNReal.ofReal (Real.exp (v i * A / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ

/-- A linear form in fresh coordinates whose coefficients read only `T`, disjoint from the keys:
coordinate `i` has the proxy `V e i` of its coefficient energy `e i`. -/
theorem CSubG.linear {κ α : Type} [Fintype α] [DecidableEq α] (key : α → ι)
    (hinj : Function.Injective key) {V : ℝ} (hV : 0 ≤ V)
    (hlaw : ∀ x (a : ℝ), ∫⁻ y, ENNReal.ofReal (Real.exp (a * y)) ∂μ (key x) ≤
      ENNReal.ofReal (Real.exp (V * a ^ 2 / 2)))
    {c : (ι → ℤ) → κ → α → ℝ} (T : Finset ι) (hT : ∀ x, key x ∉ T) (hc : DependsOn c T)
    {e : κ → ℝ} (he : ∀ ω i, ∑ x, c ω i x ^ 2 ≤ e i) :
    CSubG μ (fun ω i ↦ ∑ x, c ω i x * ω (key x)) (Finset.univ.image key) (fun i ↦ V * e i) := by
  intro i S hS a F A hF ha hA
  have hkey : ∀ x ∈ (Finset.univ : Finset α), key x ∉ S ∪ T := fun x _ h ↦ by
    rcases Finset.mem_union.mp h with h | h
    · exact Finset.disjoint_left.mp hS h (Finset.mem_image_of_mem key (Finset.mem_univ x))
    · exact hT x h
  have hdep : ∀ x ∈ (Finset.univ : Finset α),
      DependsOn (fun ω ↦ a ω * c ω i x) ((S ∪ T : Finset ι) : Set ι) := fun x _ ω ω' h ↦ by
    have h1 := ha (fun y hy ↦ h y (by simp [Finset.mem_coe.mp hy]))
    have h2 := hc (fun y hy ↦ h y (by simp [Finset.mem_coe.mp hy]))
    simp only [h1, h2]
  have h := lintegral_mul_exp_sum_key_le_fun μ key (S ∪ T) Finset.univ hinj.injOn hkey
    (hF.mono (by simp)) (c := fun x ω ↦ a ω * c ω i x) hdep (V := V) (fun x _ a ↦ hlaw x a)
  calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (a ω * ∑ x, c ω i x * ω (key x)))
        ∂Measure.infinitePi μ
      = ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (∑ x, a ω * c ω i x * ω (key x)))
        ∂Measure.infinitePi μ := by
        refine lintegral_congr fun ω ↦ ?_
        rw [Finset.mul_sum]
        simp only [mul_assoc]
    _ ≤ ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (V * ∑ x, (a ω * c ω i x) ^ 2 / 2))
        ∂Measure.infinitePi μ := h
    _ ≤ ∫⁻ ω, ENNReal.ofReal (Real.exp (V * e i * A / 2)) * F ω ∂Measure.infinitePi μ := by
        refine lintegral_mono fun ω ↦ ?_
        rw [mul_comm (ENNReal.ofReal _)]
        refine mul_le_mul_right (ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr ?_)) _
        rw [← Finset.sum_div]
        have hsq : ∑ x, (a ω * c ω i x) ^ 2 = a ω ^ 2 * ∑ x, c ω i x ^ 2 := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun x _ ↦ by ring
        rw [hsq]
        have h1 := he ω i
        have h2 := hA ω
        have h3 : a ω ^ 2 * ∑ x, c ω i x ^ 2 ≤ A * e i :=
          mul_le_mul h2 h1 (Finset.sum_nonneg fun _ _ ↦ sq_nonneg _)
            ((sq_nonneg _).trans h2)
        nlinarith
    _ = ENNReal.ofReal (Real.exp (V * e i * A / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ :=
        lintegral_const_mul' _ _ ENNReal.ofReal_ne_top

omit [∀ i, IsProbabilityMeasure (μ i)] in
/-- Hölder: an action whose coefficients read only `T`, disjoint from the block, gives coordinate
`j` the proxy `w j ^ 2` for any `w j ≥ Σ i, |m j i| √(v i)`. -/
theorem CSubG.combine {κ κ' : Type} [Fintype κ] [Nonempty κ] {X : (ι → ℤ) → κ → ℝ}
    {B : Finset ι} {v : κ → ℝ} (hX : CSubG μ X B v) (hv : ∀ i, 0 < v i) (R : Finset ι)
    (hXdep : DependsOn X R) {m : (ι → ℤ) → κ' → κ → ℝ} (T : Finset ι) (hT : Disjoint T B)
    (hm : DependsOn m T) {w : κ' → ℝ}
    (hw : ∀ ω j, ∑ i, |m ω j i| * Real.sqrt (v i) ≤ w j) :
    CSubG μ (fun ω j ↦ ∑ i, m ω j i * X ω i) B (fun j ↦ w j ^ 2) := by
  classical
  intro j S hS a F A hF ha hA
  set α : (ι → ℤ) → κ → ℝ := fun ω i ↦ a ω * m ω j i with hα
  set W : (ι → ℤ) → ℝ := fun ω ↦ ∑ i, |α ω i| * Real.sqrt (v i) with hWdef
  set p : (ι → ℤ) → κ → ℝ := fun ω i ↦
    if W ω = 0 then 1 / (Fintype.card κ : ℝ) else |α ω i| * Real.sqrt (v i) / W ω with hpdef
  set b : (ι → ℤ) → κ → ℝ := fun ω i ↦ if W ω = 0 then 0 else α ω i / p ω i with hbdef
  have hcard : (0 : ℝ) < Fintype.card κ := by exact_mod_cast Fintype.card_pos
  have hsqrt : ∀ i, 0 < Real.sqrt (v i) := fun i ↦ Real.sqrt_pos.mpr (hv i)
  have hW0 : ∀ ω, 0 ≤ W ω := fun ω ↦
    Finset.sum_nonneg fun i _ ↦ mul_nonneg (abs_nonneg _) (hsqrt i).le
  have hzero : ∀ ω, W ω = 0 → ∀ i, α ω i = 0 := by
    intro ω hW i
    have hterm := (Finset.sum_eq_zero_iff_of_nonneg fun i _ ↦
      mul_nonneg (abs_nonneg (α ω i)) (hsqrt i).le).mp hW i (Finset.mem_univ i)
    rcases mul_eq_zero.mp hterm with h | h
    · exact abs_eq_zero.mp h
    · exact absurd h (hsqrt i).ne'
  have hp0 : ∀ ω i, 0 ≤ p ω i := by
    intro ω i
    simp only [hpdef]
    split_ifs
    · positivity
    · exact div_nonneg (mul_nonneg (abs_nonneg _) (hsqrt i).le) (hW0 ω)
  have hp1 : ∀ ω, ∑ i, p ω i = 1 := by
    intro ω
    simp only [hpdef]
    split_ifs with h
    · rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]; field_simp
    · rw [← Finset.sum_div]; exact div_self h
  have hpb : ∀ ω i, p ω i * b ω i = α ω i := by
    intro ω i
    simp only [hbdef]
    split_ifs with h
    · rw [mul_zero, hzero ω h i]
    · by_cases hpi : p ω i = 0
      · have : α ω i = 0 := by
          simp only [hpdef, if_neg h] at hpi
          rcases div_eq_zero_iff.mp hpi with hpi | hpi
          · rcases mul_eq_zero.mp hpi with hpi | hpi
            · exact abs_eq_zero.mp hpi
            · exact absurd hpi (hsqrt i).ne'
          · exact absurd hpi h
        rw [hpi, zero_mul, this]
      · field_simp
  have hwj : ∀ ω, W ω ≤ |a ω| * w j := by
    intro ω
    calc W ω = |a ω| * ∑ i, |m ω j i| * Real.sqrt (v i) := by
          simp only [hWdef, hα, abs_mul, Finset.mul_sum, mul_assoc]
      _ ≤ |a ω| * w j := mul_le_mul_of_nonneg_left (hw ω j) (abs_nonneg _)
  have hb2 : ∀ ω i, b ω i ^ 2 ≤ A * w j ^ 2 / v i := by
    intro ω i
    have hAw : 0 ≤ A * w j ^ 2 / v i :=
      div_nonneg (mul_nonneg ((sq_nonneg _).trans (hA ω)) (sq_nonneg _)) (hv i).le
    simp only [hbdef]
    split_ifs with h
    · simpa using hAw
    · by_cases hai : α ω i = 0
      · rw [hai, zero_div]; simpa using hAw
      · have hWpos : 0 < W ω := lt_of_le_of_ne (hW0 ω) (Ne.symm h)
        have hpv : p ω i = |α ω i| * Real.sqrt (v i) / W ω := by simp only [hpdef, if_neg h]
        have hbv : (α ω i / p ω i) ^ 2 = W ω ^ 2 / v i := by
          rw [hpv]
          generalize α ω i = x at hai ⊢
          generalize W ω = y at hWpos ⊢
          have hx : |x| ≠ 0 := abs_ne_zero.mpr hai
          have hs := hsqrt i
          rw [div_div_eq_mul_div, div_pow, mul_pow, mul_pow, sq_abs, Real.sq_sqrt (hv i).le]
          field_simp
        rw [hbv]
        have hW2 : W ω ^ 2 ≤ A * w j ^ 2 := by
          have h1 := hwj ω
          have h2 : W ω ^ 2 ≤ (|a ω| * w j) ^ 2 := pow_le_pow_left₀ (hW0 ω) h1 2
          rw [mul_pow, sq_abs] at h2
          exact h2.trans (mul_le_mul_of_nonneg_right (hA ω) (sq_nonneg _))
        exact div_le_div_of_nonneg_right hW2 (hv i).le
  have hjensen : ∀ ω, Real.exp (∑ i, α ω i * X ω i) ≤
      ∑ i, p ω i * Real.exp (b ω i * X ω i) := by
    intro ω
    have h := convexOn_exp.map_sum_le (t := Finset.univ) (w := p ω)
      (p := fun i ↦ b ω i * X ω i) (fun i _ ↦ hp0 ω i) (hp1 ω) (fun _ _ ↦ Set.mem_univ _)
    simp only [smul_eq_mul] at h
    calc Real.exp (∑ i, α ω i * X ω i)
        = Real.exp (∑ i, p ω i * (b ω i * X ω i)) := by
          congr 1
          exact Finset.sum_congr rfl fun i _ ↦ by rw [← mul_assoc, hpb]
      _ ≤ _ := h
  -- The dependence of the weights and directions.
  have hSTdisj : Disjoint (S ∪ T) B := Finset.disjoint_union_left.mpr ⟨hS, hT⟩
  have hαdep : ∀ i, DependsOn (fun ω ↦ α ω i) ((S ∪ T : Finset ι) : Set ι) := fun i ω ω' h ↦ by
    have h1 := ha (fun y hy ↦ h y (by simp [Finset.mem_coe.mp hy]))
    have h2 := hm (fun y hy ↦ h y (by simp [Finset.mem_coe.mp hy]))
    simp only [hα, h1, h2]
  have hWdep : DependsOn W ((S ∪ T : Finset ι) : Set ι) := fun ω ω' h ↦ by
    simp only [hWdef]
    exact Finset.sum_congr rfl fun i _ ↦ by
      have h1 : α ω i = α ω' i := hαdep i h
      rw [h1]
  have hpdep : ∀ i, DependsOn (fun ω ↦ p ω i) ((S ∪ T : Finset ι) : Set ι) := fun i ω ω' h ↦ by
    have h1 : α ω i = α ω' i := hαdep i h
    have h2 : W ω = W ω' := hWdep h
    simp only [hpdef, h1, h2]
  have hbdep : ∀ i, DependsOn (fun ω ↦ b ω i) ((S ∪ T : Finset ι) : Set ι) := fun i ω ω' h ↦ by
    have h1 : α ω i = α ω' i := hαdep i h
    have h2 : W ω = W ω' := hWdep h
    have h3 : p ω i = p ω' i := hpdep i h
    simp only [hbdef, h1, h2, h3]
  have hFdep : DependsOn F ((S ∪ T : Finset ι) : Set ι) := hF.mono (by simp)
  set G : κ → (ι → ℤ) → ENNReal := fun i ω ↦ F ω * ENNReal.ofReal (p ω i)
  have hGdep : ∀ i, DependsOn (G i) ((S ∪ T : Finset ι) : Set ι) := fun i ω ω' h ↦ by
    simp only [G, hFdep h, hpdep i h]
  have hmeas : ∀ i, Measurable fun ω ↦ G i ω * ENNReal.ofReal (Real.exp (b ω i * X ω i)) :=
    fun i ↦ measurable_of_dependsOn (S ∪ T ∪ R) fun ω ω' h ↦ by
      have h1 := hGdep i (fun y hy ↦ h y (by simp only [Finset.coe_union, Set.mem_union] at hy ⊢; tauto))
      have h2 := hbdep i (fun y hy ↦ h y (by simp only [Finset.coe_union, Set.mem_union] at hy ⊢; tauto))
      have h3 := hXdep (fun y hy ↦ h y (by simp only [Finset.coe_union, Set.mem_union] at hy ⊢; tauto))
      simp only [h1, h2, h3]
  have hstep : ∀ i, ∫⁻ ω, G i ω * ENNReal.ofReal (Real.exp (b ω i * X ω i))
      ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (A * w j ^ 2 / 2)) * ∫⁻ ω, G i ω ∂Measure.infinitePi μ := by
    intro i
    have h := hX i (S ∪ T) hSTdisj (fun ω ↦ b ω i) (G i) (A * w j ^ 2 / v i) (hGdep i)
      (hbdep i) (fun ω ↦ hb2 ω i)
    have he : v i * (A * w j ^ 2 / v i) / 2 = A * w j ^ 2 / 2 := by
      field_simp [(hv i).ne']
    rwa [he] at h
  calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (a ω * ∑ i, m ω j i * X ω i)) ∂Measure.infinitePi μ
      ≤ ∫⁻ ω, ∑ i, G i ω * ENNReal.ofReal (Real.exp (b ω i * X ω i)) ∂Measure.infinitePi μ := by
        refine lintegral_mono fun ω ↦ ?_
        have hsum : a ω * ∑ i, m ω j i * X ω i = ∑ i, α ω i * X ω i := by
          rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ ↦ by simp only [hα]; ring
        rw [hsum]
        simp only [G, mul_assoc, ← Finset.mul_sum]
        refine mul_le_mul_right ?_ _
        calc ENNReal.ofReal (Real.exp (∑ i, α ω i * X ω i))
            ≤ ENNReal.ofReal (∑ i, p ω i * Real.exp (b ω i * X ω i)) :=
              ENNReal.ofReal_le_ofReal (hjensen ω)
          _ = ∑ i, ENNReal.ofReal (p ω i * Real.exp (b ω i * X ω i)) :=
              ENNReal.ofReal_sum_of_nonneg fun i _ ↦ mul_nonneg (hp0 ω i) (Real.exp_pos _).le
          _ = ∑ i, ENNReal.ofReal (p ω i) * ENNReal.ofReal (Real.exp (b ω i * X ω i)) :=
              Finset.sum_congr rfl fun i _ ↦ ENNReal.ofReal_mul (hp0 ω i)
    _ = ∑ i, ∫⁻ ω, G i ω * ENNReal.ofReal (Real.exp (b ω i * X ω i)) ∂Measure.infinitePi μ :=
        lintegral_finset_sum _ fun i _ ↦ hmeas i
    _ ≤ ∑ i, ENNReal.ofReal (Real.exp (A * w j ^ 2 / 2)) * ∫⁻ ω, G i ω ∂Measure.infinitePi μ :=
        Finset.sum_le_sum fun i _ ↦ hstep i
    _ = ENNReal.ofReal (Real.exp (w j ^ 2 * A / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
        rw [← Finset.mul_sum, ← lintegral_finset_sum _ fun i _ ↦
          measurable_of_dependsOn (S ∪ T) (hGdep i)]
        congr 1
        · congr 2; ring
        · refine lintegral_congr fun ω ↦ ?_
          simp only [G, ← Finset.mul_sum]
          rw [← ENNReal.ofReal_sum_of_nonneg fun i _ ↦ hp0 ω i, hp1 ω, ENNReal.ofReal_one,
            mul_one]

omit [∀ i, IsProbabilityMeasure (μ i)] in
/-- Two coordinatewise sub-Gaussian vectors over disjoint blocks, the second reading no
coordinate of the first, add their proxies. -/
theorem CSubG.add {κ : Type} {X Y : (ι → ℤ) → κ → ℝ} {B₁ B₂ : Finset ι} {v₁ v₂ : κ → ℝ}
    (hX : CSubG μ X B₁ v₁) (hY : CSubG μ Y B₂ v₂) (T : Finset ι) (hT : Disjoint T B₁)
    (hYdep : DependsOn Y T) :
    CSubG μ (fun ω i ↦ X ω i + Y ω i) (B₁ ∪ B₂) (fun i ↦ v₁ i + v₂ i) := by
  intro i S hS a F A hF ha hA
  have hS₁ : Disjoint S B₁ := (Finset.disjoint_union_right.mp hS).1
  have hS₂ : Disjoint S B₂ := (Finset.disjoint_union_right.mp hS).2
  set G : (ι → ℤ) → ENNReal := fun ω ↦ F ω * ENNReal.ofReal (Real.exp (a ω * Y ω i))
  have hG : DependsOn G ((S ∪ T : Finset ι) : Set ι) := by
    intro ω ω' hω
    have h1 := hF (fun x hx ↦ hω x (by simp [Finset.mem_coe.mp hx]))
    have h2 := ha (fun x hx ↦ hω x (by simp [Finset.mem_coe.mp hx]))
    have h3 := hYdep (fun x hx ↦ hω x (by simp [Finset.mem_coe.mp hx]))
    simp only [G, h1, h2, h3]
  have h1 := hX i (S ∪ T) (Finset.disjoint_union_left.mpr ⟨hS₁, hT⟩) a G A hG
    (ha.mono (by simp)) hA
  have h2 := hY i S hS₂ a F A hF ha hA
  calc ∫⁻ ω, F ω * ENNReal.ofReal (Real.exp (a ω * (X ω i + Y ω i))) ∂Measure.infinitePi μ
      = ∫⁻ ω, G ω * ENNReal.ofReal (Real.exp (a ω * X ω i)) ∂Measure.infinitePi μ := by
        refine lintegral_congr fun ω ↦ ?_
        simp only [G, mul_add, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
        ring
    _ ≤ ENNReal.ofReal (Real.exp (v₁ i * A / 2)) * ∫⁻ ω, G ω ∂Measure.infinitePi μ := h1
    _ ≤ ENNReal.ofReal (Real.exp (v₁ i * A / 2)) *
          (ENNReal.ofReal (Real.exp (v₂ i * A / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ) := by
        gcongr
    _ = ENNReal.ofReal (Real.exp ((v₁ i + v₂ i) * A / 2)) * ∫⁻ ω, F ω ∂Measure.infinitePi μ := by
        rw [← mul_assoc, ← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 3
        ring

omit [DecidableEq ι] [∀ i, IsProbabilityMeasure (μ i)] in
/-- A larger proxy. -/
theorem CSubG.mono {κ : Type} {X : (ι → ℤ) → κ → ℝ} {B : Finset ι} {v v' : κ → ℝ}
    (hX : CSubG μ X B v) (hv : ∀ i, v i ≤ v' i) : CSubG μ X B v' := by
  intro i S hS a F A hF ha hA
  refine (hX i S hS a F A hF ha hA).trans (mul_le_mul_left (ENNReal.ofReal_le_ofReal
    (Real.exp_le_exp.mpr ?_)) _)
  have hA0 : 0 ≤ A := (sq_nonneg _).trans (hA (fun _ ↦ 0))
  nlinarith [hv i]

/-- A coordinate of a coordinatewise sub-Gaussian vector reading finitely many coordinates has a
two-sided Gaussian tail. -/
theorem CSubG.tail {κ : Type} {X : (ι → ℤ) → κ → ℝ} {B : Finset ι} {v : κ → ℝ}
    (hX : CSubG μ X B v) (R : Finset ι) (hXdep : DependsOn X R) (i : κ) (hv : 0 < v i)
    {t : ℝ} (ht : 0 ≤ t) :
    Measure.infinitePi μ {ω | t ≤ |X ω i|} ≤ ENNReal.ofReal (2 * Real.exp (-t ^ 2 / (2 * v i))) := by
  have hmeas : Measurable fun ω ↦ X ω i :=
    measurable_of_dependsOn R fun ω ω' h ↦ by rw [hXdep h]
  refine measure_le_abs_le _ hmeas hv (fun a ↦ ?_) ht
  have h := hX i ∅ (Finset.disjoint_empty_left _) (fun _ ↦ a) (fun _ ↦ 1) (a ^ 2)
    (fun _ _ _ ↦ rfl) (fun _ _ _ ↦ rfl) (fun _ ↦ le_rfl)
  simp only [one_mul, lintegral_const, measure_univ, mul_one] at h
  exact h

/-- The tail of a linear form in fresh coordinates whose coefficients read only `T`, disjoint from
the keys, is at most the probability that the coefficient energy exceeds `E` plus the Gaussian
tail of proxy `V E`: on the complement, the form equals the form with truncated coefficients. -/
theorem tail_linear_truncated {α : Type} [Fintype α] [DecidableEq α] (key : α → ι)
    (hinj : Function.Injective key) {V : ℝ} (hV : 0 < V)
    (hlaw : ∀ x (a : ℝ), ∫⁻ y, ENNReal.ofReal (Real.exp (a * y)) ∂μ (key x) ≤
      ENNReal.ofReal (Real.exp (V * a ^ 2 / 2)))
    {c : (ι → ℤ) → α → ℝ} (T : Finset ι) (hT : ∀ x, key x ∉ T) (hc : DependsOn c T)
    {E : ℝ} (hE : 0 < E) {m : ℝ} (hm : 0 ≤ m) :
    Measure.infinitePi μ {ω | m ≤ |∑ x, c ω x * ω (key x)|} ≤
      Measure.infinitePi μ {ω | E < ∑ x, c ω x ^ 2} +
        ENNReal.ofReal (2 * Real.exp (-m ^ 2 / (2 * (V * E)))) := by
  classical
  set c' : (ι → ℤ) → Unit → α → ℝ := fun ω _ x ↦ if ∑ y, c ω y ^ 2 ≤ E then c ω x else 0 with hc'
  have hc'dep : DependsOn c' T := by
    intro ω ω' h
    have := hc h
    funext u x
    simp only [hc', this]
  have henergy : ∀ ω (u : Unit), ∑ x, c' ω u x ^ 2 ≤ E := by
    intro ω u
    simp only [hc']
    split_ifs with h
    · exact h
    · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, Finset.sum_const_zero]
      exact hE.le
  have hlin := CSubG.linear (μ := μ) key hinj hV.le hlaw T hT hc'dep (e := fun _ ↦ E) henergy
  set X : (ι → ℤ) → Unit → ℝ := fun ω u ↦ ∑ x, c' ω u x * ω (key x) with hX
  have hXdep : DependsOn X ((T ∪ Finset.univ.image key : Finset ι) : Set ι) := by
    intro ω ω' h
    have h1 := hc'dep (fun i hi ↦ h i (by simp [Finset.mem_coe.mp hi]))
    funext u
    simp only [hX, h1]
    refine Finset.sum_congr rfl fun x _ ↦ ?_
    rw [h (key x) (by simp)]
  have htail := hlin.tail (T ∪ Finset.univ.image key) hXdep () (mul_pos hV hE) hm
  have hsub : {ω | m ≤ |∑ x, c ω x * ω (key x)|} ⊆
      {ω | E < ∑ x, c ω x ^ 2} ∪ {ω | m ≤ |X ω ()|} := by
    intro ω hω
    by_cases hω' : ∑ x, c ω x ^ 2 ≤ E
    · right
      simp only [Set.mem_setOf_eq, hX, hc', if_pos hω'] at hω ⊢
      exact hω
    · left
      exact lt_of_not_ge hω'
  exact (measure_mono hsub).trans ((measure_union_le _ _).trans (by gcongr))

/-- `tail_linear_truncated` with a proxy `V x` for every coordinate: the energy is
`Σ V x c x²`. -/
theorem tail_linear_truncated' {α : Type} [Fintype α] [DecidableEq α] (key : α → ι)
    (hinj : Function.Injective key) {V : α → ℝ}
    (hlaw : ∀ x (a : ℝ), ∫⁻ y, ENNReal.ofReal (Real.exp (a * y)) ∂μ (key x) ≤
      ENNReal.ofReal (Real.exp (V x * a ^ 2 / 2)))
    {c : (ι → ℤ) → α → ℝ} (T : Finset ι) (hT : ∀ x, key x ∉ T) (hc : DependsOn c T)
    {E : ℝ} (hE : 0 < E) {m : ℝ} (hm : 0 ≤ m) :
    Measure.infinitePi μ {ω | m ≤ |∑ x, c ω x * ω (key x)|} ≤
      Measure.infinitePi μ {ω | E < ∑ x, V x * c ω x ^ 2} +
        ENNReal.ofReal (2 * Real.exp (-m ^ 2 / (2 * E))) := by
  classical
  set c' : (ι → ℤ) → α → ℝ := fun ω x ↦ if ∑ y, V y * c ω y ^ 2 ≤ E then c ω x else 0 with hc'
  have hc'dep : DependsOn c' T := by
    intro ω ω' h
    have := hc h
    funext x
    simp only [hc', this]
  have henergy : ∀ ω, ∑ x, V x * c' ω x ^ 2 ≤ E := by
    intro ω
    simp only [hc']
    split_ifs with h
    · exact h
    · simp only [ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, mul_zero,
        Finset.sum_const_zero]
      exact hE.le
  set X : (ι → ℤ) → ℝ := fun ω ↦ ∑ x, c' ω x * ω (key x) with hX
  have hXdep : DependsOn X ((T ∪ Finset.univ.image key : Finset ι) : Set ι) := by
    intro ω ω' h
    have h1 := hc'dep (fun i hi ↦ h i (by simp [Finset.mem_coe.mp hi]))
    simp only [hX, h1]
    refine Finset.sum_congr rfl fun x _ ↦ ?_
    rw [h (key x) (by simp)]
  have hmeas : Measurable X := measurable_of_dependsOn _ hXdep
  have hmgf : ∀ a : ℝ, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * X ω)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (E * a ^ 2 / 2)) := by
    intro a
    have hkey : ∀ x ∈ (Finset.univ : Finset α), key x ∉ T := fun x _ ↦ hT x
    have h := lintegral_mul_exp_sum_key_le_fun' μ key T Finset.univ hinj.injOn hkey
      (F := fun _ ↦ 1) (fun _ _ _ ↦ rfl) (c := fun x ω ↦ a * c' ω x)
      (fun x _ ω ω' hω ↦ by simp only [hc'dep hω]) (V := V) (fun x _ b ↦ hlaw x b)
    simp only [one_mul] at h
    calc ∫⁻ ω, ENNReal.ofReal (Real.exp (a * X ω)) ∂Measure.infinitePi μ
        = ∫⁻ ω, ENNReal.ofReal (Real.exp (∑ x, a * c' ω x * ω (key x))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          simp only [hX, Finset.mul_sum, mul_assoc]
      _ ≤ ∫⁻ ω, ENNReal.ofReal (Real.exp (∑ x, V x * (a * c' ω x) ^ 2 / 2))
            ∂Measure.infinitePi μ := h
      _ ≤ ∫⁻ _ω, ENNReal.ofReal (Real.exp (E * a ^ 2 / 2)) ∂Measure.infinitePi μ := by
          refine lintegral_mono fun ω ↦ ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr ?_)
          have hsq : ∑ x, V x * (a * c' ω x) ^ 2 / 2 = a ^ 2 * (∑ x, V x * c' ω x ^ 2) / 2 := by
            rw [Finset.mul_sum, Finset.sum_div]
            exact Finset.sum_congr rfl fun x _ ↦ by ring
          rw [hsq]
          have := henergy ω
          have ha : 0 ≤ a ^ 2 := sq_nonneg a
          nlinarith
      _ = ENNReal.ofReal (Real.exp (E * a ^ 2 / 2)) := by simp
  have htail := measure_le_abs_le (Measure.infinitePi μ) hmeas hE hmgf hm
  have hsub : {ω | m ≤ |∑ x, c ω x * ω (key x)|} ⊆
      {ω | E < ∑ x, V x * c ω x ^ 2} ∪ {ω | m ≤ |X ω|} := by
    intro ω hω
    by_cases hω' : ∑ x, V x * c ω x ^ 2 ≤ E
    · right
      simp only [Set.mem_setOf_eq, hX, hc', if_pos hω'] at hω ⊢
      exact hω
    · left
      exact lt_of_not_ge hω'
  exact (measure_mono hsub).trans ((measure_union_le _ _).trans (by gcongr))

omit [DecidableEq ι] [∀ i, IsProbabilityMeasure (μ i)] in
/-- The coefficient energy exceeds `E` only if some coefficient exceeds its share `b`, when
`Σ b² ≤ E`. -/
theorem measure_energy_gt_le {α : Type} [Fintype α] (c : (ι → ℤ) → α → ℝ) {b : α → ℝ}
    {E : ℝ} (hb : ∑ x, b x ^ 2 ≤ E) :
    Measure.infinitePi μ {ω | E < ∑ x, c ω x ^ 2} ≤
      ∑ x, Measure.infinitePi μ {ω | |b x| < |c ω x|} := by
  classical
  have hsub : {ω | E < ∑ x, c ω x ^ 2} ⊆ ⋃ x ∈ (Finset.univ : Finset α), {ω | |b x| < |c ω x|} := by
    intro ω hω
    by_contra hnot
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, Finset.mem_univ, exists_true_left, not_exists,
      not_lt] at hnot
    have hle : ∑ x, c ω x ^ 2 ≤ ∑ x, b x ^ 2 := Finset.sum_le_sum fun x _ ↦ by
      rw [← sq_abs (c ω x), ← sq_abs (b x)]
      exact pow_le_pow_left₀ (abs_nonneg _) (hnot x) 2
    exact absurd (hω.trans_le (hle.trans hb)) (lt_irrefl _)
  exact (measure_mono hsub).trans (measure_biUnion_finset_le _ _)

end MxxRuntime
