import TfheNand
import TfheMS

/-!
# The input noise as a function of the tape

`msErr` is the sum over the LWE secret of the rounding errors `colRound j` of the NAND input mask,
and the mask entry `j` is `Σᵢ A_{ij} Dᵢ mod q`, where `A_{ij}` is the key-switching mask read at
the tape coordinate `maskKey j i` and `Dᵢ` the sum of the key-switching digits of the two fresh
encryptions. Column `j` of the masks is read by nothing else.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity

/-- The tape coordinate of key-switching mask entry `(i, j)`. -/
def maskKey (j : Fin 720) (i : Fin 8192) : Key := (4, i.val * 720 + j.val)

/-- Key-switching mask entry `(i, j)` from the tape. -/
def maskA (ω : Key → ℤ) (i : Fin 8192) (j : Fin 720) : Int :=
  (SamplerLaw.residue Qmask).clamp (ω (maskKey j i)) % 4294967296

/-- The digit sum of the two fresh encryptions. -/
noncomputable def dsum (messages : Fin 2 → Int) (ω : Key → ℤ) (i : Fin 8192) : Int :=
  digits0 messages ω i + digits1 messages ω i

/-- The rounding error of NAND input mask entry `j`. -/
noncomputable def colRound (messages : Fin 2 → Int) (ω : Key → ℤ) (j : Fin 720) : Int :=
  msRound (∑ i, maskA ω i j * dsum messages ω i)

theorem round_eq_msRound (x : Int) :
    2097152 * ((x * 2048 + 2147483648) / 4294967296) - x = msRound x := by
  unfold msRound; omega

theorem kskAOf_eq (ω : Key → ℤ) (f : Fin 5898240) :
    kskAOf ω f = (SamplerLaw.residue Qmask).clamp (ω (4, f.val)) % 4294967296 := by
  letI : Fact (1 < Qmask) := ⟨by decide⟩
  rw [kskAOf, kskMaskOf, maskVals]
  congr 1
  rw [smp, polynomialOfCoefficients_coeff (by decide) (by decide)]
  have hidx : ((finProdFinEquiv (finProdFinEquiv ((0 : Fin 1), (⟨f.val / 1024, by omega⟩ :
      Fin 5760)), (⟨f.val % 1024, by omega⟩ : Fin 1024))) : Fin (1 * 5760 * 1024)).val = f.val := by
    rw [finProdFinEquiv_apply_val, finProdFinEquiv_apply_val]
    simp only [Fin.val_zero, Fin.val_mk]
    omega
  rw [hidx]
  have h := SamplerLaw.clamp_supports (.residue Qmask) (ω (4, f.val))
  simp only [SamplerLaw.Supports] at h
  rw [if_pos (by decide)] at h
  rw [ZMod.val_intCast, Int.emod_eq_of_lt h.1 (by exact_mod_cast h.2)]

theorem nandA_cong (messages : Fin 2 → Int) (ω : Key → ℤ) (j : Fin 720) :
    ((nandA messages ω j : Int) : ZMod q) = ((∑ i, maskA ω i j * dsum messages ω i : Int) : ZMod q) := by
  have hcol (d : Fin 8192 → Int) : intMatrixVectorProduct true (kskAOf ω) d j =
      ∑ i : Fin 8192, maskA ω i j * d i := by
    rw [intMatrixVectorProduct_columns (by rfl)]
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [kskAOf_eq]
    rfl
  show ((((0 - ((0 - intMatrixVectorProduct true (kskAOf ω) (digits0 messages ω) j) % 4294967296 +
    (0 - intMatrixVectorProduct true (kskAOf ω) (digits1 messages ω) j) % 4294967296) %
      4294967296) % 4294967296 : Int)) : ZMod q) = _
  rw [hcol, hcol]
  simp only [cast_emod_q, Int.cast_sub, Int.cast_add, Int.cast_zero, Int.cast_sum, Int.cast_mul,
    dsum]
  simp only [mul_add, Finset.sum_add_distrib]
  ring

theorem msErr_eq (messages : Fin 2 → Int) (ω : Key → ℤ) :
    msErr messages ω = ∑ j, secretOf ω j * colRound messages ω j := by
  unfold msErr
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [round_eq_msRound]
  congr 1
  obtain ⟨t, ht⟩ := exists_of_cast_eq (nandA_cong messages ω j)
  rw [colRound, ht, show ((q : Nat) : Int) = 2097152 * 2048 from rfl]
  rw [show (∑ i, maskA ω i j * dsum messages ω i) + 2097152 * 2048 * t =
    (∑ i, maskA ω i j * dsum messages ω i) + 2097152 * (2048 * t) by ring, msRound_periodic]

/-! ## Which coordinates the noise reads -/

/-- The coordinates of the sites `L`. -/
def region (L : List Nat) : Finset Key := L.toFinset.biUnion fun s ↦ siteSet s siteBound

theorem mem_siteSet_iff {s m : Nat} {p : Key} : p ∈ siteSet s m ↔ p.1 = s ∧ p.2 < m := by
  rw [siteSet_def]
  constructor
  · rintro hp
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.mp hp
    exact ⟨rfl, j.isLt⟩
  · rintro ⟨h1, h2⟩
    exact Finset.mem_image.mpr ⟨⟨p.2, h2⟩, Finset.mem_univ _, by ext <;> simp [h1]⟩

theorem mem_region {L : List Nat} {p : Key} : p ∈ region L ↔ p.1 ∈ L ∧ p.2 < siteBound := by
  rw [region, Finset.mem_biUnion]
  constructor
  · rintro ⟨s, hs, hp⟩
    obtain ⟨h1, h2⟩ := mem_siteSet_iff.mp hp
    exact ⟨h1 ▸ List.mem_toFinset.mp hs, h2⟩
  · rintro ⟨h1, h2⟩
    exact ⟨p.1, List.mem_toFinset.mpr h1, mem_siteSet_iff.mpr ⟨rfl, h2⟩⟩

theorem smp_congr {q n rows columns : Nat} (law : SamplerLaw) {L : List Nat} {s : Nat} (hs : s ∈ L)
    (hsize : rows * columns * n ≤ siteBound) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ (region L : Set Key), ω k = ω' k) :
    (smp law s ω : ExactMatrix q n rows columns) = smp law s ω' := by
  funext row column
  simp only [smp]
  congr 1
  funext k
  rw [h _ (Finset.mem_coe.mpr (mem_region.mpr ⟨hs, lt_of_lt_of_le (Fin.isLt _) hsize⟩))]

/-- The sites the fresh encryptions' digits and the LWE secret read. -/
abbrev inSites : List Nat := [1, 13, 15, 16, 18, 19]

theorem dsum_congr (messages : Fin 2 → Int) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ (region inSites : Set Key), ω k = ω' k) : dsum messages ω = dsum messages ω' := by
  have h13 := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) (.residue Q)
    (s := 13) (by decide) (by decide) h
  have h15 := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) bitLaw
    (s := 15) (by decide) (by decide) h
  have h16 := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) ringLaw
    (s := 16) (by decide) (by decide) h
  have h18 := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) bitLaw
    (s := 18) (by decide) (by decide) h
  have h19 := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) ringLaw
    (s := 19) (by decide) (by decide) h
  funext i
  show digitAt (extractAt (coeffVals (encC0 (smp (.residue Q) 13 ω) (smp bitLaw 15 ω)
      (smp ringLaw (15 + 1) ω)))) i + digitAt (extractAt (coeffVals (encC0 (smp (.residue Q) 13 ω)
      (smp bitLaw 18 ω) (smp ringLaw (18 + 1) ω)))) i =
    digitAt (extractAt (coeffVals (encC0 (smp (.residue Q) 13 ω') (smp bitLaw 15 ω')
      (smp ringLaw (15 + 1) ω')))) i + digitAt (extractAt (coeffVals (encC0 (smp (.residue Q) 13 ω')
      (smp bitLaw 18 ω') (smp ringLaw (18 + 1) ω')))) i
  rw [h13, h15, h18, show 15 + 1 = 16 from rfl, show 18 + 1 = 19 from rfl, h16, h19]

theorem secretOf_congr {ω ω' : Key → ℤ} (h : ∀ k ∈ (region inSites : Set Key), ω k = ω' k) :
    secretOf ω = secretOf ω' := by
  funext j
  rw [secretOf_eq, secretOf_eq, h _ (Finset.mem_coe.mpr (mem_region.mpr ⟨by simp,
    lt_of_lt_of_le j.isLt (by decide)⟩))]

theorem digitAt_range (a : Fin 1024 → Int) (i : Fin 8192) : 0 ≤ digitAt a i ∧ digitAt a i ≤ 3 := by
  unfold digitAt; omega

theorem dsum_range (messages : Fin 2 → Int) (ω : Key → ℤ) (i : Fin 8192) :
    0 ≤ dsum messages ω i ∧ dsum messages ω i ≤ 6 := by
  have h0 : 0 ≤ digits0 messages ω i ∧ digits0 messages ω i ≤ 3 := digitAt_range _ i
  have h1 : 0 ≤ digits1 messages ω i ∧ digits1 messages ω i ≤ 3 := digitAt_range _ i
  unfold dsum
  constructor <;> linarith [h0.1, h0.2, h1.1, h1.2]

/-! ## One column of the masks -/

open MeasureTheory Function

/-- Column `j` of the key-switching masks. -/
def colBlock (j : Fin 720) : Finset Key := Finset.univ.image (maskKey j)

theorem maskKey_injective (j : Fin 720) : Injective (maskKey j) := by
  intro i i' h
  simp only [maskKey, Prod.mk.injEq] at h
  exact Fin.ext (by omega)

theorem mem_colBlock {j : Fin 720} {p : Key} : p ∈ colBlock j ↔ ∃ i, maskKey j i = p := by
  simp [colBlock]

theorem region_disjoint_colBlock (j : Fin 720) : Disjoint (region inSites) (colBlock j) := by
  rw [Finset.disjoint_left]
  intro p hp hc
  obtain ⟨i, rfl⟩ := mem_colBlock.mp hc
  have := (mem_region.mp hp).1
  simp [maskKey] at this

theorem colBlock_disjoint {j j' : Fin 720} (h : j ≠ j') : Disjoint (colBlock j) (colBlock j') := by
  rw [Finset.disjoint_left]
  intro p hp hp'
  obtain ⟨i, rfl⟩ := mem_colBlock.mp hp
  obtain ⟨i', hi'⟩ := mem_colBlock.mp hp'
  simp only [maskKey, Prod.mk.injEq] at hi'
  apply h
  exact Fin.ext (by have := j.isLt; have := j'.isLt; omega)

theorem colRound_congr (messages : Fin 2 → Int) (j : Fin 720) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ ((region inSites ∪ colBlock j : Finset Key) : Set Key), ω k = ω' k) :
    colRound messages ω j = colRound messages ω' j := by
  have hr : ∀ k ∈ (region inSites : Set Key), ω k = ω' k :=
    fun k hk ↦ h k (by simp only [Finset.coe_union, Set.mem_union]; exact Or.inl hk)
  unfold colRound
  rw [dsum_congr messages hr]
  congr 1
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  have hk : maskKey j i ∈ ((region inSites ∪ colBlock j : Finset Key) : Set Key) :=
    Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inr (mem_colBlock.mpr ⟨i, rfl⟩)))
  unfold maskA
  rw [h (maskKey j i) hk]

/-- One column of rounding errors, weighted by its secret bit, has the moment of `msRound`. -/
theorem column_mgf (messages : Fin 2 → Int) (j : Fin 720) (a : ℝ) (ω : Key → ℤ) :
    ∫⁻ ω', ENNReal.ofReal (Real.exp (a * ((secretOf ((colBlock j).piecewise ω' ω) j *
      colRound messages ((colBlock j).piecewise ω' ω) j : Int) : ℝ)))
      ∂Measure.infinitePi μ ≤ ENNReal.ofReal (Real.exp (msExp a + maskExcess)) := by
  have hK0 : (1 : ENNReal) ≤ ENNReal.ofReal (Real.exp (msExp a + maskExcess)) := by
    rw [← ENNReal.ofReal_one]
    have h1 := msExp_nonneg a
    have h2 : (0 : ℝ) ≤ maskExcess := by positivity
    exact ENNReal.ofReal_le_ofReal (Real.one_le_exp (by linarith))
  have hreg : ∀ ω', ∀ k ∈ (region inSites : Set Key), (colBlock j).piecewise ω' ω k = ω k := by
    intro ω' k hk
    exact Finset.piecewise_eq_of_notMem _ _ _
      (Finset.disjoint_left.mp (region_disjoint_colBlock j) (Finset.mem_coe.mp hk))
  have hs : ∀ ω', secretOf ((colBlock j).piecewise ω' ω) j = secretOf ω j :=
    fun ω' ↦ congrFun (secretOf_congr (hreg ω')) j
  have hD : ∀ ω', dsum messages ((colBlock j).piecewise ω' ω) = dsum messages ω :=
    fun ω' ↦ dsum_congr messages (hreg ω')
  have hA : ∀ ω' i, maskA ((colBlock j).piecewise ω' ω) i j = maskA ω' i j := by
    intro ω' i
    unfold maskA
    rw [Finset.piecewise_eq_of_mem _ _ _ (mem_colBlock.mpr ⟨i, rfl⟩)]
  simp only [hs, colRound, hD, hA]
  set D := dsum messages ω
  rcases secretOf_binary ω j with h0 | h1
  · simp only [h0, zero_mul, Int.cast_zero, mul_zero, Real.exp_zero, ENNReal.ofReal_one,
      lintegral_const, measure_univ, mul_one]
    exact hK0
  by_cases hall : ∀ i, D i = 0
  · simp only [h1, hall, mul_zero, Finset.sum_const_zero, one_mul]
    simp only [msRound, zero_add, Int.reduceMod, sub_self, Int.cast_zero, mul_zero,
      Real.exp_zero, ENNReal.ofReal_one, lintegral_const, measure_univ, mul_one]
    exact hK0
  push_neg at hall
  obtain ⟨i0, hi0⟩ := hall
  have hrange : 0 ≤ D i0 ∧ D i0 ≤ 6 := dsum_range messages ω i0
  have hD0 : 1 ≤ D i0 ∧ D i0 ≤ 6 := ⟨by omega, hrange.2⟩
  simp only [h1, one_mul]
  set k0 := maskKey j i0
  set C : (Key → ℤ) → Int := fun ω' ↦ ∑ i ∈ Finset.univ.erase i0, maskA ω' i j * D i
  have hsplit : ∀ ω', ∑ i, maskA ω' i j * D i = maskA ω' i0 j * D i0 + C ω' :=
    fun ω' ↦ (Finset.add_sum_erase _ _ (Finset.mem_univ i0)).symm
  have hmeas : Measurable fun ω' : Key → ℤ ↦ ENNReal.ofReal (Real.exp (a * (msRound
      (∑ i, maskA ω' i j * D i) : ℝ))) := by
    refine measurable_of_dependsOn (colBlock j) fun ω₁ ω₂ h ↦ ?_
    have hsum : ∑ i, maskA ω₁ i j * D i = ∑ i, maskA ω₂ i j * D i := by
      refine Finset.sum_congr rfl fun i _ ↦ ?_
      unfold maskA
      rw [h (maskKey j i) (Finset.mem_coe.mpr (mem_colBlock.mpr ⟨i, rfl⟩))]
    simp only [hsum]
  rw [lintegral_infinitePi_update μ k0 hmeas]
  have hinner : ∀ ω', ∫⁻ x, ENNReal.ofReal (Real.exp (a * (msRound
      (∑ i, maskA (update ω' k0 x) i j * D i) : ℝ))) ∂μ k0 ≤
      ENNReal.ofReal (Real.exp (msExp a + maskExcess)) := by
    intro ω'
    have hC : ∀ x, C (update ω' k0 x) = C ω' := by
      intro x
      refine Finset.sum_congr rfl fun i hi ↦ ?_
      have hne : maskKey j i ≠ k0 := fun h ↦ (Finset.ne_of_mem_erase hi) (maskKey_injective j h)
      unfold maskA
      rw [update_of_ne hne]
    have hx : ∀ x, maskA (update ω' k0 x) i0 j = (SamplerLaw.residue Qmask).clamp x % 4294967296 := by
      intro x
      unfold maskA
      rw [update_self]
    simp only [hsplit, hx, hC]
    exact ms_lintegral_le (D i0) hD0 (C ω') a
  calc ∫⁻ ω', ∫⁻ x, ENNReal.ofReal (Real.exp (a * (msRound
        (∑ i, maskA (update ω' k0 x) i j * D i) : ℝ))) ∂μ k0 ∂Measure.infinitePi μ
      ≤ ∫⁻ _ω', ENNReal.ofReal (Real.exp (msExp a + maskExcess)) ∂Measure.infinitePi μ :=
        lintegral_mono hinner
    _ = ENNReal.ofReal (Real.exp (msExp a + maskExcess)) := by simp

/-! ## All columns -/

/-- The rounding errors of the columns before `n`. -/
noncomputable def msPartial (messages : Fin 2 → Int) (ω : Key → ℤ) (n : Nat) : Int :=
  ∑ j : Fin 720, if j.val < n then secretOf ω j * colRound messages ω j else 0

/-- The columns before `n`. -/
def colsBefore (n : Nat) : Finset Key :=
  (Finset.univ.filter fun j : Fin 720 ↦ j.val < n).biUnion colBlock

theorem msPartial_congr (messages : Fin 2 → Int) (n : Nat) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ ((region inSites ∪ colsBefore n : Finset Key) : Set Key), ω k = ω' k) :
    msPartial messages ω n = msPartial messages ω' n := by
  have hr : ∀ k ∈ (region inSites : Set Key), ω k = ω' k :=
    fun k hk ↦ h k (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inl hk)))
  unfold msPartial
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  split_ifs with hj
  · rw [secretOf_congr hr]
    rw [colRound_congr messages j fun k hk ↦ h k ?_]
    rcases Finset.mem_union.mp (Finset.mem_coe.mp hk) with h1 | h1
    · exact Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inl h1))
    · refine Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inr ?_))
      exact Finset.mem_biUnion.mpr ⟨j, by simp [hj], h1⟩
  · rfl

theorem msPartial_succ (messages : Fin 2 → Int) (ω : Key → ℤ) (n : Nat) (hn : n < 720) :
    msPartial messages ω (n + 1) =
      msPartial messages ω n + secretOf ω ⟨n, hn⟩ * colRound messages ω ⟨n, hn⟩ := by
  unfold msPartial
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨n, hn⟩ : Fin 720)),
    ← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨n, hn⟩ : Fin 720))]
  simp only [lt_add_one, if_true, lt_irrefl, if_false, zero_add]
  rw [add_comm]
  congr 1
  refine Finset.sum_congr rfl fun j hj ↦ ?_
  have hne : j.val ≠ n := fun h ↦ Finset.ne_of_mem_erase hj (Fin.ext h)
  by_cases hjn : j.val < n
  · rw [if_pos hjn, if_pos (by omega)]
  · rw [if_neg hjn, if_neg (by omega)]

theorem msPartial_mgf (messages : Fin 2 → Int) (a : ℝ) :
    ∀ n ≤ 720, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msPartial messages ω n : ℝ)))
      ∂Measure.infinitePi μ ≤ ENNReal.ofReal (Real.exp (n * (msExp a + maskExcess))) := by
  intro n
  induction n with
  | zero =>
    intro _
    simp [msPartial]
  | succ n ih =>
    intro hn
    have hlt : n < 720 := by omega
    set j : Fin 720 := ⟨n, hlt⟩
    have hdisj : Disjoint (region inSites ∪ colsBefore n) (colBlock j) := by
      rw [Finset.disjoint_union_left]
      refine ⟨region_disjoint_colBlock j, ?_⟩
      rw [Finset.disjoint_left]
      intro p hp hpj
      obtain ⟨j', hj', hp'⟩ := Finset.mem_biUnion.mp hp
      have : j' ≠ j := fun h ↦ by
        rw [h] at hj'
        simp [j] at hj'
      exact Finset.disjoint_left.mp (colBlock_disjoint this) hp' hpj
    have hblock := lintegral_mul_exp_block_le (μ := μ) (colBlock j) (region inSites ∪ colsBefore n)
      hdisj (F := fun ω ↦ ENNReal.ofReal (Real.exp (a * (msPartial messages ω n : ℝ))))
      (fun ω ω' h ↦ by simp only [msPartial_congr messages n h])
      (g := fun ω ↦ a * ((secretOf ω j * colRound messages ω j : Int) : ℝ))
      (fun ω ω' h ↦ by
        have hr : ∀ k ∈ (region inSites : Set Key), ω k = ω' k := fun k hk ↦ h k (by
          simp only [Finset.coe_union, Set.mem_union, Finset.mem_coe] at hk ⊢
          exact Or.inl (Or.inl hk))
        have hc : ∀ k ∈ ((region inSites ∪ colBlock j : Finset Key) : Set Key), ω k = ω' k :=
          fun k hk ↦ h k (by
            simp only [Finset.coe_union, Set.mem_union, Finset.mem_coe] at hk ⊢
            rcases hk with hk | hk
            · exact Or.inl (Or.inl hk)
            · exact Or.inr hk)
        simp only [secretOf_congr hr, colRound_congr messages j hc])
      (K := msExp a + maskExcess) (fun ω ↦ column_mgf messages j a ω)
    calc ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msPartial messages ω (n + 1) : ℝ)))
          ∂Measure.infinitePi μ
        = ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msPartial messages ω n : ℝ))) *
            ENNReal.ofReal (Real.exp (a * ((secretOf ω j * colRound messages ω j : Int) : ℝ)))
            ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [msPartial_succ messages ω n hlt, ← ENNReal.ofReal_mul (Real.exp_pos _).le,
            ← Real.exp_add]
          push_cast
          ring_nf
          rfl
      _ ≤ ENNReal.ofReal (Real.exp (msExp a + maskExcess)) *
            ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msPartial messages ω n : ℝ))) ∂Measure.infinitePi μ :=
          hblock
      _ ≤ ENNReal.ofReal (Real.exp (msExp a + maskExcess)) *
            ENNReal.ofReal (Real.exp (n * (msExp a + maskExcess))) := by
          gcongr
          exact ih (by omega)
      _ = ENNReal.ofReal (Real.exp (↑(n + 1) * (msExp a + maskExcess))) := by
          rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
          push_cast
          ring_nf

theorem msErr_eq_partial (messages : Fin 2 → Int) (ω : Key → ℤ) :
    msErr messages ω = msPartial messages ω 720 := by
  rw [msErr_eq, msPartial]
  exact Finset.sum_congr rfl fun j _ ↦ by rw [if_pos j.isLt]

/-! ## The input noise tail -/

/-- The tape coordinate of key-switching error `i`. -/
def kkey (i : Fin 8192) : Key := (5 + i.val / 1024, i.val % 1024)

theorem kkey_injective : Injective kkey := by
  intro i i' h
  simp only [kkey, Prod.mk.injEq] at h
  exact Fin.ext (by omega)

theorem lawOf_kkey (i : Fin 8192) : lawOf (kkey i).1 = lweLaw := by
  simp only [kkey]
  have : i.val / 1024 < 8 := by omega
  interval_cases h : i.val / 1024 <;> rfl

theorem kskErr_eq (ω : Key → ℤ) (i : Fin 8192) : kskErr ω i = lweLaw.clamp (ω (kkey i)) := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  rw [kskErr, kskErrs, coeffVals, smp_entry, reducePoly_coeff (by decide) (by decide), ip_coeff]
  have hb := lwe_clamp (ω (kkey i))
  rw [abs_le] at hb
  simp only [kkey] at hb ⊢
  rw [ZMod.val_intCast]
  unfold centerQ
  have hQ : ((Q : Nat) : Int) = 5234636801 := rfl
  rw [hQ]
  split_ifs <;> omega

theorem kkey_not_mem (i : Fin 8192) : kkey i ∉ region inSites ∪ colsBefore 720 := by
  rw [Finset.mem_union, not_or]
  constructor
  · intro h
    have := (mem_region.mp h).1
    simp only [kkey] at this
    have : i.val / 1024 < 8 := by omega
    simp at *
    omega
  · intro h
    obtain ⟨j, -, hj⟩ := Finset.mem_biUnion.mp h
    obtain ⟨i', hi'⟩ := mem_colBlock.mp hj
    simp only [maskKey, kkey, Prod.mk.injEq] at hi'
    omega

/-- The input noise with the key-switching errors read raw from the tape. -/
noncomputable def inNoiseRaw (messages : Fin 2 → Int) (ω : Key → ℤ) : ℝ :=
  (msErr messages ω : ℝ) - ∑ i, (dsum messages ω i : ℝ) * (ω (kkey i) : ℝ)

theorem inNoise_eq_raw (messages : Fin 2 → Int) (ω : Key → ℤ)
    (hω : ∀ k, (lawOf k.1).clamp (ω k) = ω k) : inNoise messages ω = inNoiseRaw messages ω := by
  unfold inNoise inNoiseRaw
  congr 1
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [kskErr_eq, ← lawOf_kkey i, hω]
  rfl

/-- The variance proxy of the input noise. -/
noncomputable abbrev inProxy : ℝ := 720 * 1048578 ^ 2 + 8192 * 36 * ((32768 / 1 : ℚ) : ℝ) ^ 2

theorem inNoiseRaw_mgf (messages : Fin 2 → Int) (a : ℝ) :
    ∫⁻ ω, ENNReal.ofReal (Real.exp (a * inNoiseRaw messages ω)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (720 * maskExcess + 1440 * |a| + inProxy * a ^ 2 / 2)) := by
  set S := region inSites ∪ colsBefore 720
  have hF : DependsOn (fun ω ↦ ENNReal.ofReal (Real.exp (a * (msErr messages ω : ℝ)))) S := by
    intro ω ω' h
    simp only [msErr_eq_partial, msPartial_congr messages 720 h]
  have hc : ∀ i ∈ (Finset.univ : Finset (Fin 8192)),
      DependsOn (fun ω ↦ -a * (dsum messages ω i : ℝ)) S := by
    intro i _ ω ω' h
    have hr : ∀ k ∈ (region inSites : Set Key), ω k = ω' k :=
      fun k hk ↦ h k (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inl hk)))
    simp only [dsum_congr messages hr]
  have hlaw : ∀ i ∈ (Finset.univ : Finset (Fin 8192)), ∀ b : ℝ,
      ∫⁻ y, ENNReal.ofReal (Real.exp (b * y)) ∂μ (kkey i) ≤
        ENNReal.ofReal (Real.exp (((32768 / 1 : ℚ) : ℝ) ^ 2 * b ^ 2 / 2)) := by
    intro i _ b
    show ∫⁻ y, ENNReal.ofReal (Real.exp (b * y)) ∂(lawOf (kkey i).1).measure ≤ _
    rw [lawOf_kkey]
    exact SamplerLaw.lintegral_exp_gaussian_le _ _ b
  have hstep := lintegral_mul_exp_sum_key_le_fun' μ kkey S Finset.univ kkey_injective.injOn
    (fun i _ ↦ kkey_not_mem i) hF hc hlaw
  have hsq : ∀ ω, ∑ i ∈ (Finset.univ : Finset (Fin 8192)),
      ((32768 / 1 : ℚ) : ℝ) ^ 2 * (-a * (dsum messages ω i : ℝ)) ^ 2 / 2 ≤
      8192 * 36 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2 := by
    intro ω
    have hterm : ∀ i : Fin 8192, ((32768 / 1 : ℚ) : ℝ) ^ 2 * (-a * (dsum messages ω i : ℝ)) ^ 2 / 2 ≤
        36 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2 := by
      intro i
      have hr := dsum_range messages ω i
      have hd : (dsum messages ω i : ℝ) ^ 2 ≤ 36 := by
        have h0 : (0 : ℝ) ≤ dsum messages ω i := by exact_mod_cast hr.1
        have h6 : (dsum messages ω i : ℝ) ≤ 6 := by exact_mod_cast hr.2
        nlinarith
      have : (-a * (dsum messages ω i : ℝ)) ^ 2 = a ^ 2 * (dsum messages ω i : ℝ) ^ 2 := by ring
      rw [this]
      have ha := sq_nonneg a
      have hs := sq_nonneg ((32768 / 1 : ℚ) : ℝ)
      nlinarith
    calc _ ≤ ∑ _i : Fin 8192, 36 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2 :=
          Finset.sum_le_sum fun i _ ↦ hterm i
      _ = _ := by simp; ring
  calc ∫⁻ ω, ENNReal.ofReal (Real.exp (a * inNoiseRaw messages ω)) ∂Measure.infinitePi μ
      = ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msErr messages ω : ℝ))) *
          ENNReal.ofReal (Real.exp (∑ i ∈ Finset.univ, -a * (dsum messages ω i : ℝ) *
            (ω (kkey i) : ℝ))) ∂Measure.infinitePi μ := by
        refine lintegral_congr fun ω ↦ ?_
        rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 2
        unfold inNoiseRaw
        rw [mul_sub, Finset.mul_sum, sub_eq_add_neg, ← Finset.sum_neg_distrib]
        congr 1
        exact Finset.sum_congr rfl fun i _ ↦ by ring
    _ ≤ ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msErr messages ω : ℝ))) *
          ENNReal.ofReal (Real.exp (∑ i ∈ Finset.univ,
            ((32768 / 1 : ℚ) : ℝ) ^ 2 * (-a * (dsum messages ω i : ℝ)) ^ 2 / 2))
          ∂Measure.infinitePi μ := hstep
    _ ≤ ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msErr messages ω : ℝ))) *
          ENNReal.ofReal (Real.exp (8192 * 36 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2))
          ∂Measure.infinitePi μ := by
        refine lintegral_mono fun ω ↦ ?_
        gcongr
        exact hsq ω
    _ = ENNReal.ofReal (Real.exp (8192 * 36 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2)) *
          ∫⁻ ω, ENNReal.ofReal (Real.exp (a * (msPartial messages ω 720 : ℝ)))
            ∂Measure.infinitePi μ := by
        simp_rw [msErr_eq_partial]
        rw [lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, mul_comm]
    _ ≤ ENNReal.ofReal (Real.exp (8192 * 36 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2)) *
          ENNReal.ofReal (Real.exp ((720 : Nat) * (msExp a + maskExcess))) := by
        gcongr
        exact msPartial_mgf messages a 720 le_rfl
    _ = ENNReal.ofReal (Real.exp (720 * maskExcess + 1440 * |a| + inProxy * a ^ 2 / 2)) := by
        rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 2
        simp only [msExp, inProxy]
        push_cast
        ring

theorem ae_supp : ∀ᵐ ω ∂Measure.infinitePi μ, ∀ k : Key, (lawOf k.1).clamp (ω k) = ω k :=
  ae_clamp_eq (fun k : Key ↦ lawOf k.1) fun _ ↦ rfl

theorem inNoiseRaw_measurable (messages : Fin 2 → Int) : Measurable (inNoiseRaw messages) := by
  refine measurable_of_dependsOn (region inSites ∪ colsBefore 720 ∪ Finset.univ.image kkey)
    fun ω ω' h ↦ ?_
  have h1 : ∀ k ∈ ((region inSites ∪ colsBefore 720 : Finset Key) : Set Key), ω k = ω' k :=
    fun k hk ↦ h k (by simp only [Finset.coe_union, Set.mem_union] at hk ⊢; exact Or.inl hk)
  have hr : ∀ k ∈ (region inSites : Set Key), ω k = ω' k :=
    fun k hk ↦ h1 k (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inl hk)))
  have hk : ∀ i, ω (kkey i) = ω' (kkey i) :=
    fun i ↦ h _ (by simp only [Finset.coe_union, Set.mem_union, Finset.coe_image,
      Finset.coe_univ, Set.image_univ, Set.mem_range]; exact Or.inr ⟨i, rfl⟩)
  unfold inNoiseRaw
  rw [msErr_eq_partial, msErr_eq_partial, msPartial_congr messages 720 h1, dsum_congr messages hr]
  simp only [hk]

theorem inNoise_tail (messages : Fin 2 → Int) :
    Measure.infinitePi μ {ω | inThreshold < |inNoise messages ω|} ≤
      ENNReal.ofReal (2 * Real.exp (720 * maskExcess - (inThreshold - 1440) ^ 2 / (2 * inProxy))) := by
  have hae : {ω | inThreshold < |inNoise messages ω|} =ᵐ[Measure.infinitePi μ]
      {ω | inThreshold < |inNoiseRaw messages ω|} := by
    filter_upwards [ae_supp] with ω hω
    simp only [inNoise_eq_raw messages ω hω]
  rw [measure_congr hae]
  refine (measure_mono (t := {ω | inThreshold ≤ |inNoiseRaw messages ω|})
    fun ω (hω : inThreshold < |inNoiseRaw messages ω|) ↦ le_of_lt hω).trans ?_
  have h := measure_le_abs_le_lin (Measure.infinitePi μ) (inNoiseRaw_measurable messages)
    (V := inProxy) (b := 1440) (c := 720 * maskExcess) (by unfold inProxy; positivity)
    (fun a ↦ inNoiseRaw_mgf messages a) (t := inThreshold) (by norm_num [inThreshold])
  exact h

end MxxFheTfhe
