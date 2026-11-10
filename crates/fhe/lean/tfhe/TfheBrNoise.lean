import TfheInput

/-!
# The blind-rotation error as a linear form

The constant coefficient of `errSeq … 720` is `Σ_{i,c,k} gammaCoef i c k · Eint_{12 i + c, k}`: the
bootstrapping-key error of step `i` weighted by the step's digits, rotated by the later steps.
Every coefficient is at most `32` in absolute value.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity

section Expansion

variable (A E : ExactMatrix Q N 1 8640) (Eint : ErrorMatrix N 1 8640) (zM : ExactMatrix Q N 1 1)
  (s : Fin 720 → Int) (rot : Fin 720 → Int) (init : ExactMatrix Q N 2 1)

def shiftAt (j : Nat) : Nat := if h : j < 720 then stepShift s rot ⟨j, h⟩ else 0

noncomputable def noiseAt (j : Nat) : ErrorPoly N :=
  if h : j < 720 then stepNoise A E Eint zM s rot init ⟨j, h⟩ else 0

/-- The rotation applied to the error of step `i` by the steps after it, up to step `n`. -/
def suffixShift (i n : Nat) : Nat := ∑ j ∈ Finset.Ico (i + 1) n, shiftAt s rot j

theorem errSeq_expand (n : Nat) :
    errSeq A E Eint zM s rot init n = ∑ i ∈ Finset.range n,
      rootZ ^ suffixShift s rot i n * noiseAt A E Eint zM s rot init i := by
  induction n with
  | zero => simp [errSeq]
  | succ n ih =>
    have hsuffix : ∀ i ∈ Finset.range n,
        suffixShift s rot i (n + 1) = suffixShift s rot i n + shiftAt s rot n := fun i hi ↦ by
      simp only [Finset.mem_range] at hi
      simp only [suffixShift]
      rw [Finset.sum_Ico_succ_top (by omega)]
    have hlast : suffixShift s rot n (n + 1) = 0 := by simp [suffixShift]
    rw [Finset.sum_range_succ, hlast, pow_zero, one_mul, Finset.sum_congr rfl fun i hi ↦ by
      rw [hsuffix i hi, pow_add]]
    by_cases hn : n < 720
    · have herr : errSeq A E Eint zM s rot init (n + 1) =
          rootZ ^ stepShift s rot ⟨n, hn⟩ * errSeq A E Eint zM s rot init n +
            stepNoise A E Eint zM s rot init ⟨n, hn⟩ := by simp only [errSeq, dif_pos hn]
      rw [herr, ih, Finset.mul_sum]
      simp only [shiftAt, noiseAt, dif_pos hn]
      congr 1
      exact Finset.sum_congr rfl fun i _ ↦ by ring
    · have herr : errSeq A E Eint zM s rot init (n + 1) = errSeq A E Eint zM s rot init n := by
        simp only [errSeq, dif_neg hn]
      rw [herr, ih]
      simp only [shiftAt, noiseAt, dif_neg hn, pow_zero, mul_one, add_zero]

/-- The coefficient of bootstrapping-key error `(i, c, k)` in the constant coefficient. -/
noncomputable def gammaCoef (i : Fin 720) (c : Fin 12) (k : Fin N) : Int :=
  (if k.val = 0 then 1 else -1) *
    (rootZ ^ suffixShift s rot i 720 * stepDigits A E zM s rot init i c 0).coeff
      ⟨(N - k.val) % N, Nat.mod_lt _ (by decide)⟩

theorem errSeq_coeff_zero :
    (errSeq A E Eint zM s rot init 720).coeff 0 =
      ∑ i : Fin 720, ∑ c : Fin 12, ∑ k : Fin N, gammaCoef A E zM s rot init i c k *
        (Eint 0 ⟨12 * i.val + c.val, by omega⟩).coeff k := by
  rw [errSeq_expand, Finset.sum_range (fun i ↦ rootZ ^ suffixShift s rot i 720 *
    noiseAt A E Eint zM s rot init i), Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  simp only [noiseAt, dif_pos i.isLt, Fin.eta, stepNoise, Finset.mul_sum, Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun c _ ↦ ?_
  rw [show rootZ ^ suffixShift s rot i 720 * (Eint 0 ⟨12 * i.val + c.val, by omega⟩ *
      stepDigits A E zM s rot init i c 0) = (rootZ ^ suffixShift s rot i 720 *
      stepDigits A E zM s rot init i c 0) * Eint 0 ⟨12 * i.val + c.val, by omega⟩ by ring,
    show (0 : Fin N) = ⟨0, by decide⟩ from rfl, coeff_zero_mul (by decide)]
  rfl

theorem gammaCoef_bound (i : Fin 720) (c : Fin 12) (k : Fin N) :
    |gammaCoef A E zM s rot init i c k| ≤ 32 := by
  have hD : polyNorm (stepDigits A E zM s rot init i c 0) ≤ 32 :=
    polyNorm_le_of_coeff fun x ↦ by
      have := digitInts_bound (diffOf rot i (accOf A E zM s rot init i)) c 0 x
      rw [Int.abs_eq_natAbs] at this
      exact_mod_cast this
  have hrot := (polyNorm_root_pow_mul_le (by decide) (suffixShift s rot i 720)
    (stepDigits A E zM s rot init i c 0)).trans hD
  have hcoeff := coeff_natAbs_le_polyNorm (rootZ ^ suffixShift s rot i 720 *
    stepDigits A E zM s rot init i c 0) ⟨(N - k.val) % N, Nat.mod_lt _ (by decide)⟩
  have hsign : |(if k.val = 0 then (1 : Int) else -1)| = 1 := by split_ifs <;> simp
  rw [gammaCoef, abs_mul, hsign, one_mul, Int.abs_eq_natAbs]
  exact_mod_cast hcoeff.trans hrot

end Expansion

/-! ## Which errors each step reads -/

theorem accOf_congr_E (A E E' : ExactMatrix Q N 1 8640) (zM : ExactMatrix Q N 1 1)
    (s rot : Fin 720 → Int) (init : ExactMatrix Q N 2 1) (n : Nat)
    (hE : ∀ j < n, ∀ c : Fin 12, ∀ hj : j < 720,
      E 0 ⟨12 * j + c.val, by omega⟩ = E' 0 ⟨12 * j + c.val, by omega⟩) :
    ∀ i ≤ n, accOf A E zM s rot init i = accOf A E' zM s rot init i := by
  intro i hi
  induction i with
  | zero => rfl
  | succ i ih =>
    have hacc := ih (by omega)
    by_cases h : i < 720
    · have hgsw : gswOf A E zM (s ⟨i, h⟩) ⟨i, h⟩ = gswOf A E' zM (s ⟨i, h⟩) ⟨i, h⟩ := by
        have hcol : (fun (r : Fin 1) (c : Fin 12) ↦ E r ⟨12 * i + c.val, by omega⟩) =
            fun r c ↦ E' r ⟨12 * i + c.val, by omega⟩ := by
          funext r c
          have hr : r = 0 := Subsingleton.elim _ _
          subst hr
          exact hE i (by omega) c h
        simp only [gswOf, hcol]
      simp only [accOf, brSeq, dif_pos h] at hacc ⊢
      rw [hacc]
      simp only [brStep]
      rw [show (gswOf A E zM (s ⟨i, h⟩) ⟨i, h⟩).1 = (gswOf A E' zM (s ⟨i, h⟩) ⟨i, h⟩).1 by rw [hgsw],
        show (gswOf A E zM (s ⟨i, h⟩) ⟨i, h⟩).2.1 = (gswOf A E' zM (s ⟨i, h⟩) ⟨i, h⟩).2.1 by rw [hgsw]]
    · simp only [accOf, brSeq, dif_neg h] at hacc ⊢
      exact hacc

theorem gammaCoef_congr_E {A E E' : ExactMatrix Q N 1 8640} {zM : ExactMatrix Q N 1 1}
    {s rot : Fin 720 → Int} {init : ExactMatrix Q N 2 1} {i : Fin 720}
    (hacc : accOf A E zM s rot init i = accOf A E' zM s rot init i) :
    gammaCoef A E zM s rot init i = gammaCoef A E' zM s rot init i := by
  funext c k
  unfold gammaCoef stepDigits
  rw [hacc]

/-! ## The tape coordinates of the bootstrapping-key errors -/

/-- The sites the blind rotation reads, apart from the bootstrapping-key errors. -/
abbrev outSites : List Nat := [0, 1, 2, 4, 13, 15, 16, 18, 19]

/-- The tape coordinate of bootstrapping-key error `(i, c, k)`. -/
def bkey (i : Fin 720) (c : Fin 12) (k : Fin N) : Key := (3, (12 * i.val + c.val) * 1024 + k.val)

/-- The bootstrapping-key errors of the steps before `n`. -/
def bskBefore (n : Nat) : Finset Key :=
  (Finset.univ.filter fun i : Fin 720 ↦ i.val < n).biUnion fun i ↦
    Finset.univ.image fun p : Fin 12 × Fin N ↦ bkey i p.1 p.2

/-- What step `n` may read. -/
def readKeys (n : Nat) : Finset Key := region outSites ∪ bskBefore n

theorem mem_bskBefore {n : Nat} {i : Fin 720} (hi : i.val < n) (c : Fin 12) (k : Fin N) :
    bkey i c k ∈ bskBefore n :=
  Finset.mem_biUnion.mpr ⟨i, by simp [hi], Finset.mem_image.mpr ⟨(c, k), Finset.mem_univ _, rfl⟩⟩

theorem readKeys_region {n : Nat} {ω ω' : Key → ℤ} (h : ∀ k ∈ (readKeys n : Set Key), ω k = ω' k) :
    ∀ k ∈ (region outSites : Set Key), ω k = ω' k :=
  fun k hk ↦ h k (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inl hk)))

theorem region_inSites_sub {ω ω' : Key → ℤ} (h : ∀ k ∈ (region outSites : Set Key), ω k = ω' k) :
    ∀ k ∈ (region inSites : Set Key), ω k = ω' k := by
  intro k hk
  apply h k
  obtain ⟨h1, h2⟩ := mem_region.mp (Finset.mem_coe.mp hk)
  refine Finset.mem_coe.mpr (mem_region.mpr ⟨?_, h2⟩)
  simp only [inSites, outSites, List.mem_cons, List.not_mem_nil, or_false] at h1 ⊢
  omega

theorem nandA_congr (messages : Fin 2 → Int) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ (region outSites : Set Key), ω k = ω' k) : nandA messages ω = nandA messages ω' := by
  have hr := region_inSites_sub h
  have hd := dsum_congr messages hr
  have hA : kskAOf ω = kskAOf ω' := by
    unfold kskAOf
    rw [smp_congr (q := Qmask) (n := N) (rows := 1) (columns := 5760) (.residue Qmask)
      (s := 4) (by decide) (by decide) h]
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
  show nandMask (ksMask (kskAOf ω) (digitAt (extractAt (coeffVals (encC0 (smp (.residue Q) 13 ω)
      (smp bitLaw 15 ω) (smp ringLaw (15 + 1) ω)))))) (ksMask (kskAOf ω) (digitAt (extractAt
      (coeffVals (encC0 (smp (.residue Q) 13 ω) (smp bitLaw 18 ω) (smp ringLaw (18 + 1) ω)))))) =
    nandMask (ksMask (kskAOf ω') (digitAt (extractAt (coeffVals (encC0 (smp (.residue Q) 13 ω')
      (smp bitLaw 15 ω') (smp ringLaw (15 + 1) ω')))))) (ksMask (kskAOf ω') (digitAt (extractAt
      (coeffVals (encC0 (smp (.residue Q) 13 ω') (smp bitLaw 18 ω') (smp ringLaw (18 + 1) ω'))))))
  rw [hA, h13, h15, h18, show 15 + 1 = 16 from rfl, show 18 + 1 = 19 from rfl, h16, h19]

section Tape

variable (messages : Fin 2 → Int) (v : Int)

/-- The coefficient of bootstrapping-key error `(i, c, k)` on the tape `ω`. -/
noncomputable def gammaT (ω : Key → ℤ) (i : Fin 720) (c : Fin 12) (k : Fin N) : Int :=
  gammaCoef (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smp bitLaw 2 ω) (secretOf ω)
    (rotOf (nandA messages ω)) (brInit v) i c k

theorem smpE_col (ω : Key → ℤ) (j : Fin 720) (c : Fin 12) :
    (smp ringLaw 3 ω : ExactMatrix Q N 1 8640) 0 ⟨12 * j.val + c.val, by omega⟩ =
      polynomialOfCoefficients fun k ↦ ringLaw.clamp (ω (bkey j c k)) := by
  simp only [smp]
  congr 1
  funext k
  congr 2
  simp only [bkey, finProdFinEquiv_apply_val, Fin.val_zero]
  ring_nf

theorem smpInt3_coeff (ω : Key → ℤ) (j : Fin 720) (c : Fin 12) (k : Fin N) :
    ((smpInt ringLaw 3 ω : ErrorMatrix N 1 8640) 0 ⟨12 * j.val + c.val, by omega⟩).coeff k =
      ringLaw.clamp (ω (bkey j c k)) := by
  rw [smpInt_coeff (by decide)]
  congr 2
  simp only [bkey, finProdFinEquiv_apply_val, Fin.val_zero]
  ring_nf

theorem gammaT_congr (i : Fin 720) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ (readKeys i.val : Set Key), ω k = ω' k) : gammaT messages v ω i = gammaT messages v ω' i := by
  have hreg := readKeys_region h
  have hA := smp_congr (q := Q) (n := N) (rows := 1) (columns := 8640) (.residue Q)
    (s := 0) (by decide) (by decide) hreg
  have hz := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) bitLaw
    (s := 2) (by decide) (by decide) hreg
  have hs := secretOf_congr (region_inSites_sub hreg)
  have ha := nandA_congr messages hreg
  unfold gammaT
  rw [hA, hz, hs, ha]
  apply gammaCoef_congr_E
  refine accOf_congr_E _ _ _ _ _ _ _ i.val (fun j hj c hj720 ↦ ?_) i.val le_rfl
  have := smpE_col ω ⟨j, hj720⟩ c
  have := smpE_col ω' ⟨j, hj720⟩ c
  simp only at *
  rw [smpE_col ω ⟨j, hj720⟩ c, smpE_col ω' ⟨j, hj720⟩ c]
  congr 1
  funext k
  rw [h (bkey ⟨j, hj720⟩ c k) (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inr
    (mem_bskBefore hj c k))))]

theorem brErr_eq (ω : Key → ℤ) :
    brErr ω (nandA messages ω) v = ∑ i : Fin 720, ∑ c : Fin 12, ∑ k : Fin N,
      gammaT messages v ω i c k * ringLaw.clamp (ω (bkey i c k)) := by
  rw [brErr, errSeq_coeff_zero]
  simp only [smpInt3_coeff]
  rfl

/-- The bootstrapping-key part of the output noise for the steps before `n`, with the errors read
raw from the tape. -/
noncomputable def brSumRaw (ω : Key → ℤ) (n : Nat) : ℝ :=
  ∑ i : Fin 720, if i.val < n then
    ∑ c : Fin 12, ∑ k : Fin N, (gammaT messages v ω i c k : ℝ) * (ω (bkey i c k) : ℝ) else 0

theorem brSumRaw_congr (n : Nat) {ω ω' : Key → ℤ} (h : ∀ k ∈ (readKeys n : Set Key), ω k = ω' k) :
    brSumRaw messages v ω n = brSumRaw messages v ω' n := by
  unfold brSumRaw
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  split_ifs with hi
  · have hsub : ∀ k ∈ (readKeys i.val : Set Key), ω k = ω' k := by
      intro k hk
      apply h k
      rcases Finset.mem_union.mp (Finset.mem_coe.mp hk) with hk | hk
      · exact Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inl hk))
      · obtain ⟨j, hj, hjk⟩ := Finset.mem_biUnion.mp hk
        refine Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inr (Finset.mem_biUnion.mpr
          ⟨j, ?_, hjk⟩)))
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
        omega
    rw [gammaT_congr messages v i hsub]
    refine Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
    rw [h (bkey i c k) (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inr (mem_bskBefore hi c k))))]
  · rfl

theorem brSumRaw_succ (ω : Key → ℤ) (n : Nat) (hn : n < 720) :
    brSumRaw messages v ω (n + 1) = brSumRaw messages v ω n +
      ∑ p : Fin 12 × Fin N, (gammaT messages v ω ⟨n, hn⟩ p.1 p.2 : ℝ) * (ω (bkey ⟨n, hn⟩ p.1 p.2) : ℝ) := by
  unfold brSumRaw
  rw [← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨n, hn⟩ : Fin 720)),
    ← Finset.add_sum_erase _ _ (Finset.mem_univ (⟨n, hn⟩ : Fin 720))]
  simp only [lt_add_one, if_true, lt_irrefl, if_false, zero_add, Fintype.sum_prod_type]
  rw [add_comm]
  congr 1
  refine Finset.sum_congr rfl fun j hj ↦ ?_
  have hne : j.val ≠ n := fun h ↦ Finset.ne_of_mem_erase hj (Fin.ext h)
  by_cases hjn : j.val < n
  · rw [if_pos hjn, if_pos (by omega)]
  · rw [if_neg hjn, if_neg (by omega)]

end Tape

open MeasureTheory in
theorem gaussian_mgf_le {sigma : ℚ} {cutoff : ℤ} {c C : ℝ} (hc : |c| ≤ C) :
    ∫⁻ x, ENNReal.ofReal (Real.exp (c * x)) ∂(SamplerLaw.gaussian sigma cutoff).measure ≤
      ENNReal.ofReal (Real.exp ((sigma : ℝ) ^ 2 * C ^ 2 / 2)) := by
  refine (SamplerLaw.lintegral_exp_gaussian_le sigma cutoff c).trans
    (ENNReal.ofReal_le_ofReal (Real.exp_le_exp.mpr ?_))
  have hc2 : c ^ 2 ≤ C ^ 2 := by
    rw [← sq_abs c]
    exact pow_le_pow_left₀ (abs_nonneg c) hc 2
  have : 0 ≤ (sigma : ℝ) ^ 2 := sq_nonneg _
  nlinarith

/-- The per-coordinate exponent of the bootstrapping-key errors. -/
noncomputable def brExp (a : ℝ) : ℝ := ((156 / 1 : ℚ) : ℝ) ^ 2 * (32 * |a|) ^ 2 / 2

theorem bkey_not_mem (n : Fin 720) (c : Fin 12) (k : Fin N) : bkey n c k ∉ readKeys n.val := by
  rw [readKeys, Finset.mem_union, not_or]
  constructor
  · intro h
    have := (mem_region.mp h).1
    simp [bkey] at this
  · intro h
    obtain ⟨j, hj, hjk⟩ := Finset.mem_biUnion.mp h
    obtain ⟨⟨c', k'⟩, -, he⟩ := Finset.mem_image.mp hjk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj
    simp only [bkey, Prod.mk.injEq] at he
    have := c.isLt; have := c'.isLt; have := k.isLt; have := k'.isLt
    have hN : (N : Nat) = 1024 := rfl
    omega

theorem bkey_injective (n : Fin 720) :
    Set.InjOn (fun p : Fin 12 × Fin N ↦ bkey n p.1 p.2) (Finset.univ : Finset (Fin 12 × Fin N)) := by
  rintro ⟨c, k⟩ _ ⟨c', k'⟩ _ h
  simp only [bkey, Prod.mk.injEq] at h
  have h1 : c.val < 12 := c.isLt
  have h2 : c'.val < 12 := c'.isLt
  have h3 : k.val < 1024 := k.isLt
  have h4 : k'.val < 1024 := k'.isLt
  have hN : (N : Nat) = 1024 := rfl
  refine Prod.ext (Fin.ext ?_) (Fin.ext ?_) <;> simp only <;> omega

open MeasureTheory in
theorem brSumRaw_mgf (messages : Fin 2 → Int) (v : Int) (a : ℝ) :
    ∀ n ≤ 720, ∫⁻ ω, ENNReal.ofReal (Real.exp (a * brSumRaw messages v ω n)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (n * (12 * 1024 * brExp a))) := by
  intro n
  induction n with
  | zero =>
    intro _
    simp [brSumRaw]
  | succ n ih =>
    intro hn
    have hlt : n < 720 := by omega
    set i : Fin 720 := ⟨n, hlt⟩
    have hblock := MxxRuntime.lintegral_mul_exp_sum_key_le μ
      (fun p : Fin 12 × Fin N ↦ bkey i p.1 p.2) (readKeys n) Finset.univ (bkey_injective i)
      (fun p _ ↦ bkey_not_mem i p.1 p.2)
      (F := fun ω ↦ ENNReal.ofReal (Real.exp (a * brSumRaw messages v ω n)))
      (fun ω ω' h ↦ by simp only [brSumRaw_congr messages v n h])
      (c := fun p ω ↦ a * gammaT messages v ω i p.1 p.2)
      (fun p _ ω ω' h ↦ by simp only [gammaT_congr messages v i h])
      (K := fun _ ↦ brExp a)
      (fun p _ ω ↦ by
        show ∫⁻ y, _ ∂(lawOf 3).measure ≤ _
        refine gaussian_mgf_le (sigma := 156 / 1) (cutoff := 2496) ?_
        rw [abs_mul]
        have hg := gammaCoef_bound (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smp bitLaw 2 ω)
          (secretOf ω) (rotOf (nandA messages ω)) (brInit v) i p.1 p.2
        have h32 : |(gammaT messages v ω i p.1 p.2 : ℝ)| ≤ 32 := by exact_mod_cast hg
        nlinarith [abs_nonneg a])
    calc ∫⁻ ω, ENNReal.ofReal (Real.exp (a * brSumRaw messages v ω (n + 1))) ∂Measure.infinitePi μ
        = ∫⁻ ω, ENNReal.ofReal (Real.exp (a * brSumRaw messages v ω n)) *
            ENNReal.ofReal (Real.exp (∑ p ∈ (Finset.univ : Finset (Fin 12 × Fin N)),
              a * gammaT messages v ω i p.1 p.2 * (ω (bkey i p.1 p.2) : ℝ))) ∂Measure.infinitePi μ := by
          refine lintegral_congr fun ω ↦ ?_
          rw [brSumRaw_succ messages v ω n hlt, ← ENNReal.ofReal_mul (Real.exp_pos _).le,
            ← Real.exp_add, mul_add, Finset.mul_sum]
          congr 3
          exact Finset.sum_congr rfl fun p _ ↦ by ring
      _ ≤ ENNReal.ofReal (Real.exp (∑ _p ∈ (Finset.univ : Finset (Fin 12 × Fin N)), brExp a)) *
            ∫⁻ ω, ENNReal.ofReal (Real.exp (a * brSumRaw messages v ω n)) ∂Measure.infinitePi μ :=
          hblock
      _ ≤ ENNReal.ofReal (Real.exp (12 * 1024 * brExp a)) *
            ENNReal.ofReal (Real.exp (n * (12 * 1024 * brExp a))) := by
          gcongr
          · apply le_of_eq
            simp
          · exact ih (by omega)
      _ = ENNReal.ofReal (Real.exp (↑(n + 1) * (12 * 1024 * brExp a))) := by
          rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
          push_cast
          ring_nf

end MxxFheTfhe
