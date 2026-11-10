import TfheBrNoise

/-!
# The output noise tail

For every initial rotation `v`, the output noise `outNoise v` is, almost surely, the
bootstrapping-key part `brSumRaw` switched to `q` minus the key-switching errors weighted by the
output digits, which read no key-switching error. Its moment is sub-Gaussian with proxy
`outProxy`, so its tail is Gaussian.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open MeasureTheory Mxx.Primitives MxxRuntime MxxSecurity Function

theorem outDigits_congr (messages : Fin 2 → Int) (v : Int) {ω ω' : Key → ℤ}
    (h : ∀ k ∈ (readKeys 720 : Set Key), ω k = ω' k) :
    outDigits ω (nandA messages ω) v = outDigits ω' (nandA messages ω') v := by
  have hreg := readKeys_region h
  have hA := smp_congr (q := Q) (n := N) (rows := 1) (columns := 8640) (.residue Q)
    (s := 0) (by decide) (by decide) hreg
  have hz := smp_congr (q := Q) (n := N) (rows := 1) (columns := 1) bitLaw
    (s := 2) (by decide) (by decide) hreg
  have hs := secretOf_congr (region_inSites_sub hreg)
  have ha := nandA_congr messages hreg
  unfold outDigits brFinal
  rw [hA, hz, hs, ha]
  rw [accOf_congr_E _ _ _ _ _ _ _ 720 (fun j hj c hj720 ↦ ?_) 720 le_rfl]
  rw [smpE_col ω ⟨j, hj720⟩ c, smpE_col ω' ⟨j, hj720⟩ c]
  congr 1
  funext k
  rw [h (bkey ⟨j, hj720⟩ c k) (Finset.mem_coe.mpr (Finset.mem_union.mpr (Or.inr
    (mem_bskBefore hj c k))))]

/-- The output noise at the initial rotation `v`, with the errors read raw from the tape. -/
noncomputable def outNoiseRaw (messages : Fin 2 → Int) (v : Int) (ω : Key → ℤ) : ℝ :=
  (4294967296 / 5234636801 : ℝ) * brSumRaw messages v ω 720 -
    ∑ i, (outDigits ω (nandA messages ω) v i : ℝ) * (ω (kkey i) : ℝ)

theorem outNoise_eq_raw (messages : Fin 2 → Int) (v : Int) (ω : Key → ℤ)
    (hω : ∀ k, (lawOf k.1).clamp (ω k) = ω k) :
    outNoise messages ω v = outNoiseRaw messages v ω := by
  unfold outNoise outNoiseRaw
  have hbr : (brErr ω (nandA messages ω) v : ℝ) = brSumRaw messages v ω 720 := by
    rw [brErr_eq, brSumRaw]
    push_cast
    refine Finset.sum_congr rfl fun i _ ↦ ?_
    rw [if_pos i.isLt]
    refine Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
    rw [show ringLaw.clamp (ω (bkey i c k)) = (lawOf (bkey i c k).1).clamp (ω (bkey i c k)) from rfl,
      hω]
  rw [hbr]
  congr 1
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [kskErr_eq, ← lawOf_kkey i, hω]

theorem kkey_not_mem_read (i : Fin 8192) : kkey i ∉ readKeys 720 := by
  rw [readKeys, Finset.mem_union, not_or]
  constructor
  · intro h
    have := (mem_region.mp h).1
    simp only [kkey] at this
    have : i.val / 1024 < 8 := by omega
    simp at *
    omega
  · intro h
    obtain ⟨j, -, hjk⟩ := Finset.mem_biUnion.mp h
    obtain ⟨⟨c, k⟩, -, he⟩ := Finset.mem_image.mp hjk
    simp only [bkey, kkey, Prod.mk.injEq] at he
    omega

/-- The variance proxy of the output noise. -/
noncomputable abbrev outProxy : ℝ :=
  8192 * 9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 +
    720 * (12 * 1024) * (((156 / 1 : ℚ) : ℝ) ^ 2 * (32 * (4294967296 / 5234636801)) ^ 2)

theorem outDigits_bound (messages : Fin 2 → Int) (v : Int) (ω : Key → ℤ) (i : Fin 8192) :
    0 ≤ outDigits ω (nandA messages ω) v i ∧ outDigits ω (nandA messages ω) v i ≤ 3 :=
  digitAt_range _ i

theorem outNoiseRaw_mgf (messages : Fin 2 → Int) (v : Int) (a : ℝ) :
    ∫⁻ ω, ENNReal.ofReal (Real.exp (a * outNoiseRaw messages v ω)) ∂Measure.infinitePi μ ≤
      ENNReal.ofReal (Real.exp (outProxy * a ^ 2 / 2)) := by
  set a' := a * (4294967296 / 5234636801 : ℝ)
  have hF : DependsOn (fun ω ↦ ENNReal.ofReal (Real.exp (a' * brSumRaw messages v ω 720)))
      (readKeys 720) := by
    intro ω ω' h
    simp only [brSumRaw_congr messages v 720 h]
  have hc : ∀ i ∈ (Finset.univ : Finset (Fin 8192)),
      DependsOn (fun ω ↦ -a * (outDigits ω (nandA messages ω) v i : ℝ)) (readKeys 720) := by
    intro i _ ω ω' h
    simp only [outDigits_congr messages v h]
  have hlaw : ∀ i ∈ (Finset.univ : Finset (Fin 8192)), ∀ b : ℝ,
      ∫⁻ y, ENNReal.ofReal (Real.exp (b * y)) ∂μ (kkey i) ≤
        ENNReal.ofReal (Real.exp (((32768 / 1 : ℚ) : ℝ) ^ 2 * b ^ 2 / 2)) := by
    intro i _ b
    show ∫⁻ y, ENNReal.ofReal (Real.exp (b * y)) ∂(lawOf (kkey i).1).measure ≤ _
    rw [lawOf_kkey]
    exact SamplerLaw.lintegral_exp_gaussian_le _ _ b
  have hstep := lintegral_mul_exp_sum_key_le_fun' μ kkey (readKeys 720) Finset.univ
    kkey_injective.injOn (fun i _ ↦ kkey_not_mem_read i) hF hc hlaw
  have hsq : ∀ ω, ∑ i ∈ (Finset.univ : Finset (Fin 8192)),
      ((32768 / 1 : ℚ) : ℝ) ^ 2 * (-a * (outDigits ω (nandA messages ω) v i : ℝ)) ^ 2 / 2 ≤
      8192 * 9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2 := by
    intro ω
    have hterm : ∀ i : Fin 8192, ((32768 / 1 : ℚ) : ℝ) ^ 2 *
        (-a * (outDigits ω (nandA messages ω) v i : ℝ)) ^ 2 / 2 ≤
        9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2 := by
      intro i
      have hr := outDigits_bound messages v ω i
      have hd : (outDigits ω (nandA messages ω) v i : ℝ) ^ 2 ≤ 9 := by
        have h0 : (0 : ℝ) ≤ outDigits ω (nandA messages ω) v i := by exact_mod_cast hr.1
        have h3 : (outDigits ω (nandA messages ω) v i : ℝ) ≤ 3 := by exact_mod_cast hr.2
        nlinarith
      have : (-a * (outDigits ω (nandA messages ω) v i : ℝ)) ^ 2 =
          a ^ 2 * (outDigits ω (nandA messages ω) v i : ℝ) ^ 2 := by ring
      rw [this]
      have ha := sq_nonneg a
      have hs := sq_nonneg ((32768 / 1 : ℚ) : ℝ)
      nlinarith
    calc _ ≤ ∑ _i : Fin 8192, 9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2 :=
          Finset.sum_le_sum fun i _ ↦ hterm i
      _ = _ := by simp; ring
  calc ∫⁻ ω, ENNReal.ofReal (Real.exp (a * outNoiseRaw messages v ω)) ∂Measure.infinitePi μ
      = ∫⁻ ω, ENNReal.ofReal (Real.exp (a' * brSumRaw messages v ω 720)) *
          ENNReal.ofReal (Real.exp (∑ i ∈ Finset.univ, -a * (outDigits ω (nandA messages ω) v i : ℝ) *
            (ω (kkey i) : ℝ))) ∂Measure.infinitePi μ := by
        refine lintegral_congr fun ω ↦ ?_
        rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 2
        unfold outNoiseRaw
        rw [mul_sub, Finset.mul_sum, sub_eq_add_neg, ← Finset.sum_neg_distrib]
        simp only [a']
        congr 1
        · ring
        · exact Finset.sum_congr rfl fun i _ ↦ by ring
    _ ≤ ∫⁻ ω, ENNReal.ofReal (Real.exp (a' * brSumRaw messages v ω 720)) *
          ENNReal.ofReal (Real.exp (∑ i ∈ Finset.univ,
            ((32768 / 1 : ℚ) : ℝ) ^ 2 * (-a * (outDigits ω (nandA messages ω) v i : ℝ)) ^ 2 / 2))
          ∂Measure.infinitePi μ := hstep
    _ ≤ ∫⁻ ω, ENNReal.ofReal (Real.exp (a' * brSumRaw messages v ω 720)) *
          ENNReal.ofReal (Real.exp (8192 * 9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2))
          ∂Measure.infinitePi μ := by
        refine lintegral_mono fun ω ↦ ?_
        gcongr
        exact hsq ω
    _ = ENNReal.ofReal (Real.exp (8192 * 9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2)) *
          ∫⁻ ω, ENNReal.ofReal (Real.exp (a' * brSumRaw messages v ω 720)) ∂Measure.infinitePi μ := by
        rw [lintegral_mul_const' _ _ ENNReal.ofReal_ne_top, mul_comm]
    _ ≤ ENNReal.ofReal (Real.exp (8192 * 9 * ((32768 / 1 : ℚ) : ℝ) ^ 2 * a ^ 2 / 2)) *
          ENNReal.ofReal (Real.exp ((720 : Nat) * (12 * 1024 * brExp a'))) := by
        gcongr
        exact brSumRaw_mgf messages v a' 720 le_rfl
    _ = ENNReal.ofReal (Real.exp (outProxy * a ^ 2 / 2)) := by
        rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 2
        simp only [brExp, outProxy, a', mul_pow, sq_abs]
        push_cast
        ring

end MxxFheTfhe
