import TfheDependence
import RuntimeGaussian

/-!
The failure probability of the NAND gate. The blind-rotation error's constant coefficient is a
linear form in the bootstrapping key errors whose coefficients, the rotated digits, read only
earlier keys; `noiseY` adds the key-switching errors weighted by digits that read no
key-switching error. Peeling one block of fresh Gaussian errors at a time bounds the exponential
moment of `noiseY`, Chernoff's bound gives its tail, and Hoeffding's bound the tail of the LWE
secret's Hamming weight.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MeasureTheory

/-! ## The blind-rotation error as a linear form -/

def shiftAt (W : World) (t : SampleTape) (j : Nat) : Nat :=
  if h : j < lweN then stepShift W t ⟨j, h⟩ else 0

noncomputable def noiseAt (W : World) (t : SampleTape) (j : Nat) : ErrorPoly N :=
  if h : j < lweN then stepNoise W t ⟨j, h⟩ else 0

/-- The rotation applied to the error of step `i` by the steps after it. -/
def suffixShift (W : World) (t : SampleTape) (i n : Nat) : Nat :=
  ∑ j ∈ Finset.Ico (i + 1) n, shiftAt W t j

theorem errSeq_expand (W : World) (t : SampleTape) (n : Nat) :
    errSeq W t n = ∑ i ∈ Finset.range n, rootZ ^ suffixShift W t i n * noiseAt W t i := by
  induction n with
  | zero => simp [errSeq]
  | succ n ih =>
    have hsuffix : ∀ i ∈ Finset.range n,
        suffixShift W t i (n + 1) = suffixShift W t i n + shiftAt W t n := fun i hi ↦ by
      simp only [Finset.mem_range] at hi
      simp only [suffixShift]
      rw [Finset.sum_Ico_succ_top (by omega)]
    have hlast : suffixShift W t n (n + 1) = 0 := by simp [suffixShift]
    rw [Finset.sum_range_succ, hlast, pow_zero, one_mul, Finset.sum_congr rfl fun i hi ↦ by
      rw [hsuffix i hi, pow_add]]
    by_cases hn : n < lweN
    · have herr : errSeq W t (n + 1) = rootZ ^ stepShift W t ⟨n, hn⟩ * errSeq W t n +
          stepNoise W t ⟨n, hn⟩ := by simp only [errSeq, dif_pos hn]
      rw [herr, ih, Finset.mul_sum]
      simp only [shiftAt, noiseAt, dif_pos hn]
      congr 1
      exact Finset.sum_congr rfl fun i _ ↦ by ring
    · have herr : errSeq W t (n + 1) = errSeq W t n := by simp only [errSeq, dif_neg hn]
      rw [herr, ih]
      simp only [shiftAt, noiseAt, dif_neg hn, pow_zero, mul_one, add_zero]

/-- The coefficient of bootstrapping key error `(i, c, k)` in the extracted error. -/
noncomputable def gammaCoef (W : World) (t : SampleTape) (i : Fin lweN) (c : Fin 12) (k : Fin N) :
    Int :=
  (if k.val = 0 then 1 else -1) *
    (rootZ ^ suffixShift W t i lweN * stepDigits W t i c 0).coeff
      ⟨(N - k.val) % N, Nat.mod_lt _ (by decide)⟩

theorem errSeq_coeff_zero (W : World) (t : SampleTape) :
    (errSeq W t lweN).coeff ⟨0, by decide⟩ =
      ∑ i : Fin lweN, ∑ c : Fin 12, ∑ k : Fin N, gammaCoef W t i c k * t (bskErrorKey i c k) := by
  rw [errSeq_expand, Finset.sum_range (fun i ↦ rootZ ^ suffixShift W t i lweN * noiseAt W t i),
    Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  simp only [noiseAt, dif_pos i.isLt, Fin.eta, stepNoise, Finset.mul_sum, Negacyclic.coeff_sum]
  refine Finset.sum_congr rfl fun c _ ↦ ?_
  rw [show rootZ ^ suffixShift W t i lweN * (bskError t i 0 c * stepDigits W t i c 0) =
    (rootZ ^ suffixShift W t i lweN * stepDigits W t i c 0) * bskError t i 0 c by ring,
    coeff_zero_mul (by decide)]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [gammaCoef, bskError, intPoly_coeff (by decide)]

theorem gammaCoef_bound (W : World) (t : SampleTape) (i : Fin lweN) (c : Fin 12) (k : Fin N) :
    |gammaCoef W t i c k| ≤ 32 := by
  have hD : polyNorm (stepDigits W t i c 0) ≤ 32 :=
    polyNorm_le_of_coeff fun x ↦ by
      have := digitInts_bound (diffOf W i (accSeq W t i)) c 0 x
      rw [Int.abs_eq_natAbs] at this
      exact_mod_cast this
  have hrot := (polyNorm_root_pow_mul_le (by decide) (suffixShift W t i lweN)
    (stepDigits W t i c 0)).trans hD
  have hcoeff := coeff_natAbs_le_polyNorm (rootZ ^ suffixShift W t i lweN *
    stepDigits W t i c 0) ⟨(N - k.val) % N, Nat.mod_lt _ (by decide)⟩
  have hsign : |(if k.val = 0 then (1 : Int) else -1)| = 1 := by split_ifs <;> simp
  rw [gammaCoef, abs_mul, hsign, one_mul, Int.abs_eq_natAbs]
  exact_mod_cast hcoeff.trans hrot

/-! ## Fresh keys -/

theorem mem_lweKeys_iff {k : SampleKey} : k ∈ lweKeys ↔ ∃ j : Fin lweN, lweKey j = k := by
  rw [lweKeys_def]; simp
theorem mem_ringKeys_iff {k : SampleKey} : k ∈ ringKeys ↔ ∃ j : Fin N, ringKey j = k := by
  rw [ringKeys_def]; simp
theorem mem_maskKeys_iff {k : SampleKey} :
    k ∈ maskKeys ↔ ∃ (i : Fin lweN) (c : Fin 12) (x : Fin N), bskMaskKey i c x = k := by
  rw [maskKeys_def]; simp
theorem mem_encKeys_iff {k : SampleKey} : k ∈ encKeys ↔ k = encErrorKey 1 ∨ k = encErrorKey 2 := by
  rw [encKeys_def]; simp

/-- The errors of step `n` are not read before step `n`. -/
theorem bskErrorKey_not_mem_read (n : Nat) (c : Fin 12) (k : Fin N) :
    bskErrorKey n c k ∉ readKeys n := by
  rw [mem_readKeys, mem_lweKeys_iff, mem_ringKeys_iff, mem_maskKeys_iff, mem_encKeys_iff]
  simp only [not_or, not_exists]
  refine ⟨fun j h ↦ ?_, fun j h ↦ ?_, fun i c' x h ↦ ?_, ⟨fun h ↦ ?_, fun h ↦ ?_⟩,
    fun i ⟨hi, h⟩ ↦ ?_⟩
  · simp [lweKey, bskErrorKey] at h
  · simp [ringKey, bskErrorKey] at h
  · simp [bskMaskKey, bskErrorKey] at h
  · simp [encErrorKey, bskErrorKey] at h
  · simp [encErrorKey, bskErrorKey] at h
  · obtain ⟨c', x, hx⟩ := mem_stepKeys_iff.mp h
    simp only [bskErrorKey, SampleKey.mk.injEq, List.cons.injEq] at hx
    omega

/-- Key switching errors are read by nothing before key switching. -/
theorem kskErrorKey_not_mem_read (e : Nat) : kskErrorKey e ∉ readKeys lweN := by
  rw [mem_readKeys, mem_lweKeys_iff, mem_ringKeys_iff, mem_maskKeys_iff, mem_encKeys_iff]
  simp only [not_or, not_exists]
  refine ⟨fun j h ↦ ?_, fun j h ↦ ?_, fun i c' x h ↦ ?_, ⟨fun h ↦ ?_, fun h ↦ ?_⟩,
    fun i ⟨hi, h⟩ ↦ ?_⟩
  · simp only [lweKey, kskErrorKey, SampleKey.mk.injEq, List.cons.injEq] at h; omega
  · simp only [ringKey, kskErrorKey, SampleKey.mk.injEq, List.cons.injEq] at h; omega
  · simp [bskMaskKey, kskErrorKey] at h
  · simp [encErrorKey, kskErrorKey] at h
  · simp [encErrorKey, kskErrorKey] at h
  · obtain ⟨c', x, hx⟩ := mem_stepKeys_iff.mp h
    simp [bskErrorKey, kskErrorKey] at hx

theorem bskErrorKey_injective (n : Nat) :
    Set.InjOn (fun x : Fin 12 × Fin N ↦ bskErrorKey n x.1 x.2)
      ((Finset.univ : Finset (Fin 12 × Fin N)) : Set (Fin 12 × Fin N)) := by
  rintro ⟨c, k⟩ _ ⟨c', k'⟩ _ h
  simp only [bskErrorKey, SampleKey.mk.injEq] at h
  obtain ⟨_, _, hc, hk, _⟩ := h
  exact Prod.ext (Fin.ext hc) (Fin.ext hk)

theorem kskErrorKey_injective :
    Set.InjOn (fun e : Fin 8192 ↦ kskErrorKey e) ((Finset.univ : Finset (Fin 8192)) : Set (Fin 8192)) := by
  rintro e _ e' _ h
  simp only [kskErrorKey, SampleKey.mk.injEq, List.cons.injEq] at h
  obtain ⟨⟨_, hs, _⟩, _, _, hk, _⟩ := h
  exact Fin.ext (by omega)

/-! ## Dependence of the coefficients -/

theorem gammaCoef_congr (W : World) {t t' : SampleTape} (i : Fin lweN)
    (h : Agree (readKeys i) t t') (c : Fin 12) (k : Fin N) :
    gammaCoef W t i c k = gammaCoef W t' i c k := by
  have hshift : ∀ j, shiftAt W t j = shiftAt W t' j := fun j ↦ by
    unfold shiftAt
    split_ifs with hj
    · exact stepShift_congr W h _
    · rfl
  unfold gammaCoef suffixShift
  simp only [hshift, stepDigits_congr W i h]

/-- The blind-rotation part of the noise after `n` steps. -/
noncomputable def brSum (W : World) (t : SampleTape) (n : Nat) : ℝ :=
  ∑ i ∈ Finset.range n, if h : i < lweN then
    ∑ c : Fin 12, ∑ k : Fin N, (gammaCoef W t ⟨i, h⟩ c k : ℝ) * (t (bskErrorKey i c k) : ℝ)
  else 0

theorem brSum_congr (W : World) {n : Nat} {t t' : SampleTape}
    (h : Agree (readKeys n) t t') : brSum W t n = brSum W t' n := by
  unfold brSum
  refine Finset.sum_congr rfl fun i hi ↦ ?_
  simp only [Finset.mem_range] at hi
  split_ifs with hl
  · refine Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ ?_
    rw [gammaCoef_congr W ⟨i, hl⟩ (fun x hx ↦ h x (readKeys_mono (by simp; omega) hx)) c k,
      h _ (errorKey_mem_read hi c k)]
  · rfl

/-! ## Exponential moments -/

theorem tapeMeasure_eq : tapeMeasure = Measure.infinitePi fun key : SampleKey ↦ key.law.measure :=
  rfl

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

/-- The per-coordinate exponent of the blind-rotation errors. -/
noncomputable def brStep (a : ℝ) : ℝ := ((156 / 1 : ℚ) : ℝ) ^ 2 * (32 * |a|) ^ 2 / 2

theorem brSum_mgf (W : World) (a : ℝ) {n : Nat} (hn : n ≤ lweN) :
    ∫⁻ t, ENNReal.ofReal (Real.exp (a * brSum W t n)) ∂tapeMeasure ≤
      ENNReal.ofReal (Real.exp (n * (12 * 1024 * brStep a))) := by
  induction n with
  | zero => simp [brSum]
  | succ n ih =>
    have hlt : n < lweN := by omega
    have hsplit (t : SampleTape) : a * brSum W t (n + 1) = a * brSum W t n +
        ∑ x : Fin 12 × Fin N, (a * gammaCoef W t ⟨n, hlt⟩ x.1 x.2) *
          (t (bskErrorKey n x.1 x.2) : ℝ) := by
      rw [brSum, Finset.sum_range_succ, dif_pos hlt, ← brSum, mul_add, Fintype.sum_prod_type,
        Finset.mul_sum]
      refine congrArg₂ (· + ·) rfl (Finset.sum_congr rfl fun c _ ↦ ?_)
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun k _ ↦ by ring
    have hblock := lintegral_mul_exp_sum_key_le (fun key : SampleKey ↦ key.law.measure)
      (fun x : Fin 12 × Fin N ↦ bskErrorKey n x.1 x.2) (readKeys n) Finset.univ
      (bskErrorKey_injective n) (fun x _ ↦ bskErrorKey_not_mem_read n x.1 x.2)
      (F := fun t ↦ ENNReal.ofReal (Real.exp (a * brSum W t n)))
      (fun t t' h ↦ by
        simp only
        rw [brSum_congr W (fun k hk ↦ h k (Finset.mem_coe.mpr hk))])
      (c := fun x t ↦ a * gammaCoef W t ⟨n, hlt⟩ x.1 x.2)
      (fun x _ t t' h ↦ by
        simp only
        rw [gammaCoef_congr W ⟨n, hlt⟩ (fun k hk ↦ h k (Finset.mem_coe.mpr hk))])
      (K := fun _ ↦ brStep a)
      (fun x _ t ↦ gaussian_mgf_le (sigma := 156 / 1) (cutoff := 2496) (by
        rw [abs_mul]
        have := gammaCoef_bound W t ⟨n, hlt⟩ x.1 x.2
        have h32 : |(gammaCoef W t ⟨n, hlt⟩ x.1 x.2 : ℝ)| ≤ 32 := by exact_mod_cast this
        nlinarith [abs_nonneg a]))
    rw [← tapeMeasure_eq] at hblock
    calc ∫⁻ t, ENNReal.ofReal (Real.exp (a * brSum W t (n + 1))) ∂tapeMeasure
        = ∫⁻ t, ENNReal.ofReal (Real.exp (a * brSum W t n)) *
            ENNReal.ofReal (Real.exp (∑ x : Fin 12 × Fin N,
              (a * gammaCoef W t ⟨n, hlt⟩ x.1 x.2) * (t (bskErrorKey n x.1 x.2) : ℝ)))
            ∂tapeMeasure := by
          refine lintegral_congr fun t ↦ ?_
          rw [hsplit, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
      _ ≤ ENNReal.ofReal (Real.exp (∑ _x : Fin 12 × Fin N, brStep a)) *
            ∫⁻ t, ENNReal.ofReal (Real.exp (a * brSum W t n)) ∂tapeMeasure := hblock
      _ ≤ ENNReal.ofReal (Real.exp (12 * 1024 * brStep a)) *
            ENNReal.ofReal (Real.exp (n * (12 * 1024 * brStep a))) := by
          gcongr
          · apply le_of_eq
            simp
          · exact ih (by omega)
      _ = ENNReal.ofReal (Real.exp (↑(n + 1) * (12 * 1024 * brStep a))) := by
          rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
          push_cast
          ring_nf

/-- The per-coordinate exponent of the key-switching errors. -/
noncomputable def ksStep (a : ℝ) : ℝ := ((131072 / 1 : ℚ) : ℝ) ^ 2 * (3 * |a|) ^ 2 / 2

theorem ksDigit_bound (W : World) (t : SampleTape) (e : Fin 8192) : |ksDigit W t e| ≤ 3 := by
  unfold ksDigit
  rw [abs_le]
  omega

theorem noiseY_eq (W : World) (t : SampleTape) (a : ℝ) :
    a * noiseY W t = a * (4294967296 / 5234636801) * brSum W t lweN +
      ∑ e : Fin 8192, (-(a * ksDigit W t e)) * (t (kskErrorKey e) : ℝ) := by
  have hbr : ((errSeq W t lweN).coeff ⟨0, by decide⟩ : ℝ) = brSum W t lweN := by
    rw [errSeq_coeff_zero, brSum, Finset.sum_range (fun i ↦ if h : i < lweN then
      ∑ c : Fin 12, ∑ k : Fin N, (gammaCoef W t ⟨i, h⟩ c k : ℝ) * (t (bskErrorKey i c k) : ℝ)
      else 0)]
    push_cast
    exact Finset.sum_congr rfl fun i _ ↦ by rw [dif_pos i.isLt]
  rw [noiseY, hbr, mul_sub, Finset.mul_sum, sub_eq_add_neg, ← Finset.sum_neg_distrib]
  exact congrArg₂ (· + ·) (by ring) (Finset.sum_congr rfl fun e _ ↦ by
    simp only [kskError]
    ring)

/-- The exponential moment of the noise: sub-Gaussian with variance proxy `noiseProxy`. -/
noncomputable def noiseProxy : ℝ :=
  lweN * (12 * 1024 * (((156 / 1 : ℚ) : ℝ) ^ 2 * (32 * (4294967296 / 5234636801)) ^ 2)) +
    8192 * (((131072 / 1 : ℚ) : ℝ) ^ 2 * 3 ^ 2)

theorem noiseY_mgf (W : World) (a : ℝ) :
    ∫⁻ t, ENNReal.ofReal (Real.exp (a * noiseY W t)) ∂tapeMeasure ≤
      ENNReal.ofReal (Real.exp (noiseProxy * a ^ 2 / 2)) := by
  set a' := a * (4294967296 / 5234636801)
  have hblock := lintegral_mul_exp_sum_key_le (fun key : SampleKey ↦ key.law.measure)
    (fun e : Fin 8192 ↦ kskErrorKey e) (readKeys lweN) Finset.univ kskErrorKey_injective
    (fun e _ ↦ kskErrorKey_not_mem_read e)
    (F := fun t ↦ ENNReal.ofReal (Real.exp (a' * brSum W t lweN)))
    (fun t t' h ↦ by
      simp only
      rw [brSum_congr W (fun k hk ↦ h k (Finset.mem_coe.mpr hk))])
    (c := fun e t ↦ -(a * ksDigit W t e))
    (fun e _ t t' h ↦ by
      simp only
      rw [ksDigit_congr W (fun k hk ↦ h k (Finset.mem_coe.mpr hk))])
    (K := fun _ ↦ ksStep a)
    (fun e _ t ↦ gaussian_mgf_le (sigma := 131072 / 1) (cutoff := 2097152) (by
      rw [abs_neg, abs_mul]
      have := ksDigit_bound W t e
      have h3 : |(ksDigit W t e : ℝ)| ≤ 3 := by exact_mod_cast this
      nlinarith [abs_nonneg a]))
  rw [← tapeMeasure_eq] at hblock
  calc ∫⁻ t, ENNReal.ofReal (Real.exp (a * noiseY W t)) ∂tapeMeasure
      = ∫⁻ t, ENNReal.ofReal (Real.exp (a' * brSum W t lweN)) *
          ENNReal.ofReal (Real.exp (∑ e : Fin 8192,
            (-(a * ksDigit W t e)) * (t (kskErrorKey e) : ℝ))) ∂tapeMeasure := by
        refine lintegral_congr fun t ↦ ?_
        rw [noiseY_eq, Real.exp_add, ENNReal.ofReal_mul (Real.exp_pos _).le]
    _ ≤ ENNReal.ofReal (Real.exp (∑ _e : Fin 8192, ksStep a)) *
          ∫⁻ t, ENNReal.ofReal (Real.exp (a' * brSum W t lweN)) ∂tapeMeasure := hblock
    _ ≤ ENNReal.ofReal (Real.exp (8192 * ksStep a)) *
          ENNReal.ofReal (Real.exp (lweN * (12 * 1024 * brStep a'))) := by
        gcongr
        · apply le_of_eq
          simp
        · exact brSum_mgf W a' le_rfl
    _ = ENNReal.ofReal (Real.exp (noiseProxy * a ^ 2 / 2)) := by
        rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add]
        congr 2
        simp only [ksStep, brStep, noiseProxy, a', mul_pow, sq_abs]
        push_cast
        ring

theorem noiseY_measurable (W : World) : Measurable (noiseY W) := by
  refine measurable_of_dependsOn (readKeys lweN ∪ kskKeys) fun t t' h ↦ ?_
  have hread : Agree (readKeys lweN) t t' := fun k hk ↦ h k (by simp [hk])
  have hks (e : Fin 8192) : t (kskErrorKey e) = t' (kskErrorKey e) :=
    h _ (by
      simp only [Finset.coe_union, Set.mem_union, Finset.mem_coe]
      right
      rw [kskKeys_def]
      exact Finset.mem_image.mpr ⟨e, by simp, rfl⟩)
  have h1 := noiseY_eq W t 1
  have h2 := noiseY_eq W t' 1
  simp only [one_mul] at h1 h2
  rw [h1, h2, brSum_congr W hread]
  simp only [ksDigit_congr W hread, hks]

/-- The noise tail. -/
theorem noiseY_tail (W : World) {x : ℝ} (hx : 0 ≤ x) :
    tapeMeasure {t | x ≤ |noiseY W t|} ≤
      ENNReal.ofReal (2 * Real.exp (-x ^ 2 / (2 * noiseProxy))) :=
  measure_le_abs_le tapeMeasure (noiseY_measurable W) (by unfold noiseProxy; positivity)
    (noiseY_mgf W) hx

/-! ## The Hamming weight of the LWE secret -/

theorem lweKey_injective :
    Set.InjOn (fun j : Fin lweN ↦ lweKey j) ((Finset.univ : Finset (Fin lweN)) : Set (Fin lweN)) := by
  rintro j _ j' _ h
  simp only [lweKey, SampleKey.mk.injEq] at h
  exact Fin.ext h.2.2.2.1

theorem hamming_tail :
    tapeMeasure {t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j} ≤
      ENNReal.ofReal (Real.exp (-(36864 / 315))) := by
  set a : ℝ := 128 / 105
  have hblock := lintegral_mul_exp_sum_key_le (fun key : SampleKey ↦ key.law.measure)
    (fun j : Fin lweN ↦ lweKey j) ∅ Finset.univ lweKey_injective (fun _ _ ↦ by simp)
    (F := fun _ ↦ 1) (fun _ _ _ ↦ rfl) (c := fun _ _ ↦ a) (fun _ _ _ _ _ ↦ rfl)
    (K := fun _ ↦ a / 2 + a ^ 2 / 8) (fun _ _ _ ↦ SamplerLaw.lintegral_exp_bit_le a)
  rw [← tapeMeasure_eq] at hblock
  simp only [one_mul, lintegral_const, measure_univ, mul_one, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hblock
  have hmeas : Measurable fun t : SampleTape ↦
      ENNReal.ofReal (Real.exp (∑ j : Fin lweN, a * (t (lweKey j) : ℝ))) :=
    measurable_of_dependsOn (Finset.univ.image fun j : Fin lweN ↦ lweKey j) fun t t' h ↦ by
      have hj : ∀ j : Fin lweN, t (lweKey j) = t' (lweKey j) := fun j ↦ h _ (by simp)
      simp only [hj]
  have hmarkov := mul_meas_ge_le_lintegral₀ hmeas.aemeasurable
    (ENNReal.ofReal (Real.exp (507 * a))) (μ := tapeMeasure)
  have hsub : {t : SampleTape | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j} ⊆
      {t | ENNReal.ofReal (Real.exp (507 * a)) ≤
        ENNReal.ofReal (Real.exp (∑ j : Fin lweN, a * (t (lweKey j) : ℝ)))} := by
    intro t ht
    simp only [Set.mem_setOf_eq] at ht ⊢
    apply ENNReal.ofReal_le_ofReal
    apply Real.exp_le_exp.mpr
    rw [← Finset.mul_sum]
    have : (507 : ℝ) ≤ ∑ j : Fin lweN, (t (lweKey j) : ℝ) := by
      have := ht
      simp only [lweSecret] at this
      exact_mod_cast this
    have ha : 0 < a := by norm_num [a]
    calc 507 * a = a * 507 := by ring
      _ ≤ a * ∑ j : Fin lweN, (t (lweKey j) : ℝ) := mul_le_mul_of_nonneg_left this ha.le
  have hbound : ENNReal.ofReal (Real.exp (507 * a)) *
      tapeMeasure {t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j} ≤
      ENNReal.ofReal (Real.exp (↑lweN * (a / 2 + a ^ 2 / 8))) :=
    ((mul_le_mul_right (measure_mono hsub) _).trans hmarkov).trans (by simpa using hblock)
  have hpos : ENNReal.ofReal (Real.exp (507 * a)) ≠ 0 := by simpa using Real.exp_pos (507 * a)
  calc tapeMeasure {t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j}
      = (ENNReal.ofReal (Real.exp (507 * a)))⁻¹ * (ENNReal.ofReal (Real.exp (507 * a)) *
          tapeMeasure {t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j}) := by
        rw [← mul_assoc, ENNReal.inv_mul_cancel hpos ENNReal.ofReal_ne_top, one_mul]
    _ ≤ (ENNReal.ofReal (Real.exp (507 * a)))⁻¹ *
          ENNReal.ofReal (Real.exp (↑lweN * (a / 2 + a ^ 2 / 8))) := by gcongr
    _ = ENNReal.ofReal (Real.exp (-(36864 / 315))) := by
        rw [← ENNReal.ofReal_inv_of_pos (Real.exp_pos _), ← ENNReal.ofReal_mul (by positivity),
          ← Real.exp_neg, ← Real.exp_add]
        congr 2
        simp only [a, lweN]
        norm_num

end MxxFheTfhe
