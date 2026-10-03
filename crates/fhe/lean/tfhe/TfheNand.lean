import TfheRotation

/-!
The bootstrapped NAND stage. Its output phase under the LWE secret is `+Δ` or `-Δ`, by the NAND
of the input bits, plus an error `η`. When the LWE secret has at most `506` ones the lookup table
is selected correctly, and `η` differs from the explicit linear form `noiseY` of the sampled key
errors by at most the deterministic rounding allowance.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime

/-! ## Helpers -/

/-- A packed table whose entries all equal `floor(Q / 8)` is the lookup table. -/
theorem packed_lut {table : Nat}
    (h : checkBelow (fun i ↦ packedEntry 30 table i == 654329600) 1024 = true) :
    (packedPolynomial (q := Q) (n := N) 30 1024 table) 0 0 = lutPoly := by
  letI : Fact (1 < Q) := ⟨by decide⟩
  apply negacyclic_ext (by decide)
  intro i
  rw [lutPoly, packedPolynomial_coeff (by decide) (by decide),
    reducePoly_coeff (by decide) (by decide), intPoly_coeff (by decide)]
  rw [if_pos (show i.val < 1024 from i.isLt)]
  have hi := checkBelow_spec h i.val i.isLt
  simp only [beq_iff_eq] at hi
  rw [hi]
  rfl

theorem cast_emod_q (a : Int) : ((a % 4294967296 : Int) : ZMod q) = a := ZMod.intCast_mod a q

theorem cast_emod_Q (a : Int) : ((a % 5234636801 : Int) : ZMod Q) = a := ZMod.intCast_mod a Q

theorem cast_emod_2048 (a : Int) : ((a % 2048 : Int) : ZMod 2048) = a := ZMod.intCast_mod a 2048

/-- Rounding `x q / Q` to the nearest integer moves `Q` times the result by at most `Q / 2`. -/
theorem switch_bound (x : Int) :
    |5234636801 * ((x * 4294967296 + 2617318400) / 5234636801) - 4294967296 * x| ≤
      2617318400 := by
  rw [abs_le]
  omega

/-- The key-switching digit gadget `4^k` selected by `k = i % 8`. -/
theorem select_powers {i : Nat} {w : Int}
    (h : select ((i : Int) % 8) [1 + 0, 4 + 0, 16 + 0, 64 + 0, 256 + 0, 1024 + 0, 4096 + 0,
      16384 + 0] w) : w = 4 ^ (i % 8) := by
  obtain ⟨p, hp, rfl⟩ := h
  have hp' : i % 8 = p.val := by
    have := p.isLt
    simp only [List.length_cons, List.length_nil] at this hp
    omega
  rw [hp']
  fin_cases p <;> rfl

/-- Eight base-four digits rebuild a 16-bit value. -/
theorem digits_four {t : Int} (h0 : 0 ≤ t) (h1 : t < 65536) :
    ∑ k : Fin 8, t / 4 ^ k.val % 4 * 4 ^ k.val = t := by
  simp only [Fin.sum_univ_eight, Fin.isValue, Fin.val_zero, Fin.val_one, Fin.val_two]
  norm_num
  omega

/-- Regroup a sum over flat key-switching digits by ring coefficient. -/
theorem sum_digits_reindex (f : Fin 8192 → Int) (g : Fin 1024 → Int) :
    ∑ i : Fin 8192, f i * (g ⟨i.val / 8, by omega⟩ * 4 ^ (i.val % 8)) =
      ∑ j : Fin 1024, g j * ∑ k : Fin 8, f ⟨j.val * 8 + k.val, by omega⟩ * 4 ^ k.val := by
  rw [← (finProdFinEquiv : Fin 1024 × Fin 8 ≃ Fin 8192).sum_comp, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  have hjk : (finProdFinEquiv (j, k) : Fin 8192) = ⟨j.val * 8 + k.val, by omega⟩ := by
    apply Fin.ext
    simp only [finProdFinEquiv_apply_val]
    ring
  rw [hjk]
  have hdiv : (⟨(j.val * 8 + k.val) / 8, by omega⟩ : Fin 1024) = j := Fin.ext (by simp only; omega)
  have hmod : (j.val * 8 + k.val) % 8 = k.val := by omega
  simp only [hdiv, hmod]
  ring

/-! ## The explicit key-switching digits and the noise linear form -/

/-- The canonical coefficients of the accumulator's mask after blind rotation. -/
noncomputable def accMaskValues (W : World) (t : SampleTape) (k : Fin N) : Int :=
  ((accSeq W t lweN 0 0).coeff k).val

/-- Sample extraction and the switch to `q`: entry `j` of the extracted mask. -/
def extractedMask (values : Fin N → Int) (j : Fin N) : Int :=
  ((values ⟨(N - j.val) % N, Nat.mod_lt _ (by decide)⟩ * (if j.val = 0 then 1 else -1)) %
    5234636801 * 4294967296 + 2617318400) / 5234636801 % 4294967296

/-- The top 16 bits of a mask entry, rounded. -/
def ksTop (x : Int) : Int := (x + 32768) / 65536 % 65536

/-- Key-switching digit `e`: digit `e % 8` of the rounded mask entry `e / 8`. -/
noncomputable def ksDigit (W : World) (t : SampleTape) (e : Fin 8192) : Int :=
  ksTop (extractedMask (accMaskValues W t) ⟨e.val / 8, by simp only [N]; omega⟩) /
    4 ^ (e.val % 8) % 4

/-- The noise of the NAND output that the sampled key errors contribute, before the deterministic
rounding terms: the blind-rotation error switched to `q`, minus the key-switching errors weighted
by their digits. -/
noncomputable def noiseY (W : World) (t : SampleTape) : ℝ :=
  (4294967296 / 5234636801 : ℝ) * ((errSeq W t lweN).coeff ⟨0, by decide⟩ : ℝ) -
    ∑ e : Fin 8192, (ksDigit W t e : ℝ) * (kskError t e : ℝ)

/-- The deterministic rounding allowance: the `Q`-to-`q` switch and the key-switching rounding. -/
abbrev allowance : ℝ := 33555458

/-! ## The NAND stage -/

set_option maxHeartbeats 16000000 in
/-- The bootstrapped NAND output phase is `±Δ + η`, and `η` is the noise linear form up to the
rounding allowance, when the LWE secret has at most `506` ones. -/
theorem nand_spec {W : World} {t : SampleTape} (hs : ∀ j, lweSecret t j = 0 ∨ lweSecret t j = 1)
    (hZ : ∀ k : Fin N, t (ringKey k) = 0 ∨ t (ringKey k) = 1)
    (hm1 : W.messageL = 0 ∨ W.messageL = 1) (hm2 : W.messageR = 0 ∨ W.messageR = 1)
    (hW : ∑ j, lweSecret t j ≤ 506)
    {kska : Fin 5160960 → Int} {kskb : Fin 8192 → Int}
    (hks : ∀ e : Fin 8192, kskb e = ((∑ j : Fin lweN, kska ⟨e.val * 630 + j.val, by
        have := e.isLt; have := j.isLt; simp only [lweN] at *; omega⟩ * lweSecret t j) +
        t (ringKey (e.val / 8)) * (4 ^ (e.val % 8) * 65536) + kskError t e) % q)
    {left right : (Fin lweN → Int) × Int × Unit}
    (hl : EncryptionFacts t 1 (lweSecret t) W.messageL left) (hlm : left.1 = W.maskL)
    (hr : EncryptionFacts t 2 (lweSecret t) W.messageR right) (hrm : right.1 = W.maskR)
    {A B : Fin lweN → ExactMatrix Q N 1 12} (hA : A = gswA t) (hB : B = gswB t) {outputs}
    (h : Stage_nand.generatedRoot FheBackend.backend t [3] { unit := () }
      (kska, left.2.1, right.2.1, left.1, right.1, A, B, kskb, ()) outputs) :
    ∃ η : Int, ((outputs.2.1 - ∑ j : Fin lweN, outputs.1 j * lweSecret t j : Int) : ZMod q) =
        (((1 - W.messageL * W.messageR) * 2 - 1) * Δ + η : Int) ∧
      |(η : ℝ) - noiseY W t| ≤ allowance := by
  obtain ⟨witness, _, _, _, _, _, _, _, hsum, _, hneg, hbr, hav, _, hext, _, hdig, _, houta, hbv,
    hb0, _, _, _, hksb, _, hout⟩ := h
  subst hout
  dsimp only at hsum hbr houta hksb ⊢
  obtain ⟨_, he1, hb1⟩ := hl
  obtain ⟨_, he2, hb2⟩ := hr
  set e1 := encError t 1
  set e2 := encError t 2
  simp only [q, Δ, Nat.cast_ofNat] at hb1 hb2
  have hw31 (i : Fin lweN) : witness.w_31_0 i = (left.1 i + right.1 i) % 4294967296 := (hsum i).2
  have hw32 (i : Fin lweN) : witness.w_32_0 i = (0 - witness.w_31_0 i) % 4294967296 := (hneg i).2
  have hw32W : witness.w_32_0 = combinedMask W := by
    funext i
    rw [hw32, hw31, hlm, hrm]
    rfl
  -- The combined input phase `(3 - 2 m1 - 2 m2) Δ - e1 - e2`.
  set bp : Int := ((0 - (left.2.1 + right.2.1) % 4294967296) % 4294967296 + 536870912) %
    4294967296 with hbp
  have hbpW : bp = combinedBody W t := by
    rw [hbp, hb1, hb2, hlm, hrm]
    rfl
  have hbp_range : 0 ≤ bp ∧ bp < 4294967296 := by omega
  have hap_range (i : Fin lweN) : 0 ≤ witness.w_32_0 i ∧ witness.w_32_0 i < 4294967296 := by
    rw [hw32]; omega
  have hμ : ((bp - ∑ j, witness.w_32_0 j * lweSecret t j : Int) : ZMod q) =
      (((3 - 2 * W.messageL - 2 * W.messageR) * 536870912 - e1 - e2 : Int) : ZMod q) := by
    simp only [hbp, hw32, hw31, hb1, hb2, Int.cast_sub, Int.cast_add, Int.cast_mul, Int.cast_sum,
      cast_emod_q, Int.cast_zero]
    simp only [zero_sub, neg_add, neg_mul, add_mul, Finset.sum_neg_distrib, Finset.sum_add_distrib]
    push_cast
    ring
  -- Rounding to `Z_2048`: `2^21` times each rounded value is within `2^20` of the input.
  set rb : Int := (bp * 2048 + 2147483648) / 4294967296 with hrb
  have hrb_bound : -1048576 < 2097152 * rb - bp ∧ 2097152 * rb - bp ≤ 1048576 := by omega
  have hra_bound (i : Fin lweN) :
      |2097152 * ((witness.w_32_0 i * 2048 + 2147483648) / 4294967296) - witness.w_32_0 i| ≤
        1048576 := by
    have := hap_range i
    rw [abs_le]
    omega
  -- Blind rotation follows the explicit model.
  obtain ⟨hw35a, hw35b⟩ := blind_rotation_runs (W := W) hA hB (packed_lut (by decide +kernel))
    (by rw [initialExp, roundRing, ← hbpW]) hw32W hbr
  simp only at hw35a hw35b
  have hphase := accSeq_phase W t hs lweN
  rw [← hw35a, ← hw35b] at hphase
  -- The rotation exponent is minus the rounded phase modulo `2 N`.
  set T : Nat := expSeq W t lweN with hT
  set sr : Int := ∑ i, lweSecret t i * ((witness.w_32_0 i * 2048 + 2147483648) / 4294967296) with hsr
  have hTint : ((T : Nat) : Int) = initialExp W t + ∑ i : Fin lweN, lweSecret t i * rotation W i := by
    rw [hT, expSeq, Nat.cast_add, Int.toNat_of_nonneg (initialExp_range W t).1,
      Finset.sum_range (fun j ↦ if h : j < lweN then stepShift W t ⟨j, h⟩ else 0), Nat.cast_sum]
    refine congrArg₂ (· + ·) rfl (Finset.sum_congr rfl fun i _ ↦ ?_)
    rw [dif_pos i.isLt]
    simp only [Fin.eta, stepShift, Nat.cast_mul]
    rw [Int.toNat_of_nonneg (show 0 ≤ lweSecret t i by rcases hs i with h | h <;> rw [h]; norm_num),
      Int.toNat_of_nonneg (show 0 ≤ rotation W i from (roundRing_range _).1)]
  have hTmod : ((T % (2 * N) : Nat) : Int) = (-(rb - sr)) % 2048 := by
    have hcast : ((T : Int) : ZMod 2048) = ((-(rb - sr) : Int) : ZMod 2048) := by
      rw [hTint, initialExp, ← hbpW]
      simp only [hsr, rotation, roundRing, hw32W, Int.cast_add, Int.cast_sum, Int.cast_mul,
        Int.cast_sub, Int.cast_neg, Int.cast_zero, cast_emod_2048]
      ring
    have := (ZMod.intCast_eq_intCast_iff' _ _ _).mp hcast
    simp only [N, Nat.cast_ofNat] at this ⊢
    omega
  set sw : Int := ∑ j, witness.w_32_0 j * lweSecret t j with hsw
  have hρ : |2097152 * sr - sw| ≤ 506 * 1048576 := by
    have : 2097152 * sr - sw = ∑ i, lweSecret t i * (2097152 *
        ((witness.w_32_0 i * 2048 + 2147483648) / 4294967296) - witness.w_32_0 i) := by
      rw [hsr, hsw, Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ ↦ by ring
    rw [this]
    refine (abs_sum_binary_le _ _ hs hra_bound).trans ?_
    exact mul_le_mul_of_nonneg_right hW (by norm_num)
  obtain ⟨k, hk⟩ := exists_of_cast_eq hμ
  have he1' := he1
  have he2' := he2
  rw [abs_le] at he1' he2'
  have hsign : (if T % (2 * N) = 0 ∨ N < T % (2 * N) then lutValue else -lutValue) =
      ((1 - W.messageL * W.messageR) * 2 - 1) * lutValue := by
    have hρ' := abs_le.mp hρ
    clear_value T sr sw rb bp
    simp only [q, Nat.cast_ofNat] at hk
    simp only [N] at hTmod ⊢
    rcases hm1 with h1 | h1 <;> rcases hm2 with h2 | h2 <;> rw [h1, h2] at hk ⊢ <;>
      split_ifs with hc <;> first | (exfalso; omega) | norm_num
  -- Sample extraction: the constant coefficient of the accumulator phase.
  set E0 : Int := (errSeq W t lweN).coeff ⟨0, by decide⟩ with hE0
  have hα (i : Fin N) : witness.w_36_0 i = accMaskValues W t i := by
    simp only [polynomialValues, Bool.false_eq_true, if_false] at hav
    rw [hav i, hw35a]
    rfl
  have hβ (i : Fin N) : witness.w_51_0 i = ((witness.w_35_1 0 0).coeff i).val := by
    simp only [polynomialValues, Bool.false_eq_true, if_false] at hbv
    exact hbv i
  have hw52 : witness.w_52_0 = witness.w_51_0 ⟨0, by decide⟩ := by
    obtain ⟨p, hp, h52⟩ := hb0
    have hp0 : p = 0 := by exact_mod_cast hp
    rw [h52, hp0]
    rfl
  have hcoef : ((witness.w_52_0 - ∑ j : Fin N, (if j.val = 0 then 1 else -1) *
      accMaskValues W t ⟨(N - j.val) % N, Nat.mod_lt _ (by decide)⟩ * t (ringKey j) : Int) :
        ZMod Q) = ((((1 - W.messageL * W.messageR) * 2 - 1) * lutValue + E0 : Int) : ZMod Q) := by
    letI : Fact (1 < Q) := ⟨by decide⟩
    have hred : reducePoly Q N (canonicalLift (witness.w_35_1 0 0) -
        canonicalLift (witness.w_35_0 0 0) * ringSecret t) = reducePoly Q N
          (rootZ ^ T * intPoly (fun _ ↦ lutValue) + errSeq W t lweN) := by
      rw [map_sub, map_mul, reducePoly_canonicalLift (by decide) (by decide),
        reducePoly_canonicalLift (by decide) (by decide), mul_comm (witness.w_35_0 0 0), hphase,
        map_add, map_mul, reducePoly_rootZ_pow]
      rfl
    have h0 := congrArg (fun p ↦ Negacyclic.coeff p ⟨0, by decide⟩) hred
    simp only [reducePoly_coeff (by decide : 1 < Q) (by decide : 0 < N), Negacyclic.coeff_sub,
      Negacyclic.coeff_add, coeff_zero_root_pow_mul_const_mod (by decide : 0 < N), hsign] at h0
    simp only [coeff_zero_mul (by decide : 0 < N), canonicalLift_coeff (by decide : 0 < N),
      ringSecret_coeff] at h0
    rw [hw52, hβ]
    simp only [accMaskValues, ← hw35a]
    exact h0
  -- Switch from `Q` to `q`.
  have hw38 (i : Fin N) : witness.w_38_0 i = extractedMask (accMaskValues W t) i := by
    obtain ⟨w6, _, _, _, ⟨p, hp, hw6⟩, _, _, _, hout⟩ := hext i
    have hp' : p = ⟨(N - i.val) % N, Nat.mod_lt _ (by decide)⟩ := by
      apply Fin.ext
      have := i.isLt
      simp only [Int.ofNat_eq_natCast, N] at hp this ⊢
      omega
    subst hp'
    simp only at hw6
    rw [hout, hw6, hα]
    unfold extractedMask
    by_cases hi : i.val = 0 <;> simp [hi]
  set γ : Fin N → Int := fun j ↦ (accMaskValues W t ⟨(N - j.val) % N, Nat.mod_lt _ (by decide)⟩ *
    (if j.val = 0 then 1 else -1)) % 5234636801 with hγ
  set rγ : Fin N → Int := fun j ↦ (γ j * 4294967296 + 2617318400) / 5234636801 with hrγ
  have hw38γ (j : Fin N) : witness.w_38_0 j = rγ j % 4294967296 := hw38 j
  have hβ0 : 0 ≤ witness.w_52_0 ∧ witness.w_52_0 < 5234636801 := by
    rw [hw52, hβ]
    have := ZMod.val_lt ((witness.w_35_1 0 0).coeff ⟨0, by decide⟩)
    omega
  have hcoefγ : ((witness.w_52_0 - ∑ j, γ j * t (ringKey j) : Int) : ZMod Q) =
      ((((1 - W.messageL * W.messageR) * 2 - 1) * lutValue + E0 : Int) : ZMod Q) := by
    rw [← hcoef]
    simp only [hγ, Int.cast_sub, Int.cast_sum, Int.cast_mul, cast_emod_Q]
    exact congrArg₂ (· - ·) rfl (Finset.sum_congr rfl fun j _ ↦ by ring)
  obtain ⟨l, hl⟩ := exists_of_cast_eq hcoefγ
  set rbQ : Int := (witness.w_52_0 * 4294967296 + 2617318400) / 5234636801 with hrbQ
  set P : Int := rbQ - ∑ j, rγ j * t (ringKey j) - 4294967296 * l with hP
  -- `Q (P - s Δ) = D + q E0` with `|D|` at most the switching rounding.
  set D : Int := (5234636801 * rbQ - 4294967296 * witness.w_52_0) -
    ∑ j, (5234636801 * rγ j - 4294967296 * γ j) * t (ringKey j) -
    ((1 - W.messageL * W.messageR) * 2 - 1) * 536870912 with hD
  have hPD : 5234636801 * (P - ((1 - W.messageL * W.messageR) * 2 - 1) * 536870912) =
      D + 4294967296 * E0 := by
    have hσdef : ∑ j, (5234636801 * rγ j - 4294967296 * γ j) * t (ringKey j) =
        5234636801 * ∑ j, rγ j * t (ringKey j) - 4294967296 * ∑ j, γ j * t (ringKey j) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ ↦ by ring
    simp only [Q, Nat.cast_ofNat, lutValue] at hl
    rw [hD, hσdef, hP]
    linear_combination 4294967296 * hl
  have hDbound : |D| ≤ 1025 * 2617318400 + 536870912 := by
    have hσ : |∑ j, (5234636801 * rγ j - 4294967296 * γ j) * t (ringKey j)| ≤
        1024 * 2617318400 := by
      refine (abs_sum_le_card _ 2617318400 fun j ↦ ?_).trans (by simp [N])
      rw [mul_comm]
      exact abs_binary_mul_le (hZ j) (switch_bound _) (by norm_num)
    have hb := switch_bound witness.w_52_0
    have hsgn : |((1 - W.messageL * W.messageR) * 2 - 1) * (536870912 : Int)| ≤ 536870912 := by
      rcases hm1 with h1 | h1 <;> rcases hm2 with h2 | h2 <;> rw [h1, h2] <;> norm_num
    rw [hD]
    linarith [abs_sub (5234636801 * rbQ - 4294967296 * witness.w_52_0)
      (∑ j, (5234636801 * rγ j - 4294967296 * γ j) * t (ringKey j)),
      abs_sub ((5234636801 * rbQ - 4294967296 * witness.w_52_0) -
        ∑ j, (5234636801 * rγ j - 4294967296 * γ j) * t (ringKey j))
        (((1 - W.messageL * W.messageR) * 2 - 1) * 536870912)]
  -- Key switching: the rounded top 16 bits of each extracted mask entry, in base-four digits.
  set tt : Fin N → Int := fun j ↦ (witness.w_38_0 j + 32768) / 65536 % 65536 with htt
  have hw47 (i : Fin 8192) :
      witness.w_47_0 i = tt ⟨i.val / 8, by simp only [N]; omega⟩ / 4 ^ (i.val % 8) % 4 := by
    obtain ⟨w4, w37, _, _, _, ⟨p, hp, hw4⟩, _, _, _, _, _, _, hsel, _, _, hout⟩ := hdig i
    have hp' : p = ⟨i.val / 8, by omega⟩ := by
      apply Fin.ext
      show p.val = i.val / 8
      rw [Int.ofNat_eq_natCast] at hp
      omega
    rw [hp'] at hw4
    rw [hout, hw4, select_powers hsel]
  have hw47K (i : Fin 8192) : witness.w_47_0 i = ksDigit W t i := by
    rw [hw47, ksDigit, ksTop, htt]
    dsimp only
    rw [hw38]
  have hw47_range (i : Fin 8192) : 0 ≤ witness.w_47_0 i ∧ witness.w_47_0 i < 4 := by
    rw [hw47]
    omega
  have hdigits (j : Fin N) :
      ∑ k : Fin 8, witness.w_47_0 ⟨j.val * 8 + k.val, by simp only [N] at *; omega⟩ * 4 ^ k.val =
        tt j := by
    have hterm (k : Fin 8) : witness.w_47_0 ⟨j.val * 8 + k.val, by simp only [N] at *; omega⟩ =
        tt j / 4 ^ k.val % 4 := by
      rw [hw47]
      have hdiv : (⟨(j.val * 8 + k.val) / 8, by simp only [N] at *; omega⟩ : Fin N) = j :=
        Fin.ext (by simp only; omega)
      have hmod : (j.val * 8 + k.val) % 8 = k.val := by omega
      simp only [hdiv, hmod]
    simp only [hterm]
    exact digits_four (by simp only [htt]; omega) (by simp only [htt]; omega)
  have hw50 (j : Fin lweN) : witness.w_50_0 j =
      (0 - ∑ i : Fin 8192, kska ⟨i.val * 630 + j.val, by simp only [lweN] at *; omega⟩ *
        witness.w_47_0 i) % 4294967296 := by
    rw [(houta j).2, intMatrixVectorProduct_columns (by rfl)]
  have hw64 : witness.w_64_0 = ∑ i : Fin 8192, kskb i * witness.w_47_0 i := by
    obtain ⟨p, hp, h64⟩ := hksb
    have hp0 : p = 0 := Fin.ext (by have := p.isLt; omega)
    rw [h64, hp0, intMatrixVectorProduct_columns (by rfl)]
    exact Finset.sum_congr rfl fun i _ ↦ by simp
  have hkb := hks
  simp only [q, Nat.cast_ofNat] at hkb
  have hq0 : ((4294967296 : Int) : ZMod q) = 0 := by exact_mod_cast ZMod.natCast_self q
  -- Rounding each mask entry to its top 16 bits.
  set ρ : Fin N → Int := fun j ↦ 65536 * ((witness.w_38_0 j + 32768) / 65536) - witness.w_38_0 j
    with hρdef
  have hρbound (j : Fin N) : |ρ j| ≤ 32768 := by
    have : 0 ≤ witness.w_38_0 j ∧ witness.w_38_0 j < 4294967296 := by rw [hw38γ j]; omega
    simp only [hρdef]
    rw [abs_le]
    omega
  have ht_cast (j : Fin N) : ((65536 * tt j : Int) : ZMod q) = ((rγ j + ρ j : Int) : ZMod q) := by
    have : 65536 * tt j = witness.w_38_0 j + ρ j -
        4294967296 * ((witness.w_38_0 j + 32768) / 65536 / 65536) := by
      simp only [htt, hρdef]
      omega
    rw [this, Int.cast_sub, Int.cast_mul, hq0, zero_mul, sub_zero, Int.cast_add, Int.cast_add,
      hw38γ j, cast_emod_q]
  -- The output phase.
  have hreindex : ∑ i : Fin 8192, witness.w_47_0 i *
      (t (ringKey (i.val / 8)) * (4 ^ (i.val % 8) * 65536) + kskError t i) =
      ∑ j : Fin N, t (ringKey j) * (65536 * tt j) + ∑ i, witness.w_47_0 i * kskError t i := by
    have h := sum_digits_reindex witness.w_47_0 (fun j ↦ t (ringKey j))
    simp only [hdigits] at h
    calc _ = (∑ i : Fin 8192, witness.w_47_0 i *
          (t (ringKey (i.val / 8)) * 4 ^ (i.val % 8))) * 65536 +
          ∑ i, witness.w_47_0 i * kskError t i := by
          rw [Finset.sum_mul, ← Finset.sum_add_distrib]
          exact Finset.sum_congr rfl fun i _ ↦ by ring
      _ = _ := by
          rw [show (∑ i : Fin 8192, witness.w_47_0 i * (t (ringKey (i.val / 8)) *
              4 ^ (i.val % 8))) = ∑ j : Fin N, t (ringKey j) * tt j from h, Finset.sum_mul]
          exact congrArg₂ (· + ·) (Finset.sum_congr rfl fun j _ ↦ by ring) rfl
  rw [Int.emod_eq_of_lt hβ0.1 hβ0.2, ← hrbQ]
  refine ⟨P - ((1 - W.messageL * W.messageR) * 2 - 1) * 536870912 - ∑ j, ρ j * t (ringKey j) -
    ∑ i, witness.w_47_0 i * kskError t i, ?_, ?_⟩
  · have hL : (((rbQ % 4294967296 - witness.w_64_0) % 4294967296 -
        ∑ j, witness.w_50_0 j * lweSecret t j : Int) : ZMod q) =
        ((rbQ - ∑ i : Fin 8192, witness.w_47_0 i *
          (t (ringKey (i.val / 8)) * (4 ^ (i.val % 8) * 65536) + kskError t i) : Int) : ZMod q) := by
      simp only [hw64, hw50, Int.cast_sub, Int.cast_sum, Int.cast_mul, cast_emod_q, Int.cast_zero,
        hkb, Int.cast_add]
      exact ks_algebra _ _ _ _ _ _
    have hsumt : ((∑ j : Fin N, t (ringKey j) * (65536 * tt j) : Int) : ZMod q) =
        ∑ j : Fin N, ((rγ j : Int) : ZMod q) * (t (ringKey j) : ZMod q) +
          ∑ j : Fin N, ((ρ j : Int) : ZMod q) * (t (ringKey j) : ZMod q) := by
      rw [Int.cast_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun j _ ↦ by rw [Int.cast_mul, ht_cast, Int.cast_add]; ring
    have hq0' : (4294967296 : ZMod q) = 0 := by simpa using hq0
    have hPcast : ((P : Int) : ZMod q) =
        (rbQ : ZMod q) - ∑ j, ((rγ j : Int) : ZMod q) * (t (ringKey j) : ZMod q) := by
      rw [hP]
      push_cast
      rw [hq0']
      ring
    rw [hL, hreindex, Int.cast_sub, Int.cast_add, hsumt]
    simp only [Δ, Nat.cast_ofNat]
    push_cast
    rw [hPcast]
    ring
  · have hZρ : |∑ j, ρ j * t (ringKey j)| ≤ 1024 * 32768 := by
      refine (abs_sum_le_card _ 32768 fun j ↦ ?_).trans (by simp [N])
      rw [mul_comm]
      exact abs_binary_mul_le (hZ j) (hρbound j) (by norm_num)
    have hPDr : (5234636801 : ℝ) *
        (((P - ((1 - W.messageL * W.messageR) * 2 - 1) * 536870912 : Int)) : ℝ) =
        (D : ℝ) + 4294967296 * (E0 : ℝ) := by exact_mod_cast hPD
    have hY : noiseY W t = (4294967296 / 5234636801 : ℝ) * (E0 : ℝ) -
        ((∑ i, witness.w_47_0 i * kskError t i : Int) : ℝ) := by
      rw [noiseY, Int.cast_sum]
      simp only [Int.cast_mul, hw47K]
      rfl
    have hdiff : ((P - ((1 - W.messageL * W.messageR) * 2 - 1) * 536870912 -
        ∑ j, ρ j * t (ringKey j) - ∑ i, witness.w_47_0 i * kskError t i : Int) : ℝ) -
        noiseY W t = (D : ℝ) / 5234636801 - ((∑ j, ρ j * t (ringKey j) : Int) : ℝ) := by
      rw [hY]
      push_cast at hPDr ⊢
      field_simp
      linear_combination hPDr
    rw [hdiff]
    have hDr : |(D : ℝ)| ≤ 1025 * 2617318400 + 536870912 := by exact_mod_cast hDbound
    have hZρr : |((∑ j, ρ j * t (ringKey j) : Int) : ℝ)| ≤ 1024 * 32768 := by exact_mod_cast hZρ
    have hDq : |(D : ℝ)| / 5234636801 ≤ (1025 * 2617318400 + 536870912) / 5234636801 :=
      div_le_div_of_nonneg_right hDr (by norm_num)
    calc |(D : ℝ) / 5234636801 - ((∑ j, ρ j * t (ringKey j) : Int) : ℝ)|
        ≤ |(D : ℝ) / 5234636801| + |((∑ j, ρ j * t (ringKey j) : Int) : ℝ)| := abs_sub _ _
      _ = |(D : ℝ)| / 5234636801 + |((∑ j, ρ j * t (ringKey j) : Int) : ℝ)| := by
          rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 5234636801)]
      _ ≤ allowance := by
          simp only [allowance]
          linarith


end MxxFheTfhe
