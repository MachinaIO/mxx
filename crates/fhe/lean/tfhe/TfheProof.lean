import TfheBound
import Claim
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
The generated TFHE correctness claim. A run fails only if the LWE secret has more than `506`
ones or the noise linear form exceeds `Δ` minus the rounding allowance; both events are rare
over the sampling tape.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime GeneratedClaim MeasureTheory

theorem encrypt_left_mask_eq {hashModel : HashModel} {tape tape' : SampleTape} {key : ByteArray}
    {secret secret' : Fin lweN → Int} {message message' : Int} {outputs outputs'}
    (h : Stage_encrypt_left.generatedRoot hashModel tape [1] { unit := () }
      (key, secret, message, ()) outputs)
    (h' : Stage_encrypt_left.generatedRoot hashModel tape' [1] { unit := () }
      (key, secret', message', ()) outputs') : outputs.1 = outputs'.1 := by
  obtain ⟨w, hmask, _, _, _, _, _, _, _, _, hout⟩ := h
  obtain ⟨w', hmask', _, _, _, _, _, _, _, _, hout'⟩ := h'
  subst hout hout'
  exact hashIntFamily_functional hmask hmask'

theorem encrypt_right_mask_eq {hashModel : HashModel} {tape tape' : SampleTape} {key : ByteArray}
    {secret secret' : Fin lweN → Int} {message message' : Int} {outputs outputs'}
    (h : Stage_encrypt_right.generatedRoot hashModel tape [2] { unit := () }
      (key, secret, message, ()) outputs)
    (h' : Stage_encrypt_right.generatedRoot hashModel tape' [2] { unit := () }
      (key, secret', message', ()) outputs') : outputs.1 = outputs'.1 := by
  obtain ⟨w, hmask, _, _, _, _, _, _, _, _, hout⟩ := h
  obtain ⟨w', hmask', _, _, _, _, _, _, _, _, hout'⟩ := h'
  subst hout hout'
  exact hashIntFamily_functional hmask hmask'

/-- One run is correct when the LWE secret has at most `506` ones and the noise linear form is
below `Δ` minus the rounding allowance. -/
theorem run_correct {hashModel : HashModel} {external : ExternalInputs} {t : SampleTape}
    {x : Execution} (hruns : Runs hashModel external t x) (W : World)
    (hWL : x.«stage_1».1 = W.maskL) (hWR : x.«stage_2».1 = W.maskR)
    (hmL : W.messageL = external.input_1) (hmR : W.messageR = external.input_3)
    (hW : ∑ j, lweSecret t j ≤ 506) (hY : |noiseY W t| < Δ - allowance) :
    (∀ index, (observedResidual x index).natAbs < TfheSemantics.decoderRadius 4294967296) ∧
      x.«stage_4».1 = x.«ideal» := by
  obtain ⟨hvalid, hk, hl, hr, hn, hd, hi⟩ := hruns
  obtain ⟨hsec, hs, hZ, hA, hB, _, _, hks⟩ := keygen_spec hk
  obtain ⟨_, ⟨hm1a, hm1b⟩, _, ⟨hm2a, hm2b⟩, _⟩ := hvalid
  have hm1 : W.messageL = 0 ∨ W.messageL = 1 := by rw [hmL]; omega
  have hm2 : W.messageR = 0 ∨ W.messageR = 1 := by rw [hmR]; omega
  have hl' := encrypt_left_spec hl
  have hr' := encrypt_right_spec hr
  rw [hsec, ← hmL] at hl'
  rw [hsec, ← hmR] at hr'
  obtain ⟨η, hphase, hηY⟩ := nand_spec (W := W) hs hZ hm1 hm2 hW
    (fun e ↦ (hks e).2) hl' hWL hr' hWR (funext hA) (funext hB) hn
  have hη : |(η : ℝ)| < Δ := by
    have := abs_sub_abs_le_abs_sub (η : ℝ) (noiseY W t)
    simp only [allowance] at hηY hY
    linarith
  have hηZ : -536870912 < η ∧ η < 536870912 := by
    rw [abs_lt] at hη
    simp only [Δ, Nat.cast_ofNat] at hη
    have h1 := hη.1
    have h2 := hη.2
    rw [show (-536870912 : ℝ) = ((-536870912 : ℤ) : ℝ) by norm_num] at h1
    rw [show (536870912 : ℝ) = ((536870912 : ℤ) : ℝ) by norm_num] at h2
    exact ⟨Int.cast_lt.mp h1, Int.cast_lt.mp h2⟩
  obtain ⟨hdec, hbit⟩ := decrypt_spec hd
  rw [hsec] at hdec
  have hideal := ideal_spec hi
  rw [hmL, hmR] at hphase
  -- The decryption phase is the NAND output phase.
  have hv : ((x.«stage_4».2.1 : Int) : ZMod q) =
      (((1 - external.input_1 * external.input_3) * 2 - 1) * Δ + η : Int) := by
    rw [hdec, ZMod.intCast_mod]
    exact hphase
  have hvq := (ZMod.intCast_eq_intCast_iff' _ _ _).mp hv
  have hrange : 0 ≤ x.«stage_4».2.1 ∧ x.«stage_4».2.1 < 4294967296 := by
    rw [hdec]
    simp only [q, Nat.cast_ofNat]
    omega
  simp only [q, Δ, Nat.cast_ofNat] at hvq hbit
  rw [Int.emod_eq_of_lt hrange.1 hrange.2] at hvq
  have hm1' : external.input_1 = 0 ∨ external.input_1 = 1 := by omega
  have hm2' : external.input_3 = 0 ∨ external.input_3 = 1 := by omega
  constructor
  · intro index
    unfold observedResidual TfheSemantics.messageCenter TfheSemantics.decoderRadius
    have hres : ((x.«stage_4».2.1 : Int) : ZMod 4294967296) -
        (((if x.«ideal» = 1 then 536870912 else (4294967296 : Nat) - 536870912 : Int) :
          Int) : ZMod 4294967296) = (η : ZMod 4294967296) := by
      have hq0 : ((4294967296 : Int) : ZMod 4294967296) = 0 := by
        exact_mod_cast ZMod.natCast_self 4294967296
      have hq0' : (4294967296 : ZMod 4294967296) = 0 := by simpa using hq0
      rw [show (x.«stage_4».2.1 : ZMod 4294967296) = ((x.«stage_4».2.1 : Int) :
        ZMod q) from rfl, hv, hideal]
      simp only [Δ]
      rcases hm1' with h1 | h1 <;> rcases hm2' with h2 | h2 <;>
        · simp only [h1, h2]
          norm_num
          try (ring_nf; simp [hq0'])
    rw [hres, centeredLift_intCast (by decide) (by omega)]
    omega
  · rw [hbit, hideal]
    rcases hm1' with h1 | h1 <;> rcases hm2' with h2 | h2 <;> simp only [h1, h2] at hvq ⊢ <;>
      split_ifs <;> omega

/-- `exp (-x) ≤ 2^-k` when `x ≥ k · 0.6931471808`, which exceeds `k log 2`. -/
theorem exp_neg_le_half_pow {x : ℝ} {k : ℕ} (h : (k : ℝ) * 0.6931471808 ≤ x) :
    Real.exp (-x) ≤ (1 / 2) ^ k := by
  have hlog := Real.log_two_lt_d9
  have hk : (0 : ℝ) ≤ k := Nat.cast_nonneg k
  have hle : Real.exp (-x) ≤ Real.exp (-(k * Real.log 2)) :=
    Real.exp_le_exp.mpr (by nlinarith)
  have hpow : Real.exp (-(k * Real.log 2)) = (1 / 2) ^ k := by
    rw [Real.exp_neg, Real.exp_nat_mul, Real.exp_log two_pos, one_div, inv_pow]
  exact hle.trans_eq hpow

theorem final_numeric :
    ENNReal.ofReal (Real.exp (-(36864 / 315))) +
      ENNReal.ofReal (2 * Real.exp (-((Δ : ℝ) - allowance) ^ 2 / (2 * noiseProxy))) ≤
      (2 : ENNReal)⁻¹ ^ 128 := by
  have ha : Real.exp (-(36864 / 315)) ≤ (1 / 2) ^ 129 :=
    exp_neg_le_half_pow (by norm_num)
  have hy : (130 : ℕ) * (0.6931471808 : ℝ) ≤ ((Δ : ℝ) - allowance) ^ 2 / (2 * noiseProxy) := by
    have hV : 0 < noiseProxy := by unfold noiseProxy; positivity
    rw [le_div_iff₀ (by positivity)]
    simp only [noiseProxy, allowance, Δ, lweN]
    push_cast
    norm_num
  have hb : 2 * Real.exp (-((Δ : ℝ) - allowance) ^ 2 / (2 * noiseProxy)) ≤ (1 / 2) ^ 129 := by
    have := exp_neg_le_half_pow hy
    rw [neg_div]
    calc 2 * Real.exp (-(((Δ : ℝ) - allowance) ^ 2 / (2 * noiseProxy)))
        ≤ 2 * (1 / 2) ^ 130 := by gcongr
      _ = (1 / 2) ^ 129 := by norm_num
  rw [← ENNReal.ofReal_add (Real.exp_pos _).le (by positivity),
    show (2 : ENNReal)⁻¹ ^ 128 = ENNReal.ofReal ((1 / 2) ^ 128) by
      rw [ENNReal.ofReal_pow (by norm_num), one_div, ENNReal.ofReal_inv_of_pos two_pos]
      simp]
  apply ENNReal.ofReal_le_ofReal
  calc Real.exp (-(36864 / 315)) + 2 * Real.exp (-((Δ : ℝ) - allowance) ^ 2 / (2 * noiseProxy))
      ≤ (1 / 2) ^ 129 + (1 / 2) ^ 129 := add_le_add ha hb
    _ = (1 / 2) ^ 128 := by norm_num

/-- The generated claim: for every hash model and external input, the tapes with a failing run
have measure at most `2^-128`. -/
theorem correctness : CorrectnessClaim := by
  intro hashModel external
  by_cases hex : ∃ t0 x0, Runs hashModel external t0 x0
  · obtain ⟨t0, x0, h0⟩ := hex
    let W : World := ⟨x0.«stage_1».1, x0.«stage_2».1, external.input_1, external.input_3⟩
    have hsub : {tape | ∃ execution, Runs hashModel external tape execution ∧
        ¬((∀ index, (observedResidual execution index).natAbs <
          TfheSemantics.decoderRadius 4294967296) ∧ execution.«stage_4».1 = execution.«ideal»)} ⊆
        {t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j} ∪
          {t | (Δ : ℝ) - allowance ≤ |noiseY W t|} := by
      rintro t ⟨x, hx, hfail⟩
      by_contra hc
      simp only [Set.mem_union, Set.mem_setOf_eq, not_or, not_le] at hc
      apply hfail
      exact run_correct hx W (encrypt_left_mask_eq hx.2.2.1 h0.2.2.1)
        (encrypt_right_mask_eq hx.2.2.2.1 h0.2.2.2.1) rfl rfl (by omega) hc.2
    calc tapeMeasure _ ≤ tapeMeasure ({t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j} ∪
          {t | (Δ : ℝ) - allowance ≤ |noiseY W t|}) := measure_mono hsub
      _ ≤ tapeMeasure {t | (507 : ℤ) ≤ ∑ j : Fin lweN, lweSecret t j} +
          tapeMeasure {t | (Δ : ℝ) - allowance ≤ |noiseY W t|} := measure_union_le _ _
      _ ≤ ENNReal.ofReal (Real.exp (-(36864 / 315))) +
          ENNReal.ofReal (2 * Real.exp (-((Δ : ℝ) - allowance) ^ 2 / (2 * noiseProxy))) :=
        add_le_add hamming_tail (noiseY_tail W (by simp only [allowance, Δ]; norm_num))
      _ ≤ (2 : ENNReal)⁻¹ ^ 128 := final_numeric
  · have hempty : {tape | ∃ execution, Runs hashModel external tape execution ∧
        ¬((∀ index, (observedResidual execution index).natAbs <
          TfheSemantics.decoderRadius 4294967296) ∧ execution.«stage_4».1 = execution.«ideal»)} =
        ∅ := by
      ext t
      simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false, not_exists, not_and]
      intro x hx
      exact absurd ⟨t, x, hx⟩ hex
    rw [hempty, measure_empty]
    exact zero_le _

end MxxFheTfhe
