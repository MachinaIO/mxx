import LwrLut
import LwrEncrypt
import LwrInputs

/-!
# The worst-case correctness certificate

Every execution satisfying the generated claim's run relation decodes the parity of the readout of
its selected coordinates. The requirement makes the plaintext of member `k` of `a` the monomial
`X^(2 a_k)` and every block's selectors one-hot; the five blocks then accumulate `X^(2z)` for the sum
`z` of the selected coordinates, the readout LUT gives `(-1)^⌊z/16⌋ τ(z mod 16)` at the terminal
modulus, and decryption returns its parity, which is the ideal output.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-! ## Plaintexts of the run -/

/-- A one-hot family selects one cell. -/
theorem one_hot {s : Fin 14 → Int} (h01 : ∀ i, 0 ≤ s i ∧ s i ≤ 1) (hsum : ∑ i, s i = 1) :
    ∃ sel : Fin 14, ∀ i, s i = if i = sel then 1 else 0 := by
  have hex : ∃ sel, s sel = 1 := by
    by_contra hne
    push_neg at hne
    have hzero : ∀ i, s i = 0 := fun i ↦ by have := h01 i; have := hne i; omega
    simp [hzero] at hsum
  obtain ⟨sel, hsel⟩ := hex
  refine ⟨sel, fun i ↦ ?_⟩
  split
  · rename_i hi
    rw [hi, hsel]
  · rename_i hi
    by_contra hne
    have hi1 : s i = 1 := by have := h01 i; omega
    have hpair : ∑ j ∈ ({i, sel} : Finset (Fin 14)), s j ≤ ∑ j, s j :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun j _ _ ↦ (h01 j).1)
    rw [Finset.sum_pair hi, hi1, hsel, hsum] at hpair
    omega

/-- The block sum with one-hot selectors is the selected cell. -/
theorem collapse {q m p : Nat} (h : ExactMatrix q N 1 m) (sp μ : ErrorPoly N)
    (α ν : Fin 14 → ErrorPoly N) (s : Fin 14 → Int) (sel : Fin 14)
    (hs : ∀ i, s i = if i = sel then 1 else 0) (hν : ∀ i, ν i = (s i : ErrorPoly N)) :
    (∑ i : Fin 14, fun j ↦ reducePoly q N (((p : Nat) : Int) * (α i * μ * ν i) * unitPair sp j) • h) =
      fun j ↦ reducePoly q N (((p : Nat) : Int) * (α sel * μ) * unitPair sp j) • h := by
  funext j
  rw [Finset.sum_apply, Finset.sum_eq_single sel]
  · simp [hν, hs]
  · intro i _ hi
    simp [hν, hs, hi]
  · simp

/-- A one-hot block selects one coordinate. -/
theorem one_hot_sum (a : Fin 17 → Int) (b : Fin 5) {s : Fin 14 → Int} {sel : Fin 14}
    (hs : ∀ i, s i = if i = sel then 1 else 0) :
    ∑ i : Fin 14, s i * a ⟨coords b i, coords_lt b i⟩ = a ⟨coords b sel, coords_lt b sel⟩ := by
  rw [Finset.sum_eq_single sel]
  · simp [hs]
  · intro i _ hi
    simp [hs, hi]
  · simp

/-- The plaintext of a member of `a` accepted by the requirement is `X^(2a)`. -/
theorem message_member {c e a : Int} (hc : c * c = 1) (hrel : e * 2 + (1 - c) * 32 = a * 4)
    (he : 0 ≤ e ∧ e ≤ 31) : message (n := N) c e = Xe (2 * a.toNat) := by
  have hc' : c = 1 ∨ c = -1 := Int.eq_one_or_neg_one_of_mul_eq_one hc
  rcases hc' with rfl | rfl
  · exact message_one rfl (by omega) (by simp only [N]; omega)
  · rw [message, show ((2 * N : Nat) : Int) = 64 from rfl, Int.emod_eq_of_lt (by omega) (by omega)]
    have hsplit : 2 * a.toNat = e.toNat + 32 := by omega
    rw [Xe, hsplit, pow_add, Negacyclic.root_pow_n]
    push_cast
    ring

/-- A selector plaintext with exponent zero is the bit. -/
theorem message_selector {s x : Int} (hx : x = 0) : message (n := N) s x = (s : ErrorPoly N) := by
  subst hx
  simp [message]

theorem Xe_mul (a b : Nat) : Xe (n := N) a * Xe b = Xe (a + b) := by
  rw [Xe, Xe, Xe, pow_add]

theorem intCast_coeff_zero (x : Int) : ((x : ErrorPoly N)).coeff ⟨0, hN⟩ = x := by
  have h := intCast_mul_coeff x (1 : ErrorPoly N) ⟨0, hN⟩
  rw [mul_one] at h
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := N) hN ⟨0, hN⟩ ⟨0, hN⟩
  rw [show ((⟨0, hN⟩ : Fin N) : Nat) = 0 from rfl, pow_zero, if_pos rfl] at h1
  rw [h, h1, mul_one]

/-- The half-integer readout modulo `8` is the ideal value: `m` for an even `⌊z / 16⌋` and
`8 - m` for an odd one, with `m = 2 τ(z mod 16) - 1`. -/
theorem readout_mod8 (z : Nat) :
    ((lutValue z).coeff ⟨0, hN⟩) % 8 =
      ((((z : Int) % 16) * 2 / 16 + 1) * 2 - 1) +
        (z : Int) % 32 / 16 * (8 - ((((z : Int) % 16) * 2 / 16 + 1) * 2 - 1) * 2) := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.halfIntegerReadout, MxxExponentLutLwr.readout]
  have hτ : ((2 * (2 * (z % 16) / 16 + 1) - 1 : Nat) : Int) =
      (((z : Int) % 16) * 2 / 16 + 1) * 2 - 1 := by omega
  rw [hτ]
  have hb : (z : Int) % 32 / 16 = ((z / 16 : Nat) : Int) % 2 := by omega
  rw [hb]
  rcases Nat.even_or_odd (z / 16) with h | h
  · rw [h.neg_one_pow, one_mul, show ((z / 16 : Nat) : Int) % 2 = 0 by
      obtain ⟨k, hk⟩ := h; omega]
    omega
  · rw [h.neg_one_pow, neg_one_mul, show ((z / 16 : Nat) : Int) % 2 = 1 by
      obtain ⟨k, hk⟩ := h; omega]
    omega

theorem lutValue_coeff_bound (z : Nat) : |(lutValue z).coeff ⟨0, hN⟩| ≤ 3 := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.halfIntegerReadout, MxxExponentLutLwr.readout]
  have hτ : 2 * (2 * (z % 16) / 16 + 1) - 1 ≤ 3 := by omega
  rcases Nat.even_or_odd (z / 16) with h | h
  · rw [h.neg_one_pow, one_mul, abs_le]; omega
  · rw [h.neg_one_pow, neg_one_mul, abs_le]; omega

theorem secret_ternary {out : ExactMatrix Q7 N 1 1}
    (h : Stage_payload.generatedRoot GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q7 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

theorem mask_ternary {out : ExactMatrix Q7 N 1 1}
    (h : Stage_mask.generatedRoot GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q7 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly hw
  exact ⟨s, hout.trans hs, hn⟩

set_option maxRecDepth 100000 in
set_option maxHeartbeats 64000000 in
/-- Every run decodes the half-integer readout of its selected coordinates modulo `8`. -/
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution} (hruns : GeneratedClaim.Runs model external execution) :
    execution.«stage_5» = execution.«ideal» := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honlineEval, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ha, hac, hae, hs0, hx0, hs1, hx1, hs2, hx2, hs3, hx3, hs4, hx4⟩ :=
    hvalid
  obtain ⟨hmem, hsum0, hsum1, hsum2, hsum3, hsum4⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
  set a := external.input_3
  set s0 := external.input_6
  set s1 := external.input_8
  set s2 := external.input_10
  set s3 := external.input_12
  set s4 := external.input_14
  set x0 := external.input_7
  set x1 := external.input_9
  set x2 := external.input_11
  set x3 := external.input_13
  set x4 := external.input_15
  -- The selected cells and the plaintexts.
  obtain ⟨sel0, hsel0⟩ := one_hot hs0 hsum0
  obtain ⟨sel1, hsel1⟩ := one_hot hs1 hsum1
  obtain ⟨sel2, hsel2⟩ := one_hot hs2 hsum2
  obtain ⟨sel3, hsel3⟩ := one_hot hs3 hsum3
  obtain ⟨sel4, hsel4⟩ := one_hot hs4 hsum4
  let α : Fin 17 → ErrorPoly N := fun k ↦ Xe (2 * (a k).toNat)
  have hα : ∀ k, message (n := N) (external.input_4 k) (external.input_5 k) = α k := fun k ↦
    message_member (hmem k).1 (hmem k).2 (hae k)
  have hν0 : ∀ i, message (n := N) (s0 i) (x0 i) = (s0 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx0 i; omega)
  have hν1 : ∀ i, message (n := N) (s1 i) (x1 i) = (s1 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx1 i; omega)
  have hν2 : ∀ i, message (n := N) (s2 i) (x2 i) = (s2 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx2 i; omega)
  have hν3 : ∀ i, message (n := N) (s3 i) (x3 i) = (s3 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx3 i; omega)
  have hν4 : ∀ i, message (n := N) (s4 i) (x4 i) = (s4 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx4 i; omega)
  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
  -- The gadget matrices.
  obtain ⟨G14, hG14⟩ := gadget_exists (m := 14) layout7 rfl rfl rfl
  obtain ⟨g12, hg12⟩ := gadget_exists (m := 12) layout6 rfl rfl rfl
  obtain ⟨g10, hg10⟩ := gadget_exists (m := 10) layout5 rfl rfl rfl
  obtain ⟨g8, hg8⟩ := gadget_exists (m := 8) layout4 rfl rfl rfl
  obtain ⟨g6, hg6⟩ := gadget_exists (m := 6) layout3 rfl rfl rfl
  obtain ⟨g4, hg4⟩ := gadget_exists (m := 4) layout2 rfl rfl rfl
  obtain ⟨g2, hg2⟩ := gadget_exists (m := 2) layout1 rfl rfl rfl
  have hg12u : centeredRebaseRuns g12 (centeredRebase g12 : ExactMatrix Q7 N 1 12) := ⟨hQ7, rfl⟩
  have hg10u : centeredRebaseRuns g10 (centeredRebase g10 : ExactMatrix Q6 N 1 10) := ⟨hQ6, rfl⟩
  have hg8u : centeredRebaseRuns g8 (centeredRebase g8 : ExactMatrix Q5 N 1 8) := ⟨hQ5, rfl⟩
  have hg6u : centeredRebaseRuns g6 (centeredRebase g6 : ExactMatrix Q4 N 1 6) := ⟨hQ4, rfl⟩
  have hg4u : centeredRebaseRuns g4 (centeredRebase g4 : ExactMatrix Q3 N 1 4) := ⟨hQ3, rfl⟩
  have hg2u : centeredRebaseRuns g2 (centeredRebase g2 : ExactMatrix Q2 N 1 2) := ⟨hQ2, rfl⟩
  -- Encryption and preprocessing.
  have hstate := encrypt_state henc hsm hsp hg12 hg12u
  rw [hμ0] at hstate
  have hmembers := encrypt_member henc hsm hG14
  have hmemα : ∀ k : Fin 17, Rel sm (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q7 N (α k * unitPair sm j) • G14)
      ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] 26 := fun k ↦ by
    rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin 17, execution.«stage_3».2.2.2.2.1 k = fun _ _ ↦ reducePoly Q7 N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  have hhelp := encrypt_helper5 henc hsm hg2 hg2u
  have hrhs0 := preprocess_selector0 hpre hsp hg10 hg10u
  have hrhs1 := preprocess_selector1 hpre hsp hg8 hg8u
  have hrhs2 := preprocess_selector2 hpre hsp hg6 hg6u
  have hrhs3 := preprocess_selector3 hpre hsp hg4 hg4u
  have hrhs4 := preprocess_selector4 hpre hsp hg2 hg2u
  have hsw := preprocess_switch hpre hsp hg2
  -- The five blocks accumulate the monomial of the sum of the selected coordinates.
  obtain ⟨w, hb⟩ := honlineEval
  -- Block 0.
  have hb0 := block0 hb hsmn (μ := Xe 0) (α := α) (ν := fun i ↦ message (n := N) (s0 i) (x0 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg12 hg12u hmemα
    hpayα hstate hrhs0
  have hin1 := hb0.target_eq (collapse _ _ _ _ _ _ sel0 hsel0 hν0)
  rw [Xe_mul] at hin1
  obtain ⟨z1, hz1⟩ : ∃ z : Nat, z = (a ⟨coords 0 sel0, coords_lt 0 sel0⟩).toNat + 0 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 0 sel0, coords_lt 0 sel0⟩).toNat + 0 = 2 * z1 by
    rw [hz1]; ring] at hin1
  -- Block 1.
  have hb1 := block1 hb hsmn (μ := Xe (2 * z1)) (α := α) (ν := fun i ↦ message (n := N) (s1 i) (x1 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg10 hg10u hmemα
    hpayα hin1 hrhs1
  have hin2 := hb1.target_eq (collapse _ _ _ _ _ _ sel1 hsel1 hν1)
  rw [Xe_mul] at hin2
  obtain ⟨z2, hz2⟩ : ∃ z : Nat, z = (a ⟨coords 1 sel1, coords_lt 1 sel1⟩).toNat + z1 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 1 sel1, coords_lt 1 sel1⟩).toNat + 2 * z1 = 2 * z2 by
    rw [hz2]; ring] at hin2
  -- Block 2.
  have hb2 := block2 hb hsmn (μ := Xe (2 * z2)) (α := α) (ν := fun i ↦ message (n := N) (s2 i) (x2 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg8 hg8u hmemα
    hpayα hin2 hrhs2
  have hin3 := hb2.target_eq (collapse _ _ _ _ _ _ sel2 hsel2 hν2)
  rw [Xe_mul] at hin3
  obtain ⟨z3, hz3⟩ : ∃ z : Nat, z = (a ⟨coords 2 sel2, coords_lt 2 sel2⟩).toNat + z2 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 2 sel2, coords_lt 2 sel2⟩).toNat + 2 * z2 = 2 * z3 by
    rw [hz3]; ring] at hin3
  -- Block 3.
  have hb3 := block3 hb hsmn (μ := Xe (2 * z3)) (α := α) (ν := fun i ↦ message (n := N) (s3 i) (x3 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg6 hg6u hmemα
    hpayα hin3 hrhs3
  have hin4 := hb3.target_eq (collapse _ _ _ _ _ _ sel3 hsel3 hν3)
  rw [Xe_mul] at hin4
  obtain ⟨z4, hz4⟩ : ∃ z : Nat, z = (a ⟨coords 3 sel3, coords_lt 3 sel3⟩).toNat + z3 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 3 sel3, coords_lt 3 sel3⟩).toNat + 2 * z3 = 2 * z4 by
    rw [hz4]; ring] at hin4
  -- Block 4.
  have hb4 := block4 hb hsmn (μ := Xe (2 * z4)) (α := α) (ν := fun i ↦ message (n := N) (s4 i) (x4 i))
    (fun i ↦ by rw [Xe_mul]; exact polyNorm_Xe hN1 _) (fun k ↦ polyNorm_Xe hN1 _) hG14 hg4 hg4u hmemα
    hpayα hin4 hrhs4
  have hin5 := hb4.target_eq (collapse _ _ _ _ _ _ sel4 hsel4 hν4)
  rw [Xe_mul] at hin5
  obtain ⟨z5, hz5⟩ : ∃ z : Nat, z = (a ⟨coords 4 sel4, coords_lt 4 sel4⟩).toNat + z4 := ⟨_, rfl⟩
  rw [show 2 * (a ⟨coords 4 sel4, coords_lt 4 sel4⟩).toNat + 2 * z4 = 2 * z5 by
    rw [hz5]; ring] at hin5
  -- The readout LUT and the decryption.
  have hlut := lutRound hb hsmn hg2 hg2u hin5 hhelp hsw
  have hb2 := hb
  unfold Stage_online_eval.generatedRoot.body at hb2
  extract_lets at hb2
  have houtOnlineEval := hb2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨wd, hd⟩ := hdec
  unfold Stage_decode.generatedRoot.body at hd
  dsimp only at hd
  obtain ⟨hm1, hm2, -, hdec1, hout⟩ := hd
  rw [hsm] at hm1
  have hw2 := modulusReduce_poly hQ7 hQ1 hN hm1
  have hw6 := modulusReduce_one hQ7 hQ1 hN hm2
  have hbound : 2 * 8 * outBound (∑ _i : Fin 14, cellBound4 (∑ _i : Fin 14, cellBound3
      (∑ _i : Fin 14, cellBound2 (∑ _i : Fin 14, cellBound1 (∑ _i : Fin 14, cellBound0 26))))) +
      2 * 2147481921 * 3 < Q1 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    decide
  have hval := decode_terminal_scaled (t := 8) (k := 137170518016) (r := 2147481921) (W := 3)
    hQ1 hN (by norm_num) (by norm_num) hlut hbound (i := ⟨0, hN⟩) (lutValue_coeff_bound z5) (by
      rw [houtOnlineEval, hw2, hw6]
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _
      simp only [matrixNeg, matrixSub, matrixAdd, matrixMulScalarLeft_eq, Matrix.neg_apply,
        Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq,
        one_mul]
      rfl) hdec1
  rw [hout, hval, Nat.cast_ofNat, readout_mod8, hid]
  have hz : selectedSum a ![s0, s1, s2, s3, s4] = (z5 : Int) := by
    rw [selectedSum, Fin.sum_univ_five,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 0 = s0 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 1 = s1 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 2 = s2 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 3 = s3 from rfl,
      show (![s0, s1, s2, s3, s4] : Fin 5 → Fin 14 → Int) 4 = s4 from rfl]
    rw [one_hot_sum a 0 hsel0, one_hot_sum a 1 hsel1, one_hot_sum a 2 hsel2, one_hot_sum a 3 hsel3,
      one_hot_sum a 4 hsel4, hz5, hz4, hz3, hz2, hz1]
    push_cast
    rw [Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1]
    ring
  rw [hz]

/-- The worst-case correctness claim of `worst-n32-crt7x40`. -/
theorem certificate : GeneratedClaim.CorrectnessClaim :=
  fun _ _ _ hruns ↦ correct hruns

/-- info: 'LwrProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end LwrProof
