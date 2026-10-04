import LwrBlock0
import LwrBlock1
import LwrBlock2
import LwrBlock3
import LwrBlock4
import LwrBlock5
import LwrBlock6
import LwrBlock7
import LwrBlock8
import LwrBlock9
import LwrBlock10
import LwrBlock11
import LwrBlock12
import LwrBlock13
import LwrBlock14
import LwrBlock15
import LwrBlock16
import LwrBlock17
import LwrBlock18
import LwrBlock19
import LwrBlock20
import LwrBlock21
import LwrBlock22
import LwrBlock23
import LwrBlock24
import LwrBlock25
import LwrBlock26
import LwrBlock27
import LwrBlock28
import LwrBlock29
import LwrBlock30
import LwrBlock31
import LwrBlock32
import LwrBlock33
import LwrLut
import LwrEncrypt
import LwrPreprocess
import LwrInputs

/-!
# The average-case correctness certificate

Every execution on every sample tape satisfying the generated claim's run relation decodes the
parity of the readout of its selected coordinates. The requirement makes the plaintext of member
`k` of `a` the monomial `X^(512 a_k)` and every block's selectors one-hot; the 34 blocks then
accumulate `X^(512z)` for the sum `z` of the selected coordinates, the readout LUT gives
`(-1)^⌊z/128⌋ τ(z mod 128)` at the terminal modulus, and decryption returns its parity, which is the
ideal output. The plaintexts of the members of `a` and of the state are monomials, so multiplying an
error by them keeps its bound (`SmulGain _ 1`).
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {tape : SampleTape} {path : List Nat}

/-! ## Plaintexts of the run -/

/-- A one-hot family selects one cell. -/
theorem one_hot {s : Fin 113 → Int} (h01 : ∀ i, 0 ≤ s i ∧ s i ≤ 1) (hsum : ∑ i, s i = 1) :
    ∃ sel : Fin 113, ∀ i, s i = if i = sel then 1 else 0 := by
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
    have hpair : ∑ j ∈ ({i, sel} : Finset (Fin 113)), s j ≤ ∑ j, s j :=
      Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
        (fun j _ _ ↦ (h01 j).1)
    rw [Finset.sum_pair hi, hi1, hsel, hsum] at hpair
    omega

/-- The block sum with one-hot selectors is the selected cell. -/
theorem collapse {q m p : Nat} (h : ExactMatrix q N 1 m) (sp μ : ErrorPoly N)
    (α ν : Fin 113 → ErrorPoly N) (s : Fin 113 → Int) (sel : Fin 113)
    (hs : ∀ i, s i = if i = sel then 1 else 0) (hν : ∀ i, ν i = (s i : ErrorPoly N)) :
    (∑ i : Fin 113, fun j ↦ reducePoly q N (((p : Nat) : Int) * (α i * μ * ν i) * unitPair sp j) • h) =
      fun j ↦ reducePoly q N (((p : Nat) : Int) * (α sel * μ) * unitPair sp j) • h := by
  funext j
  rw [Finset.sum_apply, Finset.sum_eq_single sel]
  · simp [hν, hs]
  · intro i _ hi
    simp [hν, hs, hi]
  · simp

/-- A one-hot block selects one coordinate. -/
theorem one_hot_sum (a : Fin 1025 → Int) (b : Fin 34) {s : Fin 113 → Int} {sel : Fin 113}
    (hs : ∀ i, s i = if i = sel then 1 else 0) :
    ∑ i : Fin 113, s i * a ⟨coords b i, coords_lt b i⟩ = a ⟨coords b sel, coords_lt b sel⟩ := by
  rw [Finset.sum_eq_single sel]
  · simp [hs]
  · intro i _ hi
    simp [hs, hi]
  · simp

/-- The plaintext of a member of `a` accepted by the requirement is `X^(512a)`. -/
theorem message_member {c e a : Int} (hc : c * c = 1) (hrel : e * 2 + (1 - c) * 65536 = a * 1024)
    (he : 0 ≤ e ∧ e ≤ 65535) : message (n := N) c e = Xe (512 * a.toNat) := by
  have hc' : c = 1 ∨ c = -1 := Int.eq_one_or_neg_one_of_mul_eq_one hc
  rcases hc' with rfl | rfl
  · exact message_one rfl (by omega) (by simp only [N]; omega)
  · rw [message, show ((2 * N : Nat) : Int) = 131072 from rfl, Int.emod_eq_of_lt (by omega) (by omega)]
    have hsplit : 512 * a.toNat = e.toNat + 65536 := by omega
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

/-- The decrypted parity of the readout is the parity of `τ(z mod 128)`. -/
theorem readout_parity (z : Nat) :
    ((lutValue z).coeff ⟨0, hN⟩) % 2 = (((z : Int) % 128) * 16 / 128 + 1) % 2 := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.readout]
  have hτ : ((16 * (z % 128) / 128 + 1 : Nat) : Int) = ((z : Int) % 128) * 16 / 128 + 1 := by omega
  rw [hτ]
  rcases Nat.even_or_odd (z / 128) with h | h
  · rw [h.neg_one_pow, one_mul]
  · rw [h.neg_one_pow, neg_one_mul, Int.neg_emod_two]

/-- The readout lies in `[-16, 16]`. -/
theorem lutValue_coeff_bound (z : Nat) : |(lutValue z).coeff ⟨0, hN⟩| ≤ (16 : Nat) := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.readout]
  have hτ : 16 * (z % 128) / 128 + 1 ≤ 16 := by omega
  rcases Nat.even_or_odd (z / 128) with h | h
  · rw [h.neg_one_pow, one_mul, abs_of_nonneg (by positivity)]; omega
  · rw [h.neg_one_pow, neg_one_mul, abs_neg, abs_of_nonneg (by positivity)]; omega

theorem secret_ternary {out : ExactMatrix Q36 N 1 1}
    (h : Stage_payload.generatedRoot tape path GeneratedClaim.stage_0_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q36 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly_at hN hw
  exact ⟨s, hout.trans hs, hn⟩

theorem mask_ternary {out : ExactMatrix Q36 N 1 1}
    (h : Stage_mask.generatedRoot tape path GeneratedClaim.stage_1_params () out) :
    ∃ s : ErrorPoly N, out = (fun _ _ ↦ reducePoly Q36 N s) ∧ polyNorm s ≤ 1 := by
  obtain ⟨w, hw, hout⟩ := h
  obtain ⟨s, hs, hn⟩ := ternary_poly_at hN hw
  exact ⟨s, hout.trans hs, hn⟩

set_option maxHeartbeats 0 in
/-- Every run decodes the parity of the readout of its selected coordinates. -/
theorem correct {model : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (hruns : GeneratedClaim.Runs model external tape execution) :
    execution.«stage_5» = execution.«ideal» := by
  obtain ⟨hvalid, hpay, hmask, hpre, henc, honlineEval, hdec, hreq, hideal, hreqtrue⟩ := hruns
  obtain ⟨sp, hsp, hspn⟩ := secret_ternary hpay
  obtain ⟨sm, hsm, hsmn⟩ := mask_ternary hmask
  obtain ⟨-, ⟨hc1, hc2⟩, ⟨he1, he2⟩, ha, hac, hae, hs0, hx0, hs1, hx1, hs2, hx2, hs3, hx3, hs4, hx4,
    hs5, hx5, hs6, hx6, hs7, hx7, hs8, hx8, hs9, hx9, hs10, hx10, hs11, hx11, hs12, hx12,
    hs13, hx13, hs14, hx14, hs15, hx15, hs16, hx16, hs17, hx17, hs18, hx18, hs19, hx19, hs20, hx20,
    hs21, hx21, hs22, hx22, hs23, hx23, hs24, hx24, hs25, hx25, hs26, hx26, hs27, hx27, hs28, hx28,
    hs29, hx29, hs30, hx30, hs31, hx31, hs32, hx32, hs33, hx33⟩ := hvalid
  obtain ⟨hmem, hsum0, hsum1, hsum2, hsum3, hsum4, hsum5, hsum6, hsum7, hsum8, hsum9, hsum10,
    hsum11, hsum12, hsum13, hsum14, hsum15, hsum16, hsum17, hsum18, hsum19, hsum20, hsum21, hsum22,
    hsum23, hsum24, hsum25, hsum26, hsum27, hsum28, hsum29, hsum30, hsum31, hsum32, hsum33⟩ := requirement_holds hreq hreqtrue
  have hid := ideal_value hideal
  set a := external.input_3
  set s0 := external.input_6
  set s1 := external.input_8
  set s2 := external.input_10
  set s3 := external.input_12
  set s4 := external.input_14
  set s5 := external.input_16
  set s6 := external.input_18
  set s7 := external.input_20
  set s8 := external.input_22
  set s9 := external.input_24
  set s10 := external.input_26
  set s11 := external.input_28
  set s12 := external.input_30
  set s13 := external.input_32
  set s14 := external.input_34
  set s15 := external.input_36
  set s16 := external.input_38
  set s17 := external.input_40
  set s18 := external.input_42
  set s19 := external.input_44
  set s20 := external.input_46
  set s21 := external.input_48
  set s22 := external.input_50
  set s23 := external.input_52
  set s24 := external.input_54
  set s25 := external.input_56
  set s26 := external.input_58
  set s27 := external.input_60
  set s28 := external.input_62
  set s29 := external.input_64
  set s30 := external.input_66
  set s31 := external.input_68
  set s32 := external.input_70
  set s33 := external.input_72
  set x0 := external.input_7
  set x1 := external.input_9
  set x2 := external.input_11
  set x3 := external.input_13
  set x4 := external.input_15
  set x5 := external.input_17
  set x6 := external.input_19
  set x7 := external.input_21
  set x8 := external.input_23
  set x9 := external.input_25
  set x10 := external.input_27
  set x11 := external.input_29
  set x12 := external.input_31
  set x13 := external.input_33
  set x14 := external.input_35
  set x15 := external.input_37
  set x16 := external.input_39
  set x17 := external.input_41
  set x18 := external.input_43
  set x19 := external.input_45
  set x20 := external.input_47
  set x21 := external.input_49
  set x22 := external.input_51
  set x23 := external.input_53
  set x24 := external.input_55
  set x25 := external.input_57
  set x26 := external.input_59
  set x27 := external.input_61
  set x28 := external.input_63
  set x29 := external.input_65
  set x30 := external.input_67
  set x31 := external.input_69
  set x32 := external.input_71
  set x33 := external.input_73
  -- The selected cells and the plaintexts.
  obtain ⟨sel0, hsel0⟩ := one_hot hs0 hsum0
  obtain ⟨sel1, hsel1⟩ := one_hot hs1 hsum1
  obtain ⟨sel2, hsel2⟩ := one_hot hs2 hsum2
  obtain ⟨sel3, hsel3⟩ := one_hot hs3 hsum3
  obtain ⟨sel4, hsel4⟩ := one_hot hs4 hsum4
  obtain ⟨sel5, hsel5⟩ := one_hot hs5 hsum5
  obtain ⟨sel6, hsel6⟩ := one_hot hs6 hsum6
  obtain ⟨sel7, hsel7⟩ := one_hot hs7 hsum7
  obtain ⟨sel8, hsel8⟩ := one_hot hs8 hsum8
  obtain ⟨sel9, hsel9⟩ := one_hot hs9 hsum9
  obtain ⟨sel10, hsel10⟩ := one_hot hs10 hsum10
  obtain ⟨sel11, hsel11⟩ := one_hot hs11 hsum11
  obtain ⟨sel12, hsel12⟩ := one_hot hs12 hsum12
  obtain ⟨sel13, hsel13⟩ := one_hot hs13 hsum13
  obtain ⟨sel14, hsel14⟩ := one_hot hs14 hsum14
  obtain ⟨sel15, hsel15⟩ := one_hot hs15 hsum15
  obtain ⟨sel16, hsel16⟩ := one_hot hs16 hsum16
  obtain ⟨sel17, hsel17⟩ := one_hot hs17 hsum17
  obtain ⟨sel18, hsel18⟩ := one_hot hs18 hsum18
  obtain ⟨sel19, hsel19⟩ := one_hot hs19 hsum19
  obtain ⟨sel20, hsel20⟩ := one_hot hs20 hsum20
  obtain ⟨sel21, hsel21⟩ := one_hot hs21 hsum21
  obtain ⟨sel22, hsel22⟩ := one_hot hs22 hsum22
  obtain ⟨sel23, hsel23⟩ := one_hot hs23 hsum23
  obtain ⟨sel24, hsel24⟩ := one_hot hs24 hsum24
  obtain ⟨sel25, hsel25⟩ := one_hot hs25 hsum25
  obtain ⟨sel26, hsel26⟩ := one_hot hs26 hsum26
  obtain ⟨sel27, hsel27⟩ := one_hot hs27 hsum27
  obtain ⟨sel28, hsel28⟩ := one_hot hs28 hsum28
  obtain ⟨sel29, hsel29⟩ := one_hot hs29 hsum29
  obtain ⟨sel30, hsel30⟩ := one_hot hs30 hsum30
  obtain ⟨sel31, hsel31⟩ := one_hot hs31 hsum31
  obtain ⟨sel32, hsel32⟩ := one_hot hs32 hsum32
  obtain ⟨sel33, hsel33⟩ := one_hot hs33 hsum33
  let α : Fin 1025 → ErrorPoly N := fun k ↦ Xe (512 * (a k).toNat)
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
  have hν5 : ∀ i, message (n := N) (s5 i) (x5 i) = (s5 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx5 i; omega)
  have hν6 : ∀ i, message (n := N) (s6 i) (x6 i) = (s6 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx6 i; omega)
  have hν7 : ∀ i, message (n := N) (s7 i) (x7 i) = (s7 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx7 i; omega)
  have hν8 : ∀ i, message (n := N) (s8 i) (x8 i) = (s8 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx8 i; omega)
  have hν9 : ∀ i, message (n := N) (s9 i) (x9 i) = (s9 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx9 i; omega)
  have hν10 : ∀ i, message (n := N) (s10 i) (x10 i) = (s10 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx10 i; omega)
  have hν11 : ∀ i, message (n := N) (s11 i) (x11 i) = (s11 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx11 i; omega)
  have hν12 : ∀ i, message (n := N) (s12 i) (x12 i) = (s12 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx12 i; omega)
  have hν13 : ∀ i, message (n := N) (s13 i) (x13 i) = (s13 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx13 i; omega)
  have hν14 : ∀ i, message (n := N) (s14 i) (x14 i) = (s14 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx14 i; omega)
  have hν15 : ∀ i, message (n := N) (s15 i) (x15 i) = (s15 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx15 i; omega)
  have hν16 : ∀ i, message (n := N) (s16 i) (x16 i) = (s16 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx16 i; omega)
  have hν17 : ∀ i, message (n := N) (s17 i) (x17 i) = (s17 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx17 i; omega)
  have hν18 : ∀ i, message (n := N) (s18 i) (x18 i) = (s18 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx18 i; omega)
  have hν19 : ∀ i, message (n := N) (s19 i) (x19 i) = (s19 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx19 i; omega)
  have hν20 : ∀ i, message (n := N) (s20 i) (x20 i) = (s20 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx20 i; omega)
  have hν21 : ∀ i, message (n := N) (s21 i) (x21 i) = (s21 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx21 i; omega)
  have hν22 : ∀ i, message (n := N) (s22 i) (x22 i) = (s22 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx22 i; omega)
  have hν23 : ∀ i, message (n := N) (s23 i) (x23 i) = (s23 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx23 i; omega)
  have hν24 : ∀ i, message (n := N) (s24 i) (x24 i) = (s24 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx24 i; omega)
  have hν25 : ∀ i, message (n := N) (s25 i) (x25 i) = (s25 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx25 i; omega)
  have hν26 : ∀ i, message (n := N) (s26 i) (x26 i) = (s26 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx26 i; omega)
  have hν27 : ∀ i, message (n := N) (s27 i) (x27 i) = (s27 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx27 i; omega)
  have hν28 : ∀ i, message (n := N) (s28 i) (x28 i) = (s28 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx28 i; omega)
  have hν29 : ∀ i, message (n := N) (s29 i) (x29 i) = (s29 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx29 i; omega)
  have hν30 : ∀ i, message (n := N) (s30 i) (x30 i) = (s30 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx30 i; omega)
  have hν31 : ∀ i, message (n := N) (s31 i) (x31 i) = (s31 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx31 i; omega)
  have hν32 : ∀ i, message (n := N) (s32 i) (x32 i) = (s32 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx32 i; omega)
  have hν33 : ∀ i, message (n := N) (s33 i) (x33 i) = (s33 i : ErrorPoly N) := fun i ↦
    message_selector (by have := hx33 i; omega)
  have hμ0 : message (n := N) external.input_1 external.input_2 = Xe 0 :=
    message_one (k := 0) (by omega) (by simp; omega) (by simp only [N]; omega)
  -- The gadget matrices.
  obtain ⟨g144, hg144⟩ := gadget_exists (m := 144) layout36 rfl rfl rfl
  obtain ⟨g140, hg140⟩ := gadget_exists (m := 140) layout35 rfl rfl rfl
  obtain ⟨g136, hg136⟩ := gadget_exists (m := 136) layout34 rfl rfl rfl
  obtain ⟨g132, hg132⟩ := gadget_exists (m := 132) layout33 rfl rfl rfl
  obtain ⟨g128, hg128⟩ := gadget_exists (m := 128) layout32 rfl rfl rfl
  obtain ⟨g124, hg124⟩ := gadget_exists (m := 124) layout31 rfl rfl rfl
  obtain ⟨g120, hg120⟩ := gadget_exists (m := 120) layout30 rfl rfl rfl
  obtain ⟨g116, hg116⟩ := gadget_exists (m := 116) layout29 rfl rfl rfl
  obtain ⟨g112, hg112⟩ := gadget_exists (m := 112) layout28 rfl rfl rfl
  obtain ⟨g108, hg108⟩ := gadget_exists (m := 108) layout27 rfl rfl rfl
  obtain ⟨g104, hg104⟩ := gadget_exists (m := 104) layout26 rfl rfl rfl
  obtain ⟨g100, hg100⟩ := gadget_exists (m := 100) layout25 rfl rfl rfl
  obtain ⟨g96, hg96⟩ := gadget_exists (m := 96) layout24 rfl rfl rfl
  obtain ⟨g92, hg92⟩ := gadget_exists (m := 92) layout23 rfl rfl rfl
  obtain ⟨g88, hg88⟩ := gadget_exists (m := 88) layout22 rfl rfl rfl
  obtain ⟨g84, hg84⟩ := gadget_exists (m := 84) layout21 rfl rfl rfl
  obtain ⟨g80, hg80⟩ := gadget_exists (m := 80) layout20 rfl rfl rfl
  obtain ⟨g76, hg76⟩ := gadget_exists (m := 76) layout19 rfl rfl rfl
  obtain ⟨g72, hg72⟩ := gadget_exists (m := 72) layout18 rfl rfl rfl
  obtain ⟨g68, hg68⟩ := gadget_exists (m := 68) layout17 rfl rfl rfl
  obtain ⟨g64, hg64⟩ := gadget_exists (m := 64) layout16 rfl rfl rfl
  obtain ⟨g60, hg60⟩ := gadget_exists (m := 60) layout15 rfl rfl rfl
  obtain ⟨g56, hg56⟩ := gadget_exists (m := 56) layout14 rfl rfl rfl
  obtain ⟨g52, hg52⟩ := gadget_exists (m := 52) layout13 rfl rfl rfl
  obtain ⟨g48, hg48⟩ := gadget_exists (m := 48) layout12 rfl rfl rfl
  obtain ⟨g44, hg44⟩ := gadget_exists (m := 44) layout11 rfl rfl rfl
  obtain ⟨g40, hg40⟩ := gadget_exists (m := 40) layout10 rfl rfl rfl
  obtain ⟨g36, hg36⟩ := gadget_exists (m := 36) layout9 rfl rfl rfl
  obtain ⟨g32, hg32⟩ := gadget_exists (m := 32) layout8 rfl rfl rfl
  obtain ⟨g28, hg28⟩ := gadget_exists (m := 28) layout7 rfl rfl rfl
  obtain ⟨g24, hg24⟩ := gadget_exists (m := 24) layout6 rfl rfl rfl
  obtain ⟨g20, hg20⟩ := gadget_exists (m := 20) layout5 rfl rfl rfl
  obtain ⟨g16, hg16⟩ := gadget_exists (m := 16) layout4 rfl rfl rfl
  obtain ⟨g12, hg12⟩ := gadget_exists (m := 12) layout3 rfl rfl rfl
  obtain ⟨g8, hg8⟩ := gadget_exists (m := 8) layout2 rfl rfl rfl
  obtain ⟨g4, hg4⟩ := gadget_exists (m := 4) layout1 rfl rfl rfl
  have hg140u : centeredRebaseRuns g140 (centeredRebase g140 : ExactMatrix Q36 N 1 140) := ⟨hQ36, rfl⟩
  have hg136u : centeredRebaseRuns g136 (centeredRebase g136 : ExactMatrix Q35 N 1 136) := ⟨hQ35, rfl⟩
  have hg132u : centeredRebaseRuns g132 (centeredRebase g132 : ExactMatrix Q34 N 1 132) := ⟨hQ34, rfl⟩
  have hg128u : centeredRebaseRuns g128 (centeredRebase g128 : ExactMatrix Q33 N 1 128) := ⟨hQ33, rfl⟩
  have hg124u : centeredRebaseRuns g124 (centeredRebase g124 : ExactMatrix Q32 N 1 124) := ⟨hQ32, rfl⟩
  have hg120u : centeredRebaseRuns g120 (centeredRebase g120 : ExactMatrix Q31 N 1 120) := ⟨hQ31, rfl⟩
  have hg116u : centeredRebaseRuns g116 (centeredRebase g116 : ExactMatrix Q30 N 1 116) := ⟨hQ30, rfl⟩
  have hg112u : centeredRebaseRuns g112 (centeredRebase g112 : ExactMatrix Q29 N 1 112) := ⟨hQ29, rfl⟩
  have hg108u : centeredRebaseRuns g108 (centeredRebase g108 : ExactMatrix Q28 N 1 108) := ⟨hQ28, rfl⟩
  have hg104u : centeredRebaseRuns g104 (centeredRebase g104 : ExactMatrix Q27 N 1 104) := ⟨hQ27, rfl⟩
  have hg100u : centeredRebaseRuns g100 (centeredRebase g100 : ExactMatrix Q26 N 1 100) := ⟨hQ26, rfl⟩
  have hg96u : centeredRebaseRuns g96 (centeredRebase g96 : ExactMatrix Q25 N 1 96) := ⟨hQ25, rfl⟩
  have hg92u : centeredRebaseRuns g92 (centeredRebase g92 : ExactMatrix Q24 N 1 92) := ⟨hQ24, rfl⟩
  have hg88u : centeredRebaseRuns g88 (centeredRebase g88 : ExactMatrix Q23 N 1 88) := ⟨hQ23, rfl⟩
  have hg84u : centeredRebaseRuns g84 (centeredRebase g84 : ExactMatrix Q22 N 1 84) := ⟨hQ22, rfl⟩
  have hg80u : centeredRebaseRuns g80 (centeredRebase g80 : ExactMatrix Q21 N 1 80) := ⟨hQ21, rfl⟩
  have hg76u : centeredRebaseRuns g76 (centeredRebase g76 : ExactMatrix Q20 N 1 76) := ⟨hQ20, rfl⟩
  have hg72u : centeredRebaseRuns g72 (centeredRebase g72 : ExactMatrix Q19 N 1 72) := ⟨hQ19, rfl⟩
  have hg68u : centeredRebaseRuns g68 (centeredRebase g68 : ExactMatrix Q18 N 1 68) := ⟨hQ18, rfl⟩
  have hg64u : centeredRebaseRuns g64 (centeredRebase g64 : ExactMatrix Q17 N 1 64) := ⟨hQ17, rfl⟩
  have hg60u : centeredRebaseRuns g60 (centeredRebase g60 : ExactMatrix Q16 N 1 60) := ⟨hQ16, rfl⟩
  have hg56u : centeredRebaseRuns g56 (centeredRebase g56 : ExactMatrix Q15 N 1 56) := ⟨hQ15, rfl⟩
  have hg52u : centeredRebaseRuns g52 (centeredRebase g52 : ExactMatrix Q14 N 1 52) := ⟨hQ14, rfl⟩
  have hg48u : centeredRebaseRuns g48 (centeredRebase g48 : ExactMatrix Q13 N 1 48) := ⟨hQ13, rfl⟩
  have hg44u : centeredRebaseRuns g44 (centeredRebase g44 : ExactMatrix Q12 N 1 44) := ⟨hQ12, rfl⟩
  have hg40u : centeredRebaseRuns g40 (centeredRebase g40 : ExactMatrix Q11 N 1 40) := ⟨hQ11, rfl⟩
  have hg36u : centeredRebaseRuns g36 (centeredRebase g36 : ExactMatrix Q10 N 1 36) := ⟨hQ10, rfl⟩
  have hg32u : centeredRebaseRuns g32 (centeredRebase g32 : ExactMatrix Q9 N 1 32) := ⟨hQ9, rfl⟩
  have hg28u : centeredRebaseRuns g28 (centeredRebase g28 : ExactMatrix Q8 N 1 28) := ⟨hQ8, rfl⟩
  have hg24u : centeredRebaseRuns g24 (centeredRebase g24 : ExactMatrix Q7 N 1 24) := ⟨hQ7, rfl⟩
  have hg20u : centeredRebaseRuns g20 (centeredRebase g20 : ExactMatrix Q6 N 1 20) := ⟨hQ6, rfl⟩
  have hg16u : centeredRebaseRuns g16 (centeredRebase g16 : ExactMatrix Q5 N 1 16) := ⟨hQ5, rfl⟩
  have hg12u : centeredRebaseRuns g12 (centeredRebase g12 : ExactMatrix Q4 N 1 12) := ⟨hQ4, rfl⟩
  have hg8u : centeredRebaseRuns g8 (centeredRebase g8 : ExactMatrix Q3 N 1 8) := ⟨hQ3, rfl⟩
  have hg4u : centeredRebaseRuns g4 (centeredRebase g4 : ExactMatrix Q2 N 1 4) := ⟨hQ2, rfl⟩
  have hG := hg144
  -- Encryption and preprocessing.
  have hstate := encrypt_state henc hsm hsp hg140 hg140u
  rw [hμ0] at hstate
  have hmembers := encrypt_member henc hsm hG
  have hmemα : ∀ k : Fin 1025, Rel sm (memberKey model external.input_0 k)
      (fun j ↦ reducePoly Q36 N (α k * unitPair sm j) • g144)
      ![execution.«stage_3».2.2.1 k, execution.«stage_3».2.2.2.1 k] 26 := fun k ↦ by
    rw [← hα k]; exact (hmembers k).1
  have hpayα : ∀ k : Fin 1025, execution.«stage_3».2.2.2.2.1 k = fun _ _ ↦ reducePoly Q36 N (α k) :=
    fun k ↦ by rw [← hα k]; exact (hmembers k).2
  have hhelp := encrypt_helper henc hsm hg4 hg4u
  have hrhs0 := preprocess_selector0 hpre hsp hg136 hg136u
  have hrhs1 := preprocess_selector1 hpre hsp hg132 hg132u
  have hrhs2 := preprocess_selector2 hpre hsp hg128 hg128u
  have hrhs3 := preprocess_selector3 hpre hsp hg124 hg124u
  have hrhs4 := preprocess_selector4 hpre hsp hg120 hg120u
  have hrhs5 := preprocess_selector5 hpre hsp hg116 hg116u
  have hrhs6 := preprocess_selector6 hpre hsp hg112 hg112u
  have hrhs7 := preprocess_selector7 hpre hsp hg108 hg108u
  have hrhs8 := preprocess_selector8 hpre hsp hg104 hg104u
  have hrhs9 := preprocess_selector9 hpre hsp hg100 hg100u
  have hrhs10 := preprocess_selector10 hpre hsp hg96 hg96u
  have hrhs11 := preprocess_selector11 hpre hsp hg92 hg92u
  have hrhs12 := preprocess_selector12 hpre hsp hg88 hg88u
  have hrhs13 := preprocess_selector13 hpre hsp hg84 hg84u
  have hrhs14 := preprocess_selector14 hpre hsp hg80 hg80u
  have hrhs15 := preprocess_selector15 hpre hsp hg76 hg76u
  have hrhs16 := preprocess_selector16 hpre hsp hg72 hg72u
  have hrhs17 := preprocess_selector17 hpre hsp hg68 hg68u
  have hrhs18 := preprocess_selector18 hpre hsp hg64 hg64u
  have hrhs19 := preprocess_selector19 hpre hsp hg60 hg60u
  have hrhs20 := preprocess_selector20 hpre hsp hg56 hg56u
  have hrhs21 := preprocess_selector21 hpre hsp hg52 hg52u
  have hrhs22 := preprocess_selector22 hpre hsp hg48 hg48u
  have hrhs23 := preprocess_selector23 hpre hsp hg44 hg44u
  have hrhs24 := preprocess_selector24 hpre hsp hg40 hg40u
  have hrhs25 := preprocess_selector25 hpre hsp hg36 hg36u
  have hrhs26 := preprocess_selector26 hpre hsp hg32 hg32u
  have hrhs27 := preprocess_selector27 hpre hsp hg28 hg28u
  have hrhs28 := preprocess_selector28 hpre hsp hg24 hg24u
  have hrhs29 := preprocess_selector29 hpre hsp hg20 hg20u
  have hrhs30 := preprocess_selector30 hpre hsp hg16 hg16u
  have hrhs31 := preprocess_selector31 hpre hsp hg12 hg12u
  have hrhs32 := preprocess_selector32 hpre hsp hg8 hg8u
  have hrhs33 := preprocess_selector33 hpre hsp hg4 hg4u
  have hsw := preprocess_switch hpre hsp
  -- The 34 blocks accumulate the monomial of the sum of the selected coordinates.
  obtain ⟨w, hb⟩ := honlineEval
  -- Block 0.
  have hb0 := block0 hb hsmn (μ := Xe 0) (α := α)
    (ν := fun i ↦ message (n := N) (s0 i) (x0 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg140 hg140u hmemα hpayα hstate hrhs0
  have hin1 := hb0.target_eq (collapse _ _ _ _ _ _ sel0 hsel0 hν0)
  rw [Xe_mul] at hin1
  obtain ⟨z1, hz1⟩ : ∃ z : Nat,
      z = (a ⟨coords 0 sel0, coords_lt 0 sel0⟩).toNat + 0 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 0 sel0, coords_lt 0 sel0⟩).toNat + 0 = 512 * z1 by
    rw [hz1]; ring] at hin1
  -- Block 1.
  have hb1 := block1 hb hsmn (μ := Xe (512 * z1)) (α := α)
    (ν := fun i ↦ message (n := N) (s1 i) (x1 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg136 hg136u hmemα hpayα hin1 hrhs1
  have hin2 := hb1.target_eq (collapse _ _ _ _ _ _ sel1 hsel1 hν1)
  rw [Xe_mul] at hin2
  obtain ⟨z2, hz2⟩ : ∃ z : Nat,
      z = (a ⟨coords 1 sel1, coords_lt 1 sel1⟩).toNat + z1 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 1 sel1, coords_lt 1 sel1⟩).toNat + 512 * z1 = 512 * z2 by
    rw [hz2]; ring] at hin2
  -- Block 2.
  have hb2 := block2 hb hsmn (μ := Xe (512 * z2)) (α := α)
    (ν := fun i ↦ message (n := N) (s2 i) (x2 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg132 hg132u hmemα hpayα hin2 hrhs2
  have hin3 := hb2.target_eq (collapse _ _ _ _ _ _ sel2 hsel2 hν2)
  rw [Xe_mul] at hin3
  obtain ⟨z3, hz3⟩ : ∃ z : Nat,
      z = (a ⟨coords 2 sel2, coords_lt 2 sel2⟩).toNat + z2 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 2 sel2, coords_lt 2 sel2⟩).toNat + 512 * z2 = 512 * z3 by
    rw [hz3]; ring] at hin3
  -- Block 3.
  have hb3 := block3 hb hsmn (μ := Xe (512 * z3)) (α := α)
    (ν := fun i ↦ message (n := N) (s3 i) (x3 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg128 hg128u hmemα hpayα hin3 hrhs3
  have hin4 := hb3.target_eq (collapse _ _ _ _ _ _ sel3 hsel3 hν3)
  rw [Xe_mul] at hin4
  obtain ⟨z4, hz4⟩ : ∃ z : Nat,
      z = (a ⟨coords 3 sel3, coords_lt 3 sel3⟩).toNat + z3 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 3 sel3, coords_lt 3 sel3⟩).toNat + 512 * z3 = 512 * z4 by
    rw [hz4]; ring] at hin4
  -- Block 4.
  have hb4 := block4 hb hsmn (μ := Xe (512 * z4)) (α := α)
    (ν := fun i ↦ message (n := N) (s4 i) (x4 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg124 hg124u hmemα hpayα hin4 hrhs4
  have hin5 := hb4.target_eq (collapse _ _ _ _ _ _ sel4 hsel4 hν4)
  rw [Xe_mul] at hin5
  obtain ⟨z5, hz5⟩ : ∃ z : Nat,
      z = (a ⟨coords 4 sel4, coords_lt 4 sel4⟩).toNat + z4 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 4 sel4, coords_lt 4 sel4⟩).toNat + 512 * z4 = 512 * z5 by
    rw [hz5]; ring] at hin5
  -- Block 5.
  have hb5 := block5 hb hsmn (μ := Xe (512 * z5)) (α := α)
    (ν := fun i ↦ message (n := N) (s5 i) (x5 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg120 hg120u hmemα hpayα hin5 hrhs5
  have hin6 := hb5.target_eq (collapse _ _ _ _ _ _ sel5 hsel5 hν5)
  rw [Xe_mul] at hin6
  obtain ⟨z6, hz6⟩ : ∃ z : Nat,
      z = (a ⟨coords 5 sel5, coords_lt 5 sel5⟩).toNat + z5 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 5 sel5, coords_lt 5 sel5⟩).toNat + 512 * z5 = 512 * z6 by
    rw [hz6]; ring] at hin6
  -- Block 6.
  have hb6 := block6 hb hsmn (μ := Xe (512 * z6)) (α := α)
    (ν := fun i ↦ message (n := N) (s6 i) (x6 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg116 hg116u hmemα hpayα hin6 hrhs6
  have hin7 := hb6.target_eq (collapse _ _ _ _ _ _ sel6 hsel6 hν6)
  rw [Xe_mul] at hin7
  obtain ⟨z7, hz7⟩ : ∃ z : Nat,
      z = (a ⟨coords 6 sel6, coords_lt 6 sel6⟩).toNat + z6 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 6 sel6, coords_lt 6 sel6⟩).toNat + 512 * z6 = 512 * z7 by
    rw [hz7]; ring] at hin7
  -- Block 7.
  have hb7 := block7 hb hsmn (μ := Xe (512 * z7)) (α := α)
    (ν := fun i ↦ message (n := N) (s7 i) (x7 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg112 hg112u hmemα hpayα hin7 hrhs7
  have hin8 := hb7.target_eq (collapse _ _ _ _ _ _ sel7 hsel7 hν7)
  rw [Xe_mul] at hin8
  obtain ⟨z8, hz8⟩ : ∃ z : Nat,
      z = (a ⟨coords 7 sel7, coords_lt 7 sel7⟩).toNat + z7 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 7 sel7, coords_lt 7 sel7⟩).toNat + 512 * z7 = 512 * z8 by
    rw [hz8]; ring] at hin8
  -- Block 8.
  have hb8 := block8 hb hsmn (μ := Xe (512 * z8)) (α := α)
    (ν := fun i ↦ message (n := N) (s8 i) (x8 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg108 hg108u hmemα hpayα hin8 hrhs8
  have hin9 := hb8.target_eq (collapse _ _ _ _ _ _ sel8 hsel8 hν8)
  rw [Xe_mul] at hin9
  obtain ⟨z9, hz9⟩ : ∃ z : Nat,
      z = (a ⟨coords 8 sel8, coords_lt 8 sel8⟩).toNat + z8 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 8 sel8, coords_lt 8 sel8⟩).toNat + 512 * z8 = 512 * z9 by
    rw [hz9]; ring] at hin9
  -- Block 9.
  have hb9 := block9 hb hsmn (μ := Xe (512 * z9)) (α := α)
    (ν := fun i ↦ message (n := N) (s9 i) (x9 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg104 hg104u hmemα hpayα hin9 hrhs9
  have hin10 := hb9.target_eq (collapse _ _ _ _ _ _ sel9 hsel9 hν9)
  rw [Xe_mul] at hin10
  obtain ⟨z10, hz10⟩ : ∃ z : Nat,
      z = (a ⟨coords 9 sel9, coords_lt 9 sel9⟩).toNat + z9 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 9 sel9, coords_lt 9 sel9⟩).toNat + 512 * z9 = 512 * z10 by
    rw [hz10]; ring] at hin10
  -- Block 10.
  have hb10 := block10 hb hsmn (μ := Xe (512 * z10)) (α := α)
    (ν := fun i ↦ message (n := N) (s10 i) (x10 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg100 hg100u hmemα hpayα hin10 hrhs10
  have hin11 := hb10.target_eq (collapse _ _ _ _ _ _ sel10 hsel10 hν10)
  rw [Xe_mul] at hin11
  obtain ⟨z11, hz11⟩ : ∃ z : Nat,
      z = (a ⟨coords 10 sel10, coords_lt 10 sel10⟩).toNat + z10 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 10 sel10, coords_lt 10 sel10⟩).toNat + 512 * z10 = 512 * z11 by
    rw [hz11]; ring] at hin11
  -- Block 11.
  have hb11 := block11 hb hsmn (μ := Xe (512 * z11)) (α := α)
    (ν := fun i ↦ message (n := N) (s11 i) (x11 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg96 hg96u hmemα hpayα hin11 hrhs11
  have hin12 := hb11.target_eq (collapse _ _ _ _ _ _ sel11 hsel11 hν11)
  rw [Xe_mul] at hin12
  obtain ⟨z12, hz12⟩ : ∃ z : Nat,
      z = (a ⟨coords 11 sel11, coords_lt 11 sel11⟩).toNat + z11 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 11 sel11, coords_lt 11 sel11⟩).toNat + 512 * z11 = 512 * z12 by
    rw [hz12]; ring] at hin12
  -- Block 12.
  have hb12 := block12 hb hsmn (μ := Xe (512 * z12)) (α := α)
    (ν := fun i ↦ message (n := N) (s12 i) (x12 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg92 hg92u hmemα hpayα hin12 hrhs12
  have hin13 := hb12.target_eq (collapse _ _ _ _ _ _ sel12 hsel12 hν12)
  rw [Xe_mul] at hin13
  obtain ⟨z13, hz13⟩ : ∃ z : Nat,
      z = (a ⟨coords 12 sel12, coords_lt 12 sel12⟩).toNat + z12 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 12 sel12, coords_lt 12 sel12⟩).toNat + 512 * z12 = 512 * z13 by
    rw [hz13]; ring] at hin13
  -- Block 13.
  have hb13 := block13 hb hsmn (μ := Xe (512 * z13)) (α := α)
    (ν := fun i ↦ message (n := N) (s13 i) (x13 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg88 hg88u hmemα hpayα hin13 hrhs13
  have hin14 := hb13.target_eq (collapse _ _ _ _ _ _ sel13 hsel13 hν13)
  rw [Xe_mul] at hin14
  obtain ⟨z14, hz14⟩ : ∃ z : Nat,
      z = (a ⟨coords 13 sel13, coords_lt 13 sel13⟩).toNat + z13 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 13 sel13, coords_lt 13 sel13⟩).toNat + 512 * z13 = 512 * z14 by
    rw [hz14]; ring] at hin14
  -- Block 14.
  have hb14 := block14 hb hsmn (μ := Xe (512 * z14)) (α := α)
    (ν := fun i ↦ message (n := N) (s14 i) (x14 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg84 hg84u hmemα hpayα hin14 hrhs14
  have hin15 := hb14.target_eq (collapse _ _ _ _ _ _ sel14 hsel14 hν14)
  rw [Xe_mul] at hin15
  obtain ⟨z15, hz15⟩ : ∃ z : Nat,
      z = (a ⟨coords 14 sel14, coords_lt 14 sel14⟩).toNat + z14 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 14 sel14, coords_lt 14 sel14⟩).toNat + 512 * z14 = 512 * z15 by
    rw [hz15]; ring] at hin15
  -- Block 15.
  have hb15 := block15 hb hsmn (μ := Xe (512 * z15)) (α := α)
    (ν := fun i ↦ message (n := N) (s15 i) (x15 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg80 hg80u hmemα hpayα hin15 hrhs15
  have hin16 := hb15.target_eq (collapse _ _ _ _ _ _ sel15 hsel15 hν15)
  rw [Xe_mul] at hin16
  obtain ⟨z16, hz16⟩ : ∃ z : Nat,
      z = (a ⟨coords 15 sel15, coords_lt 15 sel15⟩).toNat + z15 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 15 sel15, coords_lt 15 sel15⟩).toNat + 512 * z15 = 512 * z16 by
    rw [hz16]; ring] at hin16
  -- Block 16.
  have hb16 := block16 hb hsmn (μ := Xe (512 * z16)) (α := α)
    (ν := fun i ↦ message (n := N) (s16 i) (x16 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg76 hg76u hmemα hpayα hin16 hrhs16
  have hin17 := hb16.target_eq (collapse _ _ _ _ _ _ sel16 hsel16 hν16)
  rw [Xe_mul] at hin17
  obtain ⟨z17, hz17⟩ : ∃ z : Nat,
      z = (a ⟨coords 16 sel16, coords_lt 16 sel16⟩).toNat + z16 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 16 sel16, coords_lt 16 sel16⟩).toNat + 512 * z16 = 512 * z17 by
    rw [hz17]; ring] at hin17
  -- Block 17.
  have hb17 := block17 hb hsmn (μ := Xe (512 * z17)) (α := α)
    (ν := fun i ↦ message (n := N) (s17 i) (x17 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg72 hg72u hmemα hpayα hin17 hrhs17
  have hin18 := hb17.target_eq (collapse _ _ _ _ _ _ sel17 hsel17 hν17)
  rw [Xe_mul] at hin18
  obtain ⟨z18, hz18⟩ : ∃ z : Nat,
      z = (a ⟨coords 17 sel17, coords_lt 17 sel17⟩).toNat + z17 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 17 sel17, coords_lt 17 sel17⟩).toNat + 512 * z17 = 512 * z18 by
    rw [hz18]; ring] at hin18
  -- Block 18.
  have hb18 := block18 hb hsmn (μ := Xe (512 * z18)) (α := α)
    (ν := fun i ↦ message (n := N) (s18 i) (x18 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg68 hg68u hmemα hpayα hin18 hrhs18
  have hin19 := hb18.target_eq (collapse _ _ _ _ _ _ sel18 hsel18 hν18)
  rw [Xe_mul] at hin19
  obtain ⟨z19, hz19⟩ : ∃ z : Nat,
      z = (a ⟨coords 18 sel18, coords_lt 18 sel18⟩).toNat + z18 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 18 sel18, coords_lt 18 sel18⟩).toNat + 512 * z18 = 512 * z19 by
    rw [hz19]; ring] at hin19
  -- Block 19.
  have hb19 := block19 hb hsmn (μ := Xe (512 * z19)) (α := α)
    (ν := fun i ↦ message (n := N) (s19 i) (x19 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg64 hg64u hmemα hpayα hin19 hrhs19
  have hin20 := hb19.target_eq (collapse _ _ _ _ _ _ sel19 hsel19 hν19)
  rw [Xe_mul] at hin20
  obtain ⟨z20, hz20⟩ : ∃ z : Nat,
      z = (a ⟨coords 19 sel19, coords_lt 19 sel19⟩).toNat + z19 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 19 sel19, coords_lt 19 sel19⟩).toNat + 512 * z19 = 512 * z20 by
    rw [hz20]; ring] at hin20
  -- Block 20.
  have hb20 := block20 hb hsmn (μ := Xe (512 * z20)) (α := α)
    (ν := fun i ↦ message (n := N) (s20 i) (x20 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg60 hg60u hmemα hpayα hin20 hrhs20
  have hin21 := hb20.target_eq (collapse _ _ _ _ _ _ sel20 hsel20 hν20)
  rw [Xe_mul] at hin21
  obtain ⟨z21, hz21⟩ : ∃ z : Nat,
      z = (a ⟨coords 20 sel20, coords_lt 20 sel20⟩).toNat + z20 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 20 sel20, coords_lt 20 sel20⟩).toNat + 512 * z20 = 512 * z21 by
    rw [hz21]; ring] at hin21
  -- Block 21.
  have hb21 := block21 hb hsmn (μ := Xe (512 * z21)) (α := α)
    (ν := fun i ↦ message (n := N) (s21 i) (x21 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg56 hg56u hmemα hpayα hin21 hrhs21
  have hin22 := hb21.target_eq (collapse _ _ _ _ _ _ sel21 hsel21 hν21)
  rw [Xe_mul] at hin22
  obtain ⟨z22, hz22⟩ : ∃ z : Nat,
      z = (a ⟨coords 21 sel21, coords_lt 21 sel21⟩).toNat + z21 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 21 sel21, coords_lt 21 sel21⟩).toNat + 512 * z21 = 512 * z22 by
    rw [hz22]; ring] at hin22
  -- Block 22.
  have hb22 := block22 hb hsmn (μ := Xe (512 * z22)) (α := α)
    (ν := fun i ↦ message (n := N) (s22 i) (x22 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg52 hg52u hmemα hpayα hin22 hrhs22
  have hin23 := hb22.target_eq (collapse _ _ _ _ _ _ sel22 hsel22 hν22)
  rw [Xe_mul] at hin23
  obtain ⟨z23, hz23⟩ : ∃ z : Nat,
      z = (a ⟨coords 22 sel22, coords_lt 22 sel22⟩).toNat + z22 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 22 sel22, coords_lt 22 sel22⟩).toNat + 512 * z22 = 512 * z23 by
    rw [hz23]; ring] at hin23
  -- Block 23.
  have hb23 := block23 hb hsmn (μ := Xe (512 * z23)) (α := α)
    (ν := fun i ↦ message (n := N) (s23 i) (x23 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg48 hg48u hmemα hpayα hin23 hrhs23
  have hin24 := hb23.target_eq (collapse _ _ _ _ _ _ sel23 hsel23 hν23)
  rw [Xe_mul] at hin24
  obtain ⟨z24, hz24⟩ : ∃ z : Nat,
      z = (a ⟨coords 23 sel23, coords_lt 23 sel23⟩).toNat + z23 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 23 sel23, coords_lt 23 sel23⟩).toNat + 512 * z23 = 512 * z24 by
    rw [hz24]; ring] at hin24
  -- Block 24.
  have hb24 := block24 hb hsmn (μ := Xe (512 * z24)) (α := α)
    (ν := fun i ↦ message (n := N) (s24 i) (x24 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg44 hg44u hmemα hpayα hin24 hrhs24
  have hin25 := hb24.target_eq (collapse _ _ _ _ _ _ sel24 hsel24 hν24)
  rw [Xe_mul] at hin25
  obtain ⟨z25, hz25⟩ : ∃ z : Nat,
      z = (a ⟨coords 24 sel24, coords_lt 24 sel24⟩).toNat + z24 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 24 sel24, coords_lt 24 sel24⟩).toNat + 512 * z24 = 512 * z25 by
    rw [hz25]; ring] at hin25
  -- Block 25.
  have hb25 := block25 hb hsmn (μ := Xe (512 * z25)) (α := α)
    (ν := fun i ↦ message (n := N) (s25 i) (x25 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg40 hg40u hmemα hpayα hin25 hrhs25
  have hin26 := hb25.target_eq (collapse _ _ _ _ _ _ sel25 hsel25 hν25)
  rw [Xe_mul] at hin26
  obtain ⟨z26, hz26⟩ : ∃ z : Nat,
      z = (a ⟨coords 25 sel25, coords_lt 25 sel25⟩).toNat + z25 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 25 sel25, coords_lt 25 sel25⟩).toNat + 512 * z25 = 512 * z26 by
    rw [hz26]; ring] at hin26
  -- Block 26.
  have hb26 := block26 hb hsmn (μ := Xe (512 * z26)) (α := α)
    (ν := fun i ↦ message (n := N) (s26 i) (x26 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg36 hg36u hmemα hpayα hin26 hrhs26
  have hin27 := hb26.target_eq (collapse _ _ _ _ _ _ sel26 hsel26 hν26)
  rw [Xe_mul] at hin27
  obtain ⟨z27, hz27⟩ : ∃ z : Nat,
      z = (a ⟨coords 26 sel26, coords_lt 26 sel26⟩).toNat + z26 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 26 sel26, coords_lt 26 sel26⟩).toNat + 512 * z26 = 512 * z27 by
    rw [hz27]; ring] at hin27
  -- Block 27.
  have hb27 := block27 hb hsmn (μ := Xe (512 * z27)) (α := α)
    (ν := fun i ↦ message (n := N) (s27 i) (x27 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg32 hg32u hmemα hpayα hin27 hrhs27
  have hin28 := hb27.target_eq (collapse _ _ _ _ _ _ sel27 hsel27 hν27)
  rw [Xe_mul] at hin28
  obtain ⟨z28, hz28⟩ : ∃ z : Nat,
      z = (a ⟨coords 27 sel27, coords_lt 27 sel27⟩).toNat + z27 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 27 sel27, coords_lt 27 sel27⟩).toNat + 512 * z27 = 512 * z28 by
    rw [hz28]; ring] at hin28
  -- Block 28.
  have hb28 := block28 hb hsmn (μ := Xe (512 * z28)) (α := α)
    (ν := fun i ↦ message (n := N) (s28 i) (x28 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg28 hg28u hmemα hpayα hin28 hrhs28
  have hin29 := hb28.target_eq (collapse _ _ _ _ _ _ sel28 hsel28 hν28)
  rw [Xe_mul] at hin29
  obtain ⟨z29, hz29⟩ : ∃ z : Nat,
      z = (a ⟨coords 28 sel28, coords_lt 28 sel28⟩).toNat + z28 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 28 sel28, coords_lt 28 sel28⟩).toNat + 512 * z28 = 512 * z29 by
    rw [hz29]; ring] at hin29
  -- Block 29.
  have hb29 := block29 hb hsmn (μ := Xe (512 * z29)) (α := α)
    (ν := fun i ↦ message (n := N) (s29 i) (x29 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg24 hg24u hmemα hpayα hin29 hrhs29
  have hin30 := hb29.target_eq (collapse _ _ _ _ _ _ sel29 hsel29 hν29)
  rw [Xe_mul] at hin30
  obtain ⟨z30, hz30⟩ : ∃ z : Nat,
      z = (a ⟨coords 29 sel29, coords_lt 29 sel29⟩).toNat + z29 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 29 sel29, coords_lt 29 sel29⟩).toNat + 512 * z29 = 512 * z30 by
    rw [hz30]; ring] at hin30
  -- Block 30.
  have hb30 := block30 hb hsmn (μ := Xe (512 * z30)) (α := α)
    (ν := fun i ↦ message (n := N) (s30 i) (x30 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg20 hg20u hmemα hpayα hin30 hrhs30
  have hin31 := hb30.target_eq (collapse _ _ _ _ _ _ sel30 hsel30 hν30)
  rw [Xe_mul] at hin31
  obtain ⟨z31, hz31⟩ : ∃ z : Nat,
      z = (a ⟨coords 30 sel30, coords_lt 30 sel30⟩).toNat + z30 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 30 sel30, coords_lt 30 sel30⟩).toNat + 512 * z30 = 512 * z31 by
    rw [hz31]; ring] at hin31
  -- Block 31.
  have hb31 := block31 hb hsmn (μ := Xe (512 * z31)) (α := α)
    (ν := fun i ↦ message (n := N) (s31 i) (x31 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg16 hg16u hmemα hpayα hin31 hrhs31
  have hin32 := hb31.target_eq (collapse _ _ _ _ _ _ sel31 hsel31 hν31)
  rw [Xe_mul] at hin32
  obtain ⟨z32, hz32⟩ : ∃ z : Nat,
      z = (a ⟨coords 31 sel31, coords_lt 31 sel31⟩).toNat + z31 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 31 sel31, coords_lt 31 sel31⟩).toNat + 512 * z31 = 512 * z32 by
    rw [hz32]; ring] at hin32
  -- Block 32.
  have hb32 := block32 hb hsmn (μ := Xe (512 * z32)) (α := α)
    (ν := fun i ↦ message (n := N) (s32 i) (x32 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg12 hg12u hmemα hpayα hin32 hrhs32
  have hin33 := hb32.target_eq (collapse _ _ _ _ _ _ sel32 hsel32 hν32)
  rw [Xe_mul] at hin33
  obtain ⟨z33, hz33⟩ : ∃ z : Nat,
      z = (a ⟨coords 32 sel32, coords_lt 32 sel32⟩).toNat + z32 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 32 sel32, coords_lt 32 sel32⟩).toNat + 512 * z32 = 512 * z33 by
    rw [hz33]; ring] at hin33
  -- Block 33.
  have hb33 := block33 hb hsmn (μ := Xe (512 * z33)) (α := α)
    (ν := fun i ↦ message (n := N) (s33 i) (x33 i)) (fun i ↦ by rw [Xe_mul]; exact smulGain_root_pow hN1 _) (fun k ↦ smulGain_root_pow hN1 _) hG
    hg8 hg8u hmemα hpayα hin33 hrhs33
  have hin34 := hb33.target_eq (collapse _ _ _ _ _ _ sel33 hsel33 hν33)
  rw [Xe_mul] at hin34
  obtain ⟨z34, hz34⟩ : ∃ z : Nat,
      z = (a ⟨coords 33 sel33, coords_lt 33 sel33⟩).toNat + z33 := ⟨_, rfl⟩
  rw [show 512 * (a ⟨coords 33 sel33, coords_lt 33 sel33⟩).toNat + 512 * z33 = 512 * z34 by
    rw [hz34]; ring] at hin34
  -- The readout LUT and the decryption.
  have hlut := lutRound hb hsmn hg4 hg4u hin34 hhelp hsw
  have hb2 := hb
  unfold Stage_online_eval.generatedRoot.body at hb2
  extract_lets at hb2
  have houtOnlineEval := hb2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  obtain ⟨wd, hd⟩ := hdec
  unfold Stage_decode.generatedRoot.body at hd
  dsimp only at hd
  obtain ⟨hm1, hm2, -, hdec1, hout⟩ := hd
  rw [hsm] at hm1
  have hw2 := modulusReduce_poly hQ36 hQ1 hN hm1
  have hw6 := modulusReduce_one hQ36 hQ1 hN hm2
  have hbound : 4 * lutBound (∑ _i : Fin 113, cellBound33 (∑ _i : Fin 113, cellBound32 (∑ _i : Fin 113, cellBound31 (∑ _i : Fin 113, cellBound30 (∑ _i : Fin 113, cellBound29 (∑ _i : Fin 113, cellBound28 (∑ _i : Fin 113, cellBound27 (∑ _i : Fin 113, cellBound26 (∑ _i : Fin 113, cellBound25 (∑ _i : Fin 113, cellBound24 (∑ _i : Fin 113, cellBound23 (∑ _i : Fin 113, cellBound22 (∑ _i : Fin 113, cellBound21 (∑ _i : Fin 113, cellBound20 (∑ _i : Fin 113, cellBound19 (∑ _i : Fin 113, cellBound18 (∑ _i : Fin 113, cellBound17 (∑ _i : Fin 113, cellBound16 (∑ _i : Fin 113, cellBound15 (∑ _i : Fin 113, cellBound14 (∑ _i : Fin 113, cellBound13 (∑ _i : Fin 113, cellBound12 (∑ _i : Fin 113, cellBound11 (∑ _i : Fin 113, cellBound10 (∑ _i : Fin 113, cellBound9 (∑ _i : Fin 113, cellBound8 (∑ _i : Fin 113, cellBound7 (∑ _i : Fin 113, cellBound6 (∑ _i : Fin 113, cellBound5 (∑ _i : Fin 113, cellBound4 (∑ _i : Fin 113, cellBound3 (∑ _i : Fin 113, cellBound2 (∑ _i : Fin 113, cellBound1 (∑ _i : Fin 113, cellBound0 26)))))))))))))))))))))))))))))))))) + 2 * 16 + 12 < Q1 := by
    simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul]
    decide
  have hval := decode_terminal_parity_bounded hQ1 (by norm_num) hN hlut hbound (k := ⟨0, hN⟩)
    (lutValue_coeff_bound z34) (by
      rw [houtOnlineEval, hw2, hw6]
      funext r c
      obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _
      simp only [matrixSub, matrixAdd, matrixMulScalarLeft_eq, Matrix.sub_apply, Matrix.add_apply,
        Matrix.smul_apply, smul_eq_mul, Matrix.one_apply_eq, one_mul]
      rfl) hdec1
  rw [hout, hval, readout_parity, hid]
  have hz : selectedSum a s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 = (z34 : Int) := by
    rw [selectedSum, one_hot_sum a 0 hsel0, one_hot_sum a 1 hsel1, one_hot_sum a 2 hsel2,
      one_hot_sum a 3 hsel3, one_hot_sum a 4 hsel4, one_hot_sum a 5 hsel5, one_hot_sum a 6 hsel6,
      one_hot_sum a 7 hsel7, one_hot_sum a 8 hsel8, one_hot_sum a 9 hsel9, one_hot_sum a 10 hsel10,
      one_hot_sum a 11 hsel11, one_hot_sum a 12 hsel12, one_hot_sum a 13 hsel13,
      one_hot_sum a 14 hsel14, one_hot_sum a 15 hsel15, one_hot_sum a 16 hsel16,
      one_hot_sum a 17 hsel17, one_hot_sum a 18 hsel18, one_hot_sum a 19 hsel19,
      one_hot_sum a 20 hsel20, one_hot_sum a 21 hsel21, one_hot_sum a 22 hsel22,
      one_hot_sum a 23 hsel23, one_hot_sum a 24 hsel24, one_hot_sum a 25 hsel25,
      one_hot_sum a 26 hsel26, one_hot_sum a 27 hsel27, one_hot_sum a 28 hsel28,
      one_hot_sum a 29 hsel29, one_hot_sum a 30 hsel30, one_hot_sum a 31 hsel31,
      one_hot_sum a 32 hsel32, one_hot_sum a 33 hsel33, hz34, hz33, hz32, hz31, hz30, hz29, hz28,
      hz27, hz26, hz25, hz24, hz23, hz22, hz21, hz20, hz19, hz18, hz17, hz16, hz15, hz14, hz13,
      hz12, hz11, hz10, hz9, hz8, hz7, hz6, hz5, hz4, hz3, hz2, hz1]
    push_cast
    rw [Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1, Int.toNat_of_nonneg (ha _).1,
      Int.toNat_of_nonneg (ha _).1]
    ring
  rw [hz]

/-- The average-case correctness claim of `average-n65536-crt36x40`: the failure set is empty, so its
tape measure is below every bound. -/
theorem certificate : GeneratedClaim.CorrectnessClaim := by
  intro model external
  have hempty : {tape | ∃ execution, GeneratedClaim.Runs model external tape execution ∧
      ¬ execution.«stage_5» = execution.«ideal»} = ∅ := by
    ext tape
    simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
    rintro ⟨execution, hruns, hne⟩
    exact hne (correct hruns)
  rw [hempty, MeasureTheory.measure_empty]
  exact zero_le _

/-- info: 'LwrProof.certificate' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms certificate

end LwrProof
