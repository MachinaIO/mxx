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
half-integer readout of its selected coordinates modulo `64`. The requirement makes the plaintext of member
`k` of `a` the monomial `X^(512 a_k)` and every block's selectors one-hot; the 34 blocks then
accumulate `X^(512z)` for the sum `z` of the selected coordinates, the readout LUT gives
`(-1)^⌊z/128⌋ (2 τ(z mod 128) - 1)` at the terminal modulus, and decryption returns it modulo `64`,
which is the ideal output. The plaintexts of the members of `a` and of the state are monomials, so multiplying an
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

/-- The half-integer readout modulo `64` is the ideal value: `m` for an even `⌊z / 128⌋` and
`64 - m` for an odd one, with `m = 2 τ(z mod 128) - 1`. -/
theorem readout_mod64 (z : Nat) :
    ((lutValue z).coeff ⟨0, hN⟩) % 64 =
      ((((z : Int) % 128) * 16 / 128 + 1) * 2 - 1) +
        (z : Int) % 256 / 128 * (64 - ((((z : Int) % 128) * 16 / 128 + 1) * 2 - 1) * 2) := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.halfIntegerReadout, MxxExponentLutLwr.readout]
  have hτ : ((2 * (16 * (z % 128) / 128 + 1) - 1 : Nat) : Int) =
      (((z : Int) % 128) * 16 / 128 + 1) * 2 - 1 := by omega
  rw [hτ]
  have hb : (z : Int) % 256 / 128 = ((z / 128 : Nat) : Int) % 2 := by omega
  rw [hb]
  rcases Nat.even_or_odd (z / 128) with h | h
  · rw [h.neg_one_pow, one_mul, show ((z / 128 : Nat) : Int) % 2 = 0 by
      obtain ⟨k, hk⟩ := h; omega]
    omega
  · rw [h.neg_one_pow, neg_one_mul, show ((z / 128 : Nat) : Int) % 2 = 1 by
      obtain ⟨k, hk⟩ := h; omega]
    omega

theorem lutValue_coeff_bound (z : Nat) : |(lutValue z).coeff ⟨0, hN⟩| ≤ 31 := by
  rw [lutValue, intCast_coeff_zero, MxxExponentLutLwr.halfIntegerReadout, MxxExponentLutLwr.readout]
  have hτ : 2 * (16 * (z % 128) / 128 + 1) - 1 ≤ 31 := by omega
  rcases Nat.even_or_odd (z / 128) with h | h
  · rw [h.neg_one_pow, one_mul, abs_le]; omega
  · rw [h.neg_one_pow, neg_one_mul, abs_le]; omega

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

end LwrProof
