import RequirementFacts

/-! The requirement's check of `σ`: coefficient `i` is `1 - ⌊(z_i mod 8) / 4⌋` for the mask PRF's
readout exponent `z_i = Σ_k s(5i + k) a_k + Σ_k s'(5i + k) a_k`. -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem select_cell (i0 i : Nat) (h0 : i0 < 8) (h : i < 5) :
    MxxIR.selectInt (Int.ofNat (5 * i0 + i))
      [0 + 0, 1 + 0, 2 + 0, 3 + 0, 4 + 0, 0 + 0, 1 + 0, 2 + 0, 3 + 0, 4 + 0, 0 + 0, 1 + 0, 2 + 0, 3 + 0,
        4 + 0, 0 + 0, 1 + 0, 2 + 0, 3 + 0, 4 + 0, 0 + 0, 1 + 0, 2 + 0, 3 + 0, 4 + 0, 0 + 0, 1 + 0,
        2 + 0, 3 + 0, 4 + 0, 0 + 0, 1 + 0, 2 + 0, 3 + 0, 4 + 0, 0 + 0, 1 + 0, 2 + 0, 3 + 0, 4 + 0] =
      (i : Int) := by
  interval_cases i0 <;> interval_cases i <;> rfl

theorem reqSigmaInner10 {i0 i : Nat} (hi : i < 5) {cur next : Int} {s : Fin 40 → Int} {a : Fin 5 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_78_10 W_GeneratedClaim.requirement_0_params i0 i
      (cur, s, Int.ofNat i0, a, ()) next) :
    ∃ hk : 5 * i0 + i < 40, next = cur + s ⟨5 * i0 + i, hk⟩ * a ⟨i, hi⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_78_10 at h
  dsimp only at h
  obtain ⟨w7, w129, w130, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_78_10.constraints_0] at h
  obtain ⟨h0, hlt, hg, -, -, -, hsel, -, -, ha, hout⟩ := h
  have hidx : Int.ofNat i0 * 5 + Int.ofNat i = Int.ofNat (5 * i0 + i) := by
    simp only [Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at hg hsel h0 hlt
  have hk : 5 * i0 + i < 40 := by simp only [Int.ofNat_eq_natCast] at hlt; omega
  obtain ⟨_, e7⟩ := familyGet_eq hg
  have e129 := select_eq_selectInt hsel
  rw [select_cell i0 i (by omega) hi] at e129
  subst e129
  obtain ⟨_, e130⟩ := familyGet_eq (show familyGetDynamic a (Int.ofNat i) w130 from ha)
  refine ⟨hk, ?_⟩
  rw [hout, e7, e130]

theorem reqSigmaInner14 {i0 i : Nat} (hi : i < 5) {cur next : Int} {s : Fin 40 → Int} {a : Fin 5 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_78_14 W_GeneratedClaim.requirement_0_params i0 i
      (cur, s, Int.ofNat i0, a, ()) next) :
    ∃ hk : 5 * i0 + i < 40, next = cur + s ⟨5 * i0 + i, hk⟩ * a ⟨i, hi⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_78_14 at h
  dsimp only at h
  obtain ⟨w7, w129, w130, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_78_14.constraints_0] at h
  obtain ⟨h0, hlt, hg, -, -, -, hsel, -, -, ha, hout⟩ := h
  have hidx : Int.ofNat i0 * 5 + Int.ofNat i = Int.ofNat (5 * i0 + i) := by
    simp only [Int.ofNat_eq_natCast]; push_cast; ring
  rw [hidx] at hg hsel h0 hlt
  have hk : 5 * i0 + i < 40 := by simp only [Int.ofNat_eq_natCast] at hlt; omega
  obtain ⟨_, e7⟩ := familyGet_eq hg
  have e129 := select_eq_selectInt hsel
  rw [select_cell i0 i (by omega) hi] at e129
  subst e129
  obtain ⟨_, e130⟩ := familyGet_eq (show familyGetDynamic a (Int.ofNat i) w130 from ha)
  refine ⟨hk, ?_⟩
  rw [hout, e7, e130]

theorem sigma_sum {i : Nat} (hi : i < 8) {s : Fin 40 → Int} {a : Fin 5 → Int} {init final : Int}
    {body : Nat → Int → Int → Prop}
    (hstep : ∀ k cur next, k < 5 → body k cur next →
      ∃ hk : 5 * i + k < 40, ∃ hk5 : k < 5, next = cur + s ⟨5 * i + k, hk⟩ * a ⟨k, hk5⟩)
    (h : MxxIR.IterRuns body (Int.toNat 5) init final) :
    final = init + ∑ k : Fin 5, s (cellIndex (⟨i, hi⟩ : Fin 8) k) * a k := by
  rw [IterRuns.sum_of_step (fun k ↦ if hk : k < 5 then s ⟨5 * i + k, by omega⟩ * a ⟨k, hk⟩ else 0)
    (fun k cur next hk hb ↦ by
      obtain ⟨_, _, rfl⟩ := hstep k cur next (by simpa using hk) hb
      simp [show k < 5 by simpa using hk]) h]
  rw [show Int.toNat 5 = 5 from rfl, Finset.sum_range]
  congr 1

theorem reqSigma {i : Nat} {cur next : Bool} {σ : Fin 8 → Int} {s s' : Fin 40 → Int} {a : Fin 5 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_78 W_GeneratedClaim.requirement_0_params i
      (cur, σ, s, a, s', ()) next) (hn : next = true) :
    cur = true ∧ ∀ hi : i < 8, σ ⟨i, hi⟩ = 1 - ((∑ k : Fin 5, s (cellIndex (⟨i, hi⟩ : Fin 8) k) * a k +
      ∑ k : Fin 5, s' (cellIndex (⟨i, hi⟩ : Fin 8) k) * a k) % 8) / 4 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_78.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_78.constraints_0] at hb
  obtain ⟨h0, hlt, hg, -, hl1, -, hl2, -, -, hout⟩ := hb
  obtain ⟨hi, e4⟩ := familyGet_eq hg
  have e10 := sigma_sum hi (fun k c n hk hb ↦ by
    obtain ⟨hk', e⟩ := reqSigmaInner10 hk hb; exact ⟨hk', hk, e⟩) hl1
  have e14 := sigma_sum hi (fun k c n hk hb ↦ by
    obtain ⟨hk', e⟩ := reqSigmaInner14 hk hb; exact ⟨hk', hk, e⟩) hl2
  rw [e4, e10, e14] at hout
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi' ↦ ?_⟩
  rw [hout.2]
  ring_nf

end RefreshTopProof
