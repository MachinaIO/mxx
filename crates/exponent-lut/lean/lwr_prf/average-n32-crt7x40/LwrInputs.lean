import LwrSetup
import ExponentIterate

/-!
# The requirement and the ideal output

The requirement checks every member of `a` in one loop and sums every block's selectors in a loop;
it accepts plaintext monomials `c X^e` of `a` with `c X^e = X^(2a)` and one-hot selectors. The ideal
output sums `s_b(i) a(coords b i)` over the cells of every block in a loop, and is the parity of the
readout `τ(z mod 16)` of the total `z`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path : List Nat}

theorem familyGetDynamic_ofNat {α : Type} {count i : Nat} {f : Fin count → α} {out : α}
    (h : familyGetDynamic f (Int.ofNat i) out) : ∃ hi : i < count, out = f ⟨i, hi⟩ := by
  obtain ⟨p, hp, rfl⟩ := h
  have : p.val = i := Int.ofNat.inj hp
  subst this
  exact ⟨p.isLt, rfl⟩

theorem ite_add_ite_eq_two (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then (1 : Int) else 0) + (if q then 1 else 0) = 2) ↔ p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

/-- One iteration of the member loop keeps a true state only for an accepted member. -/
theorem member_step {ac ae a : Fin 17 → Int} {i : Nat} {current next : Bool}
    (h : Requirement_0.sequential_generatedRoot_6 tape path GeneratedClaim.requirement_0_params i
      (current, ac, ae, a, ()) next) (hnext : next = true) :
    current = true ∧ ∀ hi : i < 17, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * 32 = a ⟨i, hi⟩ * 4 := by
  unfold Requirement_0.sequential_generatedRoot_6 at h
  dsimp only at h
  obtain ⟨w4, w14, w23, -, -, h4, -, -, h14, -, -, h23, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h4
  obtain ⟨hi14, rfl⟩ := familyGetDynamic_ofNat h14
  obtain ⟨hi23, rfl⟩ := familyGetDynamic_ofNat h23
  subst hnext
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨hc, hm⟩, hr⟩ := hout
  refine ⟨by simpa using hc, fun _ ↦ ⟨hm, ?_⟩⟩
  linarith

/-- One iteration of the block-0 selector loop adds the selector of cell `i`. -/
theorem selector_step0 {s : Fin 14 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_10 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_10 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-0 sum of the ideal adds `s(i) a(coords 0 i)`. -/
theorem ideal_step0 {s : Fin 14 → Int} {a : Fin 17 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_4 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ * a ⟨coords 0 ⟨i, hi⟩, coords_lt 0 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_4 at h
  dsimp only at h
  obtain ⟨w3, w47, w48, -, -, h3, -, -, -, hsel, -, -, h48, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  have hc : w47 = Int.ofNat (coords 0 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 0) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGetDynamic_ofNat h48
  exact ⟨hi, hout⟩

/-- One iteration of the block-1 selector loop adds the selector of cell `i`. -/
theorem selector_step1 {s : Fin 14 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_20 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_20 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-2 selector loop adds the selector of cell `i`. -/
theorem selector_step2 {s : Fin 14 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_30 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_30 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-3 selector loop adds the selector of cell `i`. -/
theorem selector_step3 {s : Fin 14 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_40 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_40 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-4 selector loop adds the selector of cell `i`. -/
theorem selector_step4 {s : Fin 14 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_50 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_50 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-1 sum of the ideal adds `s(i) a(coords 1 i)`. -/
theorem ideal_step1 {s : Fin 14 → Int} {a : Fin 17 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_8 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ * a ⟨coords 1 ⟨i, hi⟩, coords_lt 1 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_8 at h
  dsimp only at h
  obtain ⟨w3, w47, w48, -, -, h3, -, -, -, hsel, -, -, h48, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  have hc : w47 = Int.ofNat (coords 1 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 1) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGetDynamic_ofNat h48
  exact ⟨hi, hout⟩

/-- One iteration of the block-2 sum of the ideal adds `s(i) a(coords 2 i)`. -/
theorem ideal_step2 {s : Fin 14 → Int} {a : Fin 17 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_12 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ * a ⟨coords 2 ⟨i, hi⟩, coords_lt 2 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_12 at h
  dsimp only at h
  obtain ⟨w3, w47, w48, -, -, h3, -, -, -, hsel, -, -, h48, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  have hc : w47 = Int.ofNat (coords 2 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 2) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGetDynamic_ofNat h48
  exact ⟨hi, hout⟩

/-- One iteration of the block-3 sum of the ideal adds `s(i) a(coords 3 i)`. -/
theorem ideal_step3 {s : Fin 14 → Int} {a : Fin 17 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_16 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ * a ⟨coords 3 ⟨i, hi⟩, coords_lt 3 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_16 at h
  dsimp only at h
  obtain ⟨w3, w47, w48, -, -, h3, -, -, -, hsel, -, -, h48, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  have hc : w47 = Int.ofNat (coords 3 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 3) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGetDynamic_ofNat h48
  exact ⟨hi, hout⟩

/-- One iteration of the block-4 sum of the ideal adds `s(i) a(coords 4 i)`. -/
theorem ideal_step4 {s : Fin 14 → Int} {a : Fin 17 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_20 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 14, next = current + s ⟨i, hi⟩ * a ⟨coords 4 ⟨i, hi⟩, coords_lt 4 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_20 at h
  dsimp only at h
  obtain ⟨w3, w47, w48, -, -, h3, -, -, -, hsel, -, -, h48, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGetDynamic_ofNat h3
  have hc : w47 = Int.ofNat (coords 4 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 4) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGetDynamic_ofNat h48
  exact ⟨hi, hout⟩

/-- A selector loop sums its family. -/
theorem selector_sum {s : Fin 14 → Int} {final : Int} {body : Nat → Int → Int → Prop}
    (hstep : ∀ i current next, body i current next → ∃ hi : i < 14, next = current + s ⟨i, hi⟩)
    (h : MxxIR.IterRuns body (Int.toNat 14) 0 final) : final = ∑ i, s i := by
  rw [IterRuns.sum_of_step (fun i ↦ if h : i < 14 then s ⟨i, h⟩ else 0) (fun i current next _ hb ↦ by
    obtain ⟨hi, rfl⟩ := hstep i current next hb
    simp [hi]) h, zero_add]
  exact sum_range_fin s

/-- The sum `z = Σ_b Σ_i s_b(i) a(coords b i)` of the selected coordinates. -/
def selectedSum (a : Fin 17 → Int) (s : Fin 5 → Fin 14 → Int) : Int :=
  ∑ b : Fin 5, ∑ i : Fin 14, s b i * a ⟨coords b i, coords_lt b i⟩

/-- An ideal block loop sums `s(i) a(coords b i)`. -/
theorem block_sum {s : Fin 14 → Int} {a : Fin 17 → Int} (b : Fin 5) {final : Int}
    {body : Nat → Int → Int → Prop}
    (hstep : ∀ i current next, body i current next →
      ∃ hi : i < 14, next = current + s ⟨i, hi⟩ * a ⟨coords b ⟨i, hi⟩, coords_lt b ⟨i, hi⟩⟩)
    (h : MxxIR.IterRuns body (Int.toNat 14) 0 final) :
    final = ∑ i, s i * a ⟨coords b i, coords_lt b i⟩ := by
  rw [IterRuns.sum_of_step (fun i ↦ if h : i < 14 then s ⟨i, h⟩ * a ⟨coords b ⟨i, h⟩,
      coords_lt b ⟨i, h⟩⟩ else 0) (fun i current next _ hb ↦ by
    obtain ⟨hi, rfl⟩ := hstep i current next hb
    simp [hi]) h, zero_add]
  exact sum_range_fin (fun i ↦ s i * a ⟨coords b i, coords_lt b i⟩)

set_option maxHeartbeats 4000000 in
/-- The requirement: `c_k² = 1` and `2 e_k + 32 (1 - c_k) = 4 a_k` for every member, and every block's
selectors sum to one. -/
theorem requirement_holds {ac ae a : Fin 17 → Int} {s0 s1 s2 s3 s4 : Fin 14 → Int} {out : Bool}
    (h : Requirement_0.generatedRoot tape path GeneratedClaim.requirement_0_params
      (ac, ae, a, s0, s1, s2, s3, s4, ()) out) (ht : out = true) :
    (∀ k : Fin 17, ac k * ac k = 1 ∧ ae k * 2 + (1 - ac k) * 32 = a k * 4) ∧
      ∑ i, s0 i = 1 ∧ ∑ i, s1 i = 1 ∧ ∑ i, s2 i = 1 ∧ ∑ i, s3 i = 1 ∧ ∑ i, s4 i = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold Requirement_0.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨-, hm, -, h10, -, h20, -, h30, -, h40, -, h50, hout⟩ := hb
  subst ht
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨⟨⟨⟨hw6, e10⟩, e20⟩, e30⟩, e40⟩, e50⟩ := hout
  have hmem := (IterRuns.all_of_step (fun i ↦ ∀ hi : i < 17, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * 32 = a ⟨i, hi⟩ * 4)
    (fun i current next _ hb hn ↦ member_step hb hn) hm (by simpa using hw6)).2
  refine ⟨fun k ↦ hmem k.val (lt_of_lt_of_eq k.isLt rfl) k.isLt, ?_, ?_, ?_, ?_, ?_⟩
  · rw [← selector_sum (fun i c n hb ↦ selector_step0 hb) h10]; exact e10
  · rw [← selector_sum (fun i c n hb ↦ selector_step1 hb) h20]; exact e20
  · rw [← selector_sum (fun i c n hb ↦ selector_step2 hb) h30]; exact e30
  · rw [← selector_sum (fun i c n hb ↦ selector_step3 hb) h40]; exact e40
  · rw [← selector_sum (fun i c n hb ↦ selector_step4 hb) h50]; exact e50

set_option maxHeartbeats 4000000 in
/-- The ideal output: the parity of `τ(z mod 16) = ⌊2 (z mod 16) / 16⌋ + 1`. -/
theorem ideal_value {a : Fin 17 → Int} {s0 s1 s2 s3 s4 : Fin 14 → Int} {out : Int}
    (h : Ideal.generatedRoot tape path GeneratedClaim.ideal_params (s0, a, s1, s2, s3, s4, ()) out) :
    out = (selectedSum a ![s0, s1, s2, s3, s4] % 16 * 2 / 16 + 1) % 2 := by
  obtain ⟨w, hb⟩ := h
  unfold Ideal.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨-, h4, -, h8, -, h12, -, h16, -, h20, -, -, -, hout⟩ := hb
  rw [hout, block_sum 0 (fun i c n hb ↦ ideal_step0 hb) h4, block_sum 1 (fun i c n hb ↦ ideal_step1 hb) h8,
    block_sum 2 (fun i c n hb ↦ ideal_step2 hb) h12, block_sum 3 (fun i c n hb ↦ ideal_step3 hb) h16,
    block_sum 4 (fun i c n hb ↦ ideal_step4 hb) h20]
  simp only [selectedSum, Fin.sum_univ_five, zero_add]
  rfl

end LwrProof
