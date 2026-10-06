import LwrSetup

/-!
# The requirement and the ideal output

The requirement checks every member of `a` in one loop and sums every block's selectors in a loop;
it accepts plaintext monomials `c X^e` of `a` with `c X^e = X^(512a)` and one-hot selectors. The ideal
output sums `s_b(i) a(coords b i)` over the cells of every block in a loop, and is the half-integer
readout of the total `z` modulo `64`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxRecDepth 100000

variable {tape : SampleTape} {path : List Nat}

theorem ite_add_ite_eq_two (p q : Prop) [Decidable p] [Decidable q] :
    ((if p then (1 : Int) else 0) + (if q then 1 else 0) = 2) ↔ p ∧ q := by
  by_cases hp : p <;> by_cases hq : q <;> simp [hp, hq]

/-- One iteration of the member loop keeps a true state only for an accepted member. -/
theorem member_step {ac ae a : Fin 1025 → Int} {i : Nat} {current next : Bool}
    (h : Requirement_0.sequential_generatedRoot_6 tape path GeneratedClaim.requirement_0_params i
      (current, ac, ae, a, ()) next) (hnext : next = true) :
    current = true ∧ ∀ hi : i < 1025, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * 65536 = a ⟨i, hi⟩ * 1024 := by
  unfold Requirement_0.sequential_generatedRoot_6 at h
  dsimp only at h
  obtain ⟨w4, w14, w23, -, -, h4, -, -, h14, -, -, h23, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h4
  obtain ⟨hi14, rfl⟩ := familyGet_eq h14
  obtain ⟨hi23, rfl⟩ := familyGet_eq h23
  subst hnext
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨hc, hm⟩, hr⟩ := hout
  refine ⟨by simpa using hc, fun _ ↦ ⟨hm, ?_⟩⟩
  linarith

/-- One iteration of the block-0 selector loop adds the selector of cell `i`. -/
theorem selector_step0 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_10 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_10 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-1 selector loop adds the selector of cell `i`. -/
theorem selector_step1 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_20 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_20 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-2 selector loop adds the selector of cell `i`. -/
theorem selector_step2 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_30 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_30 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-3 selector loop adds the selector of cell `i`. -/
theorem selector_step3 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_40 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_40 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-4 selector loop adds the selector of cell `i`. -/
theorem selector_step4 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_50 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_50 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-5 selector loop adds the selector of cell `i`. -/
theorem selector_step5 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_60 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_60 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-6 selector loop adds the selector of cell `i`. -/
theorem selector_step6 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_70 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_70 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-7 selector loop adds the selector of cell `i`. -/
theorem selector_step7 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_80 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_80 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-8 selector loop adds the selector of cell `i`. -/
theorem selector_step8 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_90 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_90 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-9 selector loop adds the selector of cell `i`. -/
theorem selector_step9 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_100 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_100 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-10 selector loop adds the selector of cell `i`. -/
theorem selector_step10 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_110 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_110 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-11 selector loop adds the selector of cell `i`. -/
theorem selector_step11 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_120 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_120 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-12 selector loop adds the selector of cell `i`. -/
theorem selector_step12 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_130 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_130 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-13 selector loop adds the selector of cell `i`. -/
theorem selector_step13 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_140 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_140 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-14 selector loop adds the selector of cell `i`. -/
theorem selector_step14 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_150 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_150 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-15 selector loop adds the selector of cell `i`. -/
theorem selector_step15 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_160 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_160 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-16 selector loop adds the selector of cell `i`. -/
theorem selector_step16 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_170 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_170 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-17 selector loop adds the selector of cell `i`. -/
theorem selector_step17 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_180 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_180 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-18 selector loop adds the selector of cell `i`. -/
theorem selector_step18 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_190 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_190 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-19 selector loop adds the selector of cell `i`. -/
theorem selector_step19 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_200 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_200 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-20 selector loop adds the selector of cell `i`. -/
theorem selector_step20 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_210 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_210 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-21 selector loop adds the selector of cell `i`. -/
theorem selector_step21 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_220 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_220 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-22 selector loop adds the selector of cell `i`. -/
theorem selector_step22 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_230 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_230 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-23 selector loop adds the selector of cell `i`. -/
theorem selector_step23 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_240 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_240 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-24 selector loop adds the selector of cell `i`. -/
theorem selector_step24 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_250 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_250 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-25 selector loop adds the selector of cell `i`. -/
theorem selector_step25 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_260 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_260 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-26 selector loop adds the selector of cell `i`. -/
theorem selector_step26 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_270 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_270 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-27 selector loop adds the selector of cell `i`. -/
theorem selector_step27 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_280 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_280 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-28 selector loop adds the selector of cell `i`. -/
theorem selector_step28 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_290 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_290 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-29 selector loop adds the selector of cell `i`. -/
theorem selector_step29 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_300 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_300 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-30 selector loop adds the selector of cell `i`. -/
theorem selector_step30 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_310 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_310 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-31 selector loop adds the selector of cell `i`. -/
theorem selector_step31 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_320 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_320 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-32 selector loop adds the selector of cell `i`. -/
theorem selector_step32 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_330 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_330 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-33 selector loop adds the selector of cell `i`. -/
theorem selector_step33 {s : Fin 113 → Int} {i : Nat} {current next : Int}
    (h : Requirement_0.sequential_generatedRoot_340 tape path GeneratedClaim.requirement_0_params i
      (current, s, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ := by
  unfold Requirement_0.sequential_generatedRoot_340 at h
  dsimp only at h
  obtain ⟨w3, -, -, h3, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  exact ⟨hi, hout⟩

/-- One iteration of the block-0 sum of the ideal adds `s(i) a(coords 0 i)`. -/
theorem ideal_step0 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_4 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 0 ⟨i, hi⟩, coords_lt 0 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_4 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 0 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 0) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-1 sum of the ideal adds `s(i) a(coords 1 i)`. -/
theorem ideal_step1 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_8 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 1 ⟨i, hi⟩, coords_lt 1 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_8 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 1 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 1) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-2 sum of the ideal adds `s(i) a(coords 2 i)`. -/
theorem ideal_step2 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_12 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 2 ⟨i, hi⟩, coords_lt 2 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_12 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 2 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 2) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-3 sum of the ideal adds `s(i) a(coords 3 i)`. -/
theorem ideal_step3 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_16 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 3 ⟨i, hi⟩, coords_lt 3 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_16 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 3 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 3) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-4 sum of the ideal adds `s(i) a(coords 4 i)`. -/
theorem ideal_step4 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_20 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 4 ⟨i, hi⟩, coords_lt 4 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_20 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 4 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 4) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-5 sum of the ideal adds `s(i) a(coords 5 i)`. -/
theorem ideal_step5 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_24 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 5 ⟨i, hi⟩, coords_lt 5 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_24 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 5 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 5) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-6 sum of the ideal adds `s(i) a(coords 6 i)`. -/
theorem ideal_step6 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_28 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 6 ⟨i, hi⟩, coords_lt 6 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_28 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 6 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 6) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-7 sum of the ideal adds `s(i) a(coords 7 i)`. -/
theorem ideal_step7 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_32 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 7 ⟨i, hi⟩, coords_lt 7 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_32 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 7 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 7) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-8 sum of the ideal adds `s(i) a(coords 8 i)`. -/
theorem ideal_step8 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_36 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 8 ⟨i, hi⟩, coords_lt 8 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_36 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 8 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 8) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-9 sum of the ideal adds `s(i) a(coords 9 i)`. -/
theorem ideal_step9 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_40 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 9 ⟨i, hi⟩, coords_lt 9 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_40 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 9 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 9) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-10 sum of the ideal adds `s(i) a(coords 10 i)`. -/
theorem ideal_step10 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_44 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 10 ⟨i, hi⟩, coords_lt 10 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_44 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 10 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 10) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-11 sum of the ideal adds `s(i) a(coords 11 i)`. -/
theorem ideal_step11 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_48 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 11 ⟨i, hi⟩, coords_lt 11 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_48 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 11 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 11) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-12 sum of the ideal adds `s(i) a(coords 12 i)`. -/
theorem ideal_step12 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_52 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 12 ⟨i, hi⟩, coords_lt 12 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_52 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 12 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 12) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-13 sum of the ideal adds `s(i) a(coords 13 i)`. -/
theorem ideal_step13 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_56 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 13 ⟨i, hi⟩, coords_lt 13 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_56 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 13 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 13) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-14 sum of the ideal adds `s(i) a(coords 14 i)`. -/
theorem ideal_step14 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_60 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 14 ⟨i, hi⟩, coords_lt 14 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_60 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 14 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 14) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-15 sum of the ideal adds `s(i) a(coords 15 i)`. -/
theorem ideal_step15 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_64 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 15 ⟨i, hi⟩, coords_lt 15 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_64 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 15 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 15) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-16 sum of the ideal adds `s(i) a(coords 16 i)`. -/
theorem ideal_step16 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_68 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 16 ⟨i, hi⟩, coords_lt 16 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_68 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 16 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 16) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-17 sum of the ideal adds `s(i) a(coords 17 i)`. -/
theorem ideal_step17 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_72 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 17 ⟨i, hi⟩, coords_lt 17 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_72 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 17 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 17) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-18 sum of the ideal adds `s(i) a(coords 18 i)`. -/
theorem ideal_step18 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_76 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 18 ⟨i, hi⟩, coords_lt 18 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_76 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 18 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 18) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-19 sum of the ideal adds `s(i) a(coords 19 i)`. -/
theorem ideal_step19 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_80 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 19 ⟨i, hi⟩, coords_lt 19 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_80 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 19 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 19) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-20 sum of the ideal adds `s(i) a(coords 20 i)`. -/
theorem ideal_step20 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_84 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 20 ⟨i, hi⟩, coords_lt 20 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_84 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 20 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 20) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-21 sum of the ideal adds `s(i) a(coords 21 i)`. -/
theorem ideal_step21 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_88 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 21 ⟨i, hi⟩, coords_lt 21 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_88 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 21 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 21) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-22 sum of the ideal adds `s(i) a(coords 22 i)`. -/
theorem ideal_step22 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_92 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 22 ⟨i, hi⟩, coords_lt 22 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_92 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 22 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 22) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-23 sum of the ideal adds `s(i) a(coords 23 i)`. -/
theorem ideal_step23 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_96 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 23 ⟨i, hi⟩, coords_lt 23 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_96 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 23 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 23) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-24 sum of the ideal adds `s(i) a(coords 24 i)`. -/
theorem ideal_step24 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_100 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 24 ⟨i, hi⟩, coords_lt 24 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_100 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 24 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 24) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-25 sum of the ideal adds `s(i) a(coords 25 i)`. -/
theorem ideal_step25 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_104 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 25 ⟨i, hi⟩, coords_lt 25 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_104 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 25 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 25) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-26 sum of the ideal adds `s(i) a(coords 26 i)`. -/
theorem ideal_step26 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_108 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 26 ⟨i, hi⟩, coords_lt 26 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_108 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 26 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 26) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-27 sum of the ideal adds `s(i) a(coords 27 i)`. -/
theorem ideal_step27 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_112 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 27 ⟨i, hi⟩, coords_lt 27 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_112 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 27 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 27) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-28 sum of the ideal adds `s(i) a(coords 28 i)`. -/
theorem ideal_step28 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_116 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 28 ⟨i, hi⟩, coords_lt 28 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_116 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 28 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 28) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-29 sum of the ideal adds `s(i) a(coords 29 i)`. -/
theorem ideal_step29 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_120 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 29 ⟨i, hi⟩, coords_lt 29 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_120 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 29 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 29) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-30 sum of the ideal adds `s(i) a(coords 30 i)`. -/
theorem ideal_step30 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_124 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 30 ⟨i, hi⟩, coords_lt 30 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_124 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 30 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 30) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-31 sum of the ideal adds `s(i) a(coords 31 i)`. -/
theorem ideal_step31 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_128 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 31 ⟨i, hi⟩, coords_lt 31 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_128 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 31 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 31) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-32 sum of the ideal adds `s(i) a(coords 32 i)`. -/
theorem ideal_step32 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_132 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 32 ⟨i, hi⟩, coords_lt 32 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_132 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 32 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 32) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- One iteration of the block-33 sum of the ideal adds `s(i) a(coords 33 i)`. -/
theorem ideal_step33 {s : Fin 113 → Int} {a : Fin 1025 → Int} {i : Nat} {current next : Int}
    (h : Ideal.sequential_generatedRoot_136 tape path GeneratedClaim.ideal_params i
      (current, s, a, ()) next) :
    ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords 33 ⟨i, hi⟩, coords_lt 33 ⟨i, hi⟩⟩ := by
  unfold Ideal.sequential_generatedRoot_136 at h
  dsimp only at h
  obtain ⟨w3, w344, w345, -, -, h3, -, -, -, hsel, -, -, h345, hout⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_eq h3
  have hc : w344 = Int.ofNat (coords 33 ⟨i, hi⟩) :=
    (select_eq_selectInt hsel).trans (selectInt_map_ofNat (coordsTable 33) i)
  subst hc
  obtain ⟨hi', rfl⟩ := familyGet_eq h345
  exact ⟨hi, hout⟩

/-- A selector loop sums its family. -/
theorem selector_sum {s : Fin 113 → Int} {final : Int} {body : Nat → Int → Int → Prop}
    (hstep : ∀ i current next, body i current next → ∃ hi : i < 113, next = current + s ⟨i, hi⟩)
    (h : MxxIR.IterRuns body (Int.toNat 113) 0 final) : final = ∑ i, s i := by
  rw [IterRuns.sum_of_step (fun i ↦ if h : i < 113 then s ⟨i, h⟩ else 0) (fun i current next _ hb ↦ by
    obtain ⟨hi, rfl⟩ := hstep i current next hb
    simp [hi]) h, zero_add]
  exact sum_range_fin s

/-- An ideal block loop sums `s(i) a(coords b i)`. -/
theorem block_sum {s : Fin 113 → Int} {a : Fin 1025 → Int} (b : Fin 34) {final : Int}
    {body : Nat → Int → Int → Prop}
    (hstep : ∀ i current next, body i current next →
      ∃ hi : i < 113, next = current + s ⟨i, hi⟩ * a ⟨coords b ⟨i, hi⟩, coords_lt b ⟨i, hi⟩⟩)
    (h : MxxIR.IterRuns body (Int.toNat 113) 0 final) :
    final = ∑ i, s i * a ⟨coords b i, coords_lt b i⟩ := by
  rw [IterRuns.sum_of_step (fun i ↦ if h : i < 113 then s ⟨i, h⟩ * a ⟨coords b ⟨i, h⟩,
      coords_lt b ⟨i, h⟩⟩ else 0) (fun i current next _ hb ↦ by
    obtain ⟨hi, rfl⟩ := hstep i current next hb
    simp [hi]) h, zero_add]
  exact sum_range_fin (fun i ↦ s i * a ⟨coords b i, coords_lt b i⟩)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 100000 in
/-- The requirement: `c_k² = 1` and `2 e_k + 65536 (1 - c_k) = 1024 a_k` for every member, and every
block's selectors sum to one. -/
theorem requirement_holds {ac ae a : Fin 1025 → Int} {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 : Fin 113 → Int} {out : Bool}
    (h : Requirement_0.generatedRoot tape path GeneratedClaim.requirement_0_params
      (ac, ae, a, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17,
      s18, s19, s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, ()) out) (ht : out = true) :
    (∀ k : Fin 1025, ac k * ac k = 1 ∧ ae k * 2 + (1 - ac k) * 65536 = a k * 1024) ∧
      ∑ i, s0 i = 1 ∧ ∑ i, s1 i = 1 ∧ ∑ i, s2 i = 1 ∧ ∑ i, s3 i = 1 ∧ ∑ i, s4 i = 1 ∧
      ∑ i, s5 i = 1 ∧ ∑ i, s6 i = 1 ∧ ∑ i, s7 i = 1 ∧ ∑ i, s8 i = 1 ∧ ∑ i, s9 i = 1 ∧
      ∑ i, s10 i = 1 ∧ ∑ i, s11 i = 1 ∧ ∑ i, s12 i = 1 ∧ ∑ i, s13 i = 1 ∧ ∑ i, s14 i = 1 ∧
      ∑ i, s15 i = 1 ∧ ∑ i, s16 i = 1 ∧ ∑ i, s17 i = 1 ∧ ∑ i, s18 i = 1 ∧ ∑ i, s19 i = 1 ∧
      ∑ i, s20 i = 1 ∧ ∑ i, s21 i = 1 ∧ ∑ i, s22 i = 1 ∧ ∑ i, s23 i = 1 ∧ ∑ i, s24 i = 1 ∧
      ∑ i, s25 i = 1 ∧ ∑ i, s26 i = 1 ∧ ∑ i, s27 i = 1 ∧ ∑ i, s28 i = 1 ∧ ∑ i, s29 i = 1 ∧
      ∑ i, s30 i = 1 ∧ ∑ i, s31 i = 1 ∧ ∑ i, s32 i = 1 ∧ ∑ i, s33 i = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold Requirement_0.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨-, hm, -, h10_0, -, h20_0, -, h30_0, -, h40_0, -, h50_0, -, h60_0, -, h70_0, -, h80_0, -,
    h90_0, -, h100_0, -, h110_0, -, h120_0, -, h130_0, -, h140_0, -, h150_0, -, h160_0, -, h170_0,
    -, h180_0, -, h190_0, -, h200_0, -, h210_0, -, h220_0, -, h230_0, -, h240_0, -, h250_0, -,
    h260_0, -, h270_0, -, h280_0, -, h290_0, -, h300_0, -, h310_0, -, h320_0, -, h330_0, -, h340_0,
    hout⟩ := hb
  subst ht
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨⟨hw6, e10⟩, e20⟩, e30⟩, e40⟩, e50⟩, e60⟩, e70⟩, e80⟩, e90⟩, e100⟩, e110⟩, e120⟩, e130⟩, e140⟩, e150⟩, e160⟩, e170⟩, e180⟩, e190⟩, e200⟩, e210⟩, e220⟩, e230⟩, e240⟩, e250⟩, e260⟩, e270⟩, e280⟩, e290⟩, e300⟩, e310⟩, e320⟩, e330⟩, e340⟩ := hout
  have hmem := (IterRuns.all_of_step (fun i ↦ ∀ hi : i < 1025, ac ⟨i, hi⟩ * ac ⟨i, hi⟩ = 1 ∧
      ae ⟨i, hi⟩ * 2 + (1 - ac ⟨i, hi⟩) * 65536 = a ⟨i, hi⟩ * 1024)
    (fun i current next _ hb hn ↦ member_step hb hn) hm (by simpa using hw6)).2
  refine ⟨fun k ↦ hmem k.val (lt_of_lt_of_eq k.isLt rfl) k.isLt,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [← selector_sum (fun i c n hb ↦ selector_step0 hb) h10_0]; exact e10
  · rw [← selector_sum (fun i c n hb ↦ selector_step1 hb) h20_0]; exact e20
  · rw [← selector_sum (fun i c n hb ↦ selector_step2 hb) h30_0]; exact e30
  · rw [← selector_sum (fun i c n hb ↦ selector_step3 hb) h40_0]; exact e40
  · rw [← selector_sum (fun i c n hb ↦ selector_step4 hb) h50_0]; exact e50
  · rw [← selector_sum (fun i c n hb ↦ selector_step5 hb) h60_0]; exact e60
  · rw [← selector_sum (fun i c n hb ↦ selector_step6 hb) h70_0]; exact e70
  · rw [← selector_sum (fun i c n hb ↦ selector_step7 hb) h80_0]; exact e80
  · rw [← selector_sum (fun i c n hb ↦ selector_step8 hb) h90_0]; exact e90
  · rw [← selector_sum (fun i c n hb ↦ selector_step9 hb) h100_0]; exact e100
  · rw [← selector_sum (fun i c n hb ↦ selector_step10 hb) h110_0]; exact e110
  · rw [← selector_sum (fun i c n hb ↦ selector_step11 hb) h120_0]; exact e120
  · rw [← selector_sum (fun i c n hb ↦ selector_step12 hb) h130_0]; exact e130
  · rw [← selector_sum (fun i c n hb ↦ selector_step13 hb) h140_0]; exact e140
  · rw [← selector_sum (fun i c n hb ↦ selector_step14 hb) h150_0]; exact e150
  · rw [← selector_sum (fun i c n hb ↦ selector_step15 hb) h160_0]; exact e160
  · rw [← selector_sum (fun i c n hb ↦ selector_step16 hb) h170_0]; exact e170
  · rw [← selector_sum (fun i c n hb ↦ selector_step17 hb) h180_0]; exact e180
  · rw [← selector_sum (fun i c n hb ↦ selector_step18 hb) h190_0]; exact e190
  · rw [← selector_sum (fun i c n hb ↦ selector_step19 hb) h200_0]; exact e200
  · rw [← selector_sum (fun i c n hb ↦ selector_step20 hb) h210_0]; exact e210
  · rw [← selector_sum (fun i c n hb ↦ selector_step21 hb) h220_0]; exact e220
  · rw [← selector_sum (fun i c n hb ↦ selector_step22 hb) h230_0]; exact e230
  · rw [← selector_sum (fun i c n hb ↦ selector_step23 hb) h240_0]; exact e240
  · rw [← selector_sum (fun i c n hb ↦ selector_step24 hb) h250_0]; exact e250
  · rw [← selector_sum (fun i c n hb ↦ selector_step25 hb) h260_0]; exact e260
  · rw [← selector_sum (fun i c n hb ↦ selector_step26 hb) h270_0]; exact e270
  · rw [← selector_sum (fun i c n hb ↦ selector_step27 hb) h280_0]; exact e280
  · rw [← selector_sum (fun i c n hb ↦ selector_step28 hb) h290_0]; exact e290
  · rw [← selector_sum (fun i c n hb ↦ selector_step29 hb) h300_0]; exact e300
  · rw [← selector_sum (fun i c n hb ↦ selector_step30 hb) h310_0]; exact e310
  · rw [← selector_sum (fun i c n hb ↦ selector_step31 hb) h320_0]; exact e320
  · rw [← selector_sum (fun i c n hb ↦ selector_step32 hb) h330_0]; exact e330
  · rw [← selector_sum (fun i c n hb ↦ selector_step33 hb) h340_0]; exact e340

/-- The total `z = Σ_b Σ_i s_b(i) a(coords b i)` of the selected coordinates. -/
def selectedSum (a : Fin 1025 → Int) (s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 : Fin 113 → Int) : Int :=
  0 + ∑ i, s0 i * a ⟨coords 0 i, coords_lt 0 i⟩ + ∑ i, s1 i * a ⟨coords 1 i, coords_lt 1 i⟩ +
    ∑ i, s2 i * a ⟨coords 2 i, coords_lt 2 i⟩ + ∑ i, s3 i * a ⟨coords 3 i, coords_lt 3 i⟩ +
    ∑ i, s4 i * a ⟨coords 4 i, coords_lt 4 i⟩ + ∑ i, s5 i * a ⟨coords 5 i, coords_lt 5 i⟩ +
    ∑ i, s6 i * a ⟨coords 6 i, coords_lt 6 i⟩ + ∑ i, s7 i * a ⟨coords 7 i, coords_lt 7 i⟩ +
    ∑ i, s8 i * a ⟨coords 8 i, coords_lt 8 i⟩ + ∑ i, s9 i * a ⟨coords 9 i, coords_lt 9 i⟩ +
    ∑ i, s10 i * a ⟨coords 10 i, coords_lt 10 i⟩ + ∑ i, s11 i * a ⟨coords 11 i, coords_lt 11 i⟩ +
    ∑ i, s12 i * a ⟨coords 12 i, coords_lt 12 i⟩ + ∑ i, s13 i * a ⟨coords 13 i, coords_lt 13 i⟩ +
    ∑ i, s14 i * a ⟨coords 14 i, coords_lt 14 i⟩ + ∑ i, s15 i * a ⟨coords 15 i, coords_lt 15 i⟩ +
    ∑ i, s16 i * a ⟨coords 16 i, coords_lt 16 i⟩ + ∑ i, s17 i * a ⟨coords 17 i, coords_lt 17 i⟩ +
    ∑ i, s18 i * a ⟨coords 18 i, coords_lt 18 i⟩ + ∑ i, s19 i * a ⟨coords 19 i, coords_lt 19 i⟩ +
    ∑ i, s20 i * a ⟨coords 20 i, coords_lt 20 i⟩ + ∑ i, s21 i * a ⟨coords 21 i, coords_lt 21 i⟩ +
    ∑ i, s22 i * a ⟨coords 22 i, coords_lt 22 i⟩ + ∑ i, s23 i * a ⟨coords 23 i, coords_lt 23 i⟩ +
    ∑ i, s24 i * a ⟨coords 24 i, coords_lt 24 i⟩ + ∑ i, s25 i * a ⟨coords 25 i, coords_lt 25 i⟩ +
    ∑ i, s26 i * a ⟨coords 26 i, coords_lt 26 i⟩ + ∑ i, s27 i * a ⟨coords 27 i, coords_lt 27 i⟩ +
    ∑ i, s28 i * a ⟨coords 28 i, coords_lt 28 i⟩ + ∑ i, s29 i * a ⟨coords 29 i, coords_lt 29 i⟩ +
    ∑ i, s30 i * a ⟨coords 30 i, coords_lt 30 i⟩ + ∑ i, s31 i * a ⟨coords 31 i, coords_lt 31 i⟩ +
    ∑ i, s32 i * a ⟨coords 32 i, coords_lt 32 i⟩ + ∑ i, s33 i * a ⟨coords 33 i, coords_lt 33 i⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 100000 in
/-- The ideal output: `δ mod 64` for the half-integer readout `δ = (-1)^b m` of
`z mod 256 = r + 128 b`, `m = 2 τ(r) - 1`, `τ(r) = ⌊16 r / 128⌋ + 1`: `m` for `b = 0` and `64 - m` for
`b = 1`. -/
theorem ideal_value {a : Fin 1025 → Int} {s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 : Fin 113 → Int} {out : Int}
    (h : Ideal.generatedRoot tape path GeneratedClaim.ideal_params (s0, a, s1, s2, s3, s4, s5, s6,
      s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25,
      s26, s27, s28, s29, s30, s31, s32, s33, ()) out) :
    out = ((selectedSum a s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 % 128 * 16 / 128 + 1) * 2 - 1) +
      selectedSum a s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 % 256 / 128 *
        (64 - ((selectedSum a s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19
      s20 s21 s22 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 % 128 * 16 / 128 + 1) * 2 - 1) * 2) := by
  obtain ⟨w, hb⟩ := h
  unfold Ideal.generatedRoot.body at hb
  dsimp only at hb
  obtain ⟨-, h4_0, -, h8_0, -, h12_0, -, h16_0, -, h20_0, -, h24_0, -, h28_0, -, h32_0, -, h36_0, -,
    h40_0, -, h44_0, -, h48_0, -, h52_0, -, h56_0, -, h60_0, -, h64_0, -, h68_0, -, h72_0, -, h76_0,
    -, h80_0, -, h84_0, -, h88_0, -, h92_0, -, h96_0, -, h100_0, -, h104_0, -, h108_0, -, h112_0, -,
    h116_0, -, h120_0, -, h124_0, -, h128_0, -, h132_0, -, h136_0, h138_0, h142_0, h150_0, h152_0,
    hout⟩ := hb
  rw [hout, block_sum 0 (fun i c n hb ↦ ideal_step0 hb) h4_0,
    block_sum 1 (fun i c n hb ↦ ideal_step1 hb) h8_0,
    block_sum 2 (fun i c n hb ↦ ideal_step2 hb) h12_0,
    block_sum 3 (fun i c n hb ↦ ideal_step3 hb) h16_0,
    block_sum 4 (fun i c n hb ↦ ideal_step4 hb) h20_0,
    block_sum 5 (fun i c n hb ↦ ideal_step5 hb) h24_0,
    block_sum 6 (fun i c n hb ↦ ideal_step6 hb) h28_0,
    block_sum 7 (fun i c n hb ↦ ideal_step7 hb) h32_0,
    block_sum 8 (fun i c n hb ↦ ideal_step8 hb) h36_0,
    block_sum 9 (fun i c n hb ↦ ideal_step9 hb) h40_0,
    block_sum 10 (fun i c n hb ↦ ideal_step10 hb) h44_0,
    block_sum 11 (fun i c n hb ↦ ideal_step11 hb) h48_0,
    block_sum 12 (fun i c n hb ↦ ideal_step12 hb) h52_0,
    block_sum 13 (fun i c n hb ↦ ideal_step13 hb) h56_0,
    block_sum 14 (fun i c n hb ↦ ideal_step14 hb) h60_0,
    block_sum 15 (fun i c n hb ↦ ideal_step15 hb) h64_0,
    block_sum 16 (fun i c n hb ↦ ideal_step16 hb) h68_0,
    block_sum 17 (fun i c n hb ↦ ideal_step17 hb) h72_0,
    block_sum 18 (fun i c n hb ↦ ideal_step18 hb) h76_0,
    block_sum 19 (fun i c n hb ↦ ideal_step19 hb) h80_0,
    block_sum 20 (fun i c n hb ↦ ideal_step20 hb) h84_0,
    block_sum 21 (fun i c n hb ↦ ideal_step21 hb) h88_0,
    block_sum 22 (fun i c n hb ↦ ideal_step22 hb) h92_0,
    block_sum 23 (fun i c n hb ↦ ideal_step23 hb) h96_0,
    block_sum 24 (fun i c n hb ↦ ideal_step24 hb) h100_0,
    block_sum 25 (fun i c n hb ↦ ideal_step25 hb) h104_0,
    block_sum 26 (fun i c n hb ↦ ideal_step26 hb) h108_0,
    block_sum 27 (fun i c n hb ↦ ideal_step27 hb) h112_0,
    block_sum 28 (fun i c n hb ↦ ideal_step28 hb) h116_0,
    block_sum 29 (fun i c n hb ↦ ideal_step29 hb) h120_0,
    block_sum 30 (fun i c n hb ↦ ideal_step30 hb) h124_0,
    block_sum 31 (fun i c n hb ↦ ideal_step31 hb) h128_0,
    block_sum 32 (fun i c n hb ↦ ideal_step32 hb) h132_0,
    block_sum 33 (fun i c n hb ↦ ideal_step33 hb) h136_0]
  rfl

end LwrProof
