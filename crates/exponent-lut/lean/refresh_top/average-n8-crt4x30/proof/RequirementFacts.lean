import TopRequirement

/-! The facts the requirement checks (written by `generator/gen_requirement.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut


theorem reqInner8 {i0 i : Nat} {cur next : Int} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_8_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 240, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_8_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_8_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck8 {i : Nat} {cur next : Bool} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_8 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 48, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_8.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_8.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 240 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner8 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner10 {i0 i : Nat} {cur next : Int} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_10_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 240, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_10_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_10_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck10 {i : Nat} {cur next : Bool} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_10 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 48, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_10.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_10.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 240 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner10 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner12 {i0 i : Nat} {cur next : Int} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_12_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 240, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_12_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_12_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck12 {i : Nat} {cur next : Bool} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_12 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 48, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_12.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_12.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 240 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner12 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner14 {i0 i : Nat} {cur next : Int} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_14_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 240, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_14_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_14_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck14 {i : Nat} {cur next : Bool} {F : Fin 240 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_14 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 48, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_14.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_14.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 240 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner14 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner16 {i0 i : Nat} {cur next : Int} {F : Fin 40 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_16_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 40, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_16_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_16_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck16 {i : Nat} {cur next : Bool} {F : Fin 40 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_16 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 8, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 8) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_16.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_16.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 40 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner16 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner18 {i0 i : Nat} {cur next : Int} {F : Fin 40 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_18_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 40, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_18_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_18_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck18 {i : Nat} {cur next : Bool} {F : Fin 40 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_18 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 8, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 8) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_18.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_18.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 40 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner18 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner20 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_20_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_20_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_20_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck20 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_20 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_20.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_20.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner20 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner22 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_22_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_22_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_22_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck22 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_22 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_22.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_22.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner22 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner24 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_24_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_24_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_24_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck24 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_24 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_24.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_24.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner24 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner26 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_26_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_26_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_26_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck26 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_26 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_26.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_26.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner26 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner28 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_28_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_28_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_28_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck28 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_28 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_28.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_28.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner28 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner30 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_30_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_30_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_30_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck30 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_30 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_30.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_30.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner30 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner32 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_32_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_32_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_32_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck32 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_32 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_32.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_32.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner32 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner34 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_34_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_34_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_34_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck34 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_34 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_34.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_34.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner34 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner36 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_36_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_36_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_36_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck36 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_36 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_36.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_36.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner36 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner38 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_38_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_38_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_38_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck38 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_38 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_38.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_38.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner38 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner40 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_40_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_40_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_40_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck40 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_40 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_40.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_40.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner40 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner42 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_42_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_42_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_42_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck42 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_42 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_42.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_42.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner42 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner44 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_44_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_44_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_44_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck44 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_44 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_44.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_44.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner44 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner46 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_46_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_46_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_46_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck46 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_46 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_46.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_46.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner46 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner48 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_48_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_48_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_48_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck48 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_48 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_48.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_48.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner48 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner50 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_50_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_50_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_50_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck50 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_50 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_50.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_50.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner50 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner52 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_52_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_52_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_52_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck52 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_52 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_52.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_52.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner52 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner54 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_54_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_54_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_54_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck54 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_54 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_54.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_54.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner54 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner56 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_56_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_56_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_56_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck56 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_56 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_56.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_56.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner56 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner58 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_58_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_58_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_58_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck58 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_58 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_58.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_58.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner58 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner60 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_60_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_60_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_60_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck60 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_60 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_60.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_60.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner60 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner62 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_62_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_62_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_62_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck62 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_62 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_62.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_62.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner62 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner64 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_64_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_64_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_64_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck64 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_64 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_64.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_64.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner64 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner66 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_66_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_66_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_66_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck66 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_66 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_66.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_66.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner66 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner68 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_68_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_68_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_68_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck68 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_68 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_68.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_68.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner68 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner70 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_70_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_70_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_70_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck70 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_70 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_70.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_70.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner70 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner72 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_72_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_72_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_72_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck72 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_72 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_72.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_72.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner72 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner74 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_74_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_74_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_74_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck74 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_74 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_74.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_74.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner74 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner76 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_76_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_76_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_76_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck76 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_76 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_76.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_76.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner76 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner78 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_78_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_78_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_78_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck78 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_78 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_78.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_78.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner78 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner80 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_80_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_80_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_80_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck80 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_80 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_80.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_80.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner80 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner82 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_82_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_82_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_82_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck82 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_82 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_82.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_82.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner82 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner84 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_84_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_84_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_84_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck84 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_84 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_84.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_84.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner84 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner86 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_86_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_86_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_86_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck86 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_86 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_86.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_86.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner86 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner88 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_88_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_88_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_88_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck88 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_88 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_88.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_88.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner88 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner90 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_90_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_90_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_90_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck90 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_90 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_90.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_90.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner90 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner92 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_92_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_92_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_92_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck92 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_92 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_92.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_92.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner92 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner94 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_94_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_94_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_94_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck94 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_94 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_94.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_94.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner94 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner96 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_96_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_96_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_96_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck96 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_96 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_96.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_96.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner96 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner98 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_98_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_98_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_98_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck98 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_98 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_98.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_98.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner98 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner100 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_100_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_100_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_100_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck100 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_100 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_100.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_100.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner100 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner102 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_102_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_102_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_102_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck102 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_102 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_102.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_102.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner102 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner104 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_104_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_104_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_104_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck104 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_104 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_104.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_104.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner104 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner106 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_106_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_106_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_106_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck106 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_106 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_106.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_106.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner106 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner108 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_108_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_108_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_108_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck108 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_108 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_108.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_108.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner108 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner110 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_110_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_110_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_110_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck110 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_110 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_110.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_110.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner110 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner112 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_112_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_112_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_112_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck112 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_112 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_112.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_112.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner112 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner114 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_114_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_114_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_114_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck114 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_114 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_114.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_114.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner114 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner116 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_116_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_116_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_116_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck116 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_116 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_116.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_116.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner116 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner118 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_118_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_118_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_118_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck118 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_118 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_118.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_118.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner118 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner120 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_120_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_120_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_120_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck120 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_120 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_120.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_120.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner120 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner122 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_122_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_122_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_122_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck122 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_122 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_122.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_122.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner122 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner124 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_124_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_124_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_124_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck124 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_124 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_124.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_124.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner124 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner126 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_126_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_126_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_126_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck126 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_126 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_126.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_126.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner126 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner128 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_128_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_128_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_128_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck128 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_128 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_128.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_128.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner128 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner130 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_130_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_130_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_130_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck130 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_130 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_130.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_130.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner130 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner132 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_132_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_132_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_132_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck132 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_132 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_132.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_132.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner132 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner134 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_134_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_134_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_134_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck134 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_134 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_134.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_134.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner134 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner136 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_136_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_136_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_136_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck136 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_136 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_136.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_136.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner136 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner138 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_138_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_138_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_138_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck138 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_138 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_138.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_138.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner138 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner140 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_140_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_140_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_140_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck140 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_140 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_140.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_140.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner140 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner142 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_142_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_142_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_142_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck142 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_142 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_142.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_142.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner142 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner144 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_144_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_144_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_144_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck144 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_144 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_144.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_144.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner144 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner146 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_146_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_146_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_146_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck146 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_146 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_146.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_146.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner146 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner148 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_148_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_148_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_148_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck148 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_148 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_148.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_148.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner148 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner150 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_150_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_150_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_150_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck150 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_150 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_150.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_150.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner150 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner152 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_152_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_152_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_152_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck152 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_152 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_152.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_152.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner152 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner154 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_154_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_154_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_154_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck154 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_154 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_154.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_154.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner154 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner156 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_156_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_156_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_156_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck156 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_156 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_156.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_156.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner156 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner158 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_158_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_158_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_158_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck158 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_158 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_158.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_158.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner158 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner160 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_160_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_160_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_160_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck160 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_160 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_160.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_160.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner160 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner162 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_162_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_162_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_162_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck162 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_162 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_162.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_162.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner162 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner164 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_164_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_164_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_164_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck164 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_164 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_164.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_164.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner164 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner166 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_166_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_166_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_166_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck166 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_166 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_166.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_166.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner166 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner168 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_168_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_168_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_168_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck168 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_168 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_168.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_168.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner168 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner170 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_170_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_170_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_170_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck170 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_170 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_170.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_170.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner170 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner172 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_172_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_172_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_172_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck172 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_172 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_172.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_172.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner172 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner174 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_174_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_174_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_174_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck174 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_174 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_174.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_174.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner174 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner176 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_176_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_176_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_176_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck176 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_176 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_176.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_176.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner176 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner178 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_178_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_178_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_178_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck178 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_178 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_178.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_178.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner178 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner180 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_180_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_180_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_180_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck180 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_180 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_180.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_180.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner180 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner182 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_182_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_182_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_182_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck182 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_182 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_182.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_182.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner182 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner184 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_184_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_184_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_184_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck184 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_184 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_184.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_184.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner184 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner186 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_186_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_186_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_186_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck186 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_186 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_186.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_186.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner186 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner188 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_188_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_188_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_188_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck188 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_188 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_188.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_188.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner188 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner190 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_190_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_190_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_190_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck190 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_190 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_190.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_190.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner190 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner192 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_192_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_192_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_192_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck192 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_192 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_192.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_192.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner192 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner194 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_194_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_194_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_194_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck194 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_194 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_194.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_194.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner194 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner196 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_196_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_196_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_196_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck196 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_196 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_196.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_196.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner196 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner198 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_198_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_198_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_198_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck198 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_198 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_198.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_198.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner198 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner200 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_200_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_200_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_200_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck200 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_200 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_200.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_200.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner200 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner202 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_202_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_202_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_202_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck202 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_202 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_202.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_202.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner202 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner204 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_204_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_204_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_204_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck204 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_204 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_204.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_204.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner204 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner206 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_206_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_206_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_206_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck206 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_206 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_206.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_206.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner206 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner208 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_208_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_208_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_208_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck208 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_208 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_208.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_208.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner208 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


theorem reqInner210 {i0 i : Nat} {cur next : Int} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_sequential_generatedRoot_210_5 W_GeneratedClaim.requirement_0_params i0 i
      (cur, F, Int.ofNat i0, ()) next) :
    ∃ hk : 5 * i0 + i < 800, next = cur + F ⟨5 * i0 + i, hk⟩ := by
  unfold W_Requirement_0.sequential_sequential_generatedRoot_210_5 at h
  dsimp only at h
  obtain ⟨w7, h⟩ := h
  simp only [W_Requirement_0.sequential_sequential_generatedRoot_210_5.constraints_0] at h
  obtain ⟨h0, hlt, hg, hout⟩ := h
  obtain ⟨hk, rfl⟩ := familyGet_int hg h0 hlt
  simp only [Int.ofNat_eq_natCast] at hk ⊢
  refine ⟨by omega, ?_⟩
  rw [hout]
  congr 2
  apply Fin.ext
  show (Int.ofNat i0 * 5 + Int.ofNat i).toNat = 5 * i0 + i
  rw [Int.ofNat_eq_natCast, Int.ofNat_eq_natCast]
  omega

theorem reqCheck210 {i : Nat} {cur next : Bool} {F : Fin 800 → Int}
    (h : W_Requirement_0.sequential_generatedRoot_210 W_GeneratedClaim.requirement_0_params i (cur, F, ()) next)
    (hn : next = true) :
    cur = true ∧ ∀ hi : i < 160, ∑ k : Fin 5, F (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1 := by
  obtain ⟨w, hb⟩ := h
  unfold W_Requirement_0.sequential_generatedRoot_210.body at hb
  dsimp only at hb
  simp only [W_Requirement_0.sequential_generatedRoot_210.constraints_0] at hb
  obtain ⟨-, hloop, hout⟩ := hb
  have hsum := IterRuns.sum_of_step (fun k ↦ if h : 5 * i + k < 800 then F ⟨5 * i + k, h⟩ else 0)
    (fun k c n _ hb ↦ by obtain ⟨hk, rfl⟩ := reqInner210 hb; simp [hk]) hloop
  subst hn
  replace hout := hout.symm
  simp only [decide_eq_true_eq, ite_add_ite_eq_two] at hout
  refine ⟨by simpa using hout.1, fun hi ↦ ?_⟩
  rw [hout.2, show Int.toNat 5 = 5 from rfl, zero_add, Finset.sum_range] at hsum
  rw [hsum]
  refine Finset.sum_congr rfl fun k _ ↦ ?_
  rw [dif_pos (by omega)]
  rfl


end RefreshTopProof
