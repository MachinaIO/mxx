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


end RefreshTopProof
