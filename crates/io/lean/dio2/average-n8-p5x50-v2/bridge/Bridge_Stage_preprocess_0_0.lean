import Stage_preprocess_0_0
import W_Stage_preprocess_0_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preprocess_0_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preprocess_0_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_3_12.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs17 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_3_13.bridge hs17))), h14, h15, h16⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_3.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_3 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_3 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_3
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_3.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_4_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_4_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_4.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_4 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_4 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_4
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_4.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_5_12.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs17 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_5_13.bridge hs17))), h14, h15, h16⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_5.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_5 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_5 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_5
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_5.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_22_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_6_12.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs17 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_6_13.bridge hs17))), h14, h15, h16⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_6.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_6 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_6 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_6
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_6.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_7_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_7_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_7.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_7 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_7 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_7
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_7.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_8_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_8_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_8.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_8 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_8 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_8
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_8.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_9_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_9_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_9.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_9 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_9 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_9
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_9.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_10_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_10_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_10.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_10 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_10 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_10
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_10.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_11_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_11_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_11.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_11 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_11 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_11
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_11.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_12_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_12_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_12.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_12 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_12 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_12
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_12.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_13_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_13_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_13.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_13 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_13 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_13
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_14_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_14_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_14.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_14 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_14 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_14
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_14.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_21_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_15_10.bridge (h11 i))), h12, (Stage_preprocess_0_0.iterRuns_mono h13 (fun _ _ _ hs16 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_15_13.bridge hs16))), h14, h15⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_15.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_15 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_15 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_15
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0⟩, (Stage_preprocess_0_0.parallel_generatedRoot_15.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_36.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_36 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_36 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_36
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_69.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_69 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_69 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_69
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_79.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_79 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_79 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_79
  exact h


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_80_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_80_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_80.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_80 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_80 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_80
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_80.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_83.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_83 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_83 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_83
  exact h


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_84_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_84_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_84.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_84 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_84 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_84
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_84.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_85_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_85.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_85 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_85 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_85
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_85.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_86.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_86 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_86 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_86
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_99.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_99 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_99 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_99
  exact h


theorem Stage_preprocess_0_0.scope_89.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.scope_89 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.scope_89 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.scope_89
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_107.constraints_0.bridge {backend hashModel tape path params params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_107.constraints_0 backend hashModel tape path params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0) : W_Stage_preprocess_0_0.sequential_generatedRoot_107.constraints_0 backend hashModel params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess_0_0.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.scope_89.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.sequential_generatedRoot_107.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_107 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_107 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_107
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_6_0, wit1.w_9_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_17_0, wit1.w_18_0, wit1.w_20_0, wit1.w_20_1, wit1.w_20_2, wit1.w_20_3, wit1.w_21_0, wit1.w_25_0, wit1.w_29_0, wit1.w_33_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preprocess_0_0.sequential_generatedRoot_107.constraints_0.bridge h3), h4⟩)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_116.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_116 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_116 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_116
  exact h


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_117_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_117_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_117.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_117 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_117 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_117
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_117.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_120.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_120 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_120 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_120
  exact h


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_121_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_121_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_121.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_121 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_121 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_121
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_121.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_122_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_122.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_122 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_122 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_122
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_122.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_123_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_123.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_123 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_123 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_123
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_123.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_124.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_124 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_124 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_124
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_137.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_137 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_137 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_137
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_139.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_139 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_139 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_139
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_141.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_141 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_141 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_141
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_143.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_143 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_143 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_143
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_145.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_145 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_145 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_145
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_147.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_147 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_147 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_147
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_149.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_149 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_149 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_149
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_151.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_151 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_151 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_151
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_161.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_161 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_161 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_161
  exact h


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_162_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_162_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_162_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_162_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_162_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_162_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_162_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_162_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_162.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_162.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_162.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_162_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_162_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_162.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_162 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_162 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_162
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_162.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_165.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_165 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_165 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_165
  exact h


theorem Stage_preprocess_0_0.parallel_parallel_generatedRoot_166_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_parallel_generatedRoot_166_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_166_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_parallel_generatedRoot_166_4
  exact h


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_166_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_166_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_166_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_166_17
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_166.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_166.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_166.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_parallel_generatedRoot_166_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_166_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_166.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_166 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_166 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_166
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_166.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.sequential_parallel_generatedRoot_167_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_parallel_generatedRoot_167_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_167_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_parallel_generatedRoot_167_10
  exact h


theorem Stage_preprocess_0_0.parallel_generatedRoot_167.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_167.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_167.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_0_0.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_0_0.sequential_parallel_generatedRoot_167_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_167.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_167 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_167 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_167
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_0_0.parallel_generatedRoot_167.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_0_0.parallel_generatedRoot_168.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.parallel_generatedRoot_168 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.parallel_generatedRoot_168 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.parallel_generatedRoot_168
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_181.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_181 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_181 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_181
  exact h


theorem Stage_preprocess_0_0.scope_90.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_0_0.scope_90 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_0_0.scope_90 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_0_0.scope_90
  exact h


theorem Stage_preprocess_0_0.sequential_generatedRoot_189.constraints_0.bridge {backend hashModel tape path params params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_189.constraints_0 backend hashModel tape path params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0) : W_Stage_preprocess_0_0.sequential_generatedRoot_189.constraints_0 backend hashModel params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess_0_0.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.scope_90.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.sequential_generatedRoot_189.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_0_0.sequential_generatedRoot_189 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_0_0.sequential_generatedRoot_189 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_0_0.sequential_generatedRoot_189
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_6_0, wit1.w_9_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_17_0, wit1.w_18_0, wit1.w_20_0, wit1.w_20_1, wit1.w_20_2, wit1.w_20_3, wit1.w_21_0, wit1.w_25_0, wit1.w_29_0, wit1.w_33_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preprocess_0_0.sequential_generatedRoot_189.constraints_0.bridge h3), h4⟩)⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_7.bridge {backend hashModel tape path params params' w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_36_2 w_36_3 w_69_0 w_69_1 w_69_2 w_69_3 w_79_6 w_79_7 w_83_6 w_83_7 w_107_0 w_107_1 w_107_2 w_107_3 w_116_6 w_116_7 w_120_6 w_120_7 w_137_0 w_137_1 w_137_2 w_137_3 w_139_0 w_139_1 w_139_2 w_139_3 w_141_0 w_141_1 w_141_2 w_141_3 w_143_0 w_143_1 w_143_2 w_143_3 w_145_0 w_145_1 w_145_2 w_145_3 w_147_0 w_147_1 w_147_2 w_147_3 w_149_0 w_149_1 w_149_2 w_149_3 w_151_0 w_151_1 w_151_2 w_151_3 w_152_0 w_161_6 w_161_7 w_165_6 w_165_7 w_168_0 w_168_1 w_168_2 w_168_3 w_181_0 w_181_1 w_181_2 w_181_3 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_189_1 w_189_2 w_189_3 outputs}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_7 backend hashModel tape path params w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_36_2 w_36_3 w_69_0 w_69_1 w_69_2 w_69_3 w_79_6 w_79_7 w_83_6 w_83_7 w_107_0 w_107_1 w_107_2 w_107_3 w_116_6 w_116_7 w_120_6 w_120_7 w_137_0 w_137_1 w_137_2 w_137_3 w_139_0 w_139_1 w_139_2 w_139_3 w_141_0 w_141_1 w_141_2 w_141_3 w_143_0 w_143_1 w_143_2 w_143_3 w_145_0 w_145_1 w_145_2 w_145_3 w_147_0 w_147_1 w_147_2 w_147_3 w_149_0 w_149_1 w_149_2 w_149_3 w_151_0 w_151_1 w_151_2 w_151_3 w_152_0 w_161_6 w_161_7 w_165_6 w_165_7 w_168_0 w_168_1 w_168_2 w_168_3 w_181_0 w_181_1 w_181_2 w_181_3 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_189_1 w_189_2 w_189_3 outputs) : W_Stage_preprocess_0_0.generatedRoot.constraints_7 backend hashModel params' w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_36_2 w_36_3 w_69_0 w_69_1 w_69_2 w_69_3 w_79_6 w_79_7 w_83_6 w_83_7 w_107_0 w_107_1 w_107_2 w_107_3 w_116_6 w_116_7 w_120_6 w_120_7 w_137_0 w_137_1 w_137_2 w_137_3 w_139_0 w_139_1 w_139_2 w_139_3 w_141_0 w_141_1 w_141_2 w_141_3 w_143_0 w_143_1 w_143_2 w_143_3 w_145_0 w_145_1 w_145_2 w_145_3 w_147_0 w_147_1 w_147_2 w_147_3 w_149_0 w_149_1 w_149_2 w_149_3 w_151_0 w_151_1 w_151_2 w_151_3 w_152_0 w_161_6 w_161_7 w_165_6 w_165_7 w_168_0 w_168_1 w_168_2 w_168_3 w_181_0 w_181_1 w_181_2 w_181_3 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_189_1 w_189_2 w_189_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_0_0.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_189.bridge hs14))), h13⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_6.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_17_0 w_153_0 w_154_0 w_159_0 w_160_0 w_163_0 w_164_0 w_2_0 w_5_0 w_5_1 w_5_2 w_5_3 w_7_0 w_7_1 w_7_2 w_7_3 w_15_0 w_15_1 w_15_2 w_15_3 w_137_0 w_137_1 w_137_2 w_137_3 w_148_0 w_149_0 w_149_1 w_149_2 w_149_3 w_150_0 w_151_0 w_151_1 w_151_2 w_151_3 w_155_0 w_156_0 w_157_0 w_158_0 w_161_0 w_161_1 w_161_2 w_161_3 w_161_4 w_161_5 w_161_6 w_161_7 w_162_0 w_162_1 w_162_2 w_162_3 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_165_6 w_165_7 w_166_0 w_166_1 w_166_2 w_166_3 w_167_0 w_167_1 w_167_2 w_167_3 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_181_1 w_181_2 w_181_3}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_6 backend hashModel tape path params w_0_0 w_1_0 w_17_0 w_153_0 w_154_0 w_159_0 w_160_0 w_163_0 w_164_0 w_2_0 w_5_0 w_5_1 w_5_2 w_5_3 w_7_0 w_7_1 w_7_2 w_7_3 w_15_0 w_15_1 w_15_2 w_15_3 w_137_0 w_137_1 w_137_2 w_137_3 w_148_0 w_149_0 w_149_1 w_149_2 w_149_3 w_150_0 w_151_0 w_151_1 w_151_2 w_151_3 w_155_0 w_156_0 w_157_0 w_158_0 w_161_0 w_161_1 w_161_2 w_161_3 w_161_4 w_161_5 w_161_6 w_161_7 w_162_0 w_162_1 w_162_2 w_162_3 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_165_6 w_165_7 w_166_0 w_166_1 w_166_2 w_166_3 w_167_0 w_167_1 w_167_2 w_167_3 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_181_1 w_181_2 w_181_3) : W_Stage_preprocess_0_0.generatedRoot.constraints_6 backend hashModel params' w_0_0 w_1_0 w_17_0 w_153_0 w_154_0 w_159_0 w_160_0 w_163_0 w_164_0 w_2_0 w_5_0 w_5_1 w_5_2 w_5_3 w_7_0 w_7_1 w_7_2 w_7_3 w_15_0 w_15_1 w_15_2 w_15_3 w_137_0 w_137_1 w_137_2 w_137_3 w_148_0 w_149_0 w_149_1 w_149_2 w_149_3 w_150_0 w_151_0 w_151_1 w_151_2 w_151_3 w_155_0 w_156_0 w_157_0 w_158_0 w_161_0 w_161_1 w_161_2 w_161_3 w_161_4 w_161_5 w_161_6 w_161_7 w_162_0 w_162_1 w_162_2 w_162_3 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_165_6 w_165_7 w_166_0 w_166_1 w_166_2 w_166_3 w_167_0 w_167_1 w_167_2 w_167_3 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_181_1 w_181_2 w_181_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (Stage_preprocess_0_0.iterRuns_mono h2 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_149.bridge hs33))), h3, (Stage_preprocess_0_0.iterRuns_mono h4 (fun _ _ _ hs34 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_151.bridge hs34))), h5, h6, h7, h8, h9, h10, h11, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_161.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_162.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_165.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_166.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_167.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_168.bridge (h22 i))), h23, h24, h25, h26, h27, h28, h29, h30, h31, (Stage_preprocess_0_0.iterRuns_mono h32 (fun _ _ _ hs35 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_181.bridge hs35)))⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_5.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_17_0 w_108_0 w_118_0 w_119_0 w_2_0 w_6_0 w_6_1 w_6_2 w_6_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_116_0 w_116_1 w_116_2 w_116_3 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_122_0 w_122_1 w_122_2 w_122_3 w_123_0 w_123_1 w_123_2 w_123_3 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_137_1 w_137_2 w_137_3 w_138_0 w_139_0 w_139_1 w_139_2 w_139_3 w_140_0 w_141_0 w_141_1 w_141_2 w_141_3 w_142_0 w_143_0 w_143_1 w_143_2 w_143_3 w_144_0 w_145_0 w_145_1 w_145_2 w_145_3 w_146_0 w_147_0 w_147_1 w_147_2 w_147_3}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_5 backend hashModel tape path params w_0_0 w_1_0 w_17_0 w_108_0 w_118_0 w_119_0 w_2_0 w_6_0 w_6_1 w_6_2 w_6_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_116_0 w_116_1 w_116_2 w_116_3 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_122_0 w_122_1 w_122_2 w_122_3 w_123_0 w_123_1 w_123_2 w_123_3 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_137_1 w_137_2 w_137_3 w_138_0 w_139_0 w_139_1 w_139_2 w_139_3 w_140_0 w_141_0 w_141_1 w_141_2 w_141_3 w_142_0 w_143_0 w_143_1 w_143_2 w_143_3 w_144_0 w_145_0 w_145_1 w_145_2 w_145_3 w_146_0 w_147_0 w_147_1 w_147_2 w_147_3) : W_Stage_preprocess_0_0.generatedRoot.constraints_5 backend hashModel params' w_0_0 w_1_0 w_17_0 w_108_0 w_118_0 w_119_0 w_2_0 w_6_0 w_6_1 w_6_2 w_6_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_116_0 w_116_1 w_116_2 w_116_3 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_122_0 w_122_1 w_122_2 w_122_3 w_123_0 w_123_1 w_123_2 w_123_3 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_137_1 w_137_2 w_137_3 w_138_0 w_139_0 w_139_1 w_139_2 w_139_3 w_140_0 w_141_0 w_141_1 w_141_2 w_141_3 w_142_0 w_143_0 w_143_1 w_143_2 w_143_3 w_144_0 w_145_0 w_145_1 w_145_2 w_145_3 w_146_0 w_147_0 w_147_1 w_147_2 w_147_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_117.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_120.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_121.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_122.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_123.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_124.bridge (h12 i))), h13, h14, h15, h16, h17, h18, h19, h20, h21, (Stage_preprocess_0_0.iterRuns_mono h22 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_137.bridge hs33))), h23, (Stage_preprocess_0_0.iterRuns_mono h24 (fun _ _ _ hs34 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_139.bridge hs34))), h25, (Stage_preprocess_0_0.iterRuns_mono h26 (fun _ _ _ hs35 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_141.bridge hs35))), h27, (Stage_preprocess_0_0.iterRuns_mono h28 (fun _ _ _ hs36 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_143.bridge hs36))), h29, (Stage_preprocess_0_0.iterRuns_mono h30 (fun _ _ _ hs37 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_145.bridge hs37))), h31, (Stage_preprocess_0_0.iterRuns_mono h32 (fun _ _ _ hs38 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_147.bridge hs38)))⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_4.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_17_0 w_109_0 w_114_0 w_115_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_70_0 w_84_0 w_84_1 w_84_2 w_84_3 w_85_0 w_85_1 w_85_2 w_85_3 w_86_0 w_86_1 w_86_2 w_86_3 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_99_1 w_99_2 w_99_3 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_111_0 w_112_0 w_113_0 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_4 backend hashModel tape path params w_0_0 w_1_0 w_17_0 w_109_0 w_114_0 w_115_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_70_0 w_84_0 w_84_1 w_84_2 w_84_3 w_85_0 w_85_1 w_85_2 w_85_3 w_86_0 w_86_1 w_86_2 w_86_3 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_99_1 w_99_2 w_99_3 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_111_0 w_112_0 w_113_0 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7) : W_Stage_preprocess_0_0.generatedRoot.constraints_4 backend hashModel params' w_0_0 w_1_0 w_17_0 w_109_0 w_114_0 w_115_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_70_0 w_84_0 w_84_1 w_84_2 w_84_3 w_85_0 w_85_1 w_85_2 w_85_3 w_86_0 w_86_1 w_86_2 w_86_3 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_99_1 w_99_2 w_99_3 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_111_0 w_112_0 w_113_0 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_85.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_86.bridge (h4 i))), h5, h6, h7, h8, h9, h10, h11, h12, h13, (Stage_preprocess_0_0.iterRuns_mono h14 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_99.bridge hs33))), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, (Stage_preprocess_0_0.iterRuns_mono h26 (fun _ _ _ hs34 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_107.bridge hs34))), h27, h28, h29, h30, h31, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_116.bridge (h32 i)))⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_3.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_17_0 w_71_0 w_72_0 w_77_0 w_78_0 w_81_0 w_82_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_16_0 w_56_0 w_57_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_69_0 w_69_1 w_69_2 w_69_3 w_73_0 w_74_0 w_75_0 w_76_0 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_84_0 w_84_1 w_84_2 w_84_3}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_1_0 w_17_0 w_71_0 w_72_0 w_77_0 w_78_0 w_81_0 w_82_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_16_0 w_56_0 w_57_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_69_0 w_69_1 w_69_2 w_69_3 w_73_0 w_74_0 w_75_0 w_76_0 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_84_0 w_84_1 w_84_2 w_84_3) : W_Stage_preprocess_0_0.generatedRoot.constraints_3 backend hashModel params' w_0_0 w_1_0 w_17_0 w_71_0 w_72_0 w_77_0 w_78_0 w_81_0 w_82_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_16_0 w_56_0 w_57_0 w_62_0 w_63_0 w_64_0 w_66_0 w_67_0 w_68_0 w_69_0 w_69_1 w_69_2 w_69_3 w_73_0 w_74_0 w_75_0 w_76_0 w_79_0 w_79_1 w_79_2 w_79_3 w_79_4 w_79_5 w_79_6 w_79_7 w_80_0 w_80_1 w_80_2 w_80_3 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_84_0 w_84_1 w_84_2 w_84_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, (Stage_preprocess_0_0.iterRuns_mono h18 (fun _ _ _ hs33 ↦ (Stage_preprocess_0_0.sequential_generatedRoot_69.bridge hs33))), h19, h20, h21, h22, h23, h24, h25, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_79.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_80.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_83.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_84.bridge (h32 i)))⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_1.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_17_0 w_34_0 w_35_0 w_2_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_36_0 w_36_1 w_36_2 w_36_3 w_37_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_1_0 w_17_0 w_34_0 w_35_0 w_2_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_36_0 w_36_1 w_36_2 w_36_3 w_37_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0) : W_Stage_preprocess_0_0.generatedRoot.constraints_1 backend hashModel params' w_0_0 w_1_0 w_17_0 w_34_0 w_35_0 w_2_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_36_0 w_36_1 w_36_2 w_36_3 w_37_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_36.bridge (h21 i))), h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_17_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0}
    (h : Stage_preprocess_0_0.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_17_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0) : W_Stage_preprocess_0_0.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_17_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_6_1 w_6_2 w_6_3 w_7_0 w_7_1 w_7_2 w_7_3 w_8_0 w_8_1 w_8_2 w_8_3 w_9_0 w_9_1 w_9_2 w_9_3 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_11_1 w_11_2 w_11_3 w_12_0 w_12_1 w_12_2 w_12_3 w_13_0 w_13_1 w_13_2 w_13_3 w_14_0 w_14_1 w_14_2 w_14_3 w_15_0 w_15_1 w_15_2 w_15_3 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_3.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_4.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_5.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_6.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_7.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_8.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_9.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_10.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_11.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_12.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_13.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_14.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess_0_0.parallel_generatedRoot_15.bridge (h26 i))), h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_0_0.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_preprocess_0_0.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_preprocess_0_0.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_preprocess_0_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_3_1, wit1.w_3_2, wit1.w_3_3, wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_5_1, wit1.w_5_2, wit1.w_5_3, wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_7_0, wit1.w_7_1, wit1.w_7_2, wit1.w_7_3, wit1.w_8_0, wit1.w_8_1, wit1.w_8_2, wit1.w_8_3, wit1.w_9_0, wit1.w_9_1, wit1.w_9_2, wit1.w_9_3, wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3, wit1.w_11_0, wit1.w_11_1, wit1.w_11_2, wit1.w_11_3, wit1.w_12_0, wit1.w_12_1, wit1.w_12_2, wit1.w_12_3, wit1.w_13_0, wit1.w_13_1, wit1.w_13_2, wit1.w_13_3, wit1.w_14_0, wit1.w_14_1, wit1.w_14_2, wit1.w_14_3, wit1.w_15_0, wit1.w_15_1, wit1.w_15_2, wit1.w_15_3, wit1.w_18_0, wit1.w_19_0, wit1.w_20_0, wit1.w_21_0, wit1.w_22_0, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_28_0, wit1.w_31_0, wit1.w_32_0, wit1.w_33_0, wit1.w_36_0, wit1.w_36_1, wit1.w_36_2, wit1.w_36_3, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_41_0, wit1.w_42_0, wit1.w_44_0, wit1.w_45_0, wit1.w_48_0, wit1.w_49_0, wit1.w_50_0, wit1.w_51_0, wit1.w_52_0, wit1.w_53_0, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_59_0, wit1.w_60_0, wit1.w_62_0, wit1.w_63_0, wit1.w_64_0, wit1.w_67_0, wit1.w_68_0, wit1.w_69_0, wit1.w_69_1, wit1.w_69_2, wit1.w_69_3, wit1.w_73_0, wit1.w_74_0, wit1.w_75_0, wit1.w_76_0, wit1.w_79_0, wit1.w_79_1, wit1.w_79_2, wit1.w_79_3, wit1.w_79_4, wit1.w_79_5, wit1.w_79_6, wit1.w_79_7, wit1.w_80_0, wit1.w_80_1, wit1.w_80_2, wit1.w_80_3, wit1.w_83_0, wit1.w_83_1, wit1.w_83_2, wit1.w_83_3, wit1.w_83_4, wit1.w_83_5, wit1.w_83_6, wit1.w_83_7, wit1.w_84_0, wit1.w_84_1, wit1.w_84_2, wit1.w_84_3, wit1.w_85_0, wit1.w_85_1, wit1.w_85_2, wit1.w_85_3, wit1.w_86_0, wit1.w_86_1, wit1.w_86_2, wit1.w_86_3, wit1.w_87_0, wit1.w_88_0, wit1.w_90_0, wit1.w_91_0, wit1.w_93_0, wit1.w_94_0, wit1.w_96_0, wit1.w_97_0, wit1.w_99_0, wit1.w_99_1, wit1.w_99_2, wit1.w_99_3, wit1.w_100_0, wit1.w_102_0, wit1.w_104_0, wit1.w_106_0, wit1.w_107_0, wit1.w_107_1, wit1.w_107_2, wit1.w_107_3, wit1.w_110_0, wit1.w_111_0, wit1.w_112_0, wit1.w_113_0, wit1.w_116_0, wit1.w_116_1, wit1.w_116_2, wit1.w_116_3, wit1.w_116_4, wit1.w_116_5, wit1.w_116_6, wit1.w_116_7, wit1.w_117_0, wit1.w_117_1, wit1.w_117_2, wit1.w_117_3, wit1.w_120_0, wit1.w_120_1, wit1.w_120_2, wit1.w_120_3, wit1.w_120_4, wit1.w_120_5, wit1.w_120_6, wit1.w_120_7, wit1.w_121_0, wit1.w_121_1, wit1.w_121_2, wit1.w_121_3, wit1.w_122_0, wit1.w_122_1, wit1.w_122_2, wit1.w_122_3, wit1.w_123_0, wit1.w_123_1, wit1.w_123_2, wit1.w_123_3, wit1.w_124_0, wit1.w_124_1, wit1.w_124_2, wit1.w_124_3, wit1.w_125_0, wit1.w_126_0, wit1.w_128_0, wit1.w_129_0, wit1.w_131_0, wit1.w_132_0, wit1.w_134_0, wit1.w_135_0, wit1.w_137_0, wit1.w_137_1, wit1.w_137_2, wit1.w_137_3, wit1.w_139_0, wit1.w_139_1, wit1.w_139_2, wit1.w_139_3, wit1.w_141_0, wit1.w_141_1, wit1.w_141_2, wit1.w_141_3, wit1.w_143_0, wit1.w_143_1, wit1.w_143_2, wit1.w_143_3, wit1.w_145_0, wit1.w_145_1, wit1.w_145_2, wit1.w_145_3, wit1.w_147_0, wit1.w_147_1, wit1.w_147_2, wit1.w_147_3, wit1.w_149_0, wit1.w_149_1, wit1.w_149_2, wit1.w_149_3, wit1.w_151_0, wit1.w_151_1, wit1.w_151_2, wit1.w_151_3, wit1.w_155_0, wit1.w_156_0, wit1.w_157_0, wit1.w_158_0, wit1.w_161_0, wit1.w_161_1, wit1.w_161_2, wit1.w_161_3, wit1.w_161_4, wit1.w_161_5, wit1.w_161_6, wit1.w_161_7, wit1.w_162_0, wit1.w_162_1, wit1.w_162_2, wit1.w_162_3, wit1.w_165_0, wit1.w_165_1, wit1.w_165_2, wit1.w_165_3, wit1.w_165_4, wit1.w_165_5, wit1.w_165_6, wit1.w_165_7, wit1.w_166_0, wit1.w_166_1, wit1.w_166_2, wit1.w_166_3, wit1.w_167_0, wit1.w_167_1, wit1.w_167_2, wit1.w_167_3, wit1.w_168_0, wit1.w_168_1, wit1.w_168_2, wit1.w_168_3, wit1.w_169_0, wit1.w_170_0, wit1.w_172_0, wit1.w_173_0, wit1.w_175_0, wit1.w_176_0, wit1.w_178_0, wit1.w_179_0, wit1.w_181_0, wit1.w_181_1, wit1.w_181_2, wit1.w_181_3, wit1.w_182_0, wit1.w_184_0, wit1.w_186_0, wit1.w_188_0, wit1.w_189_0, wit1.w_189_1, wit1.w_189_2, wit1.w_189_3⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10⟩ := h2; exact ⟨(Stage_preprocess_0_0.generatedRoot.constraints_0.bridge h3), (Stage_preprocess_0_0.generatedRoot.constraints_1.bridge h4), h5, (Stage_preprocess_0_0.generatedRoot.constraints_3.bridge h6), (Stage_preprocess_0_0.generatedRoot.constraints_4.bridge h7), (Stage_preprocess_0_0.generatedRoot.constraints_5.bridge h8), (Stage_preprocess_0_0.generatedRoot.constraints_6.bridge h9), (Stage_preprocess_0_0.generatedRoot.constraints_7.bridge h10)⟩)⟩)
