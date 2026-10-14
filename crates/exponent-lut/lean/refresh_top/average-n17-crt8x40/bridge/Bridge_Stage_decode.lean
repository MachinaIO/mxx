import Stage_decode
import W_Stage_decode
import ExponentSampling

/-! Every run of the average-case relations of `Stage_decode` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_decode.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_decode.sequential_generatedRoot_4.bridge {backend tape path params params' i_0 inputs outputs}
    (h : Stage_decode.sequential_generatedRoot_4 backend tape path params i_0 inputs outputs) : W_Stage_decode.sequential_generatedRoot_4 backend params' i_0 inputs outputs := by
  delta W_Stage_decode.sequential_generatedRoot_4
  exact h


theorem Stage_decode.sequential_sequential_generatedRoot_28_6.bridge {backend tape path params params' x4 i_1 inputs outputs}
    (h : Stage_decode.sequential_sequential_generatedRoot_28_6 backend tape path params x4 i_1 inputs outputs) : W_Stage_decode.sequential_sequential_generatedRoot_28_6 backend params' x4 i_1 inputs outputs := by
  delta W_Stage_decode.sequential_sequential_generatedRoot_28_6
  exact h


theorem Stage_decode.sequential_generatedRoot_28.constraints_0.bridge {backend tape path params params' i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs}
    (h : Stage_decode.sequential_generatedRoot_28.constraints_0 backend tape path params i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs) : W_Stage_decode.sequential_generatedRoot_28.constraints_0 backend params' i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (Stage_decode.iterRuns_mono h10 (fun _ _ _ hs12 ↦ (Stage_decode.sequential_sequential_generatedRoot_28_6.bridge hs12))), h11⟩)


theorem Stage_decode.sequential_generatedRoot_28.bridge {backend tape path params params' i_0 inputs outputs}
    (h : Stage_decode.sequential_generatedRoot_28 backend tape path params i_0 inputs outputs) : W_Stage_decode.sequential_generatedRoot_28 backend params' i_0 inputs outputs := by
  delta W_Stage_decode.sequential_generatedRoot_28
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_3_0, wit1.w_6_0⟩, (Stage_decode.sequential_generatedRoot_28.constraints_0.bridge h2)⟩)


theorem Stage_decode.sequential_generatedRoot_32.bridge {backend tape path params params' i_0 inputs outputs}
    (h : Stage_decode.sequential_generatedRoot_32 backend tape path params i_0 inputs outputs) : W_Stage_decode.sequential_generatedRoot_32 backend params' i_0 inputs outputs := by
  delta W_Stage_decode.sequential_generatedRoot_32
  exact h


theorem Stage_decode.sequential_sequential_generatedRoot_46_6.bridge {backend tape path params params' x4 i_1 inputs outputs}
    (h : Stage_decode.sequential_sequential_generatedRoot_46_6 backend tape path params x4 i_1 inputs outputs) : W_Stage_decode.sequential_sequential_generatedRoot_46_6 backend params' x4 i_1 inputs outputs := by
  delta W_Stage_decode.sequential_sequential_generatedRoot_46_6
  exact h


theorem Stage_decode.sequential_generatedRoot_46.constraints_0.bridge {backend tape path params params' i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs}
    (h : Stage_decode.sequential_generatedRoot_46.constraints_0 backend tape path params i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs) : W_Stage_decode.sequential_generatedRoot_46.constraints_0 backend params' i_0 w_0_0 w_1_0 w_4_0 w_5_0 w_2_0 w_3_0 w_6_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (Stage_decode.iterRuns_mono h10 (fun _ _ _ hs12 ↦ (Stage_decode.sequential_sequential_generatedRoot_46_6.bridge hs12))), h11⟩)


theorem Stage_decode.sequential_generatedRoot_46.bridge {backend tape path params params' i_0 inputs outputs}
    (h : Stage_decode.sequential_generatedRoot_46 backend tape path params i_0 inputs outputs) : W_Stage_decode.sequential_generatedRoot_46 backend params' i_0 inputs outputs := by
  delta W_Stage_decode.sequential_generatedRoot_46
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_3_0, wit1.w_6_0⟩, (Stage_decode.sequential_generatedRoot_46.constraints_0.bridge h2)⟩)


theorem Stage_decode.generatedRoot.constraints_1.bridge {backend tape path params params' w_7_0 w_30_0 w_33_0 w_34_0 w_36_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 outputs}
    (h : Stage_decode.generatedRoot.constraints_1 backend tape path params w_7_0 w_30_0 w_33_0 w_34_0 w_36_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 outputs) : W_Stage_decode.generatedRoot.constraints_1 backend params' w_7_0 w_30_0 w_33_0 w_34_0 w_36_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, (Stage_decode.iterRuns_mono h23 (fun _ _ _ hs25 ↦ (Stage_decode.sequential_generatedRoot_46.bridge hs25))), h24⟩)


theorem Stage_decode.generatedRoot.constraints_0.bridge {backend tape path params params' w_2_0 w_5_0 w_7_0 w_1_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_11_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_31_0 w_32_0}
    (h : Stage_decode.generatedRoot.constraints_0 backend tape path params w_2_0 w_5_0 w_7_0 w_1_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_11_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_31_0 w_32_0) : W_Stage_decode.generatedRoot.constraints_0 backend params' w_2_0 w_5_0 w_7_0 w_1_0 w_3_0 w_4_0 w_6_0 w_8_0 w_9_0 w_11_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 w_28_0 w_31_0 w_32_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (Stage_decode.iterRuns_mono h3 (fun _ _ _ hs33 ↦ (Stage_decode.sequential_generatedRoot_4.bridge hs33))), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, (Stage_decode.iterRuns_mono h29 (fun _ _ _ hs34 ↦ (Stage_decode.sequential_generatedRoot_28.bridge hs34))), h30, h31, (Stage_decode.iterRuns_mono h32 (fun _ _ _ hs35 ↦ (Stage_decode.sequential_generatedRoot_32.bridge hs35)))⟩)


theorem Stage_decode.generatedRoot.bridge {backend tape path params params' inputs outputs}
    (h : Stage_decode.generatedRoot backend tape path params inputs outputs) : W_Stage_decode.generatedRoot backend params' inputs outputs := by
  delta W_Stage_decode.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_21_0, wit1.w_22_0, wit1.w_28_0, wit1.w_31_0, wit1.w_32_0, wit1.w_33_0, wit1.w_34_0, wit1.w_36_0, wit1.w_46_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_decode.generatedRoot.constraints_0.bridge h3), (Stage_decode.generatedRoot.constraints_1.bridge h4)⟩)⟩)
