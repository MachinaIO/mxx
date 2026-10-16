import Stage_preimages_0_0
import W_Stage_preimages_0_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preimages_0_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preimages_0_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preimages_0_0.parallel_generatedRoot_3.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_3 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_3 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_3
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_5.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_5 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_5 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_5
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_7.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_7 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_7 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_7
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_9.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_9 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_9 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_9
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_11.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_11 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_11 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_11
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_13.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_13 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_13 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_13
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_15.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_15 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_15 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_15
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_17.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_17 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_17 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_17
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_19.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_19 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_19 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_19
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_21.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_21 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_21 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_21
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_23.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_23 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_23 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_23
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_25.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_25 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_25 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_25
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_27.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_27 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_27 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_27
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_29.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_29 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_29 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_29
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_31.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_31 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_31 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_31
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_33.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_33 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_33 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_33
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_35.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_35 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_35 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_35
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_37.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_37 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_37 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_37
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_39.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_39 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_39 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_39
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_41.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_41 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_41 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_41
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_43.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_43 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_43 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_43
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_45.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_45 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_45 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_45
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_47.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_47 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_47 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_47
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_49.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_49 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_49 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_49
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_51.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_51 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_51 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_51
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_53.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_53 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_53 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_53
  exact h


theorem Stage_preimages_0_0.parallel_generatedRoot_55.bridge {backend tape path params params' x4 inputs outputs}
    (h : Stage_preimages_0_0.parallel_generatedRoot_55 backend tape path params x4 inputs outputs) : W_Stage_preimages_0_0.parallel_generatedRoot_55 backend params' x4 inputs outputs := by
  delta W_Stage_preimages_0_0.parallel_generatedRoot_55
  exact h


theorem Stage_preimages_0_0.generatedRoot.constraints_1.bridge {backend tape path params params' w_0_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 outputs}
    (h : Stage_preimages_0_0.generatedRoot.constraints_1 backend tape path params w_0_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 outputs) : W_Stage_preimages_0_0.generatedRoot.constraints_1 backend params' w_0_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_35.bridge (h2 i))), h3, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_37.bridge (h4 i))), h5, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_39.bridge (h6 i))), h7, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_41.bridge (h8 i))), h9, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_43.bridge (h10 i))), h11, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_45.bridge (h12 i))), h13, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_47.bridge (h14 i))), h15, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_49.bridge (h16 i))), h17, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_51.bridge (h18 i))), h19, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_53.bridge (h20 i))), h21, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_55.bridge (h22 i))), h23⟩)


theorem Stage_preimages_0_0.generatedRoot.constraints_0.bridge {backend tape path params params' w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0}
    (h : Stage_preimages_0_0.generatedRoot.constraints_0 backend tape path params w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0) : W_Stage_preimages_0_0.generatedRoot.constraints_0 backend params' w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_3.bridge (h2 i))), h3, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_5.bridge (h4 i))), h5, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_7.bridge (h6 i))), h7, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_9.bridge (h8 i))), h9, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_11.bridge (h10 i))), h11, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_13.bridge (h12 i))), h13, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_15.bridge (h14 i))), h15, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_17.bridge (h16 i))), h17, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_19.bridge (h18 i))), h19, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_21.bridge (h20 i))), h21, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_23.bridge (h22 i))), h23, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_25.bridge (h24 i))), h25, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_27.bridge (h26 i))), h27, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_29.bridge (h28 i))), h29, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_31.bridge (h30 i))), h31, (fun i ↦ (Stage_preimages_0_0.parallel_generatedRoot_33.bridge (h32 i)))⟩)


theorem Stage_preimages_0_0.generatedRoot.bridge {backend tape path params params' inputs outputs}
    (h : Stage_preimages_0_0.generatedRoot backend tape path params inputs outputs) : W_Stage_preimages_0_0.generatedRoot backend params' inputs outputs := by
  delta W_Stage_preimages_0_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_5_0, wit1.w_7_0, wit1.w_9_0, wit1.w_11_0, wit1.w_13_0, wit1.w_15_0, wit1.w_17_0, wit1.w_19_0, wit1.w_21_0, wit1.w_23_0, wit1.w_25_0, wit1.w_27_0, wit1.w_29_0, wit1.w_31_0, wit1.w_33_0, wit1.w_35_0, wit1.w_37_0, wit1.w_39_0, wit1.w_41_0, wit1.w_43_0, wit1.w_45_0, wit1.w_47_0, wit1.w_49_0, wit1.w_51_0, wit1.w_53_0, wit1.w_55_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preimages_0_0.generatedRoot.constraints_0.bridge h3), (Stage_preimages_0_0.generatedRoot.constraints_1.bridge h4)⟩)⟩)
