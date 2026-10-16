import Stage_preprocess_1_1
import W_Stage_preprocess_1_1
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preprocess_1_1` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preprocess_1_1.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_18_0 w_21_0 w_22_0 w_30_0 w_31_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_3_12.bridge (h11 i))), h12, (Stage_preprocess_1_1.iterRuns_mono h13 (fun _ _ _ hs19 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_3_13.bridge hs19))), h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), (MxxExponentLut.gaussianSampleAt_sample (by decide) h17), h18⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_3.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_3 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_3 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_3
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_12_0, wit1.w_13_0, wit1.w_17_0, wit1.w_18_0, wit1.w_21_0, wit1.w_30_0⟩, (Stage_preprocess_1_1.parallel_generatedRoot_3.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10.bridge {backend x1 tape path params params' x5 i_1 x7 outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10 backend x1 tape path params x5 i_1 x7 outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10 backend x1 params' x5 i_1 x7 outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0 backend hashModel tape path params i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0 backend hashModel params' i_0 w_0_0 w_3_0 w_1_0 w_2_0 w_4_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_17_0 w_20_0 w_21_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_4_10.bridge (h11 i))), h12, (Stage_preprocess_1_1.iterRuns_mono h13 (fun _ _ _ hs18 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_4_13.bridge hs18))), h14, (MxxExponentLut.gaussianSampleAt_sample (by decide) h15), (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_4.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_4 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_4 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_4
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_10_0, wit1.w_13_0, wit1.w_17_0, wit1.w_20_0, wit1.w_29_0⟩, (Stage_preprocess_1_1.parallel_generatedRoot_4.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_33.constraints_0.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_16_0 w_17_0 w_24_0 w_25_0 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_33.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_16_0 w_17_0 w_24_0 w_25_0 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_33.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_4_0 w_5_0 w_16_0 w_17_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h; exact ⟨h1, h2, h3, h4, (MxxExponentLut.gaussianSampleAt_sample (by decide) h5), (MxxExponentLut.gaussianSampleAt_sample (by decide) h6), h7⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_33.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_33 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_33 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_33
  exact (by obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_16_0, w_24_0, h1⟩ := h; exact ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_16_0, w_24_0, (Stage_preprocess_1_1.parallel_generatedRoot_33.constraints_0.bridge h1)⟩)


theorem Stage_preprocess_1_1.sequential_generatedRoot_66.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_generatedRoot_66 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.sequential_generatedRoot_66 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_generatedRoot_66
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_2.bridge {backend tape path w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_2 backend tape path w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_2 backend w_19_0 w_47_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_69_0 w_70_0 w_71_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_1.bridge {backend hashModel tape path i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_1 backend hashModel tape path i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_1 backend hashModel i_0 w_18_0 w_23_0 w_26_0 w_19_0 w_20_0 w_21_0 w_24_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_34_0 w_35_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, (MxxExponentLut.gaussianSampleAt_sample (by decide) h22), h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_108.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_108 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_108 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_108
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_19_0, w_20_0, w_21_0, w_24_0, w_27_0, w_30_0, w_31_0, w_32_0, w_34_0, w_35_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_63_0, w_65_0, w_66_0, w_69_0, w_70_0, w_71_0, h1, h2, h3⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_19_0, w_20_0, w_21_0, w_24_0, w_27_0, w_30_0, w_31_0, w_32_0, w_34_0, w_35_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_63_0, w_65_0, w_66_0, w_69_0, w_70_0, w_71_0, h1, (Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_1.bridge h2), (Stage_preprocess_1_1.parallel_generatedRoot_108.constraints_2.bridge h3)⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_109_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_109_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_109_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_109_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_109_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_109_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_109_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_109_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_109.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_109.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_109.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_109_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_109_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_109.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_109 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_109 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_109
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_109.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_112.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_112.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_112.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_112.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_112 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_112 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_112
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_112.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_113_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_113_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_113_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_113_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_113_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_113_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_113_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_113_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_113.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_113.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_113.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_113_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_113_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_113.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_113 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_113 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_113
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_113.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_116.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_116.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_116.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_116.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_116 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_116 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_116
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_116.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_117_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_117_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_117_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_117_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_117_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_117_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_117_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_117_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_117.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_117.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_117.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_117_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_117_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_117.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_117 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_117 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_117
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_117.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_120.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_120.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_120.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_120.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_120 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_120 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_120
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_120.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_121_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_121_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_121_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_121_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_121_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_121_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_121_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_121_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_121.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_121.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_121.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_121_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_121_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_121.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_121 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_121 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_121
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_121.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_124.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_124.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_124.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_124.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_124 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_124 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_124
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_124.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_125_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_125_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_125_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_125_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_125_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_125_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_125_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_125_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_125.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_125.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_125.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_125_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_125_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_125.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_125 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_125 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_125
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_125.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_128.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_128.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_128.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_128.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_128 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_128 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_128
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_128.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_129_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_129_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_129_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_129_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_129_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_129_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_129_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_129_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_129.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_129.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_129.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_129_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_129_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_129.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_129 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_129 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_129
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_129.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_132.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_132.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_132.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_132.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_132 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_132 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_132
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_132.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_133_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_133_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_133_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_133_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_133_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_133_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_133_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_133_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_133.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_133.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_133.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_133_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_133_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_133.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_133 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_133 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_133
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_133.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_136.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_136.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_136.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_136.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_136 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_136 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_136
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_136.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_137_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_137_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_137_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_137_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_137_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_137_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_137_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_137_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_137.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_137.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_137.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_137_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_137_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_137.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_137 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_137 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_137
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_137.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_140.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_140.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_140.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_140.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_140 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_140 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_140
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_140.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_141_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_141_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_141_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_141_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_141_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_141_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_141_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_141_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_141.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_141.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_141.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_141_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_141_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_141.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_141 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_141 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_141
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_141.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_144.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_144.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_144.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_144.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_144 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_144 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_144
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_144.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_145_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_145_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_145_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_145_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_145_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_145_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_145_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_145_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_145.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_145.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_145.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_145_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_145_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_145.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_145 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_145 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_145
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_145.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_148.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_148.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_148.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_148.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_148 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_148 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_148
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_148.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_149_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_149_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_149_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_149_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_149_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_149_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_149_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_149_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_149.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_149.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_149.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_149_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_149_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_149.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_149 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_149 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_149
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_149.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_152.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_152.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_152.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_152.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_152 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_152 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_152
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_152.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_153_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_153_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_153_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_153_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_153_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_153_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_153_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_153_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_153.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_153.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_153.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_153_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_153_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_153.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_153 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_153 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_153
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_153.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_156.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_156.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_156.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_156.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_156 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_156 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_156
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_156.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_157_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_157_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_157_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_157_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_157_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_157_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_157_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_157_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_157.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_157.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_157.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_157_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_157_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_157.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_157 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_157 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_157
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_157.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_160.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_160.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_160.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_160.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_160 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_160 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_160
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_160.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_161_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_161_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_161_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_161_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_161_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_161_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_161_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_161_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_161.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_161.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_161.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_161_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_161_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_161.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_161 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_161 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_161
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_161.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_164.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_164.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_164.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_164.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_164 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_164 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_164
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_164.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_165_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_165_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_165_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_165_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_165_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_165_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_165_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_165_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_165.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_165.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_165.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_165_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_165_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_165.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_165 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_165 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_165
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_165.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_168.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_168.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_168.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_168.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_168 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_168 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_168
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_168.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_169_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_169_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_169_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_169_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_169_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_169_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_169_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_169_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_169.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_169.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_169.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_169_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_169_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_169.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_169 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_169 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_169
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_169.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_172.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_172.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_172.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_172.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_172 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_172 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_172
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_172.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_173_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_173_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_173_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_173_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_173_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_173_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_173_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_173_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_173.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_173.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_173.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_173_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_173_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_173.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_173 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_173 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_173
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_173.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_176.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_176.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_176.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_176.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_176 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_176 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_176
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_176.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_177_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_177_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_177_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_177_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_177_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_177_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_177_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_177_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_177.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_177.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_177.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_177_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_177_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_177.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_177 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_177 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_177
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_177.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_180.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_180.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_180.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_180.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_180 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_180 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_180
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_180.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_181_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_181_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_181_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_181_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_181_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_181_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_181_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_181_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_181.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_181.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_181.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_181_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_181_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_181.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_181 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_181 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_181
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_181.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_184.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_184.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_184.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_184.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_184 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_184 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_184
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_184.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_185_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_185_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_185_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_185_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_185_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_185_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_185_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_185_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_185.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_185.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_185.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_185_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_185_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_185.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_185 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_185 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_185
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_185.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_188.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_188.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_188.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_188.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_188 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_188 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_188
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_188.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_189_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_189_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_189_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_189_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_189_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_189_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_189_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_189_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_189.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_189.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_189.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_189_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_189_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_189.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_189 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_189 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_189
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_189.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_192.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_192.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_192.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_192.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_192 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_192 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_192
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_192.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_193_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_193_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_193_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_193_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_193_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_193_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_193_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_193_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_193.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_193.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_193.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_193_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_193_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_193.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_193 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_193 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_193
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_193.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_196.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_196.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_196.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_196.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_196 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_196 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_196
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_196.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_197_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_197_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_197_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_197_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_197_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_197_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_197_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_197_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_197.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_197.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_197.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_197_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_197_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_197.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_197 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_197 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_197
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_197.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_200.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_200.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_200.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_200.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_200 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_200 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_200
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_200.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_201_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_201_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_201_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_201_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_201_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_201_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_201_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_201_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_201.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_201.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_201.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_201_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_201_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_201.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_201 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_201 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_201
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_201.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_204.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_204.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_204.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_204.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_204 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_204 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_204
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_204.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_205_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_205_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_205_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_205_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_205_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_205_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_205_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_205_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_205.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_205.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_205.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_205_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_205_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_205.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_205 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_205 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_205
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_205.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_208.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_208.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_208.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_208.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_208 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_208 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_208
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_208.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_209_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_209_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_209_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_209_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_209_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_209_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_209_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_209_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_209.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_209.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_209.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_209_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_209_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_209.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_209 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_209 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_209
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_209.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_212.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_212.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_212.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_212.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_212 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_212 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_212
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_212.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_213_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_213_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_213_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_213_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_213_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_213_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_213_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_213_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_213.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_213.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_213.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_213_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_213_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_213.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_213 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_213 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_213
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_213.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_216.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_216.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_216.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_216.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_216 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_216 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_216
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_216.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_217_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_217_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_217_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_217_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_217_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_217_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_217_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_217_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_217.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_217.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_217.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_217_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_217_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_217.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_217 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_217 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_217
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_217.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_220.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_220.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_220.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_220.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_220 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_220 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_220
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_220.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_221_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_221_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_221_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_221_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_221_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_221_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_221_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_221_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_221.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_221.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_221.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_221_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_221_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_221.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_221 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_221 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_221
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_221.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_224.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_224.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_224.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_224.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_224 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_224 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_224
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_224.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_225_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_225_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_225_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_225_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_225_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_225_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_225_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_225_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_225.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_225.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_225.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_225_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_225_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_225.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_225 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_225 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_225
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_225.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_228.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_228.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_228.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_228.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_228 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_228 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_228
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_228.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_229_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_229_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_229_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_229_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_229_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_229_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_229_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_229_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_229.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_229.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_229.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_229_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_229_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_229.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_229 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_229 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_229
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_229.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_232.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_232.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_232.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_232.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_232 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_232 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_232
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_232.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_233_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_233_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_233_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_233_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_233_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_233_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_233_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_233_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_233.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_233.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_233.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_233_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_233_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_233.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_233 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_233 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_233
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_233.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_236.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_236.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_236.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_236.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_236 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_236 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_236
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_236.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_237_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_237_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_237_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_237_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_237_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_237_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_237_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_237_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_237.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_237.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_237.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_237_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_237_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_237.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_237 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_237 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_237
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_237.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_240.constraints_2.bridge {backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_240.constraints_2 backend hashModel tape path i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess_1_1.parallel_generatedRoot_240.constraints_2 backend hashModel i_0 w_21_0 w_39_0 w_40_0 w_51_0 w_52_0 w_53_0 w_54_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_68_0 w_69_0 w_70_0 w_71_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_240.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_240 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_240 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_240
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_7_0, w_8_0, w_10_0, w_11_0, w_12_0, w_14_0, w_15_0, w_18_0, w_19_0, w_20_0, w_22_0, w_23_0, w_24_0, w_27_0, w_28_0, w_31_0, w_32_0, w_35_0, w_36_0, w_37_0, w_39_0, w_40_0, w_51_0, w_53_0, w_54_0, w_57_0, w_58_0, w_59_0, w_60_0, w_68_0, w_70_0, w_71_0, w_74_0, w_75_0, w_76_0, h1, h2, (Stage_preprocess_1_1.parallel_generatedRoot_240.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess_1_1.parallel_parallel_generatedRoot_241_4.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_parallel_generatedRoot_241_4 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_241_4 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_parallel_generatedRoot_241_4
  exact h


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_241_17.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_241_17 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_241_17 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_241_17
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_241.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_241.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_241.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_parallel_generatedRoot_241_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess_1_1.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_241_17.bridge hs14))), h13⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_241.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_241 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_241 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_241
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_241.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.sequential_parallel_generatedRoot_242_10.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_parallel_generatedRoot_242_10 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_242_10 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_parallel_generatedRoot_242_10
  exact h


theorem Stage_preprocess_1_1.parallel_generatedRoot_242.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_242.constraints_0 backend hashModel tape path params i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_242.constraints_0 backend hashModel params' i_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess_1_1.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess_1_1.sequential_parallel_generatedRoot_242_10.bridge hs4))), h3⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_242.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_242 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_242 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_242
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess_1_1.parallel_generatedRoot_242.constraints_0.bridge h2)⟩)


theorem Stage_preprocess_1_1.parallel_generatedRoot_243.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.parallel_generatedRoot_243 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.parallel_generatedRoot_243 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.parallel_generatedRoot_243
  exact h


theorem Stage_preprocess_1_1.sequential_generatedRoot_256.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_generatedRoot_256 backend x1 tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.sequential_generatedRoot_256 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_generatedRoot_256
  exact h


theorem Stage_preprocess_1_1.scope_150.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess_1_1.scope_150 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess_1_1.scope_150 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess_1_1.scope_150
  exact h


theorem Stage_preprocess_1_1.sequential_generatedRoot_264.constraints_0.bridge {backend hashModel tape path params params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0}
    (h : Stage_preprocess_1_1.sequential_generatedRoot_264.constraints_0 backend hashModel tape path params i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0) : W_Stage_preprocess_1_1.sequential_generatedRoot_264.constraints_0 backend hashModel params' i_0 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess_1_1.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess_1_1.scope_150.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.sequential_generatedRoot_264.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess_1_1.sequential_generatedRoot_264 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess_1_1.sequential_generatedRoot_264 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess_1_1.sequential_generatedRoot_264
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_6_0, wit1.w_9_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_17_0, wit1.w_18_0, wit1.w_20_0, wit1.w_20_1, wit1.w_20_2, wit1.w_20_3, wit1.w_21_0, wit1.w_25_0, wit1.w_29_0, wit1.w_33_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preprocess_1_1.sequential_generatedRoot_264.constraints_0.bridge h3), h4⟩)⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_8.bridge {backend hashModel tape path params params' w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_108_6 w_108_7 w_112_6 w_112_7 w_116_6 w_116_7 w_120_6 w_120_7 w_124_6 w_124_7 w_128_6 w_128_7 w_132_6 w_132_7 w_136_6 w_136_7 w_140_6 w_140_7 w_144_6 w_144_7 w_148_6 w_148_7 w_152_6 w_152_7 w_156_6 w_156_7 w_160_6 w_160_7 w_164_6 w_164_7 w_168_6 w_168_7 w_172_6 w_172_7 w_176_6 w_176_7 w_180_6 w_180_7 w_184_6 w_184_7 w_188_6 w_188_7 w_192_6 w_192_7 w_196_6 w_196_7 w_200_6 w_200_7 w_204_6 w_204_7 w_208_6 w_208_7 w_212_6 w_212_7 w_216_6 w_216_7 w_220_6 w_220_7 w_224_6 w_224_7 w_228_6 w_228_7 w_232_6 w_232_7 w_236_6 w_236_7 w_240_6 w_240_7 w_243_0 w_243_1 w_243_2 w_243_3 w_256_3 w_257_0 w_259_0 w_261_0 w_262_0 w_263_0 w_264_0 w_264_1 w_264_2 w_264_3 outputs}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_8 backend hashModel tape path params w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_108_6 w_108_7 w_112_6 w_112_7 w_116_6 w_116_7 w_120_6 w_120_7 w_124_6 w_124_7 w_128_6 w_128_7 w_132_6 w_132_7 w_136_6 w_136_7 w_140_6 w_140_7 w_144_6 w_144_7 w_148_6 w_148_7 w_152_6 w_152_7 w_156_6 w_156_7 w_160_6 w_160_7 w_164_6 w_164_7 w_168_6 w_168_7 w_172_6 w_172_7 w_176_6 w_176_7 w_180_6 w_180_7 w_184_6 w_184_7 w_188_6 w_188_7 w_192_6 w_192_7 w_196_6 w_196_7 w_200_6 w_200_7 w_204_6 w_204_7 w_208_6 w_208_7 w_212_6 w_212_7 w_216_6 w_216_7 w_220_6 w_220_7 w_224_6 w_224_7 w_228_6 w_228_7 w_232_6 w_232_7 w_236_6 w_236_7 w_240_6 w_240_7 w_243_0 w_243_1 w_243_2 w_243_3 w_256_3 w_257_0 w_259_0 w_261_0 w_262_0 w_263_0 w_264_0 w_264_1 w_264_2 w_264_3 outputs) : W_Stage_preprocess_1_1.generatedRoot.constraints_8 backend hashModel params' w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_33_2 w_33_3 w_66_0 w_66_1 w_66_2 w_66_3 w_108_6 w_108_7 w_112_6 w_112_7 w_116_6 w_116_7 w_120_6 w_120_7 w_124_6 w_124_7 w_128_6 w_128_7 w_132_6 w_132_7 w_136_6 w_136_7 w_140_6 w_140_7 w_144_6 w_144_7 w_148_6 w_148_7 w_152_6 w_152_7 w_156_6 w_156_7 w_160_6 w_160_7 w_164_6 w_164_7 w_168_6 w_168_7 w_172_6 w_172_7 w_176_6 w_176_7 w_180_6 w_180_7 w_184_6 w_184_7 w_188_6 w_188_7 w_192_6 w_192_7 w_196_6 w_196_7 w_200_6 w_200_7 w_204_6 w_204_7 w_208_6 w_208_7 w_212_6 w_212_7 w_216_6 w_216_7 w_220_6 w_220_7 w_224_6 w_224_7 w_228_6 w_228_7 w_232_6 w_232_7 w_236_6 w_236_7 w_240_6 w_240_7 w_243_0 w_243_1 w_243_2 w_243_3 w_256_3 w_257_0 w_259_0 w_261_0 w_262_0 w_263_0 w_264_0 w_264_1 w_264_2 w_264_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5⟩ := h; exact ⟨h1, h2, h3, (Stage_preprocess_1_1.iterRuns_mono h4 (fun _ _ _ hs6 ↦ (Stage_preprocess_1_1.sequential_generatedRoot_264.bridge hs6))), h5⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_7.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_68_0 w_69_0 w_234_0 w_235_0 w_238_0 w_239_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_67_0 w_232_0 w_232_1 w_232_2 w_232_3 w_233_0 w_233_1 w_233_2 w_233_3 w_236_0 w_236_1 w_236_2 w_236_3 w_236_4 w_236_5 w_236_6 w_236_7 w_237_0 w_237_1 w_237_2 w_237_3 w_240_0 w_240_1 w_240_2 w_240_3 w_240_4 w_240_5 w_240_6 w_240_7 w_241_0 w_241_1 w_241_2 w_241_3 w_242_0 w_242_1 w_242_2 w_242_3 w_243_0 w_243_1 w_243_2 w_243_3 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_256_1 w_256_2 w_256_3 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_7 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_68_0 w_69_0 w_234_0 w_235_0 w_238_0 w_239_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_67_0 w_232_0 w_232_1 w_232_2 w_232_3 w_233_0 w_233_1 w_233_2 w_233_3 w_236_0 w_236_1 w_236_2 w_236_3 w_236_4 w_236_5 w_236_6 w_236_7 w_237_0 w_237_1 w_237_2 w_237_3 w_240_0 w_240_1 w_240_2 w_240_3 w_240_4 w_240_5 w_240_6 w_240_7 w_241_0 w_241_1 w_241_2 w_241_3 w_242_0 w_242_1 w_242_2 w_242_3 w_243_0 w_243_1 w_243_2 w_243_3 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_256_1 w_256_2 w_256_3 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0) : W_Stage_preprocess_1_1.generatedRoot.constraints_7 backend hashModel params' w_0_0 w_1_0 w_6_0 w_68_0 w_69_0 w_234_0 w_235_0 w_238_0 w_239_0 w_2_0 w_4_0 w_4_1 w_4_2 w_4_3 w_67_0 w_232_0 w_232_1 w_232_2 w_232_3 w_233_0 w_233_1 w_233_2 w_233_3 w_236_0 w_236_1 w_236_2 w_236_3 w_236_4 w_236_5 w_236_6 w_236_7 w_237_0 w_237_1 w_237_2 w_237_3 w_240_0 w_240_1 w_240_2 w_240_3 w_240_4 w_240_5 w_240_6 w_240_7 w_241_0 w_241_1 w_241_2 w_241_3 w_242_0 w_242_1 w_242_2 w_242_3 w_243_0 w_243_1 w_243_2 w_243_3 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_256_1 w_256_2 w_256_3 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_233.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_236.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_237.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_240.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_241.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_242.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_243.bridge (h14 i))), h15, h16, h17, h18, h19, h20, h21, h22, h23, (Stage_preprocess_1_1.iterRuns_mono h24 (fun _ _ _ hs33 ↦ (Stage_preprocess_1_1.sequential_generatedRoot_256.bridge hs33))), h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_6.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_230_0 w_231_0 w_2_0 w_200_0 w_200_1 w_200_2 w_200_3 w_201_0 w_201_1 w_201_2 w_201_3 w_204_0 w_204_1 w_204_2 w_204_3 w_204_4 w_204_5 w_204_6 w_204_7 w_205_0 w_205_1 w_205_2 w_205_3 w_208_0 w_208_1 w_208_2 w_208_3 w_208_4 w_208_5 w_208_6 w_208_7 w_209_0 w_209_1 w_209_2 w_209_3 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_212_6 w_212_7 w_213_0 w_213_1 w_213_2 w_213_3 w_216_0 w_216_1 w_216_2 w_216_3 w_216_4 w_216_5 w_216_6 w_216_7 w_217_0 w_217_1 w_217_2 w_217_3 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_221_0 w_221_1 w_221_2 w_221_3 w_224_0 w_224_1 w_224_2 w_224_3 w_224_4 w_224_5 w_224_6 w_224_7 w_225_0 w_225_1 w_225_2 w_225_3 w_228_0 w_228_1 w_228_2 w_228_3 w_228_4 w_228_5 w_228_6 w_228_7 w_229_0 w_229_1 w_229_2 w_229_3 w_232_0 w_232_1 w_232_2 w_232_3 w_232_4 w_232_5 w_232_6 w_232_7}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_6 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_230_0 w_231_0 w_2_0 w_200_0 w_200_1 w_200_2 w_200_3 w_201_0 w_201_1 w_201_2 w_201_3 w_204_0 w_204_1 w_204_2 w_204_3 w_204_4 w_204_5 w_204_6 w_204_7 w_205_0 w_205_1 w_205_2 w_205_3 w_208_0 w_208_1 w_208_2 w_208_3 w_208_4 w_208_5 w_208_6 w_208_7 w_209_0 w_209_1 w_209_2 w_209_3 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_212_6 w_212_7 w_213_0 w_213_1 w_213_2 w_213_3 w_216_0 w_216_1 w_216_2 w_216_3 w_216_4 w_216_5 w_216_6 w_216_7 w_217_0 w_217_1 w_217_2 w_217_3 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_221_0 w_221_1 w_221_2 w_221_3 w_224_0 w_224_1 w_224_2 w_224_3 w_224_4 w_224_5 w_224_6 w_224_7 w_225_0 w_225_1 w_225_2 w_225_3 w_228_0 w_228_1 w_228_2 w_228_3 w_228_4 w_228_5 w_228_6 w_228_7 w_229_0 w_229_1 w_229_2 w_229_3 w_232_0 w_232_1 w_232_2 w_232_3 w_232_4 w_232_5 w_232_6 w_232_7) : W_Stage_preprocess_1_1.generatedRoot.constraints_6 backend hashModel params' w_0_0 w_1_0 w_6_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_230_0 w_231_0 w_2_0 w_200_0 w_200_1 w_200_2 w_200_3 w_201_0 w_201_1 w_201_2 w_201_3 w_204_0 w_204_1 w_204_2 w_204_3 w_204_4 w_204_5 w_204_6 w_204_7 w_205_0 w_205_1 w_205_2 w_205_3 w_208_0 w_208_1 w_208_2 w_208_3 w_208_4 w_208_5 w_208_6 w_208_7 w_209_0 w_209_1 w_209_2 w_209_3 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_212_6 w_212_7 w_213_0 w_213_1 w_213_2 w_213_3 w_216_0 w_216_1 w_216_2 w_216_3 w_216_4 w_216_5 w_216_6 w_216_7 w_217_0 w_217_1 w_217_2 w_217_3 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_221_0 w_221_1 w_221_2 w_221_3 w_224_0 w_224_1 w_224_2 w_224_3 w_224_4 w_224_5 w_224_6 w_224_7 w_225_0 w_225_1 w_225_2 w_225_3 w_228_0 w_228_1 w_228_2 w_228_3 w_228_4 w_228_5 w_228_6 w_228_7 w_229_0 w_229_1 w_229_2 w_229_3 w_232_0 w_232_1 w_232_2 w_232_3 w_232_4 w_232_5 w_232_6 w_232_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_201.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_204.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_205.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_208.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_209.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_212.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_213.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_216.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_217.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_220.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_221.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_224.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_225.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_228.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_229.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_232.bridge (h32 i)))⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_5.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_170_0 w_171_0 w_174_0 w_175_0 w_178_0 w_179_0 w_182_0 w_183_0 w_186_0 w_187_0 w_190_0 w_191_0 w_194_0 w_195_0 w_198_0 w_199_0 w_2_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_169_2 w_169_3 w_172_0 w_172_1 w_172_2 w_172_3 w_172_4 w_172_5 w_172_6 w_172_7 w_173_0 w_173_1 w_173_2 w_173_3 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_177_0 w_177_1 w_177_2 w_177_3 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_181_0 w_181_1 w_181_2 w_181_3 w_184_0 w_184_1 w_184_2 w_184_3 w_184_4 w_184_5 w_184_6 w_184_7 w_185_0 w_185_1 w_185_2 w_185_3 w_188_0 w_188_1 w_188_2 w_188_3 w_188_4 w_188_5 w_188_6 w_188_7 w_189_0 w_189_1 w_189_2 w_189_3 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_192_6 w_192_7 w_193_0 w_193_1 w_193_2 w_193_3 w_196_0 w_196_1 w_196_2 w_196_3 w_196_4 w_196_5 w_196_6 w_196_7 w_197_0 w_197_1 w_197_2 w_197_3 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_5 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_170_0 w_171_0 w_174_0 w_175_0 w_178_0 w_179_0 w_182_0 w_183_0 w_186_0 w_187_0 w_190_0 w_191_0 w_194_0 w_195_0 w_198_0 w_199_0 w_2_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_169_2 w_169_3 w_172_0 w_172_1 w_172_2 w_172_3 w_172_4 w_172_5 w_172_6 w_172_7 w_173_0 w_173_1 w_173_2 w_173_3 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_177_0 w_177_1 w_177_2 w_177_3 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_181_0 w_181_1 w_181_2 w_181_3 w_184_0 w_184_1 w_184_2 w_184_3 w_184_4 w_184_5 w_184_6 w_184_7 w_185_0 w_185_1 w_185_2 w_185_3 w_188_0 w_188_1 w_188_2 w_188_3 w_188_4 w_188_5 w_188_6 w_188_7 w_189_0 w_189_1 w_189_2 w_189_3 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_192_6 w_192_7 w_193_0 w_193_1 w_193_2 w_193_3 w_196_0 w_196_1 w_196_2 w_196_3 w_196_4 w_196_5 w_196_6 w_196_7 w_197_0 w_197_1 w_197_2 w_197_3 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7) : W_Stage_preprocess_1_1.generatedRoot.constraints_5 backend hashModel params' w_0_0 w_1_0 w_6_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_170_0 w_171_0 w_174_0 w_175_0 w_178_0 w_179_0 w_182_0 w_183_0 w_186_0 w_187_0 w_190_0 w_191_0 w_194_0 w_195_0 w_198_0 w_199_0 w_2_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_169_2 w_169_3 w_172_0 w_172_1 w_172_2 w_172_3 w_172_4 w_172_5 w_172_6 w_172_7 w_173_0 w_173_1 w_173_2 w_173_3 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_177_0 w_177_1 w_177_2 w_177_3 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_181_0 w_181_1 w_181_2 w_181_3 w_184_0 w_184_1 w_184_2 w_184_3 w_184_4 w_184_5 w_184_6 w_184_7 w_185_0 w_185_1 w_185_2 w_185_3 w_188_0 w_188_1 w_188_2 w_188_3 w_188_4 w_188_5 w_188_6 w_188_7 w_189_0 w_189_1 w_189_2 w_189_3 w_192_0 w_192_1 w_192_2 w_192_3 w_192_4 w_192_5 w_192_6 w_192_7 w_193_0 w_193_1 w_193_2 w_193_3 w_196_0 w_196_1 w_196_2 w_196_3 w_196_4 w_196_5 w_196_6 w_196_7 w_197_0 w_197_1 w_197_2 w_197_3 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_169.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_172.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_173.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_176.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_177.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_180.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_181.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_184.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_185.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_188.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_189.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_192.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_193.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_196.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_197.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_200.bridge (h32 i)))⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_4.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_138_0 w_139_0 w_142_0 w_143_0 w_146_0 w_147_0 w_150_0 w_151_0 w_154_0 w_155_0 w_158_0 w_159_0 w_162_0 w_163_0 w_166_0 w_167_0 w_2_0 w_136_0 w_136_1 w_136_2 w_136_3 w_137_0 w_137_1 w_137_2 w_137_3 w_140_0 w_140_1 w_140_2 w_140_3 w_140_4 w_140_5 w_140_6 w_140_7 w_141_0 w_141_1 w_141_2 w_141_3 w_144_0 w_144_1 w_144_2 w_144_3 w_144_4 w_144_5 w_144_6 w_144_7 w_145_0 w_145_1 w_145_2 w_145_3 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_148_7 w_149_0 w_149_1 w_149_2 w_149_3 w_152_0 w_152_1 w_152_2 w_152_3 w_152_4 w_152_5 w_152_6 w_152_7 w_153_0 w_153_1 w_153_2 w_153_3 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_157_0 w_157_1 w_157_2 w_157_3 w_160_0 w_160_1 w_160_2 w_160_3 w_160_4 w_160_5 w_160_6 w_160_7 w_161_0 w_161_1 w_161_2 w_161_3 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_164_6 w_164_7 w_165_0 w_165_1 w_165_2 w_165_3 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_168_6 w_168_7}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_4 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_138_0 w_139_0 w_142_0 w_143_0 w_146_0 w_147_0 w_150_0 w_151_0 w_154_0 w_155_0 w_158_0 w_159_0 w_162_0 w_163_0 w_166_0 w_167_0 w_2_0 w_136_0 w_136_1 w_136_2 w_136_3 w_137_0 w_137_1 w_137_2 w_137_3 w_140_0 w_140_1 w_140_2 w_140_3 w_140_4 w_140_5 w_140_6 w_140_7 w_141_0 w_141_1 w_141_2 w_141_3 w_144_0 w_144_1 w_144_2 w_144_3 w_144_4 w_144_5 w_144_6 w_144_7 w_145_0 w_145_1 w_145_2 w_145_3 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_148_7 w_149_0 w_149_1 w_149_2 w_149_3 w_152_0 w_152_1 w_152_2 w_152_3 w_152_4 w_152_5 w_152_6 w_152_7 w_153_0 w_153_1 w_153_2 w_153_3 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_157_0 w_157_1 w_157_2 w_157_3 w_160_0 w_160_1 w_160_2 w_160_3 w_160_4 w_160_5 w_160_6 w_160_7 w_161_0 w_161_1 w_161_2 w_161_3 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_164_6 w_164_7 w_165_0 w_165_1 w_165_2 w_165_3 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_168_6 w_168_7) : W_Stage_preprocess_1_1.generatedRoot.constraints_4 backend hashModel params' w_0_0 w_1_0 w_6_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_138_0 w_139_0 w_142_0 w_143_0 w_146_0 w_147_0 w_150_0 w_151_0 w_154_0 w_155_0 w_158_0 w_159_0 w_162_0 w_163_0 w_166_0 w_167_0 w_2_0 w_136_0 w_136_1 w_136_2 w_136_3 w_137_0 w_137_1 w_137_2 w_137_3 w_140_0 w_140_1 w_140_2 w_140_3 w_140_4 w_140_5 w_140_6 w_140_7 w_141_0 w_141_1 w_141_2 w_141_3 w_144_0 w_144_1 w_144_2 w_144_3 w_144_4 w_144_5 w_144_6 w_144_7 w_145_0 w_145_1 w_145_2 w_145_3 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_148_7 w_149_0 w_149_1 w_149_2 w_149_3 w_152_0 w_152_1 w_152_2 w_152_3 w_152_4 w_152_5 w_152_6 w_152_7 w_153_0 w_153_1 w_153_2 w_153_3 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_157_0 w_157_1 w_157_2 w_157_3 w_160_0 w_160_1 w_160_2 w_160_3 w_160_4 w_160_5 w_160_6 w_160_7 w_161_0 w_161_1 w_161_2 w_161_3 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_164_6 w_164_7 w_165_0 w_165_1 w_165_2 w_165_3 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_168_6 w_168_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_137.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_140.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_141.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_144.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_145.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_148.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_149.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_152.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_153.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_156.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_157.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_160.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_161.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_164.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_165.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_168.bridge (h32 i)))⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_3.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_106_0 w_107_0 w_110_0 w_111_0 w_114_0 w_115_0 w_118_0 w_119_0 w_122_0 w_123_0 w_126_0 w_127_0 w_130_0 w_131_0 w_134_0 w_135_0 w_2_0 w_102_0 w_103_0 w_104_0 w_105_0 w_108_0 w_108_1 w_108_2 w_108_3 w_108_4 w_108_5 w_108_6 w_108_7 w_109_0 w_109_1 w_109_2 w_109_3 w_112_0 w_112_1 w_112_2 w_112_3 w_112_4 w_112_5 w_112_6 w_112_7 w_113_0 w_113_1 w_113_2 w_113_3 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_124_0 w_124_1 w_124_2 w_124_3 w_124_4 w_124_5 w_124_6 w_124_7 w_125_0 w_125_1 w_125_2 w_125_3 w_128_0 w_128_1 w_128_2 w_128_3 w_128_4 w_128_5 w_128_6 w_128_7 w_129_0 w_129_1 w_129_2 w_129_3 w_132_0 w_132_1 w_132_2 w_132_3 w_132_4 w_132_5 w_132_6 w_132_7 w_133_0 w_133_1 w_133_2 w_133_3 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_106_0 w_107_0 w_110_0 w_111_0 w_114_0 w_115_0 w_118_0 w_119_0 w_122_0 w_123_0 w_126_0 w_127_0 w_130_0 w_131_0 w_134_0 w_135_0 w_2_0 w_102_0 w_103_0 w_104_0 w_105_0 w_108_0 w_108_1 w_108_2 w_108_3 w_108_4 w_108_5 w_108_6 w_108_7 w_109_0 w_109_1 w_109_2 w_109_3 w_112_0 w_112_1 w_112_2 w_112_3 w_112_4 w_112_5 w_112_6 w_112_7 w_113_0 w_113_1 w_113_2 w_113_3 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_124_0 w_124_1 w_124_2 w_124_3 w_124_4 w_124_5 w_124_6 w_124_7 w_125_0 w_125_1 w_125_2 w_125_3 w_128_0 w_128_1 w_128_2 w_128_3 w_128_4 w_128_5 w_128_6 w_128_7 w_129_0 w_129_1 w_129_2 w_129_3 w_132_0 w_132_1 w_132_2 w_132_3 w_132_4 w_132_5 w_132_6 w_132_7 w_133_0 w_133_1 w_133_2 w_133_3 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7) : W_Stage_preprocess_1_1.generatedRoot.constraints_3 backend hashModel params' w_0_0 w_1_0 w_6_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_106_0 w_107_0 w_110_0 w_111_0 w_114_0 w_115_0 w_118_0 w_119_0 w_122_0 w_123_0 w_126_0 w_127_0 w_130_0 w_131_0 w_134_0 w_135_0 w_2_0 w_102_0 w_103_0 w_104_0 w_105_0 w_108_0 w_108_1 w_108_2 w_108_3 w_108_4 w_108_5 w_108_6 w_108_7 w_109_0 w_109_1 w_109_2 w_109_3 w_112_0 w_112_1 w_112_2 w_112_3 w_112_4 w_112_5 w_112_6 w_112_7 w_113_0 w_113_1 w_113_2 w_113_3 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_116_6 w_116_7 w_117_0 w_117_1 w_117_2 w_117_3 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_121_1 w_121_2 w_121_3 w_124_0 w_124_1 w_124_2 w_124_3 w_124_4 w_124_5 w_124_6 w_124_7 w_125_0 w_125_1 w_125_2 w_125_3 w_128_0 w_128_1 w_128_2 w_128_3 w_128_4 w_128_5 w_128_6 w_128_7 w_129_0 w_129_1 w_129_2 w_129_3 w_132_0 w_132_1 w_132_2 w_132_3 w_132_4 w_132_5 w_132_6 w_132_7 w_133_0 w_133_1 w_133_2 w_133_3 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_108.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_109.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_112.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_113.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_116.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_117.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_120.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_121.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_124.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_125.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_128.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_129.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_132.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_133.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_136.bridge (h32 i)))⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_2.bridge {backend hashModel tape path params params' w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_102_0 w_103_0}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_2 backend hashModel tape path params w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_102_0 w_103_0) : W_Stage_preprocess_1_1.generatedRoot.constraints_2 backend hashModel params' w_6_0 w_3_0 w_3_1 w_3_2 w_3_3 w_5_0 w_33_1 w_33_3 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_64_0 w_65_0 w_66_0 w_66_1 w_66_2 w_66_3 w_102_0 w_103_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess_1_1.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess_1_1.sequential_generatedRoot_66.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess_1_1.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0}
    (h : Stage_preprocess_1_1.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0) : W_Stage_preprocess_1_1.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_6_0 w_31_0 w_32_0 w_2_0 w_3_0 w_3_1 w_3_2 w_3_3 w_4_0 w_4_1 w_4_2 w_4_3 w_7_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_33_0 w_33_1 w_33_2 w_33_3 w_34_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_3.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_4.bridge (h4 i))), h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (fun i ↦ (Stage_preprocess_1_1.parallel_generatedRoot_33.bridge (h31 i))), h32⟩)


theorem Stage_preprocess_1_1.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_preprocess_1_1.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_preprocess_1_1.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_preprocess_1_1.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_3_1, wit1.w_3_2, wit1.w_3_3, wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_7_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_20_0, wit1.w_23_0, wit1.w_24_0, wit1.w_25_0, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_33_0, wit1.w_33_1, wit1.w_33_2, wit1.w_33_3, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_38_0, wit1.w_39_0, wit1.w_41_0, wit1.w_42_0, wit1.w_45_0, wit1.w_46_0, wit1.w_47_0, wit1.w_48_0, wit1.w_49_0, wit1.w_50_0, wit1.w_53_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_59_0, wit1.w_60_0, wit1.w_61_0, wit1.w_64_0, wit1.w_65_0, wit1.w_66_0, wit1.w_66_1, wit1.w_66_2, wit1.w_66_3, wit1.w_102_0, wit1.w_103_0, wit1.w_104_0, wit1.w_105_0, wit1.w_108_0, wit1.w_108_1, wit1.w_108_2, wit1.w_108_3, wit1.w_108_4, wit1.w_108_5, wit1.w_108_6, wit1.w_108_7, wit1.w_109_0, wit1.w_109_1, wit1.w_109_2, wit1.w_109_3, wit1.w_112_0, wit1.w_112_1, wit1.w_112_2, wit1.w_112_3, wit1.w_112_4, wit1.w_112_5, wit1.w_112_6, wit1.w_112_7, wit1.w_113_0, wit1.w_113_1, wit1.w_113_2, wit1.w_113_3, wit1.w_116_0, wit1.w_116_1, wit1.w_116_2, wit1.w_116_3, wit1.w_116_4, wit1.w_116_5, wit1.w_116_6, wit1.w_116_7, wit1.w_117_0, wit1.w_117_1, wit1.w_117_2, wit1.w_117_3, wit1.w_120_0, wit1.w_120_1, wit1.w_120_2, wit1.w_120_3, wit1.w_120_4, wit1.w_120_5, wit1.w_120_6, wit1.w_120_7, wit1.w_121_0, wit1.w_121_1, wit1.w_121_2, wit1.w_121_3, wit1.w_124_0, wit1.w_124_1, wit1.w_124_2, wit1.w_124_3, wit1.w_124_4, wit1.w_124_5, wit1.w_124_6, wit1.w_124_7, wit1.w_125_0, wit1.w_125_1, wit1.w_125_2, wit1.w_125_3, wit1.w_128_0, wit1.w_128_1, wit1.w_128_2, wit1.w_128_3, wit1.w_128_4, wit1.w_128_5, wit1.w_128_6, wit1.w_128_7, wit1.w_129_0, wit1.w_129_1, wit1.w_129_2, wit1.w_129_3, wit1.w_132_0, wit1.w_132_1, wit1.w_132_2, wit1.w_132_3, wit1.w_132_4, wit1.w_132_5, wit1.w_132_6, wit1.w_132_7, wit1.w_133_0, wit1.w_133_1, wit1.w_133_2, wit1.w_133_3, wit1.w_136_0, wit1.w_136_1, wit1.w_136_2, wit1.w_136_3, wit1.w_136_4, wit1.w_136_5, wit1.w_136_6, wit1.w_136_7, wit1.w_137_0, wit1.w_137_1, wit1.w_137_2, wit1.w_137_3, wit1.w_140_0, wit1.w_140_1, wit1.w_140_2, wit1.w_140_3, wit1.w_140_4, wit1.w_140_5, wit1.w_140_6, wit1.w_140_7, wit1.w_141_0, wit1.w_141_1, wit1.w_141_2, wit1.w_141_3, wit1.w_144_0, wit1.w_144_1, wit1.w_144_2, wit1.w_144_3, wit1.w_144_4, wit1.w_144_5, wit1.w_144_6, wit1.w_144_7, wit1.w_145_0, wit1.w_145_1, wit1.w_145_2, wit1.w_145_3, wit1.w_148_0, wit1.w_148_1, wit1.w_148_2, wit1.w_148_3, wit1.w_148_4, wit1.w_148_5, wit1.w_148_6, wit1.w_148_7, wit1.w_149_0, wit1.w_149_1, wit1.w_149_2, wit1.w_149_3, wit1.w_152_0, wit1.w_152_1, wit1.w_152_2, wit1.w_152_3, wit1.w_152_4, wit1.w_152_5, wit1.w_152_6, wit1.w_152_7, wit1.w_153_0, wit1.w_153_1, wit1.w_153_2, wit1.w_153_3, wit1.w_156_0, wit1.w_156_1, wit1.w_156_2, wit1.w_156_3, wit1.w_156_4, wit1.w_156_5, wit1.w_156_6, wit1.w_156_7, wit1.w_157_0, wit1.w_157_1, wit1.w_157_2, wit1.w_157_3, wit1.w_160_0, wit1.w_160_1, wit1.w_160_2, wit1.w_160_3, wit1.w_160_4, wit1.w_160_5, wit1.w_160_6, wit1.w_160_7, wit1.w_161_0, wit1.w_161_1, wit1.w_161_2, wit1.w_161_3, wit1.w_164_0, wit1.w_164_1, wit1.w_164_2, wit1.w_164_3, wit1.w_164_4, wit1.w_164_5, wit1.w_164_6, wit1.w_164_7, wit1.w_165_0, wit1.w_165_1, wit1.w_165_2, wit1.w_165_3, wit1.w_168_0, wit1.w_168_1, wit1.w_168_2, wit1.w_168_3, wit1.w_168_4, wit1.w_168_5, wit1.w_168_6, wit1.w_168_7, wit1.w_169_0, wit1.w_169_1, wit1.w_169_2, wit1.w_169_3, wit1.w_172_0, wit1.w_172_1, wit1.w_172_2, wit1.w_172_3, wit1.w_172_4, wit1.w_172_5, wit1.w_172_6, wit1.w_172_7, wit1.w_173_0, wit1.w_173_1, wit1.w_173_2, wit1.w_173_3, wit1.w_176_0, wit1.w_176_1, wit1.w_176_2, wit1.w_176_3, wit1.w_176_4, wit1.w_176_5, wit1.w_176_6, wit1.w_176_7, wit1.w_177_0, wit1.w_177_1, wit1.w_177_2, wit1.w_177_3, wit1.w_180_0, wit1.w_180_1, wit1.w_180_2, wit1.w_180_3, wit1.w_180_4, wit1.w_180_5, wit1.w_180_6, wit1.w_180_7, wit1.w_181_0, wit1.w_181_1, wit1.w_181_2, wit1.w_181_3, wit1.w_184_0, wit1.w_184_1, wit1.w_184_2, wit1.w_184_3, wit1.w_184_4, wit1.w_184_5, wit1.w_184_6, wit1.w_184_7, wit1.w_185_0, wit1.w_185_1, wit1.w_185_2, wit1.w_185_3, wit1.w_188_0, wit1.w_188_1, wit1.w_188_2, wit1.w_188_3, wit1.w_188_4, wit1.w_188_5, wit1.w_188_6, wit1.w_188_7, wit1.w_189_0, wit1.w_189_1, wit1.w_189_2, wit1.w_189_3, wit1.w_192_0, wit1.w_192_1, wit1.w_192_2, wit1.w_192_3, wit1.w_192_4, wit1.w_192_5, wit1.w_192_6, wit1.w_192_7, wit1.w_193_0, wit1.w_193_1, wit1.w_193_2, wit1.w_193_3, wit1.w_196_0, wit1.w_196_1, wit1.w_196_2, wit1.w_196_3, wit1.w_196_4, wit1.w_196_5, wit1.w_196_6, wit1.w_196_7, wit1.w_197_0, wit1.w_197_1, wit1.w_197_2, wit1.w_197_3, wit1.w_200_0, wit1.w_200_1, wit1.w_200_2, wit1.w_200_3, wit1.w_200_4, wit1.w_200_5, wit1.w_200_6, wit1.w_200_7, wit1.w_201_0, wit1.w_201_1, wit1.w_201_2, wit1.w_201_3, wit1.w_204_0, wit1.w_204_1, wit1.w_204_2, wit1.w_204_3, wit1.w_204_4, wit1.w_204_5, wit1.w_204_6, wit1.w_204_7, wit1.w_205_0, wit1.w_205_1, wit1.w_205_2, wit1.w_205_3, wit1.w_208_0, wit1.w_208_1, wit1.w_208_2, wit1.w_208_3, wit1.w_208_4, wit1.w_208_5, wit1.w_208_6, wit1.w_208_7, wit1.w_209_0, wit1.w_209_1, wit1.w_209_2, wit1.w_209_3, wit1.w_212_0, wit1.w_212_1, wit1.w_212_2, wit1.w_212_3, wit1.w_212_4, wit1.w_212_5, wit1.w_212_6, wit1.w_212_7, wit1.w_213_0, wit1.w_213_1, wit1.w_213_2, wit1.w_213_3, wit1.w_216_0, wit1.w_216_1, wit1.w_216_2, wit1.w_216_3, wit1.w_216_4, wit1.w_216_5, wit1.w_216_6, wit1.w_216_7, wit1.w_217_0, wit1.w_217_1, wit1.w_217_2, wit1.w_217_3, wit1.w_220_0, wit1.w_220_1, wit1.w_220_2, wit1.w_220_3, wit1.w_220_4, wit1.w_220_5, wit1.w_220_6, wit1.w_220_7, wit1.w_221_0, wit1.w_221_1, wit1.w_221_2, wit1.w_221_3, wit1.w_224_0, wit1.w_224_1, wit1.w_224_2, wit1.w_224_3, wit1.w_224_4, wit1.w_224_5, wit1.w_224_6, wit1.w_224_7, wit1.w_225_0, wit1.w_225_1, wit1.w_225_2, wit1.w_225_3, wit1.w_228_0, wit1.w_228_1, wit1.w_228_2, wit1.w_228_3, wit1.w_228_4, wit1.w_228_5, wit1.w_228_6, wit1.w_228_7, wit1.w_229_0, wit1.w_229_1, wit1.w_229_2, wit1.w_229_3, wit1.w_232_0, wit1.w_232_1, wit1.w_232_2, wit1.w_232_3, wit1.w_232_4, wit1.w_232_5, wit1.w_232_6, wit1.w_232_7, wit1.w_233_0, wit1.w_233_1, wit1.w_233_2, wit1.w_233_3, wit1.w_236_0, wit1.w_236_1, wit1.w_236_2, wit1.w_236_3, wit1.w_236_4, wit1.w_236_5, wit1.w_236_6, wit1.w_236_7, wit1.w_237_0, wit1.w_237_1, wit1.w_237_2, wit1.w_237_3, wit1.w_240_0, wit1.w_240_1, wit1.w_240_2, wit1.w_240_3, wit1.w_240_4, wit1.w_240_5, wit1.w_240_6, wit1.w_240_7, wit1.w_241_0, wit1.w_241_1, wit1.w_241_2, wit1.w_241_3, wit1.w_242_0, wit1.w_242_1, wit1.w_242_2, wit1.w_242_3, wit1.w_243_0, wit1.w_243_1, wit1.w_243_2, wit1.w_243_3, wit1.w_244_0, wit1.w_245_0, wit1.w_247_0, wit1.w_248_0, wit1.w_250_0, wit1.w_251_0, wit1.w_253_0, wit1.w_254_0, wit1.w_256_0, wit1.w_256_1, wit1.w_256_2, wit1.w_256_3, wit1.w_257_0, wit1.w_259_0, wit1.w_261_0, wit1.w_263_0, wit1.w_264_0, wit1.w_264_1, wit1.w_264_2, wit1.w_264_3⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(by obtain ⟨h5, h6⟩ := h3; exact ⟨(by obtain ⟨h7, h8⟩ := h5; exact ⟨(Stage_preprocess_1_1.generatedRoot.constraints_0.bridge h7), h8⟩), (by obtain ⟨h9, h10⟩ := h6; exact ⟨(Stage_preprocess_1_1.generatedRoot.constraints_2.bridge h9), (Stage_preprocess_1_1.generatedRoot.constraints_3.bridge h10)⟩)⟩), (by obtain ⟨h11, h12⟩ := h4; exact ⟨(by obtain ⟨h13, h14⟩ := h11; exact ⟨(Stage_preprocess_1_1.generatedRoot.constraints_4.bridge h13), (Stage_preprocess_1_1.generatedRoot.constraints_5.bridge h14)⟩), (by obtain ⟨h15, h16⟩ := h12; exact ⟨(Stage_preprocess_1_1.generatedRoot.constraints_6.bridge h15), (by obtain ⟨h17, h18⟩ := h16; exact ⟨(Stage_preprocess_1_1.generatedRoot.constraints_7.bridge h17), (Stage_preprocess_1_1.generatedRoot.constraints_8.bridge h18)⟩)⟩)⟩)⟩)⟩)
