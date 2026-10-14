import Stage_preprocess
import W_Stage_preprocess
import ExponentSampling

/-! Every run of the average-case relations of `Stage_preprocess` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_preprocess.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_39.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_39 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_39 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_39
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_40.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_40 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_40 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_40
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_41.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_41 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_41 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_41
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_42.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_42 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_42 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_42
  exact h


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_47_11.bridge {backend x1 tape path params params' x5 x6 i_2 x8 outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_47_11 backend x1 tape path params x5 x6 i_2 x8 outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_47_11 backend x1 params' x5 x6 i_2 x8 outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_47_11
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_47_14.bridge {backend x1 tape path params params' i_0 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_47_14 backend x1 tape path params i_0 i_1 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_47_14 backend x1 params' i_0 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_47_14
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_47.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_3_0 w_26_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_21_0 w_22_0 w_27_0 w_31_0 w_32_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_47.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_3_0 w_26_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_21_0 w_22_0 w_27_0 w_31_0 w_32_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_47.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_3_0 w_26_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_21_0 w_22_0 w_27_0 w_31_0 w_32_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_47_11.bridge (h14 i))), h15, (Stage_preprocess.iterRuns_mono h16 (fun _ _ _ hs22 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_47_14.bridge hs22))), h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_47.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_47 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_47 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_47
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_5_0, wit1.w_11_0, wit1.w_14_0, wit1.w_18_0, wit1.w_21_0, wit1.w_27_0, wit1.w_31_0⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_47.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_59_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_60_0 w_66_0 w_67_0 w_68_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_59_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_60_0 w_66_0 w_67_0 w_68_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_2 backend hashModel i_0 i_1 w_0_0 w_59_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_60_0 w_66_0 w_67_0 w_68_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (MxxExponentLut.gaussianSampleAt_sample (by decide) h11), h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_1.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_22_0 w_25_0 w_32_0 w_18_0 w_19_0 w_20_0 w_23_0 w_26_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_42_0 w_43_0 w_47_0 w_48_0 w_49_0 w_50_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_1 backend hashModel tape path i_0 i_1 w_0_0 w_22_0 w_25_0 w_32_0 w_18_0 w_19_0 w_20_0 w_23_0 w_26_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_42_0 w_43_0 w_47_0 w_48_0 w_49_0 w_50_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_1 backend hashModel i_0 i_1 w_0_0 w_22_0 w_25_0 w_32_0 w_18_0 w_19_0 w_20_0 w_23_0 w_26_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_42_0 w_43_0 w_47_0 w_48_0 w_49_0 w_50_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, (MxxExponentLut.gaussianSampleAt_sample (by decide) h26), h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_54.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_54 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_54 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_54
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_23_0, w_26_0, w_29_0, w_30_0, w_31_0, w_33_0, w_34_0, w_35_0, w_43_0, w_47_0, w_49_0, w_50_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_60_0, w_66_0, w_68_0, w_69_0, w_72_0, w_73_0, w_74_0, w_75_0, w_76_0, w_77_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_23_0, w_26_0, w_29_0, w_30_0, w_31_0, w_33_0, w_34_0, w_35_0, w_43_0, w_47_0, w_49_0, w_50_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_60_0, w_66_0, w_68_0, w_69_0, w_72_0, w_73_0, w_74_0, w_75_0, w_76_0, w_77_0, h1, (Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_1.bridge h2), (Stage_preprocess.parallel_parallel_generatedRoot_211_54.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_55_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_55_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_55_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_55_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_55_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_55_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_55_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_55_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_55.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_55.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_55.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_55_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_55_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_55.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_55 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_55 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_55
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_55.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_58.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_58.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_58.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_58.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_58 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_58 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_58
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_58.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_59_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_59_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_59_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_59_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_59_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_59_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_59_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_59_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_59.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_59.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_59.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_59_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_59_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_59.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_59 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_59 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_59
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_59.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_62.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_62.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_62.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_62.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_62 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_62 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_62
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_62.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_63_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_63_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_63_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_63_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_63_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_63_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_63_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_63_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_63.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_63.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_63.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_63_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_63_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_63.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_63 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_63 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_63
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_63.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_66.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_66.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_66.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_66.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_66 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_66 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_66
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_66.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_67_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_67_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_67_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_67_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_67_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_67_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_67_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_67_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_67.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_67.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_67.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_67_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_67_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_67.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_67 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_67 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_67
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_67.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_70.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_70.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_70.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_70.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_70 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_70 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_70
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_70.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_71_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_71_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_71_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_71_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_71_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_71_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_71_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_71_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_71.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_71.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_71.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_71_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_71_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_71.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_71 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_71 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_71
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_71.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_74.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_74.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_74.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_74.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_74 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_74 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_74
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_74.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_75_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_75_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_75_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_75_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_75_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_75_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_75_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_75_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_75.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_75.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_75.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_75_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_75_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_75.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_75 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_75 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_75
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_75.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_78.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_78.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_78.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_78.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_78 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_78 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_78
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_78.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_79_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_79_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_79_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_79_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_79_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_79_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_79_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_79_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_79.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_79.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_79.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_79_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_79_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_79.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_79 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_79 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_79
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_79.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_82.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_82.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_82.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_82.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_82 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_82 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_82
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_82.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_83_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_83_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_83_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_83_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_83_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_83_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_83_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_83_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_83.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_83.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_83.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_83_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_83_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_83.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_83 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_83 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_83
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_83.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_86.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_86.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_86.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_86.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_86 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_86 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_86
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_86.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_87_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_87_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_87_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_87_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_87_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_87_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_87_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_87_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_87.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_87.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_87.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_87_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_87_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_87.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_87 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_87 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_87
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_87.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_90.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_90.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_90.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_90.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_90 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_90 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_90
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_90.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_91_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_91_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_91_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_91_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_91_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_91_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_91_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_91_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_91.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_91.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_91.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_91_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_91_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_91.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_91 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_91 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_91
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_91.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_94.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_94.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_94.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_94.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_94 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_94 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_94
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_94.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_95_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_95_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_95_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_95_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_95_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_95_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_95_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_95_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_95.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_95.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_95.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_95_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_95_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_95.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_95 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_95 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_95
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_95.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_98.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_98.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_98.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_98.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_98 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_98 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_98
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_98.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_99_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_99_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_99_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_99_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_99_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_99_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_99_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_99_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_99.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_99.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_99.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_99_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_99_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_99.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_99 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_99 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_99
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_99.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_102.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_102.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_102.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_102.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_102 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_102 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_102
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_102.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_103_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_103_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_103_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_103_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_103_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_103_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_103_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_103_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_103.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_103.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_103.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_103_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_103_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_103.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_103 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_103 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_103
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_103.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_106.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_106.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_106.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_106.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_106 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_106 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_106
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_106.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_107_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_107_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_107_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_107_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_107_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_107_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_107_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_107_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_107.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_107.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_107.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_107_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_107_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_107.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_107 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_107 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_107
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_107.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_110.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_110.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_110.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_110.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_110 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_110 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_110
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_110.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_111_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_111_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_111_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_111_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_111_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_111_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_111_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_111_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_111.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_111.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_111.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_111_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_111_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_111.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_111 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_111 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_111
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_111.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_114.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_114.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_114.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_114.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_114 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_114 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_114
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_114.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_115_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_115_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_115_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_115_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_115_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_115_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_115_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_115_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_115.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_115.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_115.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_115_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_115_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_115.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_115 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_115 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_115
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_115.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_118.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_118.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_118.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_118.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_118 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_118 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_118
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_118.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_119_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_119_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_119_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_119_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_119_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_119_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_119_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_119_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_119.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_119.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_119.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_119_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_119_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_119.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_119 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_119 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_119
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_119.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_122.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_122.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_122.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_122.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_122 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_122 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_122
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_122.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_123_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_123_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_123_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_123_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_123_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_123_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_123_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_123_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_123.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_123.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_123.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_123_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_123_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_123.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_123 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_123 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_123
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_123.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_126.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_126.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_126.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_126.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_126 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_126 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_126
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_126.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_127_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_127_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_127_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_127_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_127_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_127_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_127_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_127_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_127.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_127.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_127.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_127_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_127_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_127.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_127 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_127 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_127
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_127.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_130.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_130.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_130.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_130.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_130 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_130 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_130
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_130.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_131_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_131_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_131_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_131_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_131_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_131_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_131_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_131_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_131.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_131.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_131.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_131_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_131_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_131.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_131 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_131 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_131
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_131.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_134.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_134.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_134.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_134.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_134 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_134 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_134
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_134.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_135_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_135_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_135_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_135_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_135_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_135_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_135_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_135_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_135.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_135.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_135.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_135_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_135_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_135.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_135 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_135 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_135
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_135.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_138.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_138.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_138.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_138.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_138 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_138 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_138
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_138.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_139_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_139_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_139_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_139_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_139_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_139_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_139_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_139_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_139.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_139.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_139.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_139_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_139_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_139.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_139 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_139 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_139
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_139.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_142.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_142.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_142.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_142.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_142 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_142 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_142
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_142.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_143_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_143_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_143_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_143_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_143_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_143_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_143_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_143_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_143.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_143.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_143.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_143_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_143_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_143.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_143 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_143 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_143
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_143.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_146.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_146.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_146.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_146.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_146 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_146 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_146
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_146.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_147_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_147_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_147_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_147_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_147_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_147_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_147_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_147_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_147.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_147.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_147.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_147_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_147_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_147.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_147 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_147 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_147
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_147.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_150.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_150.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_150.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_150.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_150 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_150 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_150
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_150.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_151_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_151_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_151_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_151_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_151_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_151_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_151_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_151_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_151.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_151.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_151.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_151_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_151_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_151.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_151 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_151 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_151
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_151.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_154.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_154.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_154.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_154.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_154 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_154 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_154
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_154.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_155_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_155_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_155_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_155_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_155_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_155_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_155_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_155_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_155.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_155.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_155.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_155_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_155_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_155.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_155 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_155 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_155
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_155.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_158.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_158.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_158.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_158.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_158 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_158 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_158
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_158.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_159_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_159_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_159_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_159_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_159_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_159_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_159_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_159_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_159.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_159.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_159.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_159_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_159_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_159.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_159 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_159 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_159
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_159.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_162.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_162.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_162.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_162.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_162 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_162 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_162
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_162.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_163_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_163_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_163_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_163_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_163_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_163_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_163_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_163_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_163.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_163.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_163.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_163_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_163_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_163.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_163 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_163 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_163
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_163.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_166.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_166.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_166.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_166.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_166 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_166 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_166
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_166.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_167_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_167_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_167_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_167_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_167_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_167_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_167_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_167_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_167.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_167.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_167.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_167_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_167_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_167.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_167 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_167 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_167
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_167.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_170.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_170.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_170.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_170.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_170 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_170 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_170
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_170.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_171_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_171_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_171_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_171_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_171_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_171_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_171_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_171_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_171.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_171.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_171.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_171_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_171_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_171.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_171 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_171 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_171
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_171.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_174.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_174.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_174.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_174.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_174 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_174 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_174
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_174.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_175_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_175_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_175_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_175_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_175_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_175_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_175_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_175_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_175.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_175.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_175.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_175_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_175_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_175.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_175 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_175 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_175
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_175.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_178.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_178.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_178.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_178.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_178 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_178 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_178
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_178.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_179_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_179_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_179_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_179_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_179_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_179_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_179_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_179_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_179.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_179.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_179.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_179_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_179_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_179.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_179 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_179 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_179
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_179.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_182.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_182.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_182.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_182.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_182 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_182 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_182
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_182.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_183_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_183_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_183_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_183_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_183_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_183_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_183_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_183_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_183.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_183.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_183.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_183_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_183_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_183.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_183 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_183 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_183
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_183.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_3.bridge {backend tape path w_34_0 w_66_0 w_70_0 w_72_0 w_73_0 w_84_0 w_85_0 w_86_0 w_87_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_3 backend tape path w_34_0 w_66_0 w_70_0 w_72_0 w_73_0 w_84_0 w_85_0 w_86_0 w_87_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_3 backend w_34_0 w_66_0 w_70_0 w_72_0 w_73_0 w_84_0 w_85_0 w_86_0 w_87_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28⟩ := h; exact ⟨(MxxExponentLut.gaussianSampleAt_sample (by decide) h1), h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_186.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_186 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_186 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_186
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_2.bridge h3), (Stage_preprocess.parallel_parallel_generatedRoot_211_186.constraints_3.bridge h4)⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_204_13.bridge {backend x1 tape path params params' x5 x6 i_2 x8 outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_204_13 backend x1 tape path params x5 x6 i_2 x8 outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_204_13 backend x1 params' x5 x6 i_2 x8 outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_204_13
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_204_14.bridge {backend x1 tape path params params' i_0 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_204_14 backend x1 tape path params i_0 i_1 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_204_14 backend x1 params' i_0 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_204_14
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_204.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_3_0 w_27_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_19_0 w_22_0 w_23_0 w_28_0 w_32_0 w_33_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_204.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_3_0 w_27_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_19_0 w_22_0 w_23_0 w_28_0 w_32_0 w_33_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_204.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_3_0 w_27_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_19_0 w_22_0 w_23_0 w_28_0 w_32_0 w_33_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_204_13.bridge (h16 i))), h17, (Stage_preprocess.iterRuns_mono h18 (fun _ _ _ hs25 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_204_14.bridge hs25))), h19, h20, (MxxExponentLut.gaussianSampleAt_sample (by decide) h21), h22, (MxxExponentLut.gaussianSampleAt_sample (by decide) h23), h24⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_204.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_204 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_204 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_204
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_5_0, wit1.w_13_0, wit1.w_14_0, wit1.w_18_0, wit1.w_19_0, wit1.w_22_0, wit1.w_28_0, wit1.w_32_0⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_204.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_207.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_207 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_207 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_207
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_59_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_60_0 w_66_0 w_67_0 w_68_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_59_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_60_0 w_66_0 w_67_0 w_68_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_2 backend hashModel i_0 i_1 w_0_0 w_59_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_60_0 w_66_0 w_67_0 w_68_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, (MxxExponentLut.gaussianSampleAt_sample (by decide) h11), h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_1.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_22_0 w_25_0 w_32_0 w_18_0 w_19_0 w_20_0 w_23_0 w_26_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_42_0 w_43_0 w_47_0 w_48_0 w_49_0 w_50_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_1 backend hashModel tape path i_0 i_1 w_0_0 w_22_0 w_25_0 w_32_0 w_18_0 w_19_0 w_20_0 w_23_0 w_26_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_42_0 w_43_0 w_47_0 w_48_0 w_49_0 w_50_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_1 backend hashModel i_0 i_1 w_0_0 w_22_0 w_25_0 w_32_0 w_18_0 w_19_0 w_20_0 w_23_0 w_26_0 w_28_0 w_29_0 w_30_0 w_31_0 w_33_0 w_34_0 w_35_0 w_42_0 w_43_0 w_47_0 w_48_0 w_49_0 w_50_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, (MxxExponentLut.gaussianSampleAt_sample (by decide) h26), h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_269.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_269 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_269 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_269
  exact (by obtain ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_23_0, w_26_0, w_29_0, w_30_0, w_31_0, w_33_0, w_34_0, w_35_0, w_43_0, w_47_0, w_49_0, w_50_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_60_0, w_66_0, w_68_0, w_69_0, w_72_0, w_73_0, w_74_0, w_75_0, w_76_0, w_77_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_10_0, w_12_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_20_0, w_23_0, w_26_0, w_29_0, w_30_0, w_31_0, w_33_0, w_34_0, w_35_0, w_43_0, w_47_0, w_49_0, w_50_0, w_53_0, w_54_0, w_55_0, w_56_0, w_57_0, w_60_0, w_66_0, w_68_0, w_69_0, w_72_0, w_73_0, w_74_0, w_75_0, w_76_0, w_77_0, h1, (Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_1.bridge h2), (Stage_preprocess.parallel_parallel_generatedRoot_211_269.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_270_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_270_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_270_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_270_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_270_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_270_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_270_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_270_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_270.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_270.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_270.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_270_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_270_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_270.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_270 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_270 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_270
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_270.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_273.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_273.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_273.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_273.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_273 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_273 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_273
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_273.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_274_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_274_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_274_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_274_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_274_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_274_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_274_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_274_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_274.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_274.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_274.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_274_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_274_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_274.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_274 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_274 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_274
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_274.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_277.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_277.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_277.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_277.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_277 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_277 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_277
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_277.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_278_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_278_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_278_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_278_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_278_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_278_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_278_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_278_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_278.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_278.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_278.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_278_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_278_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_278.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_278 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_278 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_278
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_278.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_281.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_281.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_281.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_281.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_281 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_281 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_281
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_281.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_282_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_282_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_282_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_282_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_282_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_282_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_282_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_282_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_282.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_282.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_282.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_282_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_282_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_282.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_282 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_282 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_282
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_282.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_285.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_285.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_285.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_285.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_285 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_285 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_285
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_285.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_286_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_286_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_286_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_286_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_286_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_286_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_286_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_286_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_286.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_286.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_286.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_286_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_286_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_286.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_286 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_286 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_286
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_286.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_289.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_289.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_289.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_289.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_289 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_289 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_289
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_289.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_290_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_290_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_290_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_290_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_290_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_290_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_290_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_290_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_290.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_290.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_290.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_290_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_290_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_290.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_290 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_290 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_290
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_290.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_293.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_293.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_293.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_293.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_293 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_293 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_293
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_293.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_294_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_294_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_294_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_294_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_294_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_294_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_294_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_294_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_294.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_294.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_294.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_294_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_294_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_294.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_294 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_294 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_294
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_294.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_297.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_297.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_297.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_297.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_297 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_297 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_297
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_297.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_298_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_298_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_298_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_298_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_298_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_298_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_298_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_298_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_298.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_298.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_298.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_298_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_298_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_298.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_298 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_298 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_298
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_298.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_301.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_301.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_301.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_301.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_301 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_301 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_301
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_301.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_302_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_302_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_302_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_302_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_302_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_302_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_302_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_302_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_302.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_302.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_302.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_302_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_302_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_302.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_302 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_302 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_302
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_302.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_305.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_305.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_305.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_305.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_305 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_305 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_305
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_305.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_306_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_306_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_306_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_306_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_306_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_306_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_306_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_306_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_306.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_306.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_306.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_306_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_306_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_306.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_306 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_306 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_306
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_306.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_309.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_309.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_309.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_309.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_309 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_309 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_309
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_309.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_310_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_310_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_310_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_310_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_310_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_310_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_310_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_310_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_310.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_310.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_310.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_310_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_310_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_310.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_310 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_310 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_310
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_310.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_313.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_313.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_313.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_313.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_313 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_313 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_313
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_313.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_314_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_314_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_314_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_314_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_314_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_314_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_314_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_314_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_314.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_314.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_314.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_314_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_314_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_314.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_314 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_314 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_314
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_314.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_317.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_317.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_317.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_317.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_317 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_317 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_317
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_317.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_318_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_318_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_318_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_318_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_318_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_318_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_318_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_318_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_318.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_318.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_318.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_318_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_318_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_318.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_318 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_318 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_318
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_318.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_321.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_321.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_321.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_321.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_321 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_321 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_321
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_321.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_322_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_322_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_322_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_322_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_322_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_322_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_322_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_322_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_322.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_322.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_322.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_322_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_322_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_322.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_322 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_322 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_322
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_322.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_325.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_325.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_325.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_325.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_325 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_325 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_325
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_325.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_326_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_326_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_326_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_326_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_326_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_326_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_326_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_326_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_326.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_326.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_326.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_326_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_326_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_326.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_326 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_326 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_326
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_326.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_329.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_329.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_329.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_329.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_329 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_329 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_329
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_329.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_330_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_330_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_330_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_330_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_330_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_330_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_330_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_330_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_330.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_330.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_330.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_330_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_330_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_330.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_330 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_330 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_330
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_330.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_333.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_333.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_333.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_333.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_333 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_333 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_333
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_333.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_334_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_334_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_334_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_334_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_334_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_334_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_334_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_334_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_334.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_334.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_334.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_334_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_334_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_334.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_334 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_334 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_334
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_334.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_337.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_337.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_337.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_337.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_337 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_337 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_337
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_337.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_338_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_338_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_338_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_338_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_338_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_338_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_338_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_338_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_338.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_338.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_338.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_338_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_338_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_338.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_338 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_338 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_338
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_338.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_341.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_341.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_341.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_341.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_341 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_341 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_341
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_341.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_342_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_342_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_342_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_342_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_342_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_342_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_342_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_342_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_342.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_342.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_342.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_342_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_342_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_342.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_342 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_342 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_342
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_342.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_345.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_345.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_345.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_345.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_345 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_345 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_345
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_345.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_346_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_346_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_346_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_346_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_346_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_346_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_346_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_346_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_346.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_346.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_346.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_346_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_346_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_346.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_346 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_346 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_346
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_346.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_349.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_349.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_349.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_349.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_349 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_349 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_349
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_349.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_350_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_350_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_350_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_350_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_350_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_350_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_350_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_350_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_350.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_350.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_350.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_350_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_350_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_350.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_350 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_350 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_350
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_350.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_353.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_353.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_353.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_353.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_353 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_353 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_353
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_353.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_354_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_354_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_354_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_354_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_354_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_354_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_354_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_354_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_354.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_354.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_354.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_354_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_354_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_354.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_354 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_354 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_354
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_354.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_357.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_357.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_357.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_357.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_357 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_357 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_357
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_357.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_358_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_358_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_358_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_358_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_358_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_358_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_358_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_358_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_358.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_358.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_358.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_358_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_358_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_358.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_358 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_358 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_358
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_358.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_361.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_361.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_361.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_361.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_361 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_361 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_361
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_361.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_362_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_362_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_362_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_362_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_362_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_362_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_362_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_362_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_362.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_362.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_362.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_362_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_362_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_362.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_362 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_362 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_362
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_362.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_365.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_365.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_365.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_365.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_365 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_365 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_365
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_365.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_366_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_366_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_366_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_366_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_366_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_366_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_366_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_366_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_366.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_366.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_366.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_366_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_366_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_366.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_366 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_366 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_366
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_366.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_369.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_369.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_369.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_369.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_369 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_369 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_369
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_369.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_370_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_370_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_370_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_370_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_370_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_370_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_370_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_370_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_370.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_370.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_370.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_370_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_370_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_370.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_370 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_370 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_370
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_370.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_373.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_373.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_373.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_373.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_373 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_373 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_373
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_373.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_374_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_374_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_374_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_374_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_374_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_374_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_374_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_374_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_374.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_374.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_374.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_374_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_374_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_374.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_374 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_374 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_374
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_374.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_377.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_377.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_377.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_377.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_377 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_377 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_377
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_377.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_378_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_378_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_378_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_378_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_378_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_378_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_378_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_378_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_378.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_378.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_378.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_378_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_378_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_378.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_378 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_378 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_378
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_378.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_381.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_381.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_381.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_381.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_381 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_381 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_381
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_381.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_382_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_382_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_382_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_382_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_382_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_382_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_382_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_382_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_382.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_382.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_382.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_382_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_382_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_382.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_382 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_382 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_382
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_382.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_385.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_385.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_385.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_385.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_385 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_385 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_385
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_385.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_386_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_386_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_386_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_386_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_386_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_386_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_386_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_386_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_386.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_386.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_386.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_386_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_386_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_386.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_386 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_386 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_386
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_386.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_389.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_389.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_389.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_389.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_389 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_389 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_389
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_389.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_390_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_390_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_390_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_390_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_390_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_390_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_390_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_390_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_390.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_390.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_390.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_390_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_390_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_390.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_390 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_390 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_390
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_390.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_393.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_393.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_393.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_393.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_393 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_393 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_393
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_393.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_394_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_394_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_394_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_394_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_394_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_394_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_394_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_394_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_394.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_394.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_394.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_394_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_394_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_394.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_394 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_394 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_394
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_394.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_397.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_397.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_397.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_397.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_397 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_397 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_397
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_397.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_398_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_398_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_398_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_398_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_398_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_398_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_398_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_398_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_398.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_398.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_398.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_398_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_398_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_398.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_398 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_398 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_398
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_398.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_401.constraints_2.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_401.constraints_2 backend hashModel tape path i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_401.constraints_2 backend hashModel i_0 i_1 w_0_0 w_50_0 w_77_0 w_46_0 w_47_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_60_0 w_61_0 w_65_0 w_66_0 w_67_0 w_68_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_78_0 w_84_0 w_85_0 w_86_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, (MxxExponentLut.gaussianSampleAt_sample (by decide) h31), h32⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_401.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_401 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_401 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_401
  exact (by obtain ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, h3, h4⟩ := h; exact ⟨w_2_0, w_4_0, w_5_0, w_7_0, w_9_0, w_10_0, w_12_0, w_13_0, w_16_0, w_17_0, w_19_0, w_21_0, w_22_0, w_24_0, w_25_0, w_27_0, w_28_0, w_31_0, w_32_0, w_33_0, w_34_0, w_35_0, w_36_0, w_39_0, w_40_0, w_43_0, w_44_0, w_47_0, w_48_0, w_49_0, w_51_0, w_52_0, w_53_0, w_61_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_78_0, w_84_0, w_86_0, w_87_0, w_90_0, w_91_0, w_92_0, w_93_0, w_94_0, w_95_0, h1, h2, (Stage_preprocess.parallel_parallel_generatedRoot_211_401.constraints_2.bridge h3), h4⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_402_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_402_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_402_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_402_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_402_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_402_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_402_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_402_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_402.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_402.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_402.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_402_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_402_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_402.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_402 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_402 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_402
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_402.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_403_13.bridge {backend x1 tape path params params' x5 x6 i_2 x8 outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_403_13 backend x1 tape path params x5 x6 i_2 x8 outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_403_13 backend x1 params' x5 x6 i_2 x8 outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_403_13
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_403_14.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_403_14 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_403_14 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_403_14
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_403.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_3_0 w_27_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_19_0 w_22_0 w_23_0 w_28_0 w_32_0 w_33_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_403.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_3_0 w_27_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_19_0 w_22_0 w_23_0 w_28_0 w_32_0 w_33_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_403.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_3_0 w_27_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_19_0 w_22_0 w_23_0 w_28_0 w_32_0 w_33_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_403_13.bridge (h14 i))), h15, (Stage_preprocess.iterRuns_mono h16 (fun _ _ _ hs23 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_403_14.bridge hs23))), h17, h18, (MxxExponentLut.gaussianSampleAt_sample (by decide) h19), h20, (MxxExponentLut.gaussianSampleAt_sample (by decide) h21), h22⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_403.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_403 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_403 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_403
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_5_0, wit1.w_13_0, wit1.w_14_0, wit1.w_18_0, wit1.w_19_0, wit1.w_22_0, wit1.w_28_0, wit1.w_32_0⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_403.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_404_10.bridge {backend hashModel tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_404_10 backend hashModel tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_404_10 backend hashModel params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_404_10
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_404.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_404.constraints_0 backend hashModel tape path params i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_404.constraints_0 backend hashModel params' i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_12_0 w_13_0 w_14_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h; exact ⟨h1, (Stage_preprocess.iterRuns_mono h2 (fun _ _ _ hs8 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_404_10.bridge hs8))), h3, h4, h5, h6, h7⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_404.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_404 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_404 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_404
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3, wit1.w_11_0, wit1.w_12_0, wit1.w_13_0, wit1.w_14_0⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_404.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_405_11.bridge {backend x1 tape path params params' x5 x6 i_2 x8 outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_405_11 backend x1 tape path params x5 x6 i_2 x8 outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_405_11 backend x1 params' x5 x6 i_2 x8 outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_405_11
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_405_14.bridge {backend x1 tape path params params' i_0 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_405_14 backend x1 tape path params i_0 i_1 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_405_14 backend x1 params' i_0 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_405_14
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_405.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_3_0 w_26_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_21_0 w_22_0 w_27_0 w_31_0 w_32_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_405.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_3_0 w_26_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_21_0 w_22_0 w_27_0 w_31_0 w_32_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_405.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_3_0 w_26_0 w_1_0 w_2_0 w_4_0 w_5_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_18_0 w_21_0 w_22_0 w_27_0 w_31_0 w_32_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_405_11.bridge (h14 i))), h15, (Stage_preprocess.iterRuns_mono h16 (fun _ _ _ hs22 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_405_14.bridge hs22))), h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_405.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_405 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_405 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_405
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_4_0, wit1.w_5_0, wit1.w_11_0, wit1.w_14_0, wit1.w_18_0, wit1.w_21_0, wit1.w_27_0, wit1.w_31_0⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_405.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_406_10.bridge {backend hashModel tape path params params' i_0 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_406_10 backend hashModel tape path params i_0 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_406_10 backend hashModel params' i_0 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_406_10
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_406.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_406.constraints_0 backend hashModel tape path params i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_406.constraints_0 backend hashModel params' i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (Stage_preprocess.iterRuns_mono h2 (fun _ _ _ hs4 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_406_10.bridge hs4))), h3⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_406.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_406 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_406 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_406
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_406.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_407.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_407 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_407 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_407
  exact h


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_420.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_420 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_420 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_420
  exact h


theorem Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_428_20.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_428_20 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_428_20 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_428_20
  exact h


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_428.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_428.constraints_0 backend hashModel tape path params i_0 i_1 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_428.constraints_0 backend hashModel params' i_0 i_1 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_428_20.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_428.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_428 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_428 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_428
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_6_0, wit1.w_9_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_17_0, wit1.w_18_0, wit1.w_20_0, wit1.w_20_1, wit1.w_20_2, wit1.w_20_3, wit1.w_21_0, wit1.w_25_0, wit1.w_29_0, wit1.w_33_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preprocess.sequential_parallel_generatedRoot_211_428.constraints_0.bridge h3), h4⟩)⟩)


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_436.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_436 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_436 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_436
  exact h


theorem Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_439_4.bridge {backend x1 tape path params params' x5 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_439_4 backend x1 tape path params x5 i_1 i_2 inputs outputs) : W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_439_4 backend x1 params' x5 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_439_4
  exact h


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_439_17.bridge {backend x1 tape path params params' x5 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_439_17 backend x1 tape path params x5 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_439_17 backend x1 params' x5 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_439_17
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_439.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_439.constraints_0 backend hashModel tape path params i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_439.constraints_0 backend hashModel params' i_0 i_1 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_4_1 w_4_2 w_4_3 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_17_1 w_17_2 w_17_3 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_parallel_generatedRoot_211_439_4.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, (Stage_preprocess.iterRuns_mono h12 (fun _ _ _ hs14 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_439_17.bridge hs14))), h13⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_439.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_439 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_439 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_439
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_4_1, wit1.w_4_2, wit1.w_4_3, wit1.w_5_0, wit1.w_6_0, wit1.w_8_0, wit1.w_9_0, wit1.w_11_0, wit1.w_12_0, wit1.w_14_0, wit1.w_15_0, wit1.w_17_0, wit1.w_17_1, wit1.w_17_2, wit1.w_17_3⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_439.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_440_10.bridge {backend hashModel tape path params params' i_0 x6 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_440_10 backend hashModel tape path params i_0 x6 i_2 inputs outputs) : W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_440_10 backend hashModel params' i_0 x6 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_440_10
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_440.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_14_0 w_15_0 w_16_0 w_19_0 w_20_0 w_22_0 w_23_0 w_25_0 w_26_0 w_28_0 outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_440.constraints_0 backend hashModel tape path params i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_14_0 w_15_0 w_16_0 w_19_0 w_20_0 w_22_0 w_23_0 w_25_0 w_26_0 w_28_0 outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_440.constraints_0 backend hashModel params' i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_0_0 w_10_0 w_10_1 w_10_2 w_10_3 w_11_0 w_14_0 w_15_0 w_16_0 w_19_0 w_20_0 w_22_0 w_23_0 w_25_0 w_26_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h; exact ⟨h1, (Stage_preprocess.iterRuns_mono h2 (fun _ _ _ hs17 ↦ (Stage_preprocess.sequential_parallel_parallel_generatedRoot_211_440_10.bridge hs17))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩)


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_440.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_440 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_440 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_440
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_10_0, wit1.w_10_1, wit1.w_10_2, wit1.w_10_3, wit1.w_11_0, wit1.w_15_0, wit1.w_16_0, wit1.w_20_0, wit1.w_23_0, wit1.w_26_0⟩, (Stage_preprocess.parallel_parallel_generatedRoot_211_440.constraints_0.bridge h2)⟩)


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_453.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_453 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_453 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_453
  exact h


theorem Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_461_20.bridge {backend x1 tape path params params' i_0 i_1 i_2 inputs outputs}
    (h : Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_461_20 backend x1 tape path params i_0 i_1 i_2 inputs outputs) : W_Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_461_20 backend x1 params' i_0 i_1 i_2 inputs outputs := by
  delta W_Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_461_20
  exact h


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_461.constraints_0.bridge {backend hashModel tape path params params' i_0 i_1 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_461.constraints_0 backend hashModel tape path params i_0 i_1 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_461.constraints_0 backend hashModel params' i_0 i_1 w_2_0 w_8_0 w_12_0 w_16_0 w_1_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_20_1 w_20_2 w_20_3 w_21_0 w_24_0 w_25_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, (Stage_preprocess.iterRuns_mono h28 (fun _ _ _ hs33 ↦ (Stage_preprocess.sequential_sequential_parallel_generatedRoot_211_461_20.bridge hs33))), h29, h30, h31, h32⟩)


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_461.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_461 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_461 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_461
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_4_0, wit1.w_6_0, wit1.w_9_0, wit1.w_10_0, wit1.w_13_0, wit1.w_14_0, wit1.w_17_0, wit1.w_18_0, wit1.w_20_0, wit1.w_20_1, wit1.w_20_2, wit1.w_20_3, wit1.w_21_0, wit1.w_25_0, wit1.w_29_0, wit1.w_33_0⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_preprocess.sequential_parallel_generatedRoot_211_461.constraints_0.bridge h3), h4⟩)⟩)


theorem Stage_preprocess.sequential_parallel_generatedRoot_211_477.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_preprocess.sequential_parallel_generatedRoot_211_477 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_preprocess.sequential_parallel_generatedRoot_211_477 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_preprocess.sequential_parallel_generatedRoot_211_477
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_492.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_492 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_492 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_492
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_493.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_493 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_493 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_493
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_494.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_494 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_494 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_494
  exact h


theorem Stage_preprocess.parallel_parallel_generatedRoot_211_495.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_preprocess.parallel_parallel_generatedRoot_211_495 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_preprocess.parallel_parallel_generatedRoot_211_495 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_preprocess.parallel_parallel_generatedRoot_211_495
  exact h


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_14.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_47_0 w_47_1 w_47_2 w_47_3 w_54_6 w_54_7 w_58_6 w_58_7 w_62_6 w_62_7 w_66_6 w_66_7 w_70_6 w_70_7 w_74_6 w_74_7 w_78_6 w_78_7 w_82_6 w_82_7 w_86_6 w_86_7 w_90_6 w_90_7 w_94_6 w_94_7 w_98_6 w_98_7 w_102_6 w_102_7 w_106_6 w_106_7 w_110_6 w_110_7 w_114_6 w_114_7 w_118_6 w_118_7 w_122_6 w_122_7 w_126_6 w_126_7 w_130_6 w_130_7 w_134_6 w_134_7 w_138_6 w_138_7 w_142_6 w_142_7 w_146_6 w_146_7 w_150_6 w_150_7 w_154_6 w_154_7 w_158_6 w_158_7 w_162_6 w_162_7 w_166_6 w_166_7 w_170_6 w_170_7 w_174_6 w_174_7 w_178_6 w_178_7 w_182_6 w_182_7 w_186_6 w_186_7 w_187_0 w_204_0 w_204_1 w_204_2 w_204_3 w_207_0 w_207_1 w_207_2 w_207_3 w_209_0 w_269_6 w_269_7 w_273_6 w_273_7 w_277_6 w_277_7 w_281_6 w_281_7 w_285_6 w_285_7 w_289_6 w_289_7 w_293_6 w_293_7 w_297_6 w_297_7 w_301_6 w_301_7 w_305_6 w_305_7 w_309_6 w_309_7 w_313_6 w_313_7 w_317_6 w_317_7 w_321_6 w_321_7 w_325_6 w_325_7 w_329_6 w_329_7 w_333_6 w_333_7 w_337_6 w_337_7 w_341_6 w_341_7 w_345_6 w_345_7 w_349_6 w_349_7 w_353_6 w_353_7 w_357_6 w_357_7 w_361_6 w_361_7 w_365_6 w_365_7 w_369_6 w_369_7 w_373_6 w_373_7 w_377_6 w_377_7 w_381_6 w_381_7 w_385_6 w_385_7 w_389_6 w_389_7 w_393_6 w_393_7 w_397_6 w_397_7 w_401_6 w_401_7 w_403_0 w_403_1 w_403_2 w_403_3 w_405_0 w_405_1 w_405_2 w_405_3 w_469_0 w_481_0 w_482_0 w_483_0 w_484_0 w_485_0 w_486_0 w_487_0 w_488_0 w_489_0 w_490_0 w_491_0 w_492_0 w_493_0 w_494_0 w_495_0 outputs}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_14 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_47_0 w_47_1 w_47_2 w_47_3 w_54_6 w_54_7 w_58_6 w_58_7 w_62_6 w_62_7 w_66_6 w_66_7 w_70_6 w_70_7 w_74_6 w_74_7 w_78_6 w_78_7 w_82_6 w_82_7 w_86_6 w_86_7 w_90_6 w_90_7 w_94_6 w_94_7 w_98_6 w_98_7 w_102_6 w_102_7 w_106_6 w_106_7 w_110_6 w_110_7 w_114_6 w_114_7 w_118_6 w_118_7 w_122_6 w_122_7 w_126_6 w_126_7 w_130_6 w_130_7 w_134_6 w_134_7 w_138_6 w_138_7 w_142_6 w_142_7 w_146_6 w_146_7 w_150_6 w_150_7 w_154_6 w_154_7 w_158_6 w_158_7 w_162_6 w_162_7 w_166_6 w_166_7 w_170_6 w_170_7 w_174_6 w_174_7 w_178_6 w_178_7 w_182_6 w_182_7 w_186_6 w_186_7 w_187_0 w_204_0 w_204_1 w_204_2 w_204_3 w_207_0 w_207_1 w_207_2 w_207_3 w_209_0 w_269_6 w_269_7 w_273_6 w_273_7 w_277_6 w_277_7 w_281_6 w_281_7 w_285_6 w_285_7 w_289_6 w_289_7 w_293_6 w_293_7 w_297_6 w_297_7 w_301_6 w_301_7 w_305_6 w_305_7 w_309_6 w_309_7 w_313_6 w_313_7 w_317_6 w_317_7 w_321_6 w_321_7 w_325_6 w_325_7 w_329_6 w_329_7 w_333_6 w_333_7 w_337_6 w_337_7 w_341_6 w_341_7 w_345_6 w_345_7 w_349_6 w_349_7 w_353_6 w_353_7 w_357_6 w_357_7 w_361_6 w_361_7 w_365_6 w_365_7 w_369_6 w_369_7 w_373_6 w_373_7 w_377_6 w_377_7 w_381_6 w_381_7 w_385_6 w_385_7 w_389_6 w_389_7 w_393_6 w_393_7 w_397_6 w_397_7 w_401_6 w_401_7 w_403_0 w_403_1 w_403_2 w_403_3 w_405_0 w_405_1 w_405_2 w_405_3 w_469_0 w_481_0 w_482_0 w_483_0 w_484_0 w_485_0 w_486_0 w_487_0 w_488_0 w_489_0 w_490_0 w_491_0 w_492_0 w_493_0 w_494_0 w_495_0 outputs) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_14 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_47_0 w_47_1 w_47_2 w_47_3 w_54_6 w_54_7 w_58_6 w_58_7 w_62_6 w_62_7 w_66_6 w_66_7 w_70_6 w_70_7 w_74_6 w_74_7 w_78_6 w_78_7 w_82_6 w_82_7 w_86_6 w_86_7 w_90_6 w_90_7 w_94_6 w_94_7 w_98_6 w_98_7 w_102_6 w_102_7 w_106_6 w_106_7 w_110_6 w_110_7 w_114_6 w_114_7 w_118_6 w_118_7 w_122_6 w_122_7 w_126_6 w_126_7 w_130_6 w_130_7 w_134_6 w_134_7 w_138_6 w_138_7 w_142_6 w_142_7 w_146_6 w_146_7 w_150_6 w_150_7 w_154_6 w_154_7 w_158_6 w_158_7 w_162_6 w_162_7 w_166_6 w_166_7 w_170_6 w_170_7 w_174_6 w_174_7 w_178_6 w_178_7 w_182_6 w_182_7 w_186_6 w_186_7 w_187_0 w_204_0 w_204_1 w_204_2 w_204_3 w_207_0 w_207_1 w_207_2 w_207_3 w_209_0 w_269_6 w_269_7 w_273_6 w_273_7 w_277_6 w_277_7 w_281_6 w_281_7 w_285_6 w_285_7 w_289_6 w_289_7 w_293_6 w_293_7 w_297_6 w_297_7 w_301_6 w_301_7 w_305_6 w_305_7 w_309_6 w_309_7 w_313_6 w_313_7 w_317_6 w_317_7 w_321_6 w_321_7 w_325_6 w_325_7 w_329_6 w_329_7 w_333_6 w_333_7 w_337_6 w_337_7 w_341_6 w_341_7 w_345_6 w_345_7 w_349_6 w_349_7 w_353_6 w_353_7 w_357_6 w_357_7 w_361_6 w_361_7 w_365_6 w_365_7 w_369_6 w_369_7 w_373_6 w_373_7 w_377_6 w_377_7 w_381_6 w_381_7 w_385_6 w_385_7 w_389_6 w_389_7 w_393_6 w_393_7 w_397_6 w_397_7 w_401_6 w_401_7 w_403_0 w_403_1 w_403_2 w_403_3 w_405_0 w_405_1 w_405_2 w_405_3 w_469_0 w_481_0 w_482_0 w_483_0 w_484_0 w_485_0 w_486_0 w_487_0 w_488_0 w_489_0 w_490_0 w_491_0 w_492_0 w_493_0 w_494_0 w_495_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_492.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_493.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_494.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_495.bridge (h20 i))), h21⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_13.bridge {backend hashModel tape path params params' i_0 w_189_0 w_463_0 w_187_1 w_187_0 w_207_2 w_207_3 w_428_3 w_433_0 w_440_0 w_440_1 w_440_2 w_440_3 w_453_3 w_454_0 w_456_0 w_458_0 w_459_0 w_460_0 w_461_0 w_461_1 w_461_2 w_461_3 w_462_0 w_464_0 w_465_0 w_467_0 w_468_0 w_469_0 w_472_0 w_473_0 w_474_0 w_475_0 w_476_0 w_477_0 w_479_0 w_480_0 w_481_0 w_482_0}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_13 backend hashModel tape path params i_0 w_189_0 w_463_0 w_187_1 w_187_0 w_207_2 w_207_3 w_428_3 w_433_0 w_440_0 w_440_1 w_440_2 w_440_3 w_453_3 w_454_0 w_456_0 w_458_0 w_459_0 w_460_0 w_461_0 w_461_1 w_461_2 w_461_3 w_462_0 w_464_0 w_465_0 w_467_0 w_468_0 w_469_0 w_472_0 w_473_0 w_474_0 w_475_0 w_476_0 w_477_0 w_479_0 w_480_0 w_481_0 w_482_0) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_13 backend hashModel params' i_0 w_189_0 w_463_0 w_187_1 w_187_0 w_207_2 w_207_3 w_428_3 w_433_0 w_440_0 w_440_1 w_440_2 w_440_3 w_453_3 w_454_0 w_456_0 w_458_0 w_459_0 w_460_0 w_461_0 w_461_1 w_461_2 w_461_3 w_462_0 w_464_0 w_465_0 w_467_0 w_468_0 w_469_0 w_472_0 w_473_0 w_474_0 w_475_0 w_476_0 w_477_0 w_479_0 w_480_0 w_481_0 w_482_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, (Stage_preprocess.iterRuns_mono h5 (fun _ _ _ hs33 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_461.bridge hs33))), h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, (Stage_preprocess.iterRuns_mono h27 (fun _ _ _ hs34 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_477.bridge hs34))), h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_12.bridge {backend hashModel tape path params params' i_0 w_34_0 w_47_0 w_47_1 w_47_2 w_47_3 w_186_0 w_186_1 w_186_2 w_186_3 w_224_0 w_429_0 w_430_0 w_431_0 w_432_0 w_433_0 w_434_0 w_435_0 w_436_0 w_438_0 w_439_0 w_439_1 w_439_2 w_439_3 w_440_0 w_440_1 w_440_2 w_440_3 w_441_0 w_442_0 w_443_0 w_444_0 w_445_0 w_446_0 w_447_0 w_448_0 w_449_0 w_450_0 w_451_0 w_452_0 w_453_0 w_453_1 w_453_2 w_453_3 w_454_0 w_455_0 w_456_0 w_457_0 w_458_0}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_12 backend hashModel tape path params i_0 w_34_0 w_47_0 w_47_1 w_47_2 w_47_3 w_186_0 w_186_1 w_186_2 w_186_3 w_224_0 w_429_0 w_430_0 w_431_0 w_432_0 w_433_0 w_434_0 w_435_0 w_436_0 w_438_0 w_439_0 w_439_1 w_439_2 w_439_3 w_440_0 w_440_1 w_440_2 w_440_3 w_441_0 w_442_0 w_443_0 w_444_0 w_445_0 w_446_0 w_447_0 w_448_0 w_449_0 w_450_0 w_451_0 w_452_0 w_453_0 w_453_1 w_453_2 w_453_3 w_454_0 w_455_0 w_456_0 w_457_0 w_458_0) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_12 backend hashModel params' i_0 w_34_0 w_47_0 w_47_1 w_47_2 w_47_3 w_186_0 w_186_1 w_186_2 w_186_3 w_224_0 w_429_0 w_430_0 w_431_0 w_432_0 w_433_0 w_434_0 w_435_0 w_436_0 w_438_0 w_439_0 w_439_1 w_439_2 w_439_3 w_440_0 w_440_1 w_440_2 w_440_3 w_441_0 w_442_0 w_443_0 w_444_0 w_445_0 w_446_0 w_447_0 w_448_0 w_449_0 w_450_0 w_451_0 w_452_0 w_453_0 w_453_1 w_453_2 w_453_3 w_454_0 w_455_0 w_456_0 w_457_0 w_458_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, (Stage_preprocess.iterRuns_mono h9 (fun _ _ _ hs33 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_436.bridge hs33))), h10, h11, h12, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_439.bridge (h13 i))), h14, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_440.bridge (h15 i))), h16, h17, h18, h19, h20, h21, h22, h23, h24, (Stage_preprocess.iterRuns_mono h25 (fun _ _ _ hs34 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_453.bridge hs34))), h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_11.bridge {backend hashModel tape path params params' i_0 w_226_0 w_225_0 w_404_0 w_404_1 w_404_2 w_404_3 w_405_0 w_405_1 w_405_2 w_405_3 w_406_0 w_406_1 w_406_2 w_406_3 w_407_0 w_407_1 w_407_2 w_407_3 w_408_0 w_409_0 w_410_0 w_411_0 w_412_0 w_413_0 w_414_0 w_415_0 w_416_0 w_417_0 w_418_0 w_419_0 w_420_0 w_420_1 w_420_2 w_420_3 w_421_0 w_422_0 w_423_0 w_424_0 w_425_0 w_426_0 w_427_0 w_428_0 w_428_1 w_428_2 w_428_3 w_429_0}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_11 backend hashModel tape path params i_0 w_226_0 w_225_0 w_404_0 w_404_1 w_404_2 w_404_3 w_405_0 w_405_1 w_405_2 w_405_3 w_406_0 w_406_1 w_406_2 w_406_3 w_407_0 w_407_1 w_407_2 w_407_3 w_408_0 w_409_0 w_410_0 w_411_0 w_412_0 w_413_0 w_414_0 w_415_0 w_416_0 w_417_0 w_418_0 w_419_0 w_420_0 w_420_1 w_420_2 w_420_3 w_421_0 w_422_0 w_423_0 w_424_0 w_425_0 w_426_0 w_427_0 w_428_0 w_428_1 w_428_2 w_428_3 w_429_0) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_11 backend hashModel params' i_0 w_226_0 w_225_0 w_404_0 w_404_1 w_404_2 w_404_3 w_405_0 w_405_1 w_405_2 w_405_3 w_406_0 w_406_1 w_406_2 w_406_3 w_407_0 w_407_1 w_407_2 w_407_3 w_408_0 w_409_0 w_410_0 w_411_0 w_412_0 w_413_0 w_414_0 w_415_0 w_416_0 w_417_0 w_418_0 w_419_0 w_420_0 w_420_1 w_420_2 w_420_3 w_421_0 w_422_0 w_423_0 w_424_0 w_425_0 w_426_0 w_427_0 w_428_0 w_428_1 w_428_2 w_428_3 w_429_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_406.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_407.bridge (h4 i))), h5, h6, h7, h8, h9, h10, h11, h12, h13, (Stage_preprocess.iterRuns_mono h14 (fun _ _ _ hs33 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_420.bridge hs33))), h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, (Stage_preprocess.iterRuns_mono h26 (fun _ _ _ hs34 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_428.bridge hs34))), h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_10.bridge {backend hashModel tape path params params' i_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_379_0 w_380_0 w_383_0 w_384_0 w_387_0 w_388_0 w_391_0 w_392_0 w_395_0 w_396_0 w_399_0 w_400_0 w_265_0 w_268_0 w_377_0 w_377_1 w_377_2 w_377_3 w_378_0 w_378_1 w_378_2 w_378_3 w_381_0 w_381_1 w_381_2 w_381_3 w_381_4 w_381_5 w_381_6 w_381_7 w_382_0 w_382_1 w_382_2 w_382_3 w_385_0 w_385_1 w_385_2 w_385_3 w_385_4 w_385_5 w_385_6 w_385_7 w_386_0 w_386_1 w_386_2 w_386_3 w_389_0 w_389_1 w_389_2 w_389_3 w_389_4 w_389_5 w_389_6 w_389_7 w_390_0 w_390_1 w_390_2 w_390_3 w_393_0 w_393_1 w_393_2 w_393_3 w_393_4 w_393_5 w_393_6 w_393_7 w_394_0 w_394_1 w_394_2 w_394_3 w_397_0 w_397_1 w_397_2 w_397_3 w_397_4 w_397_5 w_397_6 w_397_7 w_398_0 w_398_1 w_398_2 w_398_3 w_401_0 w_401_1 w_401_2 w_401_3 w_401_4 w_401_5 w_401_6 w_401_7 w_402_0 w_402_1 w_402_2 w_402_3 w_403_0 w_403_1 w_403_2 w_403_3 w_404_0 w_404_1 w_404_2 w_404_3 w_405_0 w_405_1 w_405_2 w_405_3}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_10 backend hashModel tape path params i_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_379_0 w_380_0 w_383_0 w_384_0 w_387_0 w_388_0 w_391_0 w_392_0 w_395_0 w_396_0 w_399_0 w_400_0 w_265_0 w_268_0 w_377_0 w_377_1 w_377_2 w_377_3 w_378_0 w_378_1 w_378_2 w_378_3 w_381_0 w_381_1 w_381_2 w_381_3 w_381_4 w_381_5 w_381_6 w_381_7 w_382_0 w_382_1 w_382_2 w_382_3 w_385_0 w_385_1 w_385_2 w_385_3 w_385_4 w_385_5 w_385_6 w_385_7 w_386_0 w_386_1 w_386_2 w_386_3 w_389_0 w_389_1 w_389_2 w_389_3 w_389_4 w_389_5 w_389_6 w_389_7 w_390_0 w_390_1 w_390_2 w_390_3 w_393_0 w_393_1 w_393_2 w_393_3 w_393_4 w_393_5 w_393_6 w_393_7 w_394_0 w_394_1 w_394_2 w_394_3 w_397_0 w_397_1 w_397_2 w_397_3 w_397_4 w_397_5 w_397_6 w_397_7 w_398_0 w_398_1 w_398_2 w_398_3 w_401_0 w_401_1 w_401_2 w_401_3 w_401_4 w_401_5 w_401_6 w_401_7 w_402_0 w_402_1 w_402_2 w_402_3 w_403_0 w_403_1 w_403_2 w_403_3 w_404_0 w_404_1 w_404_2 w_404_3 w_405_0 w_405_1 w_405_2 w_405_3) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_10 backend hashModel params' i_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_379_0 w_380_0 w_383_0 w_384_0 w_387_0 w_388_0 w_391_0 w_392_0 w_395_0 w_396_0 w_399_0 w_400_0 w_265_0 w_268_0 w_377_0 w_377_1 w_377_2 w_377_3 w_378_0 w_378_1 w_378_2 w_378_3 w_381_0 w_381_1 w_381_2 w_381_3 w_381_4 w_381_5 w_381_6 w_381_7 w_382_0 w_382_1 w_382_2 w_382_3 w_385_0 w_385_1 w_385_2 w_385_3 w_385_4 w_385_5 w_385_6 w_385_7 w_386_0 w_386_1 w_386_2 w_386_3 w_389_0 w_389_1 w_389_2 w_389_3 w_389_4 w_389_5 w_389_6 w_389_7 w_390_0 w_390_1 w_390_2 w_390_3 w_393_0 w_393_1 w_393_2 w_393_3 w_393_4 w_393_5 w_393_6 w_393_7 w_394_0 w_394_1 w_394_2 w_394_3 w_397_0 w_397_1 w_397_2 w_397_3 w_397_4 w_397_5 w_397_6 w_397_7 w_398_0 w_398_1 w_398_2 w_398_3 w_401_0 w_401_1 w_401_2 w_401_3 w_401_4 w_401_5 w_401_6 w_401_7 w_402_0 w_402_1 w_402_2 w_402_3 w_403_0 w_403_1 w_403_2 w_403_3 w_404_0 w_404_1 w_404_2 w_404_3 w_405_0 w_405_1 w_405_2 w_405_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_378.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_381.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_382.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_385.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_386.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_389.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_390.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_393.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_394.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_397.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_398.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_401.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_402.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_403.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_404.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_405.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_9.bridge {backend hashModel tape path params params' i_0 w_226_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_347_0 w_348_0 w_351_0 w_352_0 w_355_0 w_356_0 w_359_0 w_360_0 w_363_0 w_364_0 w_367_0 w_368_0 w_371_0 w_372_0 w_375_0 w_376_0 w_265_0 w_268_0 w_345_0 w_345_1 w_345_2 w_345_3 w_346_0 w_346_1 w_346_2 w_346_3 w_349_0 w_349_1 w_349_2 w_349_3 w_349_4 w_349_5 w_349_6 w_349_7 w_350_0 w_350_1 w_350_2 w_350_3 w_353_0 w_353_1 w_353_2 w_353_3 w_353_4 w_353_5 w_353_6 w_353_7 w_354_0 w_354_1 w_354_2 w_354_3 w_357_0 w_357_1 w_357_2 w_357_3 w_357_4 w_357_5 w_357_6 w_357_7 w_358_0 w_358_1 w_358_2 w_358_3 w_361_0 w_361_1 w_361_2 w_361_3 w_361_4 w_361_5 w_361_6 w_361_7 w_362_0 w_362_1 w_362_2 w_362_3 w_365_0 w_365_1 w_365_2 w_365_3 w_365_4 w_365_5 w_365_6 w_365_7 w_366_0 w_366_1 w_366_2 w_366_3 w_369_0 w_369_1 w_369_2 w_369_3 w_369_4 w_369_5 w_369_6 w_369_7 w_370_0 w_370_1 w_370_2 w_370_3 w_373_0 w_373_1 w_373_2 w_373_3 w_373_4 w_373_5 w_373_6 w_373_7 w_374_0 w_374_1 w_374_2 w_374_3 w_377_0 w_377_1 w_377_2 w_377_3 w_377_4 w_377_5 w_377_6 w_377_7}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_9 backend hashModel tape path params i_0 w_226_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_347_0 w_348_0 w_351_0 w_352_0 w_355_0 w_356_0 w_359_0 w_360_0 w_363_0 w_364_0 w_367_0 w_368_0 w_371_0 w_372_0 w_375_0 w_376_0 w_265_0 w_268_0 w_345_0 w_345_1 w_345_2 w_345_3 w_346_0 w_346_1 w_346_2 w_346_3 w_349_0 w_349_1 w_349_2 w_349_3 w_349_4 w_349_5 w_349_6 w_349_7 w_350_0 w_350_1 w_350_2 w_350_3 w_353_0 w_353_1 w_353_2 w_353_3 w_353_4 w_353_5 w_353_6 w_353_7 w_354_0 w_354_1 w_354_2 w_354_3 w_357_0 w_357_1 w_357_2 w_357_3 w_357_4 w_357_5 w_357_6 w_357_7 w_358_0 w_358_1 w_358_2 w_358_3 w_361_0 w_361_1 w_361_2 w_361_3 w_361_4 w_361_5 w_361_6 w_361_7 w_362_0 w_362_1 w_362_2 w_362_3 w_365_0 w_365_1 w_365_2 w_365_3 w_365_4 w_365_5 w_365_6 w_365_7 w_366_0 w_366_1 w_366_2 w_366_3 w_369_0 w_369_1 w_369_2 w_369_3 w_369_4 w_369_5 w_369_6 w_369_7 w_370_0 w_370_1 w_370_2 w_370_3 w_373_0 w_373_1 w_373_2 w_373_3 w_373_4 w_373_5 w_373_6 w_373_7 w_374_0 w_374_1 w_374_2 w_374_3 w_377_0 w_377_1 w_377_2 w_377_3 w_377_4 w_377_5 w_377_6 w_377_7) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_9 backend hashModel params' i_0 w_226_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_347_0 w_348_0 w_351_0 w_352_0 w_355_0 w_356_0 w_359_0 w_360_0 w_363_0 w_364_0 w_367_0 w_368_0 w_371_0 w_372_0 w_375_0 w_376_0 w_265_0 w_268_0 w_345_0 w_345_1 w_345_2 w_345_3 w_346_0 w_346_1 w_346_2 w_346_3 w_349_0 w_349_1 w_349_2 w_349_3 w_349_4 w_349_5 w_349_6 w_349_7 w_350_0 w_350_1 w_350_2 w_350_3 w_353_0 w_353_1 w_353_2 w_353_3 w_353_4 w_353_5 w_353_6 w_353_7 w_354_0 w_354_1 w_354_2 w_354_3 w_357_0 w_357_1 w_357_2 w_357_3 w_357_4 w_357_5 w_357_6 w_357_7 w_358_0 w_358_1 w_358_2 w_358_3 w_361_0 w_361_1 w_361_2 w_361_3 w_361_4 w_361_5 w_361_6 w_361_7 w_362_0 w_362_1 w_362_2 w_362_3 w_365_0 w_365_1 w_365_2 w_365_3 w_365_4 w_365_5 w_365_6 w_365_7 w_366_0 w_366_1 w_366_2 w_366_3 w_369_0 w_369_1 w_369_2 w_369_3 w_369_4 w_369_5 w_369_6 w_369_7 w_370_0 w_370_1 w_370_2 w_370_3 w_373_0 w_373_1 w_373_2 w_373_3 w_373_4 w_373_5 w_373_6 w_373_7 w_374_0 w_374_1 w_374_2 w_374_3 w_377_0 w_377_1 w_377_2 w_377_3 w_377_4 w_377_5 w_377_6 w_377_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_346.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_349.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_350.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_353.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_354.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_357.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_358.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_361.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_362.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_365.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_366.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_369.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_370.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_373.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_374.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_377.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_8.bridge {backend hashModel tape path params params' i_0 w_226_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_315_0 w_316_0 w_319_0 w_320_0 w_323_0 w_324_0 w_327_0 w_328_0 w_331_0 w_332_0 w_335_0 w_336_0 w_339_0 w_340_0 w_343_0 w_344_0 w_265_0 w_268_0 w_313_0 w_313_1 w_313_2 w_313_3 w_314_0 w_314_1 w_314_2 w_314_3 w_317_0 w_317_1 w_317_2 w_317_3 w_317_4 w_317_5 w_317_6 w_317_7 w_318_0 w_318_1 w_318_2 w_318_3 w_321_0 w_321_1 w_321_2 w_321_3 w_321_4 w_321_5 w_321_6 w_321_7 w_322_0 w_322_1 w_322_2 w_322_3 w_325_0 w_325_1 w_325_2 w_325_3 w_325_4 w_325_5 w_325_6 w_325_7 w_326_0 w_326_1 w_326_2 w_326_3 w_329_0 w_329_1 w_329_2 w_329_3 w_329_4 w_329_5 w_329_6 w_329_7 w_330_0 w_330_1 w_330_2 w_330_3 w_333_0 w_333_1 w_333_2 w_333_3 w_333_4 w_333_5 w_333_6 w_333_7 w_334_0 w_334_1 w_334_2 w_334_3 w_337_0 w_337_1 w_337_2 w_337_3 w_337_4 w_337_5 w_337_6 w_337_7 w_338_0 w_338_1 w_338_2 w_338_3 w_341_0 w_341_1 w_341_2 w_341_3 w_341_4 w_341_5 w_341_6 w_341_7 w_342_0 w_342_1 w_342_2 w_342_3 w_345_0 w_345_1 w_345_2 w_345_3 w_345_4 w_345_5 w_345_6 w_345_7}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_8 backend hashModel tape path params i_0 w_226_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_315_0 w_316_0 w_319_0 w_320_0 w_323_0 w_324_0 w_327_0 w_328_0 w_331_0 w_332_0 w_335_0 w_336_0 w_339_0 w_340_0 w_343_0 w_344_0 w_265_0 w_268_0 w_313_0 w_313_1 w_313_2 w_313_3 w_314_0 w_314_1 w_314_2 w_314_3 w_317_0 w_317_1 w_317_2 w_317_3 w_317_4 w_317_5 w_317_6 w_317_7 w_318_0 w_318_1 w_318_2 w_318_3 w_321_0 w_321_1 w_321_2 w_321_3 w_321_4 w_321_5 w_321_6 w_321_7 w_322_0 w_322_1 w_322_2 w_322_3 w_325_0 w_325_1 w_325_2 w_325_3 w_325_4 w_325_5 w_325_6 w_325_7 w_326_0 w_326_1 w_326_2 w_326_3 w_329_0 w_329_1 w_329_2 w_329_3 w_329_4 w_329_5 w_329_6 w_329_7 w_330_0 w_330_1 w_330_2 w_330_3 w_333_0 w_333_1 w_333_2 w_333_3 w_333_4 w_333_5 w_333_6 w_333_7 w_334_0 w_334_1 w_334_2 w_334_3 w_337_0 w_337_1 w_337_2 w_337_3 w_337_4 w_337_5 w_337_6 w_337_7 w_338_0 w_338_1 w_338_2 w_338_3 w_341_0 w_341_1 w_341_2 w_341_3 w_341_4 w_341_5 w_341_6 w_341_7 w_342_0 w_342_1 w_342_2 w_342_3 w_345_0 w_345_1 w_345_2 w_345_3 w_345_4 w_345_5 w_345_6 w_345_7) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_8 backend hashModel params' i_0 w_226_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_315_0 w_316_0 w_319_0 w_320_0 w_323_0 w_324_0 w_327_0 w_328_0 w_331_0 w_332_0 w_335_0 w_336_0 w_339_0 w_340_0 w_343_0 w_344_0 w_265_0 w_268_0 w_313_0 w_313_1 w_313_2 w_313_3 w_314_0 w_314_1 w_314_2 w_314_3 w_317_0 w_317_1 w_317_2 w_317_3 w_317_4 w_317_5 w_317_6 w_317_7 w_318_0 w_318_1 w_318_2 w_318_3 w_321_0 w_321_1 w_321_2 w_321_3 w_321_4 w_321_5 w_321_6 w_321_7 w_322_0 w_322_1 w_322_2 w_322_3 w_325_0 w_325_1 w_325_2 w_325_3 w_325_4 w_325_5 w_325_6 w_325_7 w_326_0 w_326_1 w_326_2 w_326_3 w_329_0 w_329_1 w_329_2 w_329_3 w_329_4 w_329_5 w_329_6 w_329_7 w_330_0 w_330_1 w_330_2 w_330_3 w_333_0 w_333_1 w_333_2 w_333_3 w_333_4 w_333_5 w_333_6 w_333_7 w_334_0 w_334_1 w_334_2 w_334_3 w_337_0 w_337_1 w_337_2 w_337_3 w_337_4 w_337_5 w_337_6 w_337_7 w_338_0 w_338_1 w_338_2 w_338_3 w_341_0 w_341_1 w_341_2 w_341_3 w_341_4 w_341_5 w_341_6 w_341_7 w_342_0 w_342_1 w_342_2 w_342_3 w_345_0 w_345_1 w_345_2 w_345_3 w_345_4 w_345_5 w_345_6 w_345_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_314.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_317.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_318.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_321.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_322.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_325.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_326.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_329.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_330.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_333.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_334.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_337.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_338.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_341.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_342.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_345.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_7.bridge {backend hashModel tape path params params' i_0 w_226_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_283_0 w_284_0 w_287_0 w_288_0 w_291_0 w_292_0 w_295_0 w_296_0 w_299_0 w_300_0 w_303_0 w_304_0 w_307_0 w_308_0 w_311_0 w_312_0 w_265_0 w_268_0 w_281_0 w_281_1 w_281_2 w_281_3 w_282_0 w_282_1 w_282_2 w_282_3 w_285_0 w_285_1 w_285_2 w_285_3 w_285_4 w_285_5 w_285_6 w_285_7 w_286_0 w_286_1 w_286_2 w_286_3 w_289_0 w_289_1 w_289_2 w_289_3 w_289_4 w_289_5 w_289_6 w_289_7 w_290_0 w_290_1 w_290_2 w_290_3 w_293_0 w_293_1 w_293_2 w_293_3 w_293_4 w_293_5 w_293_6 w_293_7 w_294_0 w_294_1 w_294_2 w_294_3 w_297_0 w_297_1 w_297_2 w_297_3 w_297_4 w_297_5 w_297_6 w_297_7 w_298_0 w_298_1 w_298_2 w_298_3 w_301_0 w_301_1 w_301_2 w_301_3 w_301_4 w_301_5 w_301_6 w_301_7 w_302_0 w_302_1 w_302_2 w_302_3 w_305_0 w_305_1 w_305_2 w_305_3 w_305_4 w_305_5 w_305_6 w_305_7 w_306_0 w_306_1 w_306_2 w_306_3 w_309_0 w_309_1 w_309_2 w_309_3 w_309_4 w_309_5 w_309_6 w_309_7 w_310_0 w_310_1 w_310_2 w_310_3 w_313_0 w_313_1 w_313_2 w_313_3 w_313_4 w_313_5 w_313_6 w_313_7}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_7 backend hashModel tape path params i_0 w_226_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_283_0 w_284_0 w_287_0 w_288_0 w_291_0 w_292_0 w_295_0 w_296_0 w_299_0 w_300_0 w_303_0 w_304_0 w_307_0 w_308_0 w_311_0 w_312_0 w_265_0 w_268_0 w_281_0 w_281_1 w_281_2 w_281_3 w_282_0 w_282_1 w_282_2 w_282_3 w_285_0 w_285_1 w_285_2 w_285_3 w_285_4 w_285_5 w_285_6 w_285_7 w_286_0 w_286_1 w_286_2 w_286_3 w_289_0 w_289_1 w_289_2 w_289_3 w_289_4 w_289_5 w_289_6 w_289_7 w_290_0 w_290_1 w_290_2 w_290_3 w_293_0 w_293_1 w_293_2 w_293_3 w_293_4 w_293_5 w_293_6 w_293_7 w_294_0 w_294_1 w_294_2 w_294_3 w_297_0 w_297_1 w_297_2 w_297_3 w_297_4 w_297_5 w_297_6 w_297_7 w_298_0 w_298_1 w_298_2 w_298_3 w_301_0 w_301_1 w_301_2 w_301_3 w_301_4 w_301_5 w_301_6 w_301_7 w_302_0 w_302_1 w_302_2 w_302_3 w_305_0 w_305_1 w_305_2 w_305_3 w_305_4 w_305_5 w_305_6 w_305_7 w_306_0 w_306_1 w_306_2 w_306_3 w_309_0 w_309_1 w_309_2 w_309_3 w_309_4 w_309_5 w_309_6 w_309_7 w_310_0 w_310_1 w_310_2 w_310_3 w_313_0 w_313_1 w_313_2 w_313_3 w_313_4 w_313_5 w_313_6 w_313_7) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_7 backend hashModel params' i_0 w_226_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_283_0 w_284_0 w_287_0 w_288_0 w_291_0 w_292_0 w_295_0 w_296_0 w_299_0 w_300_0 w_303_0 w_304_0 w_307_0 w_308_0 w_311_0 w_312_0 w_265_0 w_268_0 w_281_0 w_281_1 w_281_2 w_281_3 w_282_0 w_282_1 w_282_2 w_282_3 w_285_0 w_285_1 w_285_2 w_285_3 w_285_4 w_285_5 w_285_6 w_285_7 w_286_0 w_286_1 w_286_2 w_286_3 w_289_0 w_289_1 w_289_2 w_289_3 w_289_4 w_289_5 w_289_6 w_289_7 w_290_0 w_290_1 w_290_2 w_290_3 w_293_0 w_293_1 w_293_2 w_293_3 w_293_4 w_293_5 w_293_6 w_293_7 w_294_0 w_294_1 w_294_2 w_294_3 w_297_0 w_297_1 w_297_2 w_297_3 w_297_4 w_297_5 w_297_6 w_297_7 w_298_0 w_298_1 w_298_2 w_298_3 w_301_0 w_301_1 w_301_2 w_301_3 w_301_4 w_301_5 w_301_6 w_301_7 w_302_0 w_302_1 w_302_2 w_302_3 w_305_0 w_305_1 w_305_2 w_305_3 w_305_4 w_305_5 w_305_6 w_305_7 w_306_0 w_306_1 w_306_2 w_306_3 w_309_0 w_309_1 w_309_2 w_309_3 w_309_4 w_309_5 w_309_6 w_309_7 w_310_0 w_310_1 w_310_2 w_310_3 w_313_0 w_313_1 w_313_2 w_313_3 w_313_4 w_313_5 w_313_6 w_313_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_282.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_285.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_286.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_289.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_290.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_293.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_294.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_297.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_298.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_301.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_302.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_305.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_306.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_309.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_310.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_313.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_6.bridge {backend hashModel tape path params params' i_0 w_43_0 w_226_0 w_257_0 w_258_0 w_259_0 w_260_0 w_266_0 w_267_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_45_0 w_216_0 w_217_0 w_219_0 w_220_0 w_221_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_268_0 w_269_0 w_269_1 w_269_2 w_269_3 w_269_4 w_269_5 w_269_6 w_269_7 w_270_0 w_270_1 w_270_2 w_270_3 w_273_0 w_273_1 w_273_2 w_273_3 w_273_4 w_273_5 w_273_6 w_273_7 w_274_0 w_274_1 w_274_2 w_274_3 w_277_0 w_277_1 w_277_2 w_277_3 w_277_4 w_277_5 w_277_6 w_277_7 w_278_0 w_278_1 w_278_2 w_278_3 w_281_0 w_281_1 w_281_2 w_281_3 w_281_4 w_281_5 w_281_6 w_281_7}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_6 backend hashModel tape path params i_0 w_43_0 w_226_0 w_257_0 w_258_0 w_259_0 w_260_0 w_266_0 w_267_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_45_0 w_216_0 w_217_0 w_219_0 w_220_0 w_221_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_268_0 w_269_0 w_269_1 w_269_2 w_269_3 w_269_4 w_269_5 w_269_6 w_269_7 w_270_0 w_270_1 w_270_2 w_270_3 w_273_0 w_273_1 w_273_2 w_273_3 w_273_4 w_273_5 w_273_6 w_273_7 w_274_0 w_274_1 w_274_2 w_274_3 w_277_0 w_277_1 w_277_2 w_277_3 w_277_4 w_277_5 w_277_6 w_277_7 w_278_0 w_278_1 w_278_2 w_278_3 w_281_0 w_281_1 w_281_2 w_281_3 w_281_4 w_281_5 w_281_6 w_281_7) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_6 backend hashModel params' i_0 w_43_0 w_226_0 w_257_0 w_258_0 w_259_0 w_260_0 w_266_0 w_267_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_45_0 w_216_0 w_217_0 w_219_0 w_220_0 w_221_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_268_0 w_269_0 w_269_1 w_269_2 w_269_3 w_269_4 w_269_5 w_269_6 w_269_7 w_270_0 w_270_1 w_270_2 w_270_3 w_273_0 w_273_1 w_273_2 w_273_3 w_273_4 w_273_5 w_273_6 w_273_7 w_274_0 w_274_1 w_274_2 w_274_3 w_277_0 w_277_1 w_277_2 w_277_3 w_277_4 w_277_5 w_277_6 w_277_7 w_278_0 w_278_1 w_278_2 w_278_3 w_281_0 w_281_1 w_281_2 w_281_3 w_281_4 w_281_5 w_281_6 w_281_7 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_269.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_270.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_273.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_274.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_277.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_278.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_281.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_5.bridge {backend hashModel tape path params params' i_0 w_43_0 w_189_0 w_193_0 w_45_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_204_1 w_204_2 w_204_3 w_205_0 w_206_0 w_207_0 w_207_1 w_207_2 w_207_3 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_5 backend hashModel tape path params i_0 w_43_0 w_189_0 w_193_0 w_45_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_204_1 w_204_2 w_204_3 w_205_0 w_206_0 w_207_0 w_207_1 w_207_2 w_207_3 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_5 backend hashModel params' i_0 w_43_0 w_189_0 w_193_0 w_45_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_204_1 w_204_2 w_204_3 w_205_0 w_206_0 w_207_0 w_207_1 w_207_2 w_207_3 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_204.bridge (h19 i))), h20, h21, h22, (Stage_preprocess.iterRuns_mono h23 (fun _ _ _ hs33 ↦ (Stage_preprocess.sequential_parallel_generatedRoot_211_207.bridge hs33))), h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_4.bridge {backend hashModel tape path params params' i_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 w_34_0 w_160_0 w_161_0 w_164_0 w_165_0 w_168_0 w_169_0 w_172_0 w_173_0 w_176_0 w_177_0 w_180_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_44_0 w_46_0 w_159_0 w_159_1 w_159_2 w_159_3 w_162_0 w_162_1 w_162_2 w_162_3 w_162_4 w_162_5 w_162_6 w_162_7 w_163_0 w_163_1 w_163_2 w_163_3 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_167_0 w_167_1 w_167_2 w_167_3 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_171_0 w_171_1 w_171_2 w_171_3 w_174_0 w_174_1 w_174_2 w_174_3 w_174_4 w_174_5 w_174_6 w_174_7 w_175_0 w_175_1 w_175_2 w_175_3 w_178_0 w_178_1 w_178_2 w_178_3 w_178_4 w_178_5 w_178_6 w_178_7 w_179_0 w_179_1 w_179_2 w_179_3 w_182_0 w_182_1 w_182_2 w_182_3 w_182_4 w_182_5 w_182_6 w_182_7 w_183_0 w_183_1 w_183_2 w_183_3 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_187_1 w_187_0 w_190_0 w_191_0}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_4 backend hashModel tape path params i_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 w_34_0 w_160_0 w_161_0 w_164_0 w_165_0 w_168_0 w_169_0 w_172_0 w_173_0 w_176_0 w_177_0 w_180_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_44_0 w_46_0 w_159_0 w_159_1 w_159_2 w_159_3 w_162_0 w_162_1 w_162_2 w_162_3 w_162_4 w_162_5 w_162_6 w_162_7 w_163_0 w_163_1 w_163_2 w_163_3 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_167_0 w_167_1 w_167_2 w_167_3 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_171_0 w_171_1 w_171_2 w_171_3 w_174_0 w_174_1 w_174_2 w_174_3 w_174_4 w_174_5 w_174_6 w_174_7 w_175_0 w_175_1 w_175_2 w_175_3 w_178_0 w_178_1 w_178_2 w_178_3 w_178_4 w_178_5 w_178_6 w_178_7 w_179_0 w_179_1 w_179_2 w_179_3 w_182_0 w_182_1 w_182_2 w_182_3 w_182_4 w_182_5 w_182_6 w_182_7 w_183_0 w_183_1 w_183_2 w_183_3 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_187_1 w_187_0 w_190_0 w_191_0) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_4 backend hashModel params' i_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 w_34_0 w_160_0 w_161_0 w_164_0 w_165_0 w_168_0 w_169_0 w_172_0 w_173_0 w_176_0 w_177_0 w_180_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_44_0 w_46_0 w_159_0 w_159_1 w_159_2 w_159_3 w_162_0 w_162_1 w_162_2 w_162_3 w_162_4 w_162_5 w_162_6 w_162_7 w_163_0 w_163_1 w_163_2 w_163_3 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_167_0 w_167_1 w_167_2 w_167_3 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_171_0 w_171_1 w_171_2 w_171_3 w_174_0 w_174_1 w_174_2 w_174_3 w_174_4 w_174_5 w_174_6 w_174_7 w_175_0 w_175_1 w_175_2 w_175_3 w_178_0 w_178_1 w_178_2 w_178_3 w_178_4 w_178_5 w_178_6 w_178_7 w_179_0 w_179_1 w_179_2 w_179_3 w_182_0 w_182_1 w_182_2 w_182_3 w_182_4 w_182_5 w_182_6 w_182_7 w_183_0 w_183_1 w_183_2 w_183_3 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_187_1 w_187_0 w_190_0 w_191_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_162.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_163.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_166.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_167.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_170.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_171.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_174.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_175.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_178.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_179.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_182.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_183.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_186.bridge (h26 i))), (MxxRuntime.trapdoorSampleAt_sample h27), h28, h29, h30, h31, h32⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_3.bridge {backend hashModel tape path params params' i_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_34_0 w_128_0 w_129_0 w_132_0 w_133_0 w_136_0 w_137_0 w_140_0 w_141_0 w_144_0 w_145_0 w_148_0 w_149_0 w_152_0 w_153_0 w_156_0 w_157_0 w_44_0 w_46_0 w_127_0 w_127_1 w_127_2 w_127_3 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_131_0 w_131_1 w_131_2 w_131_3 w_134_0 w_134_1 w_134_2 w_134_3 w_134_4 w_134_5 w_134_6 w_134_7 w_135_0 w_135_1 w_135_2 w_135_3 w_138_0 w_138_1 w_138_2 w_138_3 w_138_4 w_138_5 w_138_6 w_138_7 w_139_0 w_139_1 w_139_2 w_139_3 w_142_0 w_142_1 w_142_2 w_142_3 w_142_4 w_142_5 w_142_6 w_142_7 w_143_0 w_143_1 w_143_2 w_143_3 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_147_0 w_147_1 w_147_2 w_147_3 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_150_6 w_150_7 w_151_0 w_151_1 w_151_2 w_151_3 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_154_6 w_154_7 w_155_0 w_155_1 w_155_2 w_155_3 w_158_0 w_158_1 w_158_2 w_158_3 w_158_4 w_158_5 w_158_6 w_158_7 w_159_0 w_159_1 w_159_2 w_159_3}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_3 backend hashModel tape path params i_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_34_0 w_128_0 w_129_0 w_132_0 w_133_0 w_136_0 w_137_0 w_140_0 w_141_0 w_144_0 w_145_0 w_148_0 w_149_0 w_152_0 w_153_0 w_156_0 w_157_0 w_44_0 w_46_0 w_127_0 w_127_1 w_127_2 w_127_3 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_131_0 w_131_1 w_131_2 w_131_3 w_134_0 w_134_1 w_134_2 w_134_3 w_134_4 w_134_5 w_134_6 w_134_7 w_135_0 w_135_1 w_135_2 w_135_3 w_138_0 w_138_1 w_138_2 w_138_3 w_138_4 w_138_5 w_138_6 w_138_7 w_139_0 w_139_1 w_139_2 w_139_3 w_142_0 w_142_1 w_142_2 w_142_3 w_142_4 w_142_5 w_142_6 w_142_7 w_143_0 w_143_1 w_143_2 w_143_3 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_147_0 w_147_1 w_147_2 w_147_3 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_150_6 w_150_7 w_151_0 w_151_1 w_151_2 w_151_3 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_154_6 w_154_7 w_155_0 w_155_1 w_155_2 w_155_3 w_158_0 w_158_1 w_158_2 w_158_3 w_158_4 w_158_5 w_158_6 w_158_7 w_159_0 w_159_1 w_159_2 w_159_3) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_3 backend hashModel params' i_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_34_0 w_128_0 w_129_0 w_132_0 w_133_0 w_136_0 w_137_0 w_140_0 w_141_0 w_144_0 w_145_0 w_148_0 w_149_0 w_152_0 w_153_0 w_156_0 w_157_0 w_44_0 w_46_0 w_127_0 w_127_1 w_127_2 w_127_3 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_131_0 w_131_1 w_131_2 w_131_3 w_134_0 w_134_1 w_134_2 w_134_3 w_134_4 w_134_5 w_134_6 w_134_7 w_135_0 w_135_1 w_135_2 w_135_3 w_138_0 w_138_1 w_138_2 w_138_3 w_138_4 w_138_5 w_138_6 w_138_7 w_139_0 w_139_1 w_139_2 w_139_3 w_142_0 w_142_1 w_142_2 w_142_3 w_142_4 w_142_5 w_142_6 w_142_7 w_143_0 w_143_1 w_143_2 w_143_3 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_147_0 w_147_1 w_147_2 w_147_3 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_150_6 w_150_7 w_151_0 w_151_1 w_151_2 w_151_3 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_154_6 w_154_7 w_155_0 w_155_1 w_155_2 w_155_3 w_158_0 w_158_1 w_158_2 w_158_3 w_158_4 w_158_5 w_158_6 w_158_7 w_159_0 w_159_1 w_159_2 w_159_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_130.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_131.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_134.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_135.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_138.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_139.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_142.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_143.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_146.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_147.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_150.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_151.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_154.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_155.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_158.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_159.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_2.bridge {backend hashModel tape path params params' i_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_34_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_116_0 w_117_0 w_120_0 w_121_0 w_124_0 w_125_0 w_44_0 w_46_0 w_95_0 w_95_1 w_95_2 w_95_3 w_98_0 w_98_1 w_98_2 w_98_3 w_98_4 w_98_5 w_98_6 w_98_7 w_99_0 w_99_1 w_99_2 w_99_3 w_102_0 w_102_1 w_102_2 w_102_3 w_102_4 w_102_5 w_102_6 w_102_7 w_103_0 w_103_1 w_103_2 w_103_3 w_106_0 w_106_1 w_106_2 w_106_3 w_106_4 w_106_5 w_106_6 w_106_7 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_111_0 w_111_1 w_111_2 w_111_3 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_114_6 w_114_7 w_115_0 w_115_1 w_115_2 w_115_3 w_118_0 w_118_1 w_118_2 w_118_3 w_118_4 w_118_5 w_118_6 w_118_7 w_119_0 w_119_1 w_119_2 w_119_3 w_122_0 w_122_1 w_122_2 w_122_3 w_122_4 w_122_5 w_122_6 w_122_7 w_123_0 w_123_1 w_123_2 w_123_3 w_126_0 w_126_1 w_126_2 w_126_3 w_126_4 w_126_5 w_126_6 w_126_7 w_127_0 w_127_1 w_127_2 w_127_3}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_2 backend hashModel tape path params i_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_34_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_116_0 w_117_0 w_120_0 w_121_0 w_124_0 w_125_0 w_44_0 w_46_0 w_95_0 w_95_1 w_95_2 w_95_3 w_98_0 w_98_1 w_98_2 w_98_3 w_98_4 w_98_5 w_98_6 w_98_7 w_99_0 w_99_1 w_99_2 w_99_3 w_102_0 w_102_1 w_102_2 w_102_3 w_102_4 w_102_5 w_102_6 w_102_7 w_103_0 w_103_1 w_103_2 w_103_3 w_106_0 w_106_1 w_106_2 w_106_3 w_106_4 w_106_5 w_106_6 w_106_7 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_111_0 w_111_1 w_111_2 w_111_3 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_114_6 w_114_7 w_115_0 w_115_1 w_115_2 w_115_3 w_118_0 w_118_1 w_118_2 w_118_3 w_118_4 w_118_5 w_118_6 w_118_7 w_119_0 w_119_1 w_119_2 w_119_3 w_122_0 w_122_1 w_122_2 w_122_3 w_122_4 w_122_5 w_122_6 w_122_7 w_123_0 w_123_1 w_123_2 w_123_3 w_126_0 w_126_1 w_126_2 w_126_3 w_126_4 w_126_5 w_126_6 w_126_7 w_127_0 w_127_1 w_127_2 w_127_3) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_2 backend hashModel params' i_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_34_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_116_0 w_117_0 w_120_0 w_121_0 w_124_0 w_125_0 w_44_0 w_46_0 w_95_0 w_95_1 w_95_2 w_95_3 w_98_0 w_98_1 w_98_2 w_98_3 w_98_4 w_98_5 w_98_6 w_98_7 w_99_0 w_99_1 w_99_2 w_99_3 w_102_0 w_102_1 w_102_2 w_102_3 w_102_4 w_102_5 w_102_6 w_102_7 w_103_0 w_103_1 w_103_2 w_103_3 w_106_0 w_106_1 w_106_2 w_106_3 w_106_4 w_106_5 w_106_6 w_106_7 w_107_0 w_107_1 w_107_2 w_107_3 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_111_0 w_111_1 w_111_2 w_111_3 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_114_6 w_114_7 w_115_0 w_115_1 w_115_2 w_115_3 w_118_0 w_118_1 w_118_2 w_118_3 w_118_4 w_118_5 w_118_6 w_118_7 w_119_0 w_119_1 w_119_2 w_119_3 w_122_0 w_122_1 w_122_2 w_122_3 w_122_4 w_122_5 w_122_6 w_122_7 w_123_0 w_123_1 w_123_2 w_123_3 w_126_0 w_126_1 w_126_2 w_126_3 w_126_4 w_126_5 w_126_6 w_126_7 w_127_0 w_127_1 w_127_2 w_127_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_98.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_99.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_102.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_103.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_106.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_107.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_110.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_111.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_114.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_115.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_118.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_119.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_122.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_123.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_126.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_127.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_1.bridge {backend hashModel tape path params params' i_0 w_2_0 w_23_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_44_0 w_46_0 w_63_0 w_63_1 w_63_2 w_63_3 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_66_6 w_66_7 w_67_0 w_67_1 w_67_2 w_67_3 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_70_6 w_70_7 w_71_0 w_71_1 w_71_2 w_71_3 w_74_0 w_74_1 w_74_2 w_74_3 w_74_4 w_74_5 w_74_6 w_74_7 w_75_0 w_75_1 w_75_2 w_75_3 w_78_0 w_78_1 w_78_2 w_78_3 w_78_4 w_78_5 w_78_6 w_78_7 w_79_0 w_79_1 w_79_2 w_79_3 w_82_0 w_82_1 w_82_2 w_82_3 w_82_4 w_82_5 w_82_6 w_82_7 w_83_0 w_83_1 w_83_2 w_83_3 w_86_0 w_86_1 w_86_2 w_86_3 w_86_4 w_86_5 w_86_6 w_86_7 w_87_0 w_87_1 w_87_2 w_87_3 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_90_6 w_90_7 w_91_0 w_91_1 w_91_2 w_91_3 w_94_0 w_94_1 w_94_2 w_94_3 w_94_4 w_94_5 w_94_6 w_94_7 w_95_0 w_95_1 w_95_2 w_95_3}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_1 backend hashModel tape path params i_0 w_2_0 w_23_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_44_0 w_46_0 w_63_0 w_63_1 w_63_2 w_63_3 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_66_6 w_66_7 w_67_0 w_67_1 w_67_2 w_67_3 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_70_6 w_70_7 w_71_0 w_71_1 w_71_2 w_71_3 w_74_0 w_74_1 w_74_2 w_74_3 w_74_4 w_74_5 w_74_6 w_74_7 w_75_0 w_75_1 w_75_2 w_75_3 w_78_0 w_78_1 w_78_2 w_78_3 w_78_4 w_78_5 w_78_6 w_78_7 w_79_0 w_79_1 w_79_2 w_79_3 w_82_0 w_82_1 w_82_2 w_82_3 w_82_4 w_82_5 w_82_6 w_82_7 w_83_0 w_83_1 w_83_2 w_83_3 w_86_0 w_86_1 w_86_2 w_86_3 w_86_4 w_86_5 w_86_6 w_86_7 w_87_0 w_87_1 w_87_2 w_87_3 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_90_6 w_90_7 w_91_0 w_91_1 w_91_2 w_91_3 w_94_0 w_94_1 w_94_2 w_94_3 w_94_4 w_94_5 w_94_6 w_94_7 w_95_0 w_95_1 w_95_2 w_95_3) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_1 backend hashModel params' i_0 w_2_0 w_23_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_44_0 w_46_0 w_63_0 w_63_1 w_63_2 w_63_3 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_66_6 w_66_7 w_67_0 w_67_1 w_67_2 w_67_3 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_70_6 w_70_7 w_71_0 w_71_1 w_71_2 w_71_3 w_74_0 w_74_1 w_74_2 w_74_3 w_74_4 w_74_5 w_74_6 w_74_7 w_75_0 w_75_1 w_75_2 w_75_3 w_78_0 w_78_1 w_78_2 w_78_3 w_78_4 w_78_5 w_78_6 w_78_7 w_79_0 w_79_1 w_79_2 w_79_3 w_82_0 w_82_1 w_82_2 w_82_3 w_82_4 w_82_5 w_82_6 w_82_7 w_83_0 w_83_1 w_83_2 w_83_3 w_86_0 w_86_1 w_86_2 w_86_3 w_86_4 w_86_5 w_86_6 w_86_7 w_87_0 w_87_1 w_87_2 w_87_3 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_90_6 w_90_7 w_91_0 w_91_1 w_91_2 w_91_3 w_94_0 w_94_1 w_94_2 w_94_3 w_94_4 w_94_5 w_94_6 w_94_7 w_95_0 w_95_1 w_95_2 w_95_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_66.bridge (h2 i))), h3, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_67.bridge (h4 i))), h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_70.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_71.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_74.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_75.bridge (h12 i))), h13, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_78.bridge (h14 i))), h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_79.bridge (h16 i))), h17, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_82.bridge (h18 i))), h19, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_83.bridge (h20 i))), h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_86.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_87.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_90.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_91.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_94.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_95.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_12_0 w_34_0 w_43_0 w_52_0 w_53_0 w_56_0 w_57_0 w_60_0 w_61_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_49_0 w_50_0 w_51_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_54_6 w_54_7 w_55_0 w_55_1 w_55_2 w_55_3 w_58_0 w_58_1 w_58_2 w_58_3 w_58_4 w_58_5 w_58_6 w_58_7 w_59_0 w_59_1 w_59_2 w_59_3 w_62_0 w_62_1 w_62_2 w_62_3 w_62_4 w_62_5 w_62_6 w_62_7 w_63_0 w_63_1 w_63_2 w_63_3}
    (h : Stage_preprocess.parallel_generatedRoot_211.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_12_0 w_34_0 w_43_0 w_52_0 w_53_0 w_56_0 w_57_0 w_60_0 w_61_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_49_0 w_50_0 w_51_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_54_6 w_54_7 w_55_0 w_55_1 w_55_2 w_55_3 w_58_0 w_58_1 w_58_2 w_58_3 w_58_4 w_58_5 w_58_6 w_58_7 w_59_0 w_59_1 w_59_2 w_59_3 w_62_0 w_62_1 w_62_2 w_62_3 w_62_4 w_62_5 w_62_6 w_62_7 w_63_0 w_63_1 w_63_2 w_63_3) : W_Stage_preprocess.parallel_generatedRoot_211.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_12_0 w_34_0 w_43_0 w_52_0 w_53_0 w_56_0 w_57_0 w_60_0 w_61_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_44_0 w_45_0 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_49_0 w_50_0 w_51_0 w_54_0 w_54_1 w_54_2 w_54_3 w_54_4 w_54_5 w_54_6 w_54_7 w_55_0 w_55_1 w_55_2 w_55_3 w_58_0 w_58_1 w_58_2 w_58_3 w_58_4 w_58_5 w_58_6 w_58_7 w_59_0 w_59_1 w_59_2 w_59_3 w_62_0 w_62_1 w_62_2 w_62_3 w_62_4 w_62_5 w_62_6 w_62_7 w_63_0 w_63_1 w_63_2 w_63_3 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_39.bridge (h6 i))), h7, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_40.bridge (h8 i))), h9, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_41.bridge (h10 i))), h11, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_42.bridge (h12 i))), h13, h14, h15, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_47.bridge (h16 i))), h17, h18, h19, h20, h21, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_54.bridge (h22 i))), h23, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_55.bridge (h24 i))), h25, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_58.bridge (h26 i))), h27, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_59.bridge (h28 i))), h29, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_62.bridge (h30 i))), h31, (fun i ↦ (Stage_preprocess.parallel_parallel_generatedRoot_211_63.bridge (h32 i)))⟩)


theorem Stage_preprocess.parallel_generatedRoot_211.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_preprocess.parallel_generatedRoot_211 backend hashModel tape path params i_0 inputs outputs) : W_Stage_preprocess.parallel_generatedRoot_211 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_preprocess.parallel_generatedRoot_211
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_35_0, wit1.w_36_0, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_41_0, wit1.w_42_0, wit1.w_44_0, wit1.w_46_0, wit1.w_47_0, wit1.w_47_1, wit1.w_47_2, wit1.w_47_3, wit1.w_48_0, wit1.w_49_0, wit1.w_50_0, wit1.w_51_0, wit1.w_54_0, wit1.w_54_1, wit1.w_54_2, wit1.w_54_3, wit1.w_54_4, wit1.w_54_5, wit1.w_54_6, wit1.w_54_7, wit1.w_55_0, wit1.w_55_1, wit1.w_55_2, wit1.w_55_3, wit1.w_58_0, wit1.w_58_1, wit1.w_58_2, wit1.w_58_3, wit1.w_58_4, wit1.w_58_5, wit1.w_58_6, wit1.w_58_7, wit1.w_59_0, wit1.w_59_1, wit1.w_59_2, wit1.w_59_3, wit1.w_62_0, wit1.w_62_1, wit1.w_62_2, wit1.w_62_3, wit1.w_62_4, wit1.w_62_5, wit1.w_62_6, wit1.w_62_7, wit1.w_63_0, wit1.w_63_1, wit1.w_63_2, wit1.w_63_3, wit1.w_66_0, wit1.w_66_1, wit1.w_66_2, wit1.w_66_3, wit1.w_66_4, wit1.w_66_5, wit1.w_66_6, wit1.w_66_7, wit1.w_67_0, wit1.w_67_1, wit1.w_67_2, wit1.w_67_3, wit1.w_70_0, wit1.w_70_1, wit1.w_70_2, wit1.w_70_3, wit1.w_70_4, wit1.w_70_5, wit1.w_70_6, wit1.w_70_7, wit1.w_71_0, wit1.w_71_1, wit1.w_71_2, wit1.w_71_3, wit1.w_74_0, wit1.w_74_1, wit1.w_74_2, wit1.w_74_3, wit1.w_74_4, wit1.w_74_5, wit1.w_74_6, wit1.w_74_7, wit1.w_75_0, wit1.w_75_1, wit1.w_75_2, wit1.w_75_3, wit1.w_78_0, wit1.w_78_1, wit1.w_78_2, wit1.w_78_3, wit1.w_78_4, wit1.w_78_5, wit1.w_78_6, wit1.w_78_7, wit1.w_79_0, wit1.w_79_1, wit1.w_79_2, wit1.w_79_3, wit1.w_82_0, wit1.w_82_1, wit1.w_82_2, wit1.w_82_3, wit1.w_82_4, wit1.w_82_5, wit1.w_82_6, wit1.w_82_7, wit1.w_83_0, wit1.w_83_1, wit1.w_83_2, wit1.w_83_3, wit1.w_86_0, wit1.w_86_1, wit1.w_86_2, wit1.w_86_3, wit1.w_86_4, wit1.w_86_5, wit1.w_86_6, wit1.w_86_7, wit1.w_87_0, wit1.w_87_1, wit1.w_87_2, wit1.w_87_3, wit1.w_90_0, wit1.w_90_1, wit1.w_90_2, wit1.w_90_3, wit1.w_90_4, wit1.w_90_5, wit1.w_90_6, wit1.w_90_7, wit1.w_91_0, wit1.w_91_1, wit1.w_91_2, wit1.w_91_3, wit1.w_94_0, wit1.w_94_1, wit1.w_94_2, wit1.w_94_3, wit1.w_94_4, wit1.w_94_5, wit1.w_94_6, wit1.w_94_7, wit1.w_95_0, wit1.w_95_1, wit1.w_95_2, wit1.w_95_3, wit1.w_98_0, wit1.w_98_1, wit1.w_98_2, wit1.w_98_3, wit1.w_98_4, wit1.w_98_5, wit1.w_98_6, wit1.w_98_7, wit1.w_99_0, wit1.w_99_1, wit1.w_99_2, wit1.w_99_3, wit1.w_102_0, wit1.w_102_1, wit1.w_102_2, wit1.w_102_3, wit1.w_102_4, wit1.w_102_5, wit1.w_102_6, wit1.w_102_7, wit1.w_103_0, wit1.w_103_1, wit1.w_103_2, wit1.w_103_3, wit1.w_106_0, wit1.w_106_1, wit1.w_106_2, wit1.w_106_3, wit1.w_106_4, wit1.w_106_5, wit1.w_106_6, wit1.w_106_7, wit1.w_107_0, wit1.w_107_1, wit1.w_107_2, wit1.w_107_3, wit1.w_110_0, wit1.w_110_1, wit1.w_110_2, wit1.w_110_3, wit1.w_110_4, wit1.w_110_5, wit1.w_110_6, wit1.w_110_7, wit1.w_111_0, wit1.w_111_1, wit1.w_111_2, wit1.w_111_3, wit1.w_114_0, wit1.w_114_1, wit1.w_114_2, wit1.w_114_3, wit1.w_114_4, wit1.w_114_5, wit1.w_114_6, wit1.w_114_7, wit1.w_115_0, wit1.w_115_1, wit1.w_115_2, wit1.w_115_3, wit1.w_118_0, wit1.w_118_1, wit1.w_118_2, wit1.w_118_3, wit1.w_118_4, wit1.w_118_5, wit1.w_118_6, wit1.w_118_7, wit1.w_119_0, wit1.w_119_1, wit1.w_119_2, wit1.w_119_3, wit1.w_122_0, wit1.w_122_1, wit1.w_122_2, wit1.w_122_3, wit1.w_122_4, wit1.w_122_5, wit1.w_122_6, wit1.w_122_7, wit1.w_123_0, wit1.w_123_1, wit1.w_123_2, wit1.w_123_3, wit1.w_126_0, wit1.w_126_1, wit1.w_126_2, wit1.w_126_3, wit1.w_126_4, wit1.w_126_5, wit1.w_126_6, wit1.w_126_7, wit1.w_127_0, wit1.w_127_1, wit1.w_127_2, wit1.w_127_3, wit1.w_130_0, wit1.w_130_1, wit1.w_130_2, wit1.w_130_3, wit1.w_130_4, wit1.w_130_5, wit1.w_130_6, wit1.w_130_7, wit1.w_131_0, wit1.w_131_1, wit1.w_131_2, wit1.w_131_3, wit1.w_134_0, wit1.w_134_1, wit1.w_134_2, wit1.w_134_3, wit1.w_134_4, wit1.w_134_5, wit1.w_134_6, wit1.w_134_7, wit1.w_135_0, wit1.w_135_1, wit1.w_135_2, wit1.w_135_3, wit1.w_138_0, wit1.w_138_1, wit1.w_138_2, wit1.w_138_3, wit1.w_138_4, wit1.w_138_5, wit1.w_138_6, wit1.w_138_7, wit1.w_139_0, wit1.w_139_1, wit1.w_139_2, wit1.w_139_3, wit1.w_142_0, wit1.w_142_1, wit1.w_142_2, wit1.w_142_3, wit1.w_142_4, wit1.w_142_5, wit1.w_142_6, wit1.w_142_7, wit1.w_143_0, wit1.w_143_1, wit1.w_143_2, wit1.w_143_3, wit1.w_146_0, wit1.w_146_1, wit1.w_146_2, wit1.w_146_3, wit1.w_146_4, wit1.w_146_5, wit1.w_146_6, wit1.w_146_7, wit1.w_147_0, wit1.w_147_1, wit1.w_147_2, wit1.w_147_3, wit1.w_150_0, wit1.w_150_1, wit1.w_150_2, wit1.w_150_3, wit1.w_150_4, wit1.w_150_5, wit1.w_150_6, wit1.w_150_7, wit1.w_151_0, wit1.w_151_1, wit1.w_151_2, wit1.w_151_3, wit1.w_154_0, wit1.w_154_1, wit1.w_154_2, wit1.w_154_3, wit1.w_154_4, wit1.w_154_5, wit1.w_154_6, wit1.w_154_7, wit1.w_155_0, wit1.w_155_1, wit1.w_155_2, wit1.w_155_3, wit1.w_158_0, wit1.w_158_1, wit1.w_158_2, wit1.w_158_3, wit1.w_158_4, wit1.w_158_5, wit1.w_158_6, wit1.w_158_7, wit1.w_159_0, wit1.w_159_1, wit1.w_159_2, wit1.w_159_3, wit1.w_162_0, wit1.w_162_1, wit1.w_162_2, wit1.w_162_3, wit1.w_162_4, wit1.w_162_5, wit1.w_162_6, wit1.w_162_7, wit1.w_163_0, wit1.w_163_1, wit1.w_163_2, wit1.w_163_3, wit1.w_166_0, wit1.w_166_1, wit1.w_166_2, wit1.w_166_3, wit1.w_166_4, wit1.w_166_5, wit1.w_166_6, wit1.w_166_7, wit1.w_167_0, wit1.w_167_1, wit1.w_167_2, wit1.w_167_3, wit1.w_170_0, wit1.w_170_1, wit1.w_170_2, wit1.w_170_3, wit1.w_170_4, wit1.w_170_5, wit1.w_170_6, wit1.w_170_7, wit1.w_171_0, wit1.w_171_1, wit1.w_171_2, wit1.w_171_3, wit1.w_174_0, wit1.w_174_1, wit1.w_174_2, wit1.w_174_3, wit1.w_174_4, wit1.w_174_5, wit1.w_174_6, wit1.w_174_7, wit1.w_175_0, wit1.w_175_1, wit1.w_175_2, wit1.w_175_3, wit1.w_178_0, wit1.w_178_1, wit1.w_178_2, wit1.w_178_3, wit1.w_178_4, wit1.w_178_5, wit1.w_178_6, wit1.w_178_7, wit1.w_179_0, wit1.w_179_1, wit1.w_179_2, wit1.w_179_3, wit1.w_182_0, wit1.w_182_1, wit1.w_182_2, wit1.w_182_3, wit1.w_182_4, wit1.w_182_5, wit1.w_182_6, wit1.w_182_7, wit1.w_183_0, wit1.w_183_1, wit1.w_183_2, wit1.w_183_3, wit1.w_186_0, wit1.w_186_1, wit1.w_186_2, wit1.w_186_3, wit1.w_186_4, wit1.w_186_5, wit1.w_186_6, wit1.w_186_7, wit1.w_187_1, wit1.w_187_0, wit1.w_190_0, wit1.w_191_0, wit1.w_194_0, wit1.w_195_0, wit1.w_197_0, wit1.w_198_0, wit1.w_202_0, wit1.w_203_0, wit1.w_204_0, wit1.w_204_1, wit1.w_204_2, wit1.w_204_3, wit1.w_205_0, wit1.w_206_0, wit1.w_207_0, wit1.w_207_1, wit1.w_207_2, wit1.w_207_3, wit1.w_208_0, wit1.w_209_0, wit1.w_210_0, wit1.w_211_0, wit1.w_212_0, wit1.w_217_0, wit1.w_220_0, wit1.w_221_0, wit1.w_261_0, wit1.w_262_0, wit1.w_263_0, wit1.w_264_0, wit1.w_265_0, wit1.w_268_0, wit1.w_269_0, wit1.w_269_1, wit1.w_269_2, wit1.w_269_3, wit1.w_269_4, wit1.w_269_5, wit1.w_269_6, wit1.w_269_7, wit1.w_270_0, wit1.w_270_1, wit1.w_270_2, wit1.w_270_3, wit1.w_273_0, wit1.w_273_1, wit1.w_273_2, wit1.w_273_3, wit1.w_273_4, wit1.w_273_5, wit1.w_273_6, wit1.w_273_7, wit1.w_274_0, wit1.w_274_1, wit1.w_274_2, wit1.w_274_3, wit1.w_277_0, wit1.w_277_1, wit1.w_277_2, wit1.w_277_3, wit1.w_277_4, wit1.w_277_5, wit1.w_277_6, wit1.w_277_7, wit1.w_278_0, wit1.w_278_1, wit1.w_278_2, wit1.w_278_3, wit1.w_281_0, wit1.w_281_1, wit1.w_281_2, wit1.w_281_3, wit1.w_281_4, wit1.w_281_5, wit1.w_281_6, wit1.w_281_7, wit1.w_282_0, wit1.w_282_1, wit1.w_282_2, wit1.w_282_3, wit1.w_285_0, wit1.w_285_1, wit1.w_285_2, wit1.w_285_3, wit1.w_285_4, wit1.w_285_5, wit1.w_285_6, wit1.w_285_7, wit1.w_286_0, wit1.w_286_1, wit1.w_286_2, wit1.w_286_3, wit1.w_289_0, wit1.w_289_1, wit1.w_289_2, wit1.w_289_3, wit1.w_289_4, wit1.w_289_5, wit1.w_289_6, wit1.w_289_7, wit1.w_290_0, wit1.w_290_1, wit1.w_290_2, wit1.w_290_3, wit1.w_293_0, wit1.w_293_1, wit1.w_293_2, wit1.w_293_3, wit1.w_293_4, wit1.w_293_5, wit1.w_293_6, wit1.w_293_7, wit1.w_294_0, wit1.w_294_1, wit1.w_294_2, wit1.w_294_3, wit1.w_297_0, wit1.w_297_1, wit1.w_297_2, wit1.w_297_3, wit1.w_297_4, wit1.w_297_5, wit1.w_297_6, wit1.w_297_7, wit1.w_298_0, wit1.w_298_1, wit1.w_298_2, wit1.w_298_3, wit1.w_301_0, wit1.w_301_1, wit1.w_301_2, wit1.w_301_3, wit1.w_301_4, wit1.w_301_5, wit1.w_301_6, wit1.w_301_7, wit1.w_302_0, wit1.w_302_1, wit1.w_302_2, wit1.w_302_3, wit1.w_305_0, wit1.w_305_1, wit1.w_305_2, wit1.w_305_3, wit1.w_305_4, wit1.w_305_5, wit1.w_305_6, wit1.w_305_7, wit1.w_306_0, wit1.w_306_1, wit1.w_306_2, wit1.w_306_3, wit1.w_309_0, wit1.w_309_1, wit1.w_309_2, wit1.w_309_3, wit1.w_309_4, wit1.w_309_5, wit1.w_309_6, wit1.w_309_7, wit1.w_310_0, wit1.w_310_1, wit1.w_310_2, wit1.w_310_3, wit1.w_313_0, wit1.w_313_1, wit1.w_313_2, wit1.w_313_3, wit1.w_313_4, wit1.w_313_5, wit1.w_313_6, wit1.w_313_7, wit1.w_314_0, wit1.w_314_1, wit1.w_314_2, wit1.w_314_3, wit1.w_317_0, wit1.w_317_1, wit1.w_317_2, wit1.w_317_3, wit1.w_317_4, wit1.w_317_5, wit1.w_317_6, wit1.w_317_7, wit1.w_318_0, wit1.w_318_1, wit1.w_318_2, wit1.w_318_3, wit1.w_321_0, wit1.w_321_1, wit1.w_321_2, wit1.w_321_3, wit1.w_321_4, wit1.w_321_5, wit1.w_321_6, wit1.w_321_7, wit1.w_322_0, wit1.w_322_1, wit1.w_322_2, wit1.w_322_3, wit1.w_325_0, wit1.w_325_1, wit1.w_325_2, wit1.w_325_3, wit1.w_325_4, wit1.w_325_5, wit1.w_325_6, wit1.w_325_7, wit1.w_326_0, wit1.w_326_1, wit1.w_326_2, wit1.w_326_3, wit1.w_329_0, wit1.w_329_1, wit1.w_329_2, wit1.w_329_3, wit1.w_329_4, wit1.w_329_5, wit1.w_329_6, wit1.w_329_7, wit1.w_330_0, wit1.w_330_1, wit1.w_330_2, wit1.w_330_3, wit1.w_333_0, wit1.w_333_1, wit1.w_333_2, wit1.w_333_3, wit1.w_333_4, wit1.w_333_5, wit1.w_333_6, wit1.w_333_7, wit1.w_334_0, wit1.w_334_1, wit1.w_334_2, wit1.w_334_3, wit1.w_337_0, wit1.w_337_1, wit1.w_337_2, wit1.w_337_3, wit1.w_337_4, wit1.w_337_5, wit1.w_337_6, wit1.w_337_7, wit1.w_338_0, wit1.w_338_1, wit1.w_338_2, wit1.w_338_3, wit1.w_341_0, wit1.w_341_1, wit1.w_341_2, wit1.w_341_3, wit1.w_341_4, wit1.w_341_5, wit1.w_341_6, wit1.w_341_7, wit1.w_342_0, wit1.w_342_1, wit1.w_342_2, wit1.w_342_3, wit1.w_345_0, wit1.w_345_1, wit1.w_345_2, wit1.w_345_3, wit1.w_345_4, wit1.w_345_5, wit1.w_345_6, wit1.w_345_7, wit1.w_346_0, wit1.w_346_1, wit1.w_346_2, wit1.w_346_3, wit1.w_349_0, wit1.w_349_1, wit1.w_349_2, wit1.w_349_3, wit1.w_349_4, wit1.w_349_5, wit1.w_349_6, wit1.w_349_7, wit1.w_350_0, wit1.w_350_1, wit1.w_350_2, wit1.w_350_3, wit1.w_353_0, wit1.w_353_1, wit1.w_353_2, wit1.w_353_3, wit1.w_353_4, wit1.w_353_5, wit1.w_353_6, wit1.w_353_7, wit1.w_354_0, wit1.w_354_1, wit1.w_354_2, wit1.w_354_3, wit1.w_357_0, wit1.w_357_1, wit1.w_357_2, wit1.w_357_3, wit1.w_357_4, wit1.w_357_5, wit1.w_357_6, wit1.w_357_7, wit1.w_358_0, wit1.w_358_1, wit1.w_358_2, wit1.w_358_3, wit1.w_361_0, wit1.w_361_1, wit1.w_361_2, wit1.w_361_3, wit1.w_361_4, wit1.w_361_5, wit1.w_361_6, wit1.w_361_7, wit1.w_362_0, wit1.w_362_1, wit1.w_362_2, wit1.w_362_3, wit1.w_365_0, wit1.w_365_1, wit1.w_365_2, wit1.w_365_3, wit1.w_365_4, wit1.w_365_5, wit1.w_365_6, wit1.w_365_7, wit1.w_366_0, wit1.w_366_1, wit1.w_366_2, wit1.w_366_3, wit1.w_369_0, wit1.w_369_1, wit1.w_369_2, wit1.w_369_3, wit1.w_369_4, wit1.w_369_5, wit1.w_369_6, wit1.w_369_7, wit1.w_370_0, wit1.w_370_1, wit1.w_370_2, wit1.w_370_3, wit1.w_373_0, wit1.w_373_1, wit1.w_373_2, wit1.w_373_3, wit1.w_373_4, wit1.w_373_5, wit1.w_373_6, wit1.w_373_7, wit1.w_374_0, wit1.w_374_1, wit1.w_374_2, wit1.w_374_3, wit1.w_377_0, wit1.w_377_1, wit1.w_377_2, wit1.w_377_3, wit1.w_377_4, wit1.w_377_5, wit1.w_377_6, wit1.w_377_7, wit1.w_378_0, wit1.w_378_1, wit1.w_378_2, wit1.w_378_3, wit1.w_381_0, wit1.w_381_1, wit1.w_381_2, wit1.w_381_3, wit1.w_381_4, wit1.w_381_5, wit1.w_381_6, wit1.w_381_7, wit1.w_382_0, wit1.w_382_1, wit1.w_382_2, wit1.w_382_3, wit1.w_385_0, wit1.w_385_1, wit1.w_385_2, wit1.w_385_3, wit1.w_385_4, wit1.w_385_5, wit1.w_385_6, wit1.w_385_7, wit1.w_386_0, wit1.w_386_1, wit1.w_386_2, wit1.w_386_3, wit1.w_389_0, wit1.w_389_1, wit1.w_389_2, wit1.w_389_3, wit1.w_389_4, wit1.w_389_5, wit1.w_389_6, wit1.w_389_7, wit1.w_390_0, wit1.w_390_1, wit1.w_390_2, wit1.w_390_3, wit1.w_393_0, wit1.w_393_1, wit1.w_393_2, wit1.w_393_3, wit1.w_393_4, wit1.w_393_5, wit1.w_393_6, wit1.w_393_7, wit1.w_394_0, wit1.w_394_1, wit1.w_394_2, wit1.w_394_3, wit1.w_397_0, wit1.w_397_1, wit1.w_397_2, wit1.w_397_3, wit1.w_397_4, wit1.w_397_5, wit1.w_397_6, wit1.w_397_7, wit1.w_398_0, wit1.w_398_1, wit1.w_398_2, wit1.w_398_3, wit1.w_401_0, wit1.w_401_1, wit1.w_401_2, wit1.w_401_3, wit1.w_401_4, wit1.w_401_5, wit1.w_401_6, wit1.w_401_7, wit1.w_402_0, wit1.w_402_1, wit1.w_402_2, wit1.w_402_3, wit1.w_403_0, wit1.w_403_1, wit1.w_403_2, wit1.w_403_3, wit1.w_404_0, wit1.w_404_1, wit1.w_404_2, wit1.w_404_3, wit1.w_405_0, wit1.w_405_1, wit1.w_405_2, wit1.w_405_3, wit1.w_406_0, wit1.w_406_1, wit1.w_406_2, wit1.w_406_3, wit1.w_407_0, wit1.w_407_1, wit1.w_407_2, wit1.w_407_3, wit1.w_408_0, wit1.w_409_0, wit1.w_411_0, wit1.w_412_0, wit1.w_414_0, wit1.w_415_0, wit1.w_417_0, wit1.w_418_0, wit1.w_420_0, wit1.w_420_1, wit1.w_420_2, wit1.w_420_3, wit1.w_421_0, wit1.w_423_0, wit1.w_425_0, wit1.w_427_0, wit1.w_428_0, wit1.w_428_1, wit1.w_428_2, wit1.w_428_3, wit1.w_429_0, wit1.w_430_0, wit1.w_432_0, wit1.w_433_0, wit1.w_435_0, wit1.w_436_0, wit1.w_439_0, wit1.w_439_1, wit1.w_439_2, wit1.w_439_3, wit1.w_440_0, wit1.w_440_1, wit1.w_440_2, wit1.w_440_3, wit1.w_441_0, wit1.w_442_0, wit1.w_444_0, wit1.w_445_0, wit1.w_447_0, wit1.w_448_0, wit1.w_450_0, wit1.w_451_0, wit1.w_453_0, wit1.w_453_1, wit1.w_453_2, wit1.w_453_3, wit1.w_454_0, wit1.w_456_0, wit1.w_458_0, wit1.w_460_0, wit1.w_461_0, wit1.w_461_1, wit1.w_461_2, wit1.w_461_3, wit1.w_464_0, wit1.w_465_0, wit1.w_468_0, wit1.w_469_0, wit1.w_473_0, wit1.w_474_0, wit1.w_476_0, wit1.w_477_0, wit1.w_480_0, wit1.w_481_0, wit1.w_483_0, wit1.w_484_0, wit1.w_486_0, wit1.w_487_0, wit1.w_488_0, wit1.w_489_0, wit1.w_490_0, wit1.w_491_0, wit1.w_492_0, wit1.w_493_0, wit1.w_494_0, wit1.w_495_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h2; exact ⟨(Stage_preprocess.parallel_generatedRoot_211.constraints_0.bridge h3), (Stage_preprocess.parallel_generatedRoot_211.constraints_1.bridge h4), (Stage_preprocess.parallel_generatedRoot_211.constraints_2.bridge h5), (Stage_preprocess.parallel_generatedRoot_211.constraints_3.bridge h6), (Stage_preprocess.parallel_generatedRoot_211.constraints_4.bridge h7), (Stage_preprocess.parallel_generatedRoot_211.constraints_5.bridge h8), (Stage_preprocess.parallel_generatedRoot_211.constraints_6.bridge h9), (Stage_preprocess.parallel_generatedRoot_211.constraints_7.bridge h10), (Stage_preprocess.parallel_generatedRoot_211.constraints_8.bridge h11), (Stage_preprocess.parallel_generatedRoot_211.constraints_9.bridge h12), (Stage_preprocess.parallel_generatedRoot_211.constraints_10.bridge h13), (Stage_preprocess.parallel_generatedRoot_211.constraints_11.bridge h14), (Stage_preprocess.parallel_generatedRoot_211.constraints_12.bridge h15), (Stage_preprocess.parallel_generatedRoot_211.constraints_13.bridge h16), (Stage_preprocess.parallel_generatedRoot_211.constraints_14.bridge h17)⟩)⟩)


theorem Stage_preprocess.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_211_1 w_211_2 w_211_3 w_211_4 w_211_5 w_211_6 w_211_7 w_211_8 w_211_9 w_211_10 w_211_11 w_211_12 w_211_13 w_211_14 w_211_15 w_211_16 w_211_17 w_211_18 w_211_19 w_211_20 w_211_21 w_211_22 w_211_23 w_211_24 w_211_25 w_211_26 w_211_27 w_211_28 w_211_29 w_211_30 w_211_31 w_211_32 w_211_33 w_211_34 w_211_35 w_211_36 w_211_37 w_211_38 w_211_39 w_211_40 w_211_41 w_211_42 w_211_43 w_211_44 w_211_45 w_211_46 w_211_47 w_211_48 w_211_49 w_211_50 w_211_51 w_211_52 w_211_53 w_211_54 w_211_55 w_211_56 w_211_57 w_211_58 w_211_59 w_211_60 w_211_61 w_211_62 w_211_63 w_211_64 w_211_65 w_211_66 w_211_67 w_211_68 w_211_69 w_211_70 w_211_71 w_211_72 w_211_73 w_211_74 w_211_75 w_211_76 w_211_77 w_211_78 w_211_79 w_211_80 w_211_81 w_211_82 w_211_83 w_211_84 w_211_85 w_211_86 w_211_87 w_211_88 w_211_89 w_211_90 w_211_91 w_211_92 w_211_93 w_211_94 w_211_95 w_211_96 w_211_97 w_211_98 w_211_99 w_211_100 w_211_101 w_211_102 w_211_103 w_211_104 w_211_105 w_211_106 w_211_107 w_211_108 w_211_109 w_211_110 w_211_111 w_211_112 w_211_113 w_211_114 w_211_115 w_211_116 w_211_117 w_211_118 w_211_119 w_211_120 w_211_121 w_211_122 w_211_123 w_211_124 w_211_125 w_211_126 w_211_127 w_211_128 w_211_129 w_211_130 w_211_131 w_211_132 w_211_133 w_211_134 w_211_135 w_211_136 w_211_137 w_211_138 w_211_139 w_211_140 w_211_141 w_211_142 w_211_143 w_211_144 w_211_145 w_211_146 w_211_147 w_211_148 w_211_149 w_211_150 w_211_151 w_211_152 w_211_153 w_211_154 w_211_155 w_211_156 w_211_157 w_211_158 w_211_159 w_211_160 w_211_161 w_211_162 w_211_163 w_211_164 w_211_165 w_211_166 w_211_167 w_211_168 w_211_169 w_211_170 w_211_171 w_211_172 w_211_173 w_211_174 w_211_175 w_211_176 w_211_177 w_211_178 w_211_179 w_211_180 w_211_181 w_211_182 w_211_183 w_211_184 w_211_185 w_211_186 w_211_187 w_211_188 w_211_189 w_211_190 w_211_191 w_211_192 w_211_193 w_211_194 w_211_195 w_211_196 w_211_197 w_211_198 w_211_199 w_211_200 w_211_201 w_211_202 w_211_203 w_211_204 w_211_205 w_211_206 w_211_207 w_211_208 w_211_209 w_211_210 w_211_211 w_211_212 w_211_213 w_211_214 w_211_215 w_211_216 w_211_217 w_211_218 w_211_219 w_211_220 w_211_221 w_211_222 w_211_223 w_211_224 w_211_225 w_211_226 w_211_227 w_211_228 w_211_229 w_211_230 w_211_231 w_211_232 w_211_233 w_211_234 w_211_235 w_211_236 w_211_237 w_211_238 w_211_239 w_211_240 w_211_241 outputs}
    (h : Stage_preprocess.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_211_1 w_211_2 w_211_3 w_211_4 w_211_5 w_211_6 w_211_7 w_211_8 w_211_9 w_211_10 w_211_11 w_211_12 w_211_13 w_211_14 w_211_15 w_211_16 w_211_17 w_211_18 w_211_19 w_211_20 w_211_21 w_211_22 w_211_23 w_211_24 w_211_25 w_211_26 w_211_27 w_211_28 w_211_29 w_211_30 w_211_31 w_211_32 w_211_33 w_211_34 w_211_35 w_211_36 w_211_37 w_211_38 w_211_39 w_211_40 w_211_41 w_211_42 w_211_43 w_211_44 w_211_45 w_211_46 w_211_47 w_211_48 w_211_49 w_211_50 w_211_51 w_211_52 w_211_53 w_211_54 w_211_55 w_211_56 w_211_57 w_211_58 w_211_59 w_211_60 w_211_61 w_211_62 w_211_63 w_211_64 w_211_65 w_211_66 w_211_67 w_211_68 w_211_69 w_211_70 w_211_71 w_211_72 w_211_73 w_211_74 w_211_75 w_211_76 w_211_77 w_211_78 w_211_79 w_211_80 w_211_81 w_211_82 w_211_83 w_211_84 w_211_85 w_211_86 w_211_87 w_211_88 w_211_89 w_211_90 w_211_91 w_211_92 w_211_93 w_211_94 w_211_95 w_211_96 w_211_97 w_211_98 w_211_99 w_211_100 w_211_101 w_211_102 w_211_103 w_211_104 w_211_105 w_211_106 w_211_107 w_211_108 w_211_109 w_211_110 w_211_111 w_211_112 w_211_113 w_211_114 w_211_115 w_211_116 w_211_117 w_211_118 w_211_119 w_211_120 w_211_121 w_211_122 w_211_123 w_211_124 w_211_125 w_211_126 w_211_127 w_211_128 w_211_129 w_211_130 w_211_131 w_211_132 w_211_133 w_211_134 w_211_135 w_211_136 w_211_137 w_211_138 w_211_139 w_211_140 w_211_141 w_211_142 w_211_143 w_211_144 w_211_145 w_211_146 w_211_147 w_211_148 w_211_149 w_211_150 w_211_151 w_211_152 w_211_153 w_211_154 w_211_155 w_211_156 w_211_157 w_211_158 w_211_159 w_211_160 w_211_161 w_211_162 w_211_163 w_211_164 w_211_165 w_211_166 w_211_167 w_211_168 w_211_169 w_211_170 w_211_171 w_211_172 w_211_173 w_211_174 w_211_175 w_211_176 w_211_177 w_211_178 w_211_179 w_211_180 w_211_181 w_211_182 w_211_183 w_211_184 w_211_185 w_211_186 w_211_187 w_211_188 w_211_189 w_211_190 w_211_191 w_211_192 w_211_193 w_211_194 w_211_195 w_211_196 w_211_197 w_211_198 w_211_199 w_211_200 w_211_201 w_211_202 w_211_203 w_211_204 w_211_205 w_211_206 w_211_207 w_211_208 w_211_209 w_211_210 w_211_211 w_211_212 w_211_213 w_211_214 w_211_215 w_211_216 w_211_217 w_211_218 w_211_219 w_211_220 w_211_221 w_211_222 w_211_223 w_211_224 w_211_225 w_211_226 w_211_227 w_211_228 w_211_229 w_211_230 w_211_231 w_211_232 w_211_233 w_211_234 w_211_235 w_211_236 w_211_237 w_211_238 w_211_239 w_211_240 w_211_241 outputs) : W_Stage_preprocess.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_211_1 w_211_2 w_211_3 w_211_4 w_211_5 w_211_6 w_211_7 w_211_8 w_211_9 w_211_10 w_211_11 w_211_12 w_211_13 w_211_14 w_211_15 w_211_16 w_211_17 w_211_18 w_211_19 w_211_20 w_211_21 w_211_22 w_211_23 w_211_24 w_211_25 w_211_26 w_211_27 w_211_28 w_211_29 w_211_30 w_211_31 w_211_32 w_211_33 w_211_34 w_211_35 w_211_36 w_211_37 w_211_38 w_211_39 w_211_40 w_211_41 w_211_42 w_211_43 w_211_44 w_211_45 w_211_46 w_211_47 w_211_48 w_211_49 w_211_50 w_211_51 w_211_52 w_211_53 w_211_54 w_211_55 w_211_56 w_211_57 w_211_58 w_211_59 w_211_60 w_211_61 w_211_62 w_211_63 w_211_64 w_211_65 w_211_66 w_211_67 w_211_68 w_211_69 w_211_70 w_211_71 w_211_72 w_211_73 w_211_74 w_211_75 w_211_76 w_211_77 w_211_78 w_211_79 w_211_80 w_211_81 w_211_82 w_211_83 w_211_84 w_211_85 w_211_86 w_211_87 w_211_88 w_211_89 w_211_90 w_211_91 w_211_92 w_211_93 w_211_94 w_211_95 w_211_96 w_211_97 w_211_98 w_211_99 w_211_100 w_211_101 w_211_102 w_211_103 w_211_104 w_211_105 w_211_106 w_211_107 w_211_108 w_211_109 w_211_110 w_211_111 w_211_112 w_211_113 w_211_114 w_211_115 w_211_116 w_211_117 w_211_118 w_211_119 w_211_120 w_211_121 w_211_122 w_211_123 w_211_124 w_211_125 w_211_126 w_211_127 w_211_128 w_211_129 w_211_130 w_211_131 w_211_132 w_211_133 w_211_134 w_211_135 w_211_136 w_211_137 w_211_138 w_211_139 w_211_140 w_211_141 w_211_142 w_211_143 w_211_144 w_211_145 w_211_146 w_211_147 w_211_148 w_211_149 w_211_150 w_211_151 w_211_152 w_211_153 w_211_154 w_211_155 w_211_156 w_211_157 w_211_158 w_211_159 w_211_160 w_211_161 w_211_162 w_211_163 w_211_164 w_211_165 w_211_166 w_211_167 w_211_168 w_211_169 w_211_170 w_211_171 w_211_172 w_211_173 w_211_174 w_211_175 w_211_176 w_211_177 w_211_178 w_211_179 w_211_180 w_211_181 w_211_182 w_211_183 w_211_184 w_211_185 w_211_186 w_211_187 w_211_188 w_211_189 w_211_190 w_211_191 w_211_192 w_211_193 w_211_194 w_211_195 w_211_196 w_211_197 w_211_198 w_211_199 w_211_200 w_211_201 w_211_202 w_211_203 w_211_204 w_211_205 w_211_206 w_211_207 w_211_208 w_211_209 w_211_210 w_211_211 w_211_212 w_211_213 w_211_214 w_211_215 w_211_216 w_211_217 w_211_218 w_211_219 w_211_220 w_211_221 w_211_222 w_211_223 w_211_224 w_211_225 w_211_226 w_211_227 w_211_228 w_211_229 w_211_230 w_211_231 w_211_232 w_211_233 w_211_234 w_211_235 w_211_236 w_211_237 w_211_238 w_211_239 w_211_240 w_211_241 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (fun i ↦ (Stage_preprocess.parallel_generatedRoot_211.bridge (h2 i))), h3⟩)


theorem Stage_preprocess.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_preprocess.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_preprocess.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_preprocess.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_211_0, wit1.w_211_1, wit1.w_211_2, wit1.w_211_3, wit1.w_211_4, wit1.w_211_5, wit1.w_211_6, wit1.w_211_7, wit1.w_211_8, wit1.w_211_9, wit1.w_211_10, wit1.w_211_11, wit1.w_211_12, wit1.w_211_13, wit1.w_211_14, wit1.w_211_15, wit1.w_211_16, wit1.w_211_17, wit1.w_211_18, wit1.w_211_19, wit1.w_211_20, wit1.w_211_21, wit1.w_211_22, wit1.w_211_23, wit1.w_211_24, wit1.w_211_25, wit1.w_211_26, wit1.w_211_27, wit1.w_211_28, wit1.w_211_29, wit1.w_211_30, wit1.w_211_31, wit1.w_211_32, wit1.w_211_33, wit1.w_211_34, wit1.w_211_35, wit1.w_211_36, wit1.w_211_37, wit1.w_211_38, wit1.w_211_39, wit1.w_211_40, wit1.w_211_41, wit1.w_211_42, wit1.w_211_43, wit1.w_211_44, wit1.w_211_45, wit1.w_211_46, wit1.w_211_47, wit1.w_211_48, wit1.w_211_49, wit1.w_211_50, wit1.w_211_51, wit1.w_211_52, wit1.w_211_53, wit1.w_211_54, wit1.w_211_55, wit1.w_211_56, wit1.w_211_57, wit1.w_211_58, wit1.w_211_59, wit1.w_211_60, wit1.w_211_61, wit1.w_211_62, wit1.w_211_63, wit1.w_211_64, wit1.w_211_65, wit1.w_211_66, wit1.w_211_67, wit1.w_211_68, wit1.w_211_69, wit1.w_211_70, wit1.w_211_71, wit1.w_211_72, wit1.w_211_73, wit1.w_211_74, wit1.w_211_75, wit1.w_211_76, wit1.w_211_77, wit1.w_211_78, wit1.w_211_79, wit1.w_211_80, wit1.w_211_81, wit1.w_211_82, wit1.w_211_83, wit1.w_211_84, wit1.w_211_85, wit1.w_211_86, wit1.w_211_87, wit1.w_211_88, wit1.w_211_89, wit1.w_211_90, wit1.w_211_91, wit1.w_211_92, wit1.w_211_93, wit1.w_211_94, wit1.w_211_95, wit1.w_211_96, wit1.w_211_97, wit1.w_211_98, wit1.w_211_99, wit1.w_211_100, wit1.w_211_101, wit1.w_211_102, wit1.w_211_103, wit1.w_211_104, wit1.w_211_105, wit1.w_211_106, wit1.w_211_107, wit1.w_211_108, wit1.w_211_109, wit1.w_211_110, wit1.w_211_111, wit1.w_211_112, wit1.w_211_113, wit1.w_211_114, wit1.w_211_115, wit1.w_211_116, wit1.w_211_117, wit1.w_211_118, wit1.w_211_119, wit1.w_211_120, wit1.w_211_121, wit1.w_211_122, wit1.w_211_123, wit1.w_211_124, wit1.w_211_125, wit1.w_211_126, wit1.w_211_127, wit1.w_211_128, wit1.w_211_129, wit1.w_211_130, wit1.w_211_131, wit1.w_211_132, wit1.w_211_133, wit1.w_211_134, wit1.w_211_135, wit1.w_211_136, wit1.w_211_137, wit1.w_211_138, wit1.w_211_139, wit1.w_211_140, wit1.w_211_141, wit1.w_211_142, wit1.w_211_143, wit1.w_211_144, wit1.w_211_145, wit1.w_211_146, wit1.w_211_147, wit1.w_211_148, wit1.w_211_149, wit1.w_211_150, wit1.w_211_151, wit1.w_211_152, wit1.w_211_153, wit1.w_211_154, wit1.w_211_155, wit1.w_211_156, wit1.w_211_157, wit1.w_211_158, wit1.w_211_159, wit1.w_211_160, wit1.w_211_161, wit1.w_211_162, wit1.w_211_163, wit1.w_211_164, wit1.w_211_165, wit1.w_211_166, wit1.w_211_167, wit1.w_211_168, wit1.w_211_169, wit1.w_211_170, wit1.w_211_171, wit1.w_211_172, wit1.w_211_173, wit1.w_211_174, wit1.w_211_175, wit1.w_211_176, wit1.w_211_177, wit1.w_211_178, wit1.w_211_179, wit1.w_211_180, wit1.w_211_181, wit1.w_211_182, wit1.w_211_183, wit1.w_211_184, wit1.w_211_185, wit1.w_211_186, wit1.w_211_187, wit1.w_211_188, wit1.w_211_189, wit1.w_211_190, wit1.w_211_191, wit1.w_211_192, wit1.w_211_193, wit1.w_211_194, wit1.w_211_195, wit1.w_211_196, wit1.w_211_197, wit1.w_211_198, wit1.w_211_199, wit1.w_211_200, wit1.w_211_201, wit1.w_211_202, wit1.w_211_203, wit1.w_211_204, wit1.w_211_205, wit1.w_211_206, wit1.w_211_207, wit1.w_211_208, wit1.w_211_209, wit1.w_211_210, wit1.w_211_211, wit1.w_211_212, wit1.w_211_213, wit1.w_211_214, wit1.w_211_215, wit1.w_211_216, wit1.w_211_217, wit1.w_211_218, wit1.w_211_219, wit1.w_211_220, wit1.w_211_221, wit1.w_211_222, wit1.w_211_223, wit1.w_211_224, wit1.w_211_225, wit1.w_211_226, wit1.w_211_227, wit1.w_211_228, wit1.w_211_229, wit1.w_211_230, wit1.w_211_231, wit1.w_211_232, wit1.w_211_233, wit1.w_211_234, wit1.w_211_235, wit1.w_211_236, wit1.w_211_237, wit1.w_211_238, wit1.w_211_239, wit1.w_211_240, wit1.w_211_241⟩, (Stage_preprocess.generatedRoot.constraints_0.bridge h2)⟩)
