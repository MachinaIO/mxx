import Stage_encrypt
import W_Stage_encrypt
import ExponentSampling

/-! Every run of the average-case relations of `Stage_encrypt` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_encrypt.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_42.constraints_0.bridge {backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_42.constraints_0 backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_42.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, (MxxExponentLut.gaussianSampleAt_sample (by decide) h2), (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_42.bridge {backend x1 tape path params params' x5 x6 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_42 backend x1 tape path params x5 x6 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_42 backend x1 params' x5 x6 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_42
  exact (by obtain ⟨w_13_0, w_17_0, w_24_0, h1⟩ := h; exact ⟨w_13_0, w_17_0, w_24_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_42.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_44.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_44.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_44.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_44.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_44 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_44 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_44
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_44.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_45.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_45.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_45.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_45.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_45 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_45 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_45
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_45.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_46.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_46.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_46.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_46.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_46 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_46 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_46
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_46.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_47.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_47.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_47.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_47.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_47 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_47 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_47
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_47.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_48.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_48.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_48.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_48.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_48 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_48 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_48
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_48.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_49.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_49.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_49.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_49.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_49 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_49 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_49
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_49.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_50.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_50.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_50.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_50.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_50 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_50 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_50
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_50.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_51.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_51.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_51.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_51.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_51 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_51 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_51
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_51.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_52.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_52.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_52.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_52.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_52 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_52 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_52
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_52.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_53.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_53.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_53.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_53.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_53 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_53 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_53
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_53.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_54.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_54.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_54.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_54.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_54 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_54 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_54
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_54.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_55.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_55.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_55.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_55.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_55 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_55 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_55
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_55.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_56.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_56.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_56.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_56.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_56 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_56 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_56
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_56.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_57.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_57.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_57.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_57.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_57 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_57 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_57
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_57.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_58.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_58.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_58.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_58.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_58 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_58 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_58
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_58.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_59.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_59.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_59.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_59.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_59 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_59 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_59
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_59.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_60.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_60.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_60.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_60.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_60 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_60 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_60
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_60.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_61.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_61.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_61.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_61.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_61 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_61 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_61
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_61.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_62.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_62.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_62.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_62.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_62 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_62 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_62
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_62.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_63.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_63.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_63.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_63.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_63 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_63 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_63
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_63.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_64.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_64.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_64.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_64.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_64 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_64 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_64
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_64.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_65.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_65.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_65.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_65.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_65 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_65 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_65
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_65.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_66.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_66.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_66.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_66.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_66 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_66 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_66
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_66.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_67.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_67.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_67.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_67.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_67 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_67 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_67
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_67.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_68.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_68.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_68.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_68.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_68 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_68 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_68
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_68.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_69.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_69.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_69.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_69.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_69 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_69 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_69
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_69.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_70.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_70.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_70.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_70.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_70 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_70 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_70
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_70.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_71.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_71.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_71.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_71.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_71 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_71 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_71
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_71.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_72.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_72.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_72.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_72.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_72 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_72 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_72
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_72.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_73.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_73.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_73.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_73.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_73 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_73 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_73
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_73.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_74.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_74.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_74.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_74.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_74 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_74 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_74
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_74.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_75.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_75.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_75.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_75.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_75 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_75 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_75
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_75.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_76.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_76.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_76.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_76.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_76 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_76 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_76
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_76.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_77.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_77.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_77.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_77.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_77 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_77 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_77
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_77.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_78.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_78.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_78.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_78.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_78 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_78 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_78
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_78.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_108.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_108.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_108.constraints_0 backend hashModel i_0 i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_108.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_108 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_108 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_108
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_108.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_109.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_109.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_109.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_109.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_109 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_109 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_109
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_109.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_148.constraints_0.bridge {backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_148.constraints_0 backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_148.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, (MxxExponentLut.gaussianSampleAt_sample (by decide) h2), (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_148.bridge {backend x1 tape path params params' x5 x6 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_148 backend x1 tape path params x5 x6 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_148 backend x1 params' x5 x6 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_148
  exact (by obtain ⟨w_13_0, w_17_0, w_24_0, h1⟩ := h; exact ⟨w_13_0, w_17_0, w_24_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_148.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_150.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_150.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_150.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_150.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_150 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_150 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_150
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_150.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_151.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_151.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_151.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_151.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_151 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_151 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_151
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_151.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_152.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_152.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_152.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_152.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_152 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_152 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_152
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_152.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_153.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_153.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_153.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_153.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_153 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_153 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_153
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_153.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_154.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_154.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_154.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_154.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_154 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_154 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_154
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_154.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_155.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_155.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_155.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_155.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_155 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_155 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_155
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_155.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_156.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_156.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_156.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_156.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_156 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_156 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_156
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_156.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_157.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_157.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_157.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_157.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_157 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_157 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_157
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_157.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_158.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_158.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_158.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_158.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_158 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_158 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_158
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_158.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_159.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_159.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_159.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_159.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_159 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_159 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_159
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_159.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_160.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_160.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_160.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_160.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_160 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_160 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_160
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_160.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_161.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_161.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_161.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_161.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_161 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_161 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_161
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_161.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_162.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_162.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_162.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_162.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_162 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_162 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_162
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_162.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_163.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_163.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_163.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_163.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_163 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_163 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_163
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_163.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_164.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_164.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_164.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_164.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_164 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_164 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_164
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_164.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_165.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_165.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_165.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_165.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_165 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_165 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_165
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_165.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_166.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_166.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_166.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_166.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_166 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_166 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_166
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_166.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_167.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_167.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_167.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_167.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_167 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_167 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_167
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_167.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_168.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_168.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_168.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_168.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_168 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_168 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_168
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_168.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_169.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_169.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_169.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_169.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_169 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_169 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_169
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_169.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_170.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_170.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_170.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_170.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_170 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_170 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_170
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_170.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_171.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_171.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_171.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_171.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_171 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_171 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_171
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_171.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_172.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_172.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_172.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_172.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_172 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_172 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_172
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_172.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_173.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_173.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_173.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_173.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_173 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_173 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_173
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_173.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_174.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_174.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_174.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_174.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_174 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_174 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_174
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_174.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_175.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_175.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_175.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_175.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_175 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_175 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_175
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_175.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_176.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_176.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_176.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_176.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_176 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_176 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_176
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_176.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_177.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_177.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_177.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_177.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_177 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_177 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_177
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_177.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_178.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_178.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_178.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_178.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_178 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_178 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_178
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_178.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_179.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_179.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_179.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_179.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_179 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_179 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_179
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_179.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_180.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_180.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_180.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_180.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_180 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_180 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_180
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_180.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_181.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_181.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_181.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_181.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_181 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_181 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_181
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_181.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_182.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_182.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_182.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_182.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_182 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_182 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_182
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_182.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_183.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_183.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_183.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_183.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_183 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_183 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_183
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_183.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_184.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_184.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_184.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_184.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_184 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_184 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_184
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_184.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_185.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_185.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_185.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_185.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_185 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_185 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_185
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_185.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_186.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_186.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_186.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_186.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_186 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_186 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_186
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_186.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_187.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_187.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_187.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_187.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_187 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_187 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_187
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_187.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_188.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_188.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_188.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_188.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_188 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_188 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_188
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_188.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_189.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_189.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_189.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_189.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_189 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_189 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_189
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_189.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_190.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_190.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_190.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_190.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_190 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_190 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_190
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_190.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_191.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_191.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_191.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_191.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_191 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_191 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_191
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_191.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_192.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_192.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_192.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_192.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_192 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_192 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_192
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_192.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_193.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_193.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_193.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_193.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_193 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_193 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_193
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_193.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_194.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_194.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_194.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_194.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_194 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_194 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_194
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_194.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_195.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_195.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_195.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_195.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_195 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_195 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_195
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_195.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_196.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_196.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_196.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_196.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_196 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_196 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_196
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_196.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_197.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_197.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_197.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_197.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_197 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_197 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_197
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_197.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_198.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_198.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_198.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_198.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_198 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_198 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_198
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_198.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_199.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_199.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_199.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_199.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_199 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_199 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_199
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_199.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_200.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_200.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_200.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_200.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_200 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_200 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_200
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_200.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_201.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_201.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_201.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_201.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_201 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_201 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_201
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_201.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_202.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_202.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_202.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_202.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_202 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_202 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_202
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_202.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_203.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_203.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_203.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_203.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_203 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_203 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_203
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_203.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_204.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_204.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_204.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_204.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_204 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_204 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_204
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_204.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_205.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_205.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_205.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_205.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_205 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_205 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_205
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_205.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_206.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_206.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_206.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_206.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_206 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_206 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_206
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_206.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_207.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_207.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_207.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_207.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_207 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_207 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_207
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_207.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_208.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_208.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_208.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_208.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_208 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_208 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_208
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_208.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_209.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_209.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_209.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_209.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_209 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_209 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_209
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_209.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_210.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_210.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_210.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_210.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_210 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_210 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_210
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_210.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_211.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_211.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_211.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_211.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_211 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_211 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_211
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_211.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_212.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_212.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_212.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_212.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_212 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_212 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_212
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_212.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_213.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_213.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_213.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_213.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_213 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_213 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_213
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_213.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_214.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_214.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_214.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_214.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_214 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_214 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_214
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_214.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_215.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_215.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_215.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_215.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_215 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_215 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_215
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_215.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_216.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_216.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_216.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_216.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_216 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_216 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_216
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_216.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_217.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_217.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_217.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_217.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_217 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_217 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_217
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_217.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_218.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_218.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_218.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_218.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_218 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_218 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_218
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_218.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_219.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_219.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_219.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_219.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_219 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_219 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_219
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_219.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_220.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_220.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_220.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_220.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_220 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_220 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_220
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_220.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_221.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_221.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_221.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_32_221.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_32_221 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_32_221 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_32_221
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_32_221.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_8.bridge {backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_79_0 w_110_0 w_111_0 w_112_0 w_113_0 w_7_0 w_25_0 w_35_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_97_0 w_107_0 w_108_0 w_108_1 w_109_0 w_109_1 w_132_0 w_141_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_230_0 w_234_0 w_235_0 w_236_0 w_238_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_250_0 w_251_0 outputs}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_8 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_79_0 w_110_0 w_111_0 w_112_0 w_113_0 w_7_0 w_25_0 w_35_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_97_0 w_107_0 w_108_0 w_108_1 w_109_0 w_109_1 w_132_0 w_141_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_230_0 w_234_0 w_235_0 w_236_0 w_238_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_250_0 w_251_0 outputs) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_8 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_79_0 w_110_0 w_111_0 w_112_0 w_113_0 w_7_0 w_25_0 w_35_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_97_0 w_107_0 w_108_0 w_108_1 w_109_0 w_109_1 w_132_0 w_141_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_230_0 w_234_0 w_235_0 w_236_0 w_238_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_250_0 w_251_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14⟩ := h; exact ⟨h1, (MxxExponentLut.gaussianSampleAt_sample (by decide) h2), h3, h4, h5, h6, h7, h8, (MxxExponentLut.gaussianSampleAt_sample (by decide) h9), h10, h11, h12, (MxxExponentLut.gaussianSampleAt_sample (by decide) h13), h14⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_7.bridge {backend hashModel tape path params params' i_0 w_4_0 w_79_0 w_149_0 w_7_0 w_114_0 w_116_0 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_222_0 w_223_0 w_225_0 w_226_0}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_7 backend hashModel tape path params i_0 w_4_0 w_79_0 w_149_0 w_7_0 w_114_0 w_116_0 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_222_0 w_223_0 w_225_0 w_226_0) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_7 backend hashModel params' i_0 w_4_0 w_79_0 w_149_0 w_7_0 w_114_0 w_116_0 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_222_0 w_223_0 w_225_0 w_226_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_210.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_211.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_212.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_213.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_214.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_215.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_216.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_217.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_218.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_219.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_220.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_221.bridge (h24 i))), h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_6.bridge {backend hashModel tape path params params' i_0 w_149_0 w_114_0 w_116_0 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_6 backend hashModel tape path params i_0 w_149_0 w_114_0 w_116_0 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_6 backend hashModel params' i_0 w_149_0 w_114_0 w_116_0 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_194.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_195.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_196.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_197.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_198.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_199.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_200.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_201.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_202.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_203.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_204.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_205.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_206.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_207.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_208.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_209.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_5.bridge {backend hashModel tape path params params' i_0 w_149_0 w_114_0 w_116_0 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_5 backend hashModel tape path params i_0 w_149_0 w_114_0 w_116_0 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_5 backend hashModel params' i_0 w_149_0 w_114_0 w_116_0 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_178.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_179.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_180.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_181.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_182.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_183.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_184.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_185.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_186.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_187.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_188.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_189.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_190.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_191.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_192.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_193.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_4.bridge {backend hashModel tape path params params' i_0 w_149_0 w_114_0 w_116_0 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_4 backend hashModel tape path params i_0 w_149_0 w_114_0 w_116_0 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_4 backend hashModel params' i_0 w_149_0 w_114_0 w_116_0 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_162.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_163.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_164.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_165.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_166.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_167.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_168.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_169.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_170.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_171.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_172.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_173.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_174.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_175.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_176.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_177.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_3.bridge {backend hashModel tape path params params' i_0 w_11_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_29_0 w_114_0 w_116_0 w_119_0 w_120_0 w_121_0 w_131_0 w_136_0 w_140_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_3 backend hashModel tape path params i_0 w_11_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_29_0 w_114_0 w_116_0 w_119_0 w_120_0 w_121_0 w_131_0 w_136_0 w_140_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_3 backend hashModel params' i_0 w_11_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_29_0 w_114_0 w_116_0 w_119_0 w_120_0 w_121_0 w_131_0 w_136_0 w_140_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, (MxxExponentLut.gaussianSampleAt_sample (by decide) h4), h5, (MxxExponentLut.gaussianSampleAt_sample (by decide) h6), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_148.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_150.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_151.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_152.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_153.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_154.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_155.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_156.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_157.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_158.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_159.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_160.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_161.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_2.bridge {backend hashModel tape path params params' i_0 w_4_0 w_43_0 w_79_0 w_5_0 w_7_0 w_8_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_80_0 w_82_0 w_85_0 w_86_0 w_96_0 w_98_0 w_100_0 w_106_0 w_108_0 w_108_1 w_109_0 w_109_1 w_114_0 w_116_0}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_2 backend hashModel tape path params i_0 w_4_0 w_43_0 w_79_0 w_5_0 w_7_0 w_8_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_80_0 w_82_0 w_85_0 w_86_0 w_96_0 w_98_0 w_100_0 w_106_0 w_108_0 w_108_1 w_109_0 w_109_1 w_114_0 w_116_0) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_2 backend hashModel params' i_0 w_4_0 w_43_0 w_79_0 w_5_0 w_7_0 w_8_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_80_0 w_82_0 w_85_0 w_86_0 w_96_0 w_98_0 w_100_0 w_106_0 w_108_0 w_108_1 w_109_0 w_109_1 w_114_0 w_116_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_71.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_72.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_73.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_74.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_75.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_76.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_77.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_78.bridge (h16 i))), h17, h18, h19, h20, h21, h22, (MxxExponentLut.gaussianSampleAt_sample (by decide) h23), h24, h25, (MxxExponentLut.gaussianSampleAt_sample (by decide) h26), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_108.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_109.bridge (h30 i))), h31, h32⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_1.bridge {backend hashModel tape path params params' i_0 w_43_0 w_5_0 w_8_0 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_1 backend hashModel tape path params i_0 w_43_0 w_5_0 w_8_0 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_1 backend hashModel params' i_0 w_43_0 w_5_0 w_8_0 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_55.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_56.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_57.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_58.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_59.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_60.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_61.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_62.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_63.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_64.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_65.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_66.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_67.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_68.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_69.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_70.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.constraints_0.bridge {backend hashModel tape path params params' i_0 w_4_0 w_11_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_14_0 w_24_0 w_29_0 w_30_0 w_34_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1}
    (h : Stage_encrypt.parallel_generatedRoot_32.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_14_0 w_24_0 w_29_0 w_30_0 w_34_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1) : W_Stage_encrypt.parallel_generatedRoot_32.constraints_0 backend hashModel params' i_0 w_4_0 w_11_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_14_0 w_24_0 w_29_0 w_30_0 w_34_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.gaussianSampleAt_sample (by decide) h6), h7, (MxxExponentLut.gaussianSampleAt_sample (by decide) h8), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_42.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_44.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_45.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_46.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_47.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_48.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_49.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_50.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_51.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_52.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_53.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_32_54.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_32.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_encrypt.parallel_generatedRoot_32 backend hashModel tape path params i_0 inputs outputs) : W_Stage_encrypt.parallel_generatedRoot_32 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_encrypt.parallel_generatedRoot_32
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0, wit1.w_8_0, wit1.w_12_0, wit1.w_13_0, wit1.w_14_0, wit1.w_24_0, wit1.w_30_0, wit1.w_34_0, wit1.w_42_0, wit1.w_42_1, wit1.w_42_2, wit1.w_42_3, wit1.w_42_4, wit1.w_42_5, wit1.w_42_6, wit1.w_44_0, wit1.w_44_1, wit1.w_45_0, wit1.w_45_1, wit1.w_46_0, wit1.w_46_1, wit1.w_47_0, wit1.w_47_1, wit1.w_48_0, wit1.w_48_1, wit1.w_49_0, wit1.w_49_1, wit1.w_50_0, wit1.w_50_1, wit1.w_51_0, wit1.w_51_1, wit1.w_52_0, wit1.w_52_1, wit1.w_53_0, wit1.w_53_1, wit1.w_54_0, wit1.w_54_1, wit1.w_55_0, wit1.w_55_1, wit1.w_56_0, wit1.w_56_1, wit1.w_57_0, wit1.w_57_1, wit1.w_58_0, wit1.w_58_1, wit1.w_59_0, wit1.w_59_1, wit1.w_60_0, wit1.w_60_1, wit1.w_61_0, wit1.w_61_1, wit1.w_62_0, wit1.w_62_1, wit1.w_63_0, wit1.w_63_1, wit1.w_64_0, wit1.w_64_1, wit1.w_65_0, wit1.w_65_1, wit1.w_66_0, wit1.w_66_1, wit1.w_67_0, wit1.w_67_1, wit1.w_68_0, wit1.w_68_1, wit1.w_69_0, wit1.w_69_1, wit1.w_70_0, wit1.w_70_1, wit1.w_71_0, wit1.w_71_1, wit1.w_72_0, wit1.w_72_1, wit1.w_73_0, wit1.w_73_1, wit1.w_74_0, wit1.w_74_1, wit1.w_75_0, wit1.w_75_1, wit1.w_76_0, wit1.w_76_1, wit1.w_77_0, wit1.w_77_1, wit1.w_78_0, wit1.w_78_1, wit1.w_80_0, wit1.w_82_0, wit1.w_85_0, wit1.w_86_0, wit1.w_96_0, wit1.w_98_0, wit1.w_100_0, wit1.w_106_0, wit1.w_108_0, wit1.w_108_1, wit1.w_109_0, wit1.w_109_1, wit1.w_114_0, wit1.w_116_0, wit1.w_119_0, wit1.w_120_0, wit1.w_121_0, wit1.w_131_0, wit1.w_136_0, wit1.w_140_0, wit1.w_148_0, wit1.w_148_1, wit1.w_148_2, wit1.w_148_3, wit1.w_148_4, wit1.w_148_5, wit1.w_148_6, wit1.w_150_0, wit1.w_150_1, wit1.w_151_0, wit1.w_151_1, wit1.w_152_0, wit1.w_152_1, wit1.w_153_0, wit1.w_153_1, wit1.w_154_0, wit1.w_154_1, wit1.w_155_0, wit1.w_155_1, wit1.w_156_0, wit1.w_156_1, wit1.w_157_0, wit1.w_157_1, wit1.w_158_0, wit1.w_158_1, wit1.w_159_0, wit1.w_159_1, wit1.w_160_0, wit1.w_160_1, wit1.w_161_0, wit1.w_161_1, wit1.w_162_0, wit1.w_162_1, wit1.w_163_0, wit1.w_163_1, wit1.w_164_0, wit1.w_164_1, wit1.w_165_0, wit1.w_165_1, wit1.w_166_0, wit1.w_166_1, wit1.w_167_0, wit1.w_167_1, wit1.w_168_0, wit1.w_168_1, wit1.w_169_0, wit1.w_169_1, wit1.w_170_0, wit1.w_170_1, wit1.w_171_0, wit1.w_171_1, wit1.w_172_0, wit1.w_172_1, wit1.w_173_0, wit1.w_173_1, wit1.w_174_0, wit1.w_174_1, wit1.w_175_0, wit1.w_175_1, wit1.w_176_0, wit1.w_176_1, wit1.w_177_0, wit1.w_177_1, wit1.w_178_0, wit1.w_178_1, wit1.w_179_0, wit1.w_179_1, wit1.w_180_0, wit1.w_180_1, wit1.w_181_0, wit1.w_181_1, wit1.w_182_0, wit1.w_182_1, wit1.w_183_0, wit1.w_183_1, wit1.w_184_0, wit1.w_184_1, wit1.w_185_0, wit1.w_185_1, wit1.w_186_0, wit1.w_186_1, wit1.w_187_0, wit1.w_187_1, wit1.w_188_0, wit1.w_188_1, wit1.w_189_0, wit1.w_189_1, wit1.w_190_0, wit1.w_190_1, wit1.w_191_0, wit1.w_191_1, wit1.w_192_0, wit1.w_192_1, wit1.w_193_0, wit1.w_193_1, wit1.w_194_0, wit1.w_194_1, wit1.w_195_0, wit1.w_195_1, wit1.w_196_0, wit1.w_196_1, wit1.w_197_0, wit1.w_197_1, wit1.w_198_0, wit1.w_198_1, wit1.w_199_0, wit1.w_199_1, wit1.w_200_0, wit1.w_200_1, wit1.w_201_0, wit1.w_201_1, wit1.w_202_0, wit1.w_202_1, wit1.w_203_0, wit1.w_203_1, wit1.w_204_0, wit1.w_204_1, wit1.w_205_0, wit1.w_205_1, wit1.w_206_0, wit1.w_206_1, wit1.w_207_0, wit1.w_207_1, wit1.w_208_0, wit1.w_208_1, wit1.w_209_0, wit1.w_209_1, wit1.w_210_0, wit1.w_210_1, wit1.w_211_0, wit1.w_211_1, wit1.w_212_0, wit1.w_212_1, wit1.w_213_0, wit1.w_213_1, wit1.w_214_0, wit1.w_214_1, wit1.w_215_0, wit1.w_215_1, wit1.w_216_0, wit1.w_216_1, wit1.w_217_0, wit1.w_217_1, wit1.w_218_0, wit1.w_218_1, wit1.w_219_0, wit1.w_219_1, wit1.w_220_0, wit1.w_220_1, wit1.w_221_0, wit1.w_221_1, wit1.w_222_0, wit1.w_223_0, wit1.w_225_0, wit1.w_226_0, wit1.w_230_0, wit1.w_234_0, wit1.w_236_0, wit1.w_238_0, wit1.w_243_0, wit1.w_245_0, wit1.w_246_0, wit1.w_247_0, wit1.w_250_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := h2; exact ⟨(Stage_encrypt.parallel_generatedRoot_32.constraints_0.bridge h3), (Stage_encrypt.parallel_generatedRoot_32.constraints_1.bridge h4), (Stage_encrypt.parallel_generatedRoot_32.constraints_2.bridge h5), (Stage_encrypt.parallel_generatedRoot_32.constraints_3.bridge h6), (Stage_encrypt.parallel_generatedRoot_32.constraints_4.bridge h7), (Stage_encrypt.parallel_generatedRoot_32.constraints_5.bridge h8), (Stage_encrypt.parallel_generatedRoot_32.constraints_6.bridge h9), (Stage_encrypt.parallel_generatedRoot_32.constraints_7.bridge h10), (Stage_encrypt.parallel_generatedRoot_32.constraints_8.bridge h11)⟩)⟩)


theorem Stage_encrypt.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_1 w_32_2 w_32_3 w_32_4 w_32_5 w_32_6 w_32_7 w_32_8 w_32_9 w_32_10 w_32_11 w_32_12 w_32_13 w_32_14 w_32_15 w_32_16 w_32_17 w_32_18 w_32_19 w_32_20 w_32_21 w_32_22 w_32_23 w_32_24 w_32_25 w_32_26 w_32_27 w_32_28 w_32_29 w_32_30 w_32_31 w_32_32 w_32_33 w_32_34 w_32_35 w_32_36 w_32_37 w_32_38 w_32_39 w_32_40 w_32_41 w_32_42 w_32_43 w_32_44 w_32_45 w_32_46 w_32_47 w_32_48 w_32_49 w_32_50 w_32_51 w_32_52 w_32_53 w_32_54 w_32_55 w_32_56 w_32_57 w_32_58 w_32_59 w_32_60 w_32_61 w_32_62 w_32_63 w_32_64 w_32_65 w_32_66 w_32_67 w_32_68 w_32_69 w_32_70 w_32_71 w_32_72 w_32_73 w_32_74 w_32_75 w_32_76 w_32_77 w_32_78 w_32_79 w_32_80 w_32_81 w_32_82 w_32_83 w_32_84 w_32_85 w_32_86 w_32_87 w_32_88 w_32_89 w_32_90 w_32_91 w_32_92 w_32_93 w_32_94 w_32_95 w_32_96 w_32_97 w_32_98 w_32_99 w_32_100 w_32_101 w_32_102 w_32_103 w_32_104 w_32_105 w_32_106 w_32_107 w_32_108 w_32_109 w_32_110 w_32_111 w_32_112 w_32_113 w_32_114 w_32_115 w_32_116 w_32_117 w_32_118 w_32_119 w_32_120 w_32_121 w_32_122 w_32_123 w_32_124 w_32_125 w_32_126 w_32_127 w_32_128 w_32_129 w_32_130 w_32_131 w_32_132 w_32_133 w_32_134 w_32_135 w_32_136 w_32_137 w_32_138 w_32_139 w_32_140 w_32_141 w_32_142 w_32_143 w_32_144 w_32_145 w_32_146 w_32_147 w_32_148 w_32_149 w_32_150 w_32_151 w_32_152 w_32_153 w_32_154 w_32_155 w_32_156 w_32_157 w_32_158 w_32_159 w_32_160 w_32_161 w_32_162 w_32_163 w_32_164 w_32_165 w_32_166 w_32_167 w_32_168 w_32_169 w_32_170 w_32_171 w_32_172 w_32_173 w_32_174 w_32_175 w_32_176 w_32_177 w_32_178 w_32_179 w_32_180 w_32_181 w_32_182 w_32_183 w_32_184 w_32_185 w_32_186 w_32_187 w_32_188 w_32_189 w_32_190 w_32_191 w_32_192 w_32_193 w_32_194 w_32_195 w_32_196 w_32_197 w_32_198 w_32_199 w_32_200 w_32_201 w_32_202 w_32_203 w_32_204 w_32_205 w_32_206 w_32_207 w_32_208 w_32_209 w_32_210 w_32_211 w_32_212 w_32_213 w_32_214 w_32_215 w_32_216 w_32_217 w_32_218 w_32_219 w_32_220 w_32_221 w_32_222 w_32_223 w_32_224 w_32_225 w_32_226 w_32_227 w_32_228 w_32_229 w_32_230 w_32_231 w_32_232 w_32_233 w_32_234 w_32_235 w_32_236 w_32_237 w_32_238 w_32_239 w_32_240 w_32_241 w_32_242 w_32_243 w_32_244 w_32_245 w_32_246 w_32_247 w_32_248 outputs}
    (h : Stage_encrypt.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_1 w_32_2 w_32_3 w_32_4 w_32_5 w_32_6 w_32_7 w_32_8 w_32_9 w_32_10 w_32_11 w_32_12 w_32_13 w_32_14 w_32_15 w_32_16 w_32_17 w_32_18 w_32_19 w_32_20 w_32_21 w_32_22 w_32_23 w_32_24 w_32_25 w_32_26 w_32_27 w_32_28 w_32_29 w_32_30 w_32_31 w_32_32 w_32_33 w_32_34 w_32_35 w_32_36 w_32_37 w_32_38 w_32_39 w_32_40 w_32_41 w_32_42 w_32_43 w_32_44 w_32_45 w_32_46 w_32_47 w_32_48 w_32_49 w_32_50 w_32_51 w_32_52 w_32_53 w_32_54 w_32_55 w_32_56 w_32_57 w_32_58 w_32_59 w_32_60 w_32_61 w_32_62 w_32_63 w_32_64 w_32_65 w_32_66 w_32_67 w_32_68 w_32_69 w_32_70 w_32_71 w_32_72 w_32_73 w_32_74 w_32_75 w_32_76 w_32_77 w_32_78 w_32_79 w_32_80 w_32_81 w_32_82 w_32_83 w_32_84 w_32_85 w_32_86 w_32_87 w_32_88 w_32_89 w_32_90 w_32_91 w_32_92 w_32_93 w_32_94 w_32_95 w_32_96 w_32_97 w_32_98 w_32_99 w_32_100 w_32_101 w_32_102 w_32_103 w_32_104 w_32_105 w_32_106 w_32_107 w_32_108 w_32_109 w_32_110 w_32_111 w_32_112 w_32_113 w_32_114 w_32_115 w_32_116 w_32_117 w_32_118 w_32_119 w_32_120 w_32_121 w_32_122 w_32_123 w_32_124 w_32_125 w_32_126 w_32_127 w_32_128 w_32_129 w_32_130 w_32_131 w_32_132 w_32_133 w_32_134 w_32_135 w_32_136 w_32_137 w_32_138 w_32_139 w_32_140 w_32_141 w_32_142 w_32_143 w_32_144 w_32_145 w_32_146 w_32_147 w_32_148 w_32_149 w_32_150 w_32_151 w_32_152 w_32_153 w_32_154 w_32_155 w_32_156 w_32_157 w_32_158 w_32_159 w_32_160 w_32_161 w_32_162 w_32_163 w_32_164 w_32_165 w_32_166 w_32_167 w_32_168 w_32_169 w_32_170 w_32_171 w_32_172 w_32_173 w_32_174 w_32_175 w_32_176 w_32_177 w_32_178 w_32_179 w_32_180 w_32_181 w_32_182 w_32_183 w_32_184 w_32_185 w_32_186 w_32_187 w_32_188 w_32_189 w_32_190 w_32_191 w_32_192 w_32_193 w_32_194 w_32_195 w_32_196 w_32_197 w_32_198 w_32_199 w_32_200 w_32_201 w_32_202 w_32_203 w_32_204 w_32_205 w_32_206 w_32_207 w_32_208 w_32_209 w_32_210 w_32_211 w_32_212 w_32_213 w_32_214 w_32_215 w_32_216 w_32_217 w_32_218 w_32_219 w_32_220 w_32_221 w_32_222 w_32_223 w_32_224 w_32_225 w_32_226 w_32_227 w_32_228 w_32_229 w_32_230 w_32_231 w_32_232 w_32_233 w_32_234 w_32_235 w_32_236 w_32_237 w_32_238 w_32_239 w_32_240 w_32_241 w_32_242 w_32_243 w_32_244 w_32_245 w_32_246 w_32_247 w_32_248 outputs) : W_Stage_encrypt.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_1 w_32_2 w_32_3 w_32_4 w_32_5 w_32_6 w_32_7 w_32_8 w_32_9 w_32_10 w_32_11 w_32_12 w_32_13 w_32_14 w_32_15 w_32_16 w_32_17 w_32_18 w_32_19 w_32_20 w_32_21 w_32_22 w_32_23 w_32_24 w_32_25 w_32_26 w_32_27 w_32_28 w_32_29 w_32_30 w_32_31 w_32_32 w_32_33 w_32_34 w_32_35 w_32_36 w_32_37 w_32_38 w_32_39 w_32_40 w_32_41 w_32_42 w_32_43 w_32_44 w_32_45 w_32_46 w_32_47 w_32_48 w_32_49 w_32_50 w_32_51 w_32_52 w_32_53 w_32_54 w_32_55 w_32_56 w_32_57 w_32_58 w_32_59 w_32_60 w_32_61 w_32_62 w_32_63 w_32_64 w_32_65 w_32_66 w_32_67 w_32_68 w_32_69 w_32_70 w_32_71 w_32_72 w_32_73 w_32_74 w_32_75 w_32_76 w_32_77 w_32_78 w_32_79 w_32_80 w_32_81 w_32_82 w_32_83 w_32_84 w_32_85 w_32_86 w_32_87 w_32_88 w_32_89 w_32_90 w_32_91 w_32_92 w_32_93 w_32_94 w_32_95 w_32_96 w_32_97 w_32_98 w_32_99 w_32_100 w_32_101 w_32_102 w_32_103 w_32_104 w_32_105 w_32_106 w_32_107 w_32_108 w_32_109 w_32_110 w_32_111 w_32_112 w_32_113 w_32_114 w_32_115 w_32_116 w_32_117 w_32_118 w_32_119 w_32_120 w_32_121 w_32_122 w_32_123 w_32_124 w_32_125 w_32_126 w_32_127 w_32_128 w_32_129 w_32_130 w_32_131 w_32_132 w_32_133 w_32_134 w_32_135 w_32_136 w_32_137 w_32_138 w_32_139 w_32_140 w_32_141 w_32_142 w_32_143 w_32_144 w_32_145 w_32_146 w_32_147 w_32_148 w_32_149 w_32_150 w_32_151 w_32_152 w_32_153 w_32_154 w_32_155 w_32_156 w_32_157 w_32_158 w_32_159 w_32_160 w_32_161 w_32_162 w_32_163 w_32_164 w_32_165 w_32_166 w_32_167 w_32_168 w_32_169 w_32_170 w_32_171 w_32_172 w_32_173 w_32_174 w_32_175 w_32_176 w_32_177 w_32_178 w_32_179 w_32_180 w_32_181 w_32_182 w_32_183 w_32_184 w_32_185 w_32_186 w_32_187 w_32_188 w_32_189 w_32_190 w_32_191 w_32_192 w_32_193 w_32_194 w_32_195 w_32_196 w_32_197 w_32_198 w_32_199 w_32_200 w_32_201 w_32_202 w_32_203 w_32_204 w_32_205 w_32_206 w_32_207 w_32_208 w_32_209 w_32_210 w_32_211 w_32_212 w_32_213 w_32_214 w_32_215 w_32_216 w_32_217 w_32_218 w_32_219 w_32_220 w_32_221 w_32_222 w_32_223 w_32_224 w_32_225 w_32_226 w_32_227 w_32_228 w_32_229 w_32_230 w_32_231 w_32_232 w_32_233 w_32_234 w_32_235 w_32_236 w_32_237 w_32_238 w_32_239 w_32_240 w_32_241 w_32_242 w_32_243 w_32_244 w_32_245 w_32_246 w_32_247 w_32_248 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_generatedRoot_32.bridge (h2 i))), h3⟩)


theorem Stage_encrypt.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_encrypt.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_encrypt.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_encrypt.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_32_0, wit1.w_32_1, wit1.w_32_2, wit1.w_32_3, wit1.w_32_4, wit1.w_32_5, wit1.w_32_6, wit1.w_32_7, wit1.w_32_8, wit1.w_32_9, wit1.w_32_10, wit1.w_32_11, wit1.w_32_12, wit1.w_32_13, wit1.w_32_14, wit1.w_32_15, wit1.w_32_16, wit1.w_32_17, wit1.w_32_18, wit1.w_32_19, wit1.w_32_20, wit1.w_32_21, wit1.w_32_22, wit1.w_32_23, wit1.w_32_24, wit1.w_32_25, wit1.w_32_26, wit1.w_32_27, wit1.w_32_28, wit1.w_32_29, wit1.w_32_30, wit1.w_32_31, wit1.w_32_32, wit1.w_32_33, wit1.w_32_34, wit1.w_32_35, wit1.w_32_36, wit1.w_32_37, wit1.w_32_38, wit1.w_32_39, wit1.w_32_40, wit1.w_32_41, wit1.w_32_42, wit1.w_32_43, wit1.w_32_44, wit1.w_32_45, wit1.w_32_46, wit1.w_32_47, wit1.w_32_48, wit1.w_32_49, wit1.w_32_50, wit1.w_32_51, wit1.w_32_52, wit1.w_32_53, wit1.w_32_54, wit1.w_32_55, wit1.w_32_56, wit1.w_32_57, wit1.w_32_58, wit1.w_32_59, wit1.w_32_60, wit1.w_32_61, wit1.w_32_62, wit1.w_32_63, wit1.w_32_64, wit1.w_32_65, wit1.w_32_66, wit1.w_32_67, wit1.w_32_68, wit1.w_32_69, wit1.w_32_70, wit1.w_32_71, wit1.w_32_72, wit1.w_32_73, wit1.w_32_74, wit1.w_32_75, wit1.w_32_76, wit1.w_32_77, wit1.w_32_78, wit1.w_32_79, wit1.w_32_80, wit1.w_32_81, wit1.w_32_82, wit1.w_32_83, wit1.w_32_84, wit1.w_32_85, wit1.w_32_86, wit1.w_32_87, wit1.w_32_88, wit1.w_32_89, wit1.w_32_90, wit1.w_32_91, wit1.w_32_92, wit1.w_32_93, wit1.w_32_94, wit1.w_32_95, wit1.w_32_96, wit1.w_32_97, wit1.w_32_98, wit1.w_32_99, wit1.w_32_100, wit1.w_32_101, wit1.w_32_102, wit1.w_32_103, wit1.w_32_104, wit1.w_32_105, wit1.w_32_106, wit1.w_32_107, wit1.w_32_108, wit1.w_32_109, wit1.w_32_110, wit1.w_32_111, wit1.w_32_112, wit1.w_32_113, wit1.w_32_114, wit1.w_32_115, wit1.w_32_116, wit1.w_32_117, wit1.w_32_118, wit1.w_32_119, wit1.w_32_120, wit1.w_32_121, wit1.w_32_122, wit1.w_32_123, wit1.w_32_124, wit1.w_32_125, wit1.w_32_126, wit1.w_32_127, wit1.w_32_128, wit1.w_32_129, wit1.w_32_130, wit1.w_32_131, wit1.w_32_132, wit1.w_32_133, wit1.w_32_134, wit1.w_32_135, wit1.w_32_136, wit1.w_32_137, wit1.w_32_138, wit1.w_32_139, wit1.w_32_140, wit1.w_32_141, wit1.w_32_142, wit1.w_32_143, wit1.w_32_144, wit1.w_32_145, wit1.w_32_146, wit1.w_32_147, wit1.w_32_148, wit1.w_32_149, wit1.w_32_150, wit1.w_32_151, wit1.w_32_152, wit1.w_32_153, wit1.w_32_154, wit1.w_32_155, wit1.w_32_156, wit1.w_32_157, wit1.w_32_158, wit1.w_32_159, wit1.w_32_160, wit1.w_32_161, wit1.w_32_162, wit1.w_32_163, wit1.w_32_164, wit1.w_32_165, wit1.w_32_166, wit1.w_32_167, wit1.w_32_168, wit1.w_32_169, wit1.w_32_170, wit1.w_32_171, wit1.w_32_172, wit1.w_32_173, wit1.w_32_174, wit1.w_32_175, wit1.w_32_176, wit1.w_32_177, wit1.w_32_178, wit1.w_32_179, wit1.w_32_180, wit1.w_32_181, wit1.w_32_182, wit1.w_32_183, wit1.w_32_184, wit1.w_32_185, wit1.w_32_186, wit1.w_32_187, wit1.w_32_188, wit1.w_32_189, wit1.w_32_190, wit1.w_32_191, wit1.w_32_192, wit1.w_32_193, wit1.w_32_194, wit1.w_32_195, wit1.w_32_196, wit1.w_32_197, wit1.w_32_198, wit1.w_32_199, wit1.w_32_200, wit1.w_32_201, wit1.w_32_202, wit1.w_32_203, wit1.w_32_204, wit1.w_32_205, wit1.w_32_206, wit1.w_32_207, wit1.w_32_208, wit1.w_32_209, wit1.w_32_210, wit1.w_32_211, wit1.w_32_212, wit1.w_32_213, wit1.w_32_214, wit1.w_32_215, wit1.w_32_216, wit1.w_32_217, wit1.w_32_218, wit1.w_32_219, wit1.w_32_220, wit1.w_32_221, wit1.w_32_222, wit1.w_32_223, wit1.w_32_224, wit1.w_32_225, wit1.w_32_226, wit1.w_32_227, wit1.w_32_228, wit1.w_32_229, wit1.w_32_230, wit1.w_32_231, wit1.w_32_232, wit1.w_32_233, wit1.w_32_234, wit1.w_32_235, wit1.w_32_236, wit1.w_32_237, wit1.w_32_238, wit1.w_32_239, wit1.w_32_240, wit1.w_32_241, wit1.w_32_242, wit1.w_32_243, wit1.w_32_244, wit1.w_32_245, wit1.w_32_246, wit1.w_32_247, wit1.w_32_248⟩, (Stage_encrypt.generatedRoot.constraints_0.bridge h2)⟩)
