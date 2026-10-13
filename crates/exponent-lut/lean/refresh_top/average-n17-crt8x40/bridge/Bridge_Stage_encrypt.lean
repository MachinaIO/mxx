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


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_42.constraints_0.bridge {backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_42.constraints_0 backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_42.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, (MxxExponentLut.gaussianSampleAt_sample (by decide) h2), (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_42.bridge {backend x1 tape path params params' x5 x6 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_42 backend x1 tape path params x5 x6 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_42 backend x1 params' x5 x6 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_42
  exact (by obtain ⟨w_13_0, w_17_0, w_24_0, h1⟩ := h; exact ⟨w_13_0, w_17_0, w_24_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_42.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_44.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_44.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_44.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_44.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_44 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_44 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_44
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_44.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_45.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_45.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_45.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_45.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_45 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_45 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_45
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_45.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_46.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_46.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_46.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_46.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_46 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_46 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_46
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_46.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_47.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_47.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_47.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_47.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_47 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_47 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_47
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_47.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_48.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_48.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_48.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_48.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_48 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_48 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_48
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_48.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_49.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_49.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_49.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_49.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_49 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_49 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_49
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_49.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_50.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_50.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_50.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_50.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_50 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_50 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_50
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_50.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_51.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_51.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_51.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_51.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_51 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_51 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_51
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_51.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_52.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_52.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_52.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_52.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_52 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_52 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_52
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_52.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_53.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_53.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_53.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_53.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_53 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_53 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_53
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_53.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_54.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_54.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_54.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_54.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_54 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_54 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_54
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_54.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_55.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_55.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_55.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_55.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_55 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_55 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_55
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_55.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_56.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_56.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_56.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_56.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_56 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_56 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_56
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_56.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_57.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_57.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_57.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_57.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_57 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_57 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_57
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_57.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_58.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_58.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_58.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_58.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_58 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_58 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_58
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_58.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_59.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_59.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_59.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_59.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_59 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_59 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_59
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_59.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_60.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_60.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_60.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_60.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_60 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_60 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_60
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_60.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_61.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_61.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_61.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_61.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_61 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_61 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_61
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_61.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_62.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_62.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_62.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_62.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_62 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_62 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_62
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_62.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_63.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_63.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_63.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_63.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_63 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_63 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_63
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_63.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_64.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_64.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_64.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_64.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_64 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_64 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_64
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_64.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_65.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_65.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_65.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_65.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_65 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_65 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_65
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_65.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_66.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_66.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_66.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_66.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_66 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_66 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_66
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_66.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_67.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_67.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_67.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_67.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_67 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_67 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_67
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_67.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_68.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_68.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_68.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_68.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_68 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_68 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_68
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_68.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_69.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_69.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_69.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_69.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_69 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_69 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_69
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_69.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_70.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_70.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_70.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_70.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_70 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_70 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_70
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_70.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_71.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_71.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_71.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_71.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_71 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_71 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_71
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_71.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_72.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_72.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_72.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_72.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_72 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_72 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_72
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_72.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_73.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_73.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_73.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_73.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_73 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_73 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_73
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_73.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_74.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_74.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_74.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_74.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_74 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_74 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_74
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_74.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_75.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_75.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_75.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_75.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_75 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_75 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_75
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_75.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_76.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_76.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_76.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_76.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_76 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_76 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_76
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_76.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_77.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_77.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_77.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_77.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_77 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_77 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_77
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_77.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_78.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_78.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_78.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_78.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_78 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_78 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_78
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_78.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_108.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_108.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_108.constraints_0 backend hashModel i_0 i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_108.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_108 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_108 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_108
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_108.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_109.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_109.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_109.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_109.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_109 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_109 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_109
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_109.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_148.constraints_0.bridge {backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_148.constraints_0 backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_148.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, (MxxExponentLut.gaussianSampleAt_sample (by decide) h2), (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_148.bridge {backend x1 tape path params params' x5 x6 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_148 backend x1 tape path params x5 x6 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_148 backend x1 params' x5 x6 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_148
  exact (by obtain ⟨w_13_0, w_17_0, w_24_0, h1⟩ := h; exact ⟨w_13_0, w_17_0, w_24_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_148.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_150.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_150.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_150.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_150.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_150 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_150 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_150
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_150.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_151.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_151.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_151.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_151.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_151 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_151 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_151
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_151.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_152.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_152.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_152.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_152.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_152 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_152 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_152
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_152.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_153.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_153.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_153.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_153.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_153 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_153 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_153
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_153.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_154.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_154.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_154.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_154.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_154 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_154 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_154
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_154.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_155.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_155.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_155.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_155.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_155 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_155 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_155
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_155.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_156.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_156.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_156.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_156.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_156 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_156 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_156
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_156.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_157.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_157.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_157.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_157.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_157 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_157 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_157
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_157.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_158.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_158.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_158.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_158.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_158 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_158 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_158
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_158.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_159.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_159.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_159.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_159.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_159 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_159 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_159
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_159.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_160.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_160.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_160.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_160.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_160 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_160 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_160
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_160.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_161.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_161.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_161.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_161.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_161 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_161 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_161
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_161.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_162.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_162.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_162.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_162.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_162 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_162 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_162
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_162.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_163.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_163.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_163.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_163.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_163 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_163 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_163
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_163.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_164.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_164.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_164.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_164.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_164 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_164 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_164
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_164.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_165.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_165.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_165.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_165.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_165 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_165 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_165
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_165.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_166.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_166.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_166.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_166.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_166 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_166 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_166
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_166.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_167.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_167.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_167.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_167.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_167 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_167 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_167
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_167.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_168.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_168.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_168.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_168.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_168 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_168 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_168
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_168.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_169.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_169.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_169.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_169.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_169 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_169 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_169
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_169.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_170.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_170.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_170.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_170.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_170 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_170 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_170
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_170.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_171.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_171.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_171.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_171.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_171 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_171 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_171
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_171.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_172.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_172.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_172.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_172.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_172 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_172 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_172
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_172.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_173.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_173.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_173.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_173.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_173 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_173 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_173
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_173.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_174.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_174.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_174.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_174.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_174 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_174 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_174
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_174.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_175.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_175.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_175.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_175.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_175 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_175 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_175
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_175.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_176.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_176.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_176.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_176.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_176 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_176 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_176
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_176.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_177.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_177.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_177.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_177.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_177 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_177 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_177
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_177.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_178.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_178.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_178.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_178.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_178 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_178 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_178
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_178.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_179.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_179.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_179.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_179.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_179 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_179 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_179
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_179.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_180.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_180.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_180.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_180.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_180 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_180 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_180
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_180.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_181.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_181.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_181.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_181.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_181 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_181 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_181
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_181.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_182.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_182.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_182.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_182.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_182 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_182 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_182
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_182.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_183.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_183.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_183.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_183.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_183 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_183 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_183
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_183.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_184.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_184.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_184.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_184.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_184 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_184 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_184
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_184.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_185.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_185.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_185.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_185.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_185 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_185 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_185
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_185.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_186.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_186.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_186.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_186.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_186 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_186 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_186
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_186.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_187.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_187.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_187.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_187.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_187 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_187 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_187
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_187.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_188.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_188.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_188.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_188.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_188 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_188 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_188
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_188.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_189.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_189.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_189.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_189.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_189 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_189 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_189
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_189.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_190.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_190.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_190.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_190.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_190 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_190 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_190
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_190.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_191.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_191.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_191.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_191.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_191 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_191 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_191
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_191.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_192.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_192.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_192.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_192.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_192 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_192 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_192
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_192.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_193.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_193.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_193.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_193.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_193 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_193 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_193
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_193.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_194.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_194.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_194.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_194.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_194 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_194 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_194
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_194.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_195.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_195.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_195.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_195.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_195 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_195 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_195
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_195.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_196.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_196.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_196.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_196.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_196 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_196 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_196
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_196.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_197.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_197.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_197.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_197.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_197 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_197 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_197
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_197.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_198.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_198.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_198.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_198.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_198 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_198 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_198
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_198.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_199.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_199.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_199.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_199.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_199 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_199 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_199
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_199.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_200.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_200.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_200.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_200.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_200 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_200 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_200
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_200.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_201.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_201.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_201.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_201.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_201 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_201 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_201
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_201.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_202.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_202.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_202.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_202.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_202 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_202 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_202
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_202.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_203.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_203.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_203.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_203.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_203 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_203 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_203
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_203.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_204.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_204.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_204.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_204.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_204 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_204 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_204
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_204.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_205.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_205.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_205.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_205.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_205 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_205 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_205
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_205.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_206.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_206.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_206.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_206.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_206 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_206 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_206
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_206.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_207.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_207.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_207.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_207.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_207 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_207 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_207
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_207.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_208.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_208.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_208.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_208.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_208 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_208 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_208
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_208.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_209.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_209.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_209.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_209.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_209 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_209 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_209
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_209.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_210.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_210.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_210.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_210.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_210 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_210 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_210
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_210.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_211.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_211.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_211.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_211.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_211 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_211 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_211
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_211.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_212.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_212.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_212.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_212.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_212 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_212 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_212
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_212.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_213.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_213.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_213.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_213.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_213 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_213 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_213
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_213.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_214.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_214.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_214.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_214.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_214 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_214 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_214
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_214.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_215.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_215.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_215.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_215.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_215 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_215 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_215
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_215.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_216.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_216.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_216.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_216.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_216 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_216 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_216
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_216.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_217.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_217.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_217.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_217.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_217 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_217 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_217
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_217.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_218.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_218.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_218.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_218.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_218 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_218 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_218
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_218.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_219.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_219.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_219.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_219.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_219 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_219 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_219
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_219.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_220.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_220.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_220.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_220.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_220 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_220 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_220
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_220.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_221.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_221.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_221.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_221.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_221 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_221 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_221
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_221.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_260.constraints_0.bridge {backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_260.constraints_0 backend tape path w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_260.constraints_0 backend w_0_0 w_1_0 w_2_0 w_3_0 w_12_0 w_13_0 w_17_0 w_18_0 w_24_0 w_25_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4⟩ := h; exact ⟨h1, (MxxExponentLut.gaussianSampleAt_sample (by decide) h2), (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_260.bridge {backend x1 tape path params params' x5 x6 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_260 backend x1 tape path params x5 x6 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_260 backend x1 params' x5 x6 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_260
  exact (by obtain ⟨w_13_0, w_17_0, w_24_0, h1⟩ := h; exact ⟨w_13_0, w_17_0, w_24_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_260.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_262.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_262.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_262.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_262.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_262 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_262 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_262
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_262.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_263.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_263.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_263.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_263.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_263 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_263 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_263
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_263.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_264.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_264.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_264.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_264.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_264 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_264 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_264
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_264.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_265.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_265.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_265.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_265.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_265 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_265 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_265
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_265.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_266.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_266.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_266.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_266.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_266 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_266 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_266
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_266.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_267.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_267.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_267.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_267.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_267 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_267 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_267
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_267.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_268.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_268.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_268.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_268.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_268 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_268 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_268
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_268.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_269.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_269.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_269.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_269.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_269 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_269 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_269
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_269.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_270.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_270.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_270.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_270.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_270 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_270 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_270
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_270.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_271.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_271.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_271.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_271.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_271 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_271 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_271
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_271.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_272.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_272.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_272.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_272.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_272 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_272 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_272
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_272.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_273.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_273.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_273.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_273.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_273 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_273 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_273
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_273.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_274.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_274.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_274.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_274.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_274 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_274 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_274
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_274.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_275.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_275.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_275.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_275.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_275 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_275 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_275
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_275.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_276.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_276.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_276.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_276.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_276 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_276 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_276
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_276.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_277.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_277.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_277.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_277.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_277 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_277 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_277
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_277.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_278.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_278.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_278.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_278.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_278 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_278 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_278
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_278.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_279.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_279.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_279.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_279.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_279 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_279 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_279
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_279.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_280.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_280.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_280.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_280.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_280 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_280 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_280
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_280.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_281.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_281.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_281.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_281.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_281 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_281 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_281
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_281.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_282.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_282.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_282.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_282.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_282 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_282 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_282
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_282.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_283.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_283.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_283.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_283.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_283 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_283 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_283
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_283.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_284.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_284.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_284.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_284.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_284 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_284 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_284
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_284.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_285.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_285.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_285.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_285.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_285 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_285 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_285
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_285.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_286.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_286.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_286.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_286.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_286 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_286 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_286
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_286.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_287.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_287.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_287.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_287.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_287 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_287 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_287
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_287.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_288.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_288.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_288.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_288.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_288 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_288 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_288
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_288.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_289.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_289.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_289.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_289.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_289 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_289 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_289
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_289.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_290.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_290.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_290.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_290.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_290 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_290 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_290
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_290.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_291.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_291.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_291.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_291.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_291 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_291 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_291
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_291.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_292.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_292.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_292.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_292.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_292 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_292 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_292
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_292.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_293.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_293.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_293.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_293.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_293 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_293 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_293
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_293.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_294.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_294.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_294.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_294.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_294 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_294 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_294
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_294.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_295.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_295.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_295.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_295.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_295 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_295 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_295
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_295.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_296.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_296.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_296.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_296.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_296 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_296 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_296
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_296.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_297.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_297.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_297.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_297.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_297 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_297 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_297
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_297.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_298.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_298.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_298.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_298.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_298 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_298 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_298
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_298.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_299.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_299.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_299.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_299.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_299 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_299 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_299
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_299.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_300.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_300.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_300.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_300.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_300 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_300 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_300
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_300.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_301.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_301.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_301.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_301.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_301 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_301 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_301
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_301.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_302.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_302.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_302.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_302.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_302 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_302 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_302
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_302.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_303.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_303.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_303.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_303.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_303 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_303 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_303
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_303.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_304.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_304.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_304.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_304.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_304 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_304 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_304
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_304.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_305.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_305.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_305.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_305.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_305 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_305 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_305
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_305.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_306.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_306.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_306.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_306.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_306 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_306 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_306
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_306.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_307.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_307.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_307.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_307.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_307 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_307 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_307
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_307.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_308.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_308.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_308.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_308.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_308 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_308 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_308
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_308.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_309.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_309.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_309.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_309.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_309 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_309 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_309
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_309.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_310.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_310.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_310.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_310.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_310 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_310 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_310
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_310.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_311.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_311.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_311.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_311.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_311 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_311 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_311
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_311.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_312.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_312.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_312.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_312.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_312 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_312 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_312
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_312.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_313.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_313.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_313.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_313.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_313 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_313 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_313
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_313.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_314.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_314.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_314.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_314.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_314 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_314 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_314
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_314.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_315.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_315.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_315.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_315.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_315 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_315 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_315
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_315.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_316.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_316.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_316.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_316.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_316 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_316 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_316
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_316.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_317.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_317.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_317.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_317.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_317 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_317 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_317
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_317.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_318.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_318.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_318.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_318.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_318 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_318 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_318
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_318.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_319.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_319.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_319.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_319.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_319 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_319 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_319
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_319.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_320.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_320.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_320.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_320.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_320 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_320 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_320
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_320.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_321.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_321.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_321.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_321.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_321 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_321 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_321
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_321.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_322.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_322.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_322.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_322.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_322 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_322 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_322
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_322.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_323.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_323.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_323.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_323.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_323 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_323 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_323
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_323.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_324.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_324.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_324.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_324.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_324 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_324 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_324
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_324.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_325.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_325.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_325.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_325.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_325 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_325 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_325
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_325.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_326.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_326.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_326.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_326.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_326 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_326 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_326
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_326.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_327.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_327.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_327.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_327.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_327 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_327 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_327
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_327.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_328.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_328.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_328.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_328.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_328 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_328 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_328
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_328.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_329.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_329.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_329.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_329.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_329 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_329 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_329
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_329.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_330.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_330.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_330.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_330.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_330 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_330 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_330
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_330.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_331.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_331.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_331.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_331.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_331 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_331 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_331
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_331.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_332.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_332.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_332.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_332.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_332 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_332 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_332
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_332.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_333.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_333.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_333.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_encrypt.parallel_parallel_generatedRoot_45_333.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_45_333 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_encrypt.parallel_parallel_generatedRoot_45_333 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_encrypt.parallel_parallel_generatedRoot_45_333
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_encrypt.parallel_parallel_generatedRoot_45_333.constraints_0.bridge h1)⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_13.bridge {hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_79_0 w_110_0 w_111_0 w_112_0 w_113_0 w_222_0 w_223_0 w_224_0 w_225_0 w_7_0 w_25_0 w_35_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_97_0 w_107_0 w_108_0 w_108_1 w_109_0 w_109_1 w_132_0 w_141_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_244_0 w_253_0 w_260_0 w_260_1 w_260_2 w_260_3 w_260_4 w_260_5 w_260_6 w_262_0 w_262_1 w_263_0 w_263_1 w_264_0 w_264_1 w_265_0 w_265_1 w_266_0 w_266_1 w_267_0 w_267_1 w_268_0 w_268_1 w_269_0 w_269_1 w_270_0 w_270_1 w_271_0 w_271_1 w_272_0 w_272_1 w_273_0 w_273_1 w_274_0 w_274_1 w_275_0 w_275_1 w_276_0 w_276_1 w_277_0 w_277_1 w_278_0 w_278_1 w_279_0 w_279_1 w_280_0 w_280_1 w_281_0 w_281_1 w_282_0 w_282_1 w_283_0 w_283_1 w_284_0 w_284_1 w_285_0 w_285_1 w_286_0 w_286_1 w_287_0 w_287_1 w_288_0 w_288_1 w_289_0 w_289_1 w_290_0 w_290_1 w_291_0 w_291_1 w_292_0 w_292_1 w_293_0 w_293_1 w_294_0 w_294_1 w_295_0 w_295_1 w_296_0 w_296_1 w_297_0 w_297_1 w_298_0 w_298_1 w_299_0 w_299_1 w_300_0 w_300_1 w_301_0 w_301_1 w_302_0 w_302_1 w_303_0 w_303_1 w_304_0 w_304_1 w_305_0 w_305_1 w_306_0 w_306_1 w_307_0 w_307_1 w_308_0 w_308_1 w_309_0 w_309_1 w_310_0 w_310_1 w_311_0 w_311_1 w_312_0 w_312_1 w_313_0 w_313_1 w_314_0 w_314_1 w_315_0 w_315_1 w_316_0 w_316_1 w_317_0 w_317_1 w_318_0 w_318_1 w_319_0 w_319_1 w_320_0 w_320_1 w_321_0 w_321_1 w_322_0 w_322_1 w_323_0 w_323_1 w_324_0 w_324_1 w_325_0 w_325_1 w_326_0 w_326_1 w_327_0 w_327_1 w_328_0 w_328_1 w_329_0 w_329_1 w_330_0 w_330_1 w_331_0 w_331_1 w_332_0 w_332_1 w_333_0 w_333_1 w_347_0 w_350_0 w_355_0 w_356_0 w_357_0 w_358_0 w_359_0 w_362_0 w_363_0 outputs}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_13 hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_79_0 w_110_0 w_111_0 w_112_0 w_113_0 w_222_0 w_223_0 w_224_0 w_225_0 w_7_0 w_25_0 w_35_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_97_0 w_107_0 w_108_0 w_108_1 w_109_0 w_109_1 w_132_0 w_141_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_244_0 w_253_0 w_260_0 w_260_1 w_260_2 w_260_3 w_260_4 w_260_5 w_260_6 w_262_0 w_262_1 w_263_0 w_263_1 w_264_0 w_264_1 w_265_0 w_265_1 w_266_0 w_266_1 w_267_0 w_267_1 w_268_0 w_268_1 w_269_0 w_269_1 w_270_0 w_270_1 w_271_0 w_271_1 w_272_0 w_272_1 w_273_0 w_273_1 w_274_0 w_274_1 w_275_0 w_275_1 w_276_0 w_276_1 w_277_0 w_277_1 w_278_0 w_278_1 w_279_0 w_279_1 w_280_0 w_280_1 w_281_0 w_281_1 w_282_0 w_282_1 w_283_0 w_283_1 w_284_0 w_284_1 w_285_0 w_285_1 w_286_0 w_286_1 w_287_0 w_287_1 w_288_0 w_288_1 w_289_0 w_289_1 w_290_0 w_290_1 w_291_0 w_291_1 w_292_0 w_292_1 w_293_0 w_293_1 w_294_0 w_294_1 w_295_0 w_295_1 w_296_0 w_296_1 w_297_0 w_297_1 w_298_0 w_298_1 w_299_0 w_299_1 w_300_0 w_300_1 w_301_0 w_301_1 w_302_0 w_302_1 w_303_0 w_303_1 w_304_0 w_304_1 w_305_0 w_305_1 w_306_0 w_306_1 w_307_0 w_307_1 w_308_0 w_308_1 w_309_0 w_309_1 w_310_0 w_310_1 w_311_0 w_311_1 w_312_0 w_312_1 w_313_0 w_313_1 w_314_0 w_314_1 w_315_0 w_315_1 w_316_0 w_316_1 w_317_0 w_317_1 w_318_0 w_318_1 w_319_0 w_319_1 w_320_0 w_320_1 w_321_0 w_321_1 w_322_0 w_322_1 w_323_0 w_323_1 w_324_0 w_324_1 w_325_0 w_325_1 w_326_0 w_326_1 w_327_0 w_327_1 w_328_0 w_328_1 w_329_0 w_329_1 w_330_0 w_330_1 w_331_0 w_331_1 w_332_0 w_332_1 w_333_0 w_333_1 w_347_0 w_350_0 w_355_0 w_356_0 w_357_0 w_358_0 w_359_0 w_362_0 w_363_0 outputs) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_13 hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_79_0 w_110_0 w_111_0 w_112_0 w_113_0 w_222_0 w_223_0 w_224_0 w_225_0 w_7_0 w_25_0 w_35_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_97_0 w_107_0 w_108_0 w_108_1 w_109_0 w_109_1 w_132_0 w_141_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_244_0 w_253_0 w_260_0 w_260_1 w_260_2 w_260_3 w_260_4 w_260_5 w_260_6 w_262_0 w_262_1 w_263_0 w_263_1 w_264_0 w_264_1 w_265_0 w_265_1 w_266_0 w_266_1 w_267_0 w_267_1 w_268_0 w_268_1 w_269_0 w_269_1 w_270_0 w_270_1 w_271_0 w_271_1 w_272_0 w_272_1 w_273_0 w_273_1 w_274_0 w_274_1 w_275_0 w_275_1 w_276_0 w_276_1 w_277_0 w_277_1 w_278_0 w_278_1 w_279_0 w_279_1 w_280_0 w_280_1 w_281_0 w_281_1 w_282_0 w_282_1 w_283_0 w_283_1 w_284_0 w_284_1 w_285_0 w_285_1 w_286_0 w_286_1 w_287_0 w_287_1 w_288_0 w_288_1 w_289_0 w_289_1 w_290_0 w_290_1 w_291_0 w_291_1 w_292_0 w_292_1 w_293_0 w_293_1 w_294_0 w_294_1 w_295_0 w_295_1 w_296_0 w_296_1 w_297_0 w_297_1 w_298_0 w_298_1 w_299_0 w_299_1 w_300_0 w_300_1 w_301_0 w_301_1 w_302_0 w_302_1 w_303_0 w_303_1 w_304_0 w_304_1 w_305_0 w_305_1 w_306_0 w_306_1 w_307_0 w_307_1 w_308_0 w_308_1 w_309_0 w_309_1 w_310_0 w_310_1 w_311_0 w_311_1 w_312_0 w_312_1 w_313_0 w_313_1 w_314_0 w_314_1 w_315_0 w_315_1 w_316_0 w_316_1 w_317_0 w_317_1 w_318_0 w_318_1 w_319_0 w_319_1 w_320_0 w_320_1 w_321_0 w_321_1 w_322_0 w_322_1 w_323_0 w_323_1 w_324_0 w_324_1 w_325_0 w_325_1 w_326_0 w_326_1 w_327_0 w_327_1 w_328_0 w_328_1 w_329_0 w_329_1 w_330_0 w_330_1 w_331_0 w_331_1 w_332_0 w_332_1 w_333_0 w_333_1 w_347_0 w_350_0 w_355_0 w_356_0 w_357_0 w_358_0 w_359_0 w_362_0 w_363_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, (MxxExponentLut.gaussianSampleAt_sample (by decide) h3), h4, h5, h6, (MxxExponentLut.gaussianSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_12.bridge {backend hashModel tape path params params' i_0 w_4_0 w_79_0 w_261_0 w_7_0 w_226_0 w_228_0 w_325_0 w_325_1 w_326_0 w_326_1 w_327_0 w_327_1 w_328_0 w_328_1 w_329_0 w_329_1 w_330_0 w_330_1 w_331_0 w_331_1 w_332_0 w_332_1 w_333_0 w_333_1 w_334_0 w_335_0 w_337_0 w_338_0 w_342_0 w_346_0 w_348_0}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_12 backend hashModel tape path params i_0 w_4_0 w_79_0 w_261_0 w_7_0 w_226_0 w_228_0 w_325_0 w_325_1 w_326_0 w_326_1 w_327_0 w_327_1 w_328_0 w_328_1 w_329_0 w_329_1 w_330_0 w_330_1 w_331_0 w_331_1 w_332_0 w_332_1 w_333_0 w_333_1 w_334_0 w_335_0 w_337_0 w_338_0 w_342_0 w_346_0 w_348_0) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_12 backend hashModel params' i_0 w_4_0 w_79_0 w_261_0 w_7_0 w_226_0 w_228_0 w_325_0 w_325_1 w_326_0 w_326_1 w_327_0 w_327_1 w_328_0 w_328_1 w_329_0 w_329_1 w_330_0 w_330_1 w_331_0 w_331_1 w_332_0 w_332_1 w_333_0 w_333_1 w_334_0 w_335_0 w_337_0 w_338_0 w_342_0 w_346_0 w_348_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_325.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_326.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_327.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_328.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_329.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_330.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_331.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_332.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_333.bridge (h18 i))), h19, h20, h21, h22, h23, h24, h25, h26, h27, (MxxExponentLut.gaussianSampleAt_sample (by decide) h28), h29, h30, h31, h32⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_11.bridge {backend hashModel tape path params params' i_0 w_261_0 w_226_0 w_228_0 w_309_0 w_309_1 w_310_0 w_310_1 w_311_0 w_311_1 w_312_0 w_312_1 w_313_0 w_313_1 w_314_0 w_314_1 w_315_0 w_315_1 w_316_0 w_316_1 w_317_0 w_317_1 w_318_0 w_318_1 w_319_0 w_319_1 w_320_0 w_320_1 w_321_0 w_321_1 w_322_0 w_322_1 w_323_0 w_323_1 w_324_0 w_324_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_11 backend hashModel tape path params i_0 w_261_0 w_226_0 w_228_0 w_309_0 w_309_1 w_310_0 w_310_1 w_311_0 w_311_1 w_312_0 w_312_1 w_313_0 w_313_1 w_314_0 w_314_1 w_315_0 w_315_1 w_316_0 w_316_1 w_317_0 w_317_1 w_318_0 w_318_1 w_319_0 w_319_1 w_320_0 w_320_1 w_321_0 w_321_1 w_322_0 w_322_1 w_323_0 w_323_1 w_324_0 w_324_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_11 backend hashModel params' i_0 w_261_0 w_226_0 w_228_0 w_309_0 w_309_1 w_310_0 w_310_1 w_311_0 w_311_1 w_312_0 w_312_1 w_313_0 w_313_1 w_314_0 w_314_1 w_315_0 w_315_1 w_316_0 w_316_1 w_317_0 w_317_1 w_318_0 w_318_1 w_319_0 w_319_1 w_320_0 w_320_1 w_321_0 w_321_1 w_322_0 w_322_1 w_323_0 w_323_1 w_324_0 w_324_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_309.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_310.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_311.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_312.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_313.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_314.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_315.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_316.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_317.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_318.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_319.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_320.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_321.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_322.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_323.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_324.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_10.bridge {backend hashModel tape path params params' i_0 w_261_0 w_226_0 w_228_0 w_293_0 w_293_1 w_294_0 w_294_1 w_295_0 w_295_1 w_296_0 w_296_1 w_297_0 w_297_1 w_298_0 w_298_1 w_299_0 w_299_1 w_300_0 w_300_1 w_301_0 w_301_1 w_302_0 w_302_1 w_303_0 w_303_1 w_304_0 w_304_1 w_305_0 w_305_1 w_306_0 w_306_1 w_307_0 w_307_1 w_308_0 w_308_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_10 backend hashModel tape path params i_0 w_261_0 w_226_0 w_228_0 w_293_0 w_293_1 w_294_0 w_294_1 w_295_0 w_295_1 w_296_0 w_296_1 w_297_0 w_297_1 w_298_0 w_298_1 w_299_0 w_299_1 w_300_0 w_300_1 w_301_0 w_301_1 w_302_0 w_302_1 w_303_0 w_303_1 w_304_0 w_304_1 w_305_0 w_305_1 w_306_0 w_306_1 w_307_0 w_307_1 w_308_0 w_308_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_10 backend hashModel params' i_0 w_261_0 w_226_0 w_228_0 w_293_0 w_293_1 w_294_0 w_294_1 w_295_0 w_295_1 w_296_0 w_296_1 w_297_0 w_297_1 w_298_0 w_298_1 w_299_0 w_299_1 w_300_0 w_300_1 w_301_0 w_301_1 w_302_0 w_302_1 w_303_0 w_303_1 w_304_0 w_304_1 w_305_0 w_305_1 w_306_0 w_306_1 w_307_0 w_307_1 w_308_0 w_308_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_293.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_294.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_295.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_296.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_297.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_298.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_299.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_300.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_301.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_302.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_303.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_304.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_305.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_306.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_307.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_308.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_9.bridge {backend hashModel tape path params params' i_0 w_261_0 w_226_0 w_228_0 w_277_0 w_277_1 w_278_0 w_278_1 w_279_0 w_279_1 w_280_0 w_280_1 w_281_0 w_281_1 w_282_0 w_282_1 w_283_0 w_283_1 w_284_0 w_284_1 w_285_0 w_285_1 w_286_0 w_286_1 w_287_0 w_287_1 w_288_0 w_288_1 w_289_0 w_289_1 w_290_0 w_290_1 w_291_0 w_291_1 w_292_0 w_292_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_9 backend hashModel tape path params i_0 w_261_0 w_226_0 w_228_0 w_277_0 w_277_1 w_278_0 w_278_1 w_279_0 w_279_1 w_280_0 w_280_1 w_281_0 w_281_1 w_282_0 w_282_1 w_283_0 w_283_1 w_284_0 w_284_1 w_285_0 w_285_1 w_286_0 w_286_1 w_287_0 w_287_1 w_288_0 w_288_1 w_289_0 w_289_1 w_290_0 w_290_1 w_291_0 w_291_1 w_292_0 w_292_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_9 backend hashModel params' i_0 w_261_0 w_226_0 w_228_0 w_277_0 w_277_1 w_278_0 w_278_1 w_279_0 w_279_1 w_280_0 w_280_1 w_281_0 w_281_1 w_282_0 w_282_1 w_283_0 w_283_1 w_284_0 w_284_1 w_285_0 w_285_1 w_286_0 w_286_1 w_287_0 w_287_1 w_288_0 w_288_1 w_289_0 w_289_1 w_290_0 w_290_1 w_291_0 w_291_1 w_292_0 w_292_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_277.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_278.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_279.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_280.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_281.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_282.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_283.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_284.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_285.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_286.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_287.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_288.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_289.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_290.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_291.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_292.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_8.bridge {backend hashModel tape path params params' i_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_261_0 w_226_0 w_228_0 w_260_0 w_260_1 w_260_2 w_260_3 w_260_4 w_260_5 w_260_6 w_262_0 w_262_1 w_263_0 w_263_1 w_264_0 w_264_1 w_265_0 w_265_1 w_266_0 w_266_1 w_267_0 w_267_1 w_268_0 w_268_1 w_269_0 w_269_1 w_270_0 w_270_1 w_271_0 w_271_1 w_272_0 w_272_1 w_273_0 w_273_1 w_274_0 w_274_1 w_275_0 w_275_1 w_276_0 w_276_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_8 backend hashModel tape path params i_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_261_0 w_226_0 w_228_0 w_260_0 w_260_1 w_260_2 w_260_3 w_260_4 w_260_5 w_260_6 w_262_0 w_262_1 w_263_0 w_263_1 w_264_0 w_264_1 w_265_0 w_265_1 w_266_0 w_266_1 w_267_0 w_267_1 w_268_0 w_268_1 w_269_0 w_269_1 w_270_0 w_270_1 w_271_0 w_271_1 w_272_0 w_272_1 w_273_0 w_273_1 w_274_0 w_274_1 w_275_0 w_275_1 w_276_0 w_276_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_8 backend hashModel params' i_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_261_0 w_226_0 w_228_0 w_260_0 w_260_1 w_260_2 w_260_3 w_260_4 w_260_5 w_260_6 w_262_0 w_262_1 w_263_0 w_263_1 w_264_0 w_264_1 w_265_0 w_265_1 w_266_0 w_266_1 w_267_0 w_267_1 w_268_0 w_268_1 w_269_0 w_269_1 w_270_0 w_270_1 w_271_0 w_271_1 w_272_0 w_272_1 w_273_0 w_273_1 w_274_0 w_274_1 w_275_0 w_275_1 w_276_0 w_276_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_260.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_262.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_263.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_264.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_265.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_266.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_267.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_268.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_269.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_270.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_271.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_272.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_273.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_274.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_275.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_276.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_7.bridge {backend hashModel tape path params params' i_0 w_4_0 w_11_0 w_149_0 w_7_0 w_29_0 w_114_0 w_116_0 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_226_0 w_228_0 w_231_0 w_232_0 w_233_0 w_243_0 w_248_0 w_252_0}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_7 backend hashModel tape path params i_0 w_4_0 w_11_0 w_149_0 w_7_0 w_29_0 w_114_0 w_116_0 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_226_0 w_228_0 w_231_0 w_232_0 w_233_0 w_243_0 w_248_0 w_252_0) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_7 backend hashModel params' i_0 w_4_0 w_11_0 w_149_0 w_7_0 w_29_0 w_114_0 w_116_0 w_210_0 w_210_1 w_211_0 w_211_1 w_212_0 w_212_1 w_213_0 w_213_1 w_214_0 w_214_1 w_215_0 w_215_1 w_216_0 w_216_1 w_217_0 w_217_1 w_218_0 w_218_1 w_219_0 w_219_1 w_220_0 w_220_1 w_221_0 w_221_1 w_226_0 w_228_0 w_231_0 w_232_0 w_233_0 w_243_0 w_248_0 w_252_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_210.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_211.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_212.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_213.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_214.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_215.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_216.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_217.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_218.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_219.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_220.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_221.bridge (h24 i))), h25, h26, h27, h28, h29, (MxxExponentLut.gaussianSampleAt_sample (by decide) h30), h31, (MxxExponentLut.gaussianSampleAt_sample (by decide) h32)⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_6.bridge {backend hashModel tape path params params' i_0 w_149_0 w_114_0 w_116_0 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_6 backend hashModel tape path params i_0 w_149_0 w_114_0 w_116_0 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_6 backend hashModel params' i_0 w_149_0 w_114_0 w_116_0 w_194_0 w_194_1 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_197_1 w_198_0 w_198_1 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_201_1 w_202_0 w_202_1 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_205_1 w_206_0 w_206_1 w_207_0 w_207_1 w_208_0 w_208_1 w_209_0 w_209_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_194.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_195.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_196.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_197.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_198.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_199.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_200.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_201.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_202.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_203.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_204.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_205.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_206.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_207.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_208.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_209.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_5.bridge {backend hashModel tape path params params' i_0 w_149_0 w_114_0 w_116_0 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_5 backend hashModel tape path params i_0 w_149_0 w_114_0 w_116_0 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_5 backend hashModel params' i_0 w_149_0 w_114_0 w_116_0 w_178_0 w_178_1 w_179_0 w_179_1 w_180_0 w_180_1 w_181_0 w_181_1 w_182_0 w_182_1 w_183_0 w_183_1 w_184_0 w_184_1 w_185_0 w_185_1 w_186_0 w_186_1 w_187_0 w_187_1 w_188_0 w_188_1 w_189_0 w_189_1 w_190_0 w_190_1 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_193_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_178.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_179.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_180.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_181.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_182.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_183.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_184.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_185.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_186.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_187.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_188.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_189.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_190.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_191.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_192.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_193.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_4.bridge {backend hashModel tape path params params' i_0 w_149_0 w_114_0 w_116_0 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_4 backend hashModel tape path params i_0 w_149_0 w_114_0 w_116_0 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_4 backend hashModel params' i_0 w_149_0 w_114_0 w_116_0 w_162_0 w_162_1 w_163_0 w_163_1 w_164_0 w_164_1 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_167_1 w_168_0 w_168_1 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_171_1 w_172_0 w_172_1 w_173_0 w_173_1 w_174_0 w_174_1 w_175_0 w_175_1 w_176_0 w_176_1 w_177_0 w_177_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_162.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_163.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_164.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_165.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_166.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_167.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_168.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_169.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_170.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_171.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_172.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_173.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_174.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_175.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_176.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_177.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_3.bridge {backend hashModel tape path params params' i_0 w_11_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_29_0 w_114_0 w_116_0 w_119_0 w_120_0 w_121_0 w_131_0 w_136_0 w_140_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_3 backend hashModel tape path params i_0 w_11_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_29_0 w_114_0 w_116_0 w_119_0 w_120_0 w_121_0 w_131_0 w_136_0 w_140_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_3 backend hashModel params' i_0 w_11_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_29_0 w_114_0 w_116_0 w_119_0 w_120_0 w_121_0 w_131_0 w_136_0 w_140_0 w_148_0 w_148_1 w_148_2 w_148_3 w_148_4 w_148_5 w_148_6 w_150_0 w_150_1 w_151_0 w_151_1 w_152_0 w_152_1 w_153_0 w_153_1 w_154_0 w_154_1 w_155_0 w_155_1 w_156_0 w_156_1 w_157_0 w_157_1 w_158_0 w_158_1 w_159_0 w_159_1 w_160_0 w_160_1 w_161_0 w_161_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, (MxxExponentLut.gaussianSampleAt_sample (by decide) h4), h5, (MxxExponentLut.gaussianSampleAt_sample (by decide) h6), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_148.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_150.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_151.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_152.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_153.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_154.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_155.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_156.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_157.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_158.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_159.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_160.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_161.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_2.bridge {backend hashModel tape path params params' i_0 w_4_0 w_43_0 w_79_0 w_5_0 w_7_0 w_8_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_80_0 w_82_0 w_85_0 w_86_0 w_96_0 w_98_0 w_100_0 w_106_0 w_108_0 w_108_1 w_109_0 w_109_1 w_114_0 w_116_0}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_2 backend hashModel tape path params i_0 w_4_0 w_43_0 w_79_0 w_5_0 w_7_0 w_8_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_80_0 w_82_0 w_85_0 w_86_0 w_96_0 w_98_0 w_100_0 w_106_0 w_108_0 w_108_1 w_109_0 w_109_1 w_114_0 w_116_0) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_2 backend hashModel params' i_0 w_4_0 w_43_0 w_79_0 w_5_0 w_7_0 w_8_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_80_0 w_82_0 w_85_0 w_86_0 w_96_0 w_98_0 w_100_0 w_106_0 w_108_0 w_108_1 w_109_0 w_109_1 w_114_0 w_116_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_71.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_72.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_73.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_74.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_75.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_76.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_77.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_78.bridge (h16 i))), h17, h18, h19, h20, h21, h22, (MxxExponentLut.gaussianSampleAt_sample (by decide) h23), h24, h25, (MxxExponentLut.gaussianSampleAt_sample (by decide) h26), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_108.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_109.bridge (h30 i))), h31, h32⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_1.bridge {backend hashModel tape path params params' i_0 w_43_0 w_5_0 w_8_0 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_1 backend hashModel tape path params i_0 w_43_0 w_5_0 w_8_0 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_1 backend hashModel params' i_0 w_43_0 w_5_0 w_8_0 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_55.bridge (h2 i))), h3, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_56.bridge (h4 i))), h5, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_57.bridge (h6 i))), h7, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_58.bridge (h8 i))), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_59.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_60.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_61.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_62.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_63.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_64.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_65.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_66.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_67.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_68.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_69.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_70.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.constraints_0.bridge {backend hashModel tape path params params' i_0 w_4_0 w_11_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_14_0 w_24_0 w_29_0 w_30_0 w_34_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1}
    (h : Stage_encrypt.parallel_generatedRoot_45.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_14_0 w_24_0 w_29_0 w_30_0 w_34_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1) : W_Stage_encrypt.parallel_generatedRoot_45.constraints_0 backend hashModel params' i_0 w_4_0 w_11_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_5_0 w_7_0 w_8_0 w_12_0 w_13_0 w_14_0 w_24_0 w_29_0 w_30_0 w_34_0 w_42_0 w_42_1 w_42_2 w_42_3 w_42_4 w_42_5 w_42_6 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (MxxExponentLut.gaussianSampleAt_sample (by decide) h6), h7, (MxxExponentLut.gaussianSampleAt_sample (by decide) h8), h9, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_42.bridge (h10 i))), h11, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_44.bridge (h12 i))), h13, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_45.bridge (h14 i))), h15, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_46.bridge (h16 i))), h17, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_47.bridge (h18 i))), h19, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_48.bridge (h20 i))), h21, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_49.bridge (h22 i))), h23, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_50.bridge (h24 i))), h25, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_51.bridge (h26 i))), h27, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_52.bridge (h28 i))), h29, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_53.bridge (h30 i))), h31, (fun i ↦ (Stage_encrypt.parallel_parallel_generatedRoot_45_54.bridge (h32 i)))⟩)


theorem Stage_encrypt.parallel_generatedRoot_45.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_encrypt.parallel_generatedRoot_45 backend hashModel tape path params i_0 inputs outputs) : W_Stage_encrypt.parallel_generatedRoot_45 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_encrypt.parallel_generatedRoot_45
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_5_0, wit1.w_8_0, wit1.w_12_0, wit1.w_13_0, wit1.w_14_0, wit1.w_24_0, wit1.w_30_0, wit1.w_34_0, wit1.w_42_0, wit1.w_42_1, wit1.w_42_2, wit1.w_42_3, wit1.w_42_4, wit1.w_42_5, wit1.w_42_6, wit1.w_44_0, wit1.w_44_1, wit1.w_45_0, wit1.w_45_1, wit1.w_46_0, wit1.w_46_1, wit1.w_47_0, wit1.w_47_1, wit1.w_48_0, wit1.w_48_1, wit1.w_49_0, wit1.w_49_1, wit1.w_50_0, wit1.w_50_1, wit1.w_51_0, wit1.w_51_1, wit1.w_52_0, wit1.w_52_1, wit1.w_53_0, wit1.w_53_1, wit1.w_54_0, wit1.w_54_1, wit1.w_55_0, wit1.w_55_1, wit1.w_56_0, wit1.w_56_1, wit1.w_57_0, wit1.w_57_1, wit1.w_58_0, wit1.w_58_1, wit1.w_59_0, wit1.w_59_1, wit1.w_60_0, wit1.w_60_1, wit1.w_61_0, wit1.w_61_1, wit1.w_62_0, wit1.w_62_1, wit1.w_63_0, wit1.w_63_1, wit1.w_64_0, wit1.w_64_1, wit1.w_65_0, wit1.w_65_1, wit1.w_66_0, wit1.w_66_1, wit1.w_67_0, wit1.w_67_1, wit1.w_68_0, wit1.w_68_1, wit1.w_69_0, wit1.w_69_1, wit1.w_70_0, wit1.w_70_1, wit1.w_71_0, wit1.w_71_1, wit1.w_72_0, wit1.w_72_1, wit1.w_73_0, wit1.w_73_1, wit1.w_74_0, wit1.w_74_1, wit1.w_75_0, wit1.w_75_1, wit1.w_76_0, wit1.w_76_1, wit1.w_77_0, wit1.w_77_1, wit1.w_78_0, wit1.w_78_1, wit1.w_80_0, wit1.w_82_0, wit1.w_85_0, wit1.w_86_0, wit1.w_96_0, wit1.w_98_0, wit1.w_100_0, wit1.w_106_0, wit1.w_108_0, wit1.w_108_1, wit1.w_109_0, wit1.w_109_1, wit1.w_114_0, wit1.w_116_0, wit1.w_119_0, wit1.w_120_0, wit1.w_121_0, wit1.w_131_0, wit1.w_136_0, wit1.w_140_0, wit1.w_148_0, wit1.w_148_1, wit1.w_148_2, wit1.w_148_3, wit1.w_148_4, wit1.w_148_5, wit1.w_148_6, wit1.w_150_0, wit1.w_150_1, wit1.w_151_0, wit1.w_151_1, wit1.w_152_0, wit1.w_152_1, wit1.w_153_0, wit1.w_153_1, wit1.w_154_0, wit1.w_154_1, wit1.w_155_0, wit1.w_155_1, wit1.w_156_0, wit1.w_156_1, wit1.w_157_0, wit1.w_157_1, wit1.w_158_0, wit1.w_158_1, wit1.w_159_0, wit1.w_159_1, wit1.w_160_0, wit1.w_160_1, wit1.w_161_0, wit1.w_161_1, wit1.w_162_0, wit1.w_162_1, wit1.w_163_0, wit1.w_163_1, wit1.w_164_0, wit1.w_164_1, wit1.w_165_0, wit1.w_165_1, wit1.w_166_0, wit1.w_166_1, wit1.w_167_0, wit1.w_167_1, wit1.w_168_0, wit1.w_168_1, wit1.w_169_0, wit1.w_169_1, wit1.w_170_0, wit1.w_170_1, wit1.w_171_0, wit1.w_171_1, wit1.w_172_0, wit1.w_172_1, wit1.w_173_0, wit1.w_173_1, wit1.w_174_0, wit1.w_174_1, wit1.w_175_0, wit1.w_175_1, wit1.w_176_0, wit1.w_176_1, wit1.w_177_0, wit1.w_177_1, wit1.w_178_0, wit1.w_178_1, wit1.w_179_0, wit1.w_179_1, wit1.w_180_0, wit1.w_180_1, wit1.w_181_0, wit1.w_181_1, wit1.w_182_0, wit1.w_182_1, wit1.w_183_0, wit1.w_183_1, wit1.w_184_0, wit1.w_184_1, wit1.w_185_0, wit1.w_185_1, wit1.w_186_0, wit1.w_186_1, wit1.w_187_0, wit1.w_187_1, wit1.w_188_0, wit1.w_188_1, wit1.w_189_0, wit1.w_189_1, wit1.w_190_0, wit1.w_190_1, wit1.w_191_0, wit1.w_191_1, wit1.w_192_0, wit1.w_192_1, wit1.w_193_0, wit1.w_193_1, wit1.w_194_0, wit1.w_194_1, wit1.w_195_0, wit1.w_195_1, wit1.w_196_0, wit1.w_196_1, wit1.w_197_0, wit1.w_197_1, wit1.w_198_0, wit1.w_198_1, wit1.w_199_0, wit1.w_199_1, wit1.w_200_0, wit1.w_200_1, wit1.w_201_0, wit1.w_201_1, wit1.w_202_0, wit1.w_202_1, wit1.w_203_0, wit1.w_203_1, wit1.w_204_0, wit1.w_204_1, wit1.w_205_0, wit1.w_205_1, wit1.w_206_0, wit1.w_206_1, wit1.w_207_0, wit1.w_207_1, wit1.w_208_0, wit1.w_208_1, wit1.w_209_0, wit1.w_209_1, wit1.w_210_0, wit1.w_210_1, wit1.w_211_0, wit1.w_211_1, wit1.w_212_0, wit1.w_212_1, wit1.w_213_0, wit1.w_213_1, wit1.w_214_0, wit1.w_214_1, wit1.w_215_0, wit1.w_215_1, wit1.w_216_0, wit1.w_216_1, wit1.w_217_0, wit1.w_217_1, wit1.w_218_0, wit1.w_218_1, wit1.w_219_0, wit1.w_219_1, wit1.w_220_0, wit1.w_220_1, wit1.w_221_0, wit1.w_221_1, wit1.w_226_0, wit1.w_228_0, wit1.w_231_0, wit1.w_232_0, wit1.w_233_0, wit1.w_243_0, wit1.w_248_0, wit1.w_252_0, wit1.w_260_0, wit1.w_260_1, wit1.w_260_2, wit1.w_260_3, wit1.w_260_4, wit1.w_260_5, wit1.w_260_6, wit1.w_262_0, wit1.w_262_1, wit1.w_263_0, wit1.w_263_1, wit1.w_264_0, wit1.w_264_1, wit1.w_265_0, wit1.w_265_1, wit1.w_266_0, wit1.w_266_1, wit1.w_267_0, wit1.w_267_1, wit1.w_268_0, wit1.w_268_1, wit1.w_269_0, wit1.w_269_1, wit1.w_270_0, wit1.w_270_1, wit1.w_271_0, wit1.w_271_1, wit1.w_272_0, wit1.w_272_1, wit1.w_273_0, wit1.w_273_1, wit1.w_274_0, wit1.w_274_1, wit1.w_275_0, wit1.w_275_1, wit1.w_276_0, wit1.w_276_1, wit1.w_277_0, wit1.w_277_1, wit1.w_278_0, wit1.w_278_1, wit1.w_279_0, wit1.w_279_1, wit1.w_280_0, wit1.w_280_1, wit1.w_281_0, wit1.w_281_1, wit1.w_282_0, wit1.w_282_1, wit1.w_283_0, wit1.w_283_1, wit1.w_284_0, wit1.w_284_1, wit1.w_285_0, wit1.w_285_1, wit1.w_286_0, wit1.w_286_1, wit1.w_287_0, wit1.w_287_1, wit1.w_288_0, wit1.w_288_1, wit1.w_289_0, wit1.w_289_1, wit1.w_290_0, wit1.w_290_1, wit1.w_291_0, wit1.w_291_1, wit1.w_292_0, wit1.w_292_1, wit1.w_293_0, wit1.w_293_1, wit1.w_294_0, wit1.w_294_1, wit1.w_295_0, wit1.w_295_1, wit1.w_296_0, wit1.w_296_1, wit1.w_297_0, wit1.w_297_1, wit1.w_298_0, wit1.w_298_1, wit1.w_299_0, wit1.w_299_1, wit1.w_300_0, wit1.w_300_1, wit1.w_301_0, wit1.w_301_1, wit1.w_302_0, wit1.w_302_1, wit1.w_303_0, wit1.w_303_1, wit1.w_304_0, wit1.w_304_1, wit1.w_305_0, wit1.w_305_1, wit1.w_306_0, wit1.w_306_1, wit1.w_307_0, wit1.w_307_1, wit1.w_308_0, wit1.w_308_1, wit1.w_309_0, wit1.w_309_1, wit1.w_310_0, wit1.w_310_1, wit1.w_311_0, wit1.w_311_1, wit1.w_312_0, wit1.w_312_1, wit1.w_313_0, wit1.w_313_1, wit1.w_314_0, wit1.w_314_1, wit1.w_315_0, wit1.w_315_1, wit1.w_316_0, wit1.w_316_1, wit1.w_317_0, wit1.w_317_1, wit1.w_318_0, wit1.w_318_1, wit1.w_319_0, wit1.w_319_1, wit1.w_320_0, wit1.w_320_1, wit1.w_321_0, wit1.w_321_1, wit1.w_322_0, wit1.w_322_1, wit1.w_323_0, wit1.w_323_1, wit1.w_324_0, wit1.w_324_1, wit1.w_325_0, wit1.w_325_1, wit1.w_326_0, wit1.w_326_1, wit1.w_327_0, wit1.w_327_1, wit1.w_328_0, wit1.w_328_1, wit1.w_329_0, wit1.w_329_1, wit1.w_330_0, wit1.w_330_1, wit1.w_331_0, wit1.w_331_1, wit1.w_332_0, wit1.w_332_1, wit1.w_333_0, wit1.w_333_1, wit1.w_334_0, wit1.w_335_0, wit1.w_337_0, wit1.w_338_0, wit1.w_342_0, wit1.w_346_0, wit1.w_348_0, wit1.w_350_0, wit1.w_355_0, wit1.w_357_0, wit1.w_358_0, wit1.w_359_0, wit1.w_362_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16⟩ := h2; exact ⟨(Stage_encrypt.parallel_generatedRoot_45.constraints_0.bridge h3), (Stage_encrypt.parallel_generatedRoot_45.constraints_1.bridge h4), (Stage_encrypt.parallel_generatedRoot_45.constraints_2.bridge h5), (Stage_encrypt.parallel_generatedRoot_45.constraints_3.bridge h6), (Stage_encrypt.parallel_generatedRoot_45.constraints_4.bridge h7), (Stage_encrypt.parallel_generatedRoot_45.constraints_5.bridge h8), (Stage_encrypt.parallel_generatedRoot_45.constraints_6.bridge h9), (Stage_encrypt.parallel_generatedRoot_45.constraints_7.bridge h10), (Stage_encrypt.parallel_generatedRoot_45.constraints_8.bridge h11), (Stage_encrypt.parallel_generatedRoot_45.constraints_9.bridge h12), (Stage_encrypt.parallel_generatedRoot_45.constraints_10.bridge h13), (Stage_encrypt.parallel_generatedRoot_45.constraints_11.bridge h14), (Stage_encrypt.parallel_generatedRoot_45.constraints_12.bridge h15), (Stage_encrypt.parallel_generatedRoot_45.constraints_13.bridge h16)⟩)⟩)


theorem Stage_encrypt.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_45_1 w_45_2 w_45_3 w_45_4 w_45_5 w_45_6 w_45_7 w_45_8 w_45_9 w_45_10 w_45_11 w_45_12 w_45_13 w_45_14 w_45_15 w_45_16 w_45_17 w_45_18 w_45_19 w_45_20 w_45_21 w_45_22 w_45_23 w_45_24 w_45_25 w_45_26 w_45_27 w_45_28 w_45_29 w_45_30 w_45_31 w_45_32 w_45_33 w_45_34 w_45_35 w_45_36 w_45_37 w_45_38 w_45_39 w_45_40 w_45_41 w_45_42 w_45_43 w_45_44 w_45_45 w_45_46 w_45_47 w_45_48 w_45_49 w_45_50 w_45_51 w_45_52 w_45_53 w_45_54 w_45_55 w_45_56 w_45_57 w_45_58 w_45_59 w_45_60 w_45_61 w_45_62 w_45_63 w_45_64 w_45_65 w_45_66 w_45_67 w_45_68 w_45_69 w_45_70 w_45_71 w_45_72 w_45_73 w_45_74 w_45_75 w_45_76 w_45_77 w_45_78 w_45_79 w_45_80 w_45_81 w_45_82 w_45_83 w_45_84 w_45_85 w_45_86 w_45_87 w_45_88 w_45_89 w_45_90 w_45_91 w_45_92 w_45_93 w_45_94 w_45_95 w_45_96 w_45_97 w_45_98 w_45_99 w_45_100 w_45_101 w_45_102 w_45_103 w_45_104 w_45_105 w_45_106 w_45_107 w_45_108 w_45_109 w_45_110 w_45_111 w_45_112 w_45_113 w_45_114 w_45_115 w_45_116 w_45_117 w_45_118 w_45_119 w_45_120 w_45_121 w_45_122 w_45_123 w_45_124 w_45_125 w_45_126 w_45_127 w_45_128 w_45_129 w_45_130 w_45_131 w_45_132 w_45_133 w_45_134 w_45_135 w_45_136 w_45_137 w_45_138 w_45_139 w_45_140 w_45_141 w_45_142 w_45_143 w_45_144 w_45_145 w_45_146 w_45_147 w_45_148 w_45_149 w_45_150 w_45_151 w_45_152 w_45_153 w_45_154 w_45_155 w_45_156 w_45_157 w_45_158 w_45_159 w_45_160 w_45_161 w_45_162 w_45_163 w_45_164 w_45_165 w_45_166 w_45_167 w_45_168 w_45_169 w_45_170 w_45_171 w_45_172 w_45_173 w_45_174 w_45_175 w_45_176 w_45_177 w_45_178 w_45_179 w_45_180 w_45_181 w_45_182 w_45_183 w_45_184 w_45_185 w_45_186 w_45_187 w_45_188 w_45_189 w_45_190 w_45_191 w_45_192 w_45_193 w_45_194 w_45_195 w_45_196 w_45_197 w_45_198 w_45_199 w_45_200 w_45_201 w_45_202 w_45_203 w_45_204 w_45_205 w_45_206 w_45_207 w_45_208 w_45_209 w_45_210 w_45_211 w_45_212 w_45_213 w_45_214 w_45_215 w_45_216 w_45_217 w_45_218 w_45_219 w_45_220 w_45_221 w_45_222 w_45_223 w_45_224 w_45_225 w_45_226 w_45_227 w_45_228 w_45_229 w_45_230 w_45_231 w_45_232 w_45_233 w_45_234 w_45_235 w_45_236 w_45_237 w_45_238 w_45_239 w_45_240 w_45_241 w_45_242 w_45_243 w_45_244 w_45_245 w_45_246 w_45_247 w_45_248 w_45_249 w_45_250 w_45_251 w_45_252 w_45_253 w_45_254 w_45_255 w_45_256 w_45_257 w_45_258 w_45_259 w_45_260 w_45_261 w_45_262 w_45_263 w_45_264 w_45_265 w_45_266 w_45_267 w_45_268 w_45_269 w_45_270 w_45_271 w_45_272 w_45_273 w_45_274 w_45_275 w_45_276 w_45_277 w_45_278 w_45_279 w_45_280 w_45_281 w_45_282 w_45_283 w_45_284 w_45_285 w_45_286 w_45_287 w_45_288 w_45_289 w_45_290 w_45_291 w_45_292 w_45_293 w_45_294 w_45_295 w_45_296 w_45_297 w_45_298 w_45_299 w_45_300 w_45_301 w_45_302 w_45_303 w_45_304 w_45_305 w_45_306 w_45_307 w_45_308 w_45_309 w_45_310 w_45_311 w_45_312 w_45_313 w_45_314 w_45_315 w_45_316 w_45_317 w_45_318 w_45_319 w_45_320 w_45_321 w_45_322 w_45_323 w_45_324 w_45_325 w_45_326 w_45_327 w_45_328 w_45_329 w_45_330 w_45_331 w_45_332 w_45_333 w_45_334 w_45_335 w_45_336 w_45_337 w_45_338 w_45_339 w_45_340 w_45_341 w_45_342 w_45_343 w_45_344 w_45_345 w_45_346 w_45_347 w_45_348 w_45_349 w_45_350 w_45_351 w_45_352 w_45_353 w_45_354 w_45_355 w_45_356 w_45_357 w_45_358 w_45_359 w_45_360 w_45_361 w_45_362 w_45_363 w_45_364 w_45_365 w_45_366 w_45_367 w_45_368 w_45_369 w_45_370 w_45_371 w_45_372 w_45_373 w_45_374 w_45_375 w_45_376 w_45_377 w_45_378 w_45_379 w_45_380 w_45_381 w_45_382 w_45_383 w_45_384 w_45_385 w_45_386 w_45_387 w_45_388 w_45_389 w_45_390 w_45_391 w_45_392 w_45_393 w_45_394 w_45_395 w_45_396 w_45_397 w_45_398 w_45_399 w_45_400 w_45_401 w_45_402 w_45_403 w_45_404 w_45_405 outputs}
    (h : Stage_encrypt.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_45_1 w_45_2 w_45_3 w_45_4 w_45_5 w_45_6 w_45_7 w_45_8 w_45_9 w_45_10 w_45_11 w_45_12 w_45_13 w_45_14 w_45_15 w_45_16 w_45_17 w_45_18 w_45_19 w_45_20 w_45_21 w_45_22 w_45_23 w_45_24 w_45_25 w_45_26 w_45_27 w_45_28 w_45_29 w_45_30 w_45_31 w_45_32 w_45_33 w_45_34 w_45_35 w_45_36 w_45_37 w_45_38 w_45_39 w_45_40 w_45_41 w_45_42 w_45_43 w_45_44 w_45_45 w_45_46 w_45_47 w_45_48 w_45_49 w_45_50 w_45_51 w_45_52 w_45_53 w_45_54 w_45_55 w_45_56 w_45_57 w_45_58 w_45_59 w_45_60 w_45_61 w_45_62 w_45_63 w_45_64 w_45_65 w_45_66 w_45_67 w_45_68 w_45_69 w_45_70 w_45_71 w_45_72 w_45_73 w_45_74 w_45_75 w_45_76 w_45_77 w_45_78 w_45_79 w_45_80 w_45_81 w_45_82 w_45_83 w_45_84 w_45_85 w_45_86 w_45_87 w_45_88 w_45_89 w_45_90 w_45_91 w_45_92 w_45_93 w_45_94 w_45_95 w_45_96 w_45_97 w_45_98 w_45_99 w_45_100 w_45_101 w_45_102 w_45_103 w_45_104 w_45_105 w_45_106 w_45_107 w_45_108 w_45_109 w_45_110 w_45_111 w_45_112 w_45_113 w_45_114 w_45_115 w_45_116 w_45_117 w_45_118 w_45_119 w_45_120 w_45_121 w_45_122 w_45_123 w_45_124 w_45_125 w_45_126 w_45_127 w_45_128 w_45_129 w_45_130 w_45_131 w_45_132 w_45_133 w_45_134 w_45_135 w_45_136 w_45_137 w_45_138 w_45_139 w_45_140 w_45_141 w_45_142 w_45_143 w_45_144 w_45_145 w_45_146 w_45_147 w_45_148 w_45_149 w_45_150 w_45_151 w_45_152 w_45_153 w_45_154 w_45_155 w_45_156 w_45_157 w_45_158 w_45_159 w_45_160 w_45_161 w_45_162 w_45_163 w_45_164 w_45_165 w_45_166 w_45_167 w_45_168 w_45_169 w_45_170 w_45_171 w_45_172 w_45_173 w_45_174 w_45_175 w_45_176 w_45_177 w_45_178 w_45_179 w_45_180 w_45_181 w_45_182 w_45_183 w_45_184 w_45_185 w_45_186 w_45_187 w_45_188 w_45_189 w_45_190 w_45_191 w_45_192 w_45_193 w_45_194 w_45_195 w_45_196 w_45_197 w_45_198 w_45_199 w_45_200 w_45_201 w_45_202 w_45_203 w_45_204 w_45_205 w_45_206 w_45_207 w_45_208 w_45_209 w_45_210 w_45_211 w_45_212 w_45_213 w_45_214 w_45_215 w_45_216 w_45_217 w_45_218 w_45_219 w_45_220 w_45_221 w_45_222 w_45_223 w_45_224 w_45_225 w_45_226 w_45_227 w_45_228 w_45_229 w_45_230 w_45_231 w_45_232 w_45_233 w_45_234 w_45_235 w_45_236 w_45_237 w_45_238 w_45_239 w_45_240 w_45_241 w_45_242 w_45_243 w_45_244 w_45_245 w_45_246 w_45_247 w_45_248 w_45_249 w_45_250 w_45_251 w_45_252 w_45_253 w_45_254 w_45_255 w_45_256 w_45_257 w_45_258 w_45_259 w_45_260 w_45_261 w_45_262 w_45_263 w_45_264 w_45_265 w_45_266 w_45_267 w_45_268 w_45_269 w_45_270 w_45_271 w_45_272 w_45_273 w_45_274 w_45_275 w_45_276 w_45_277 w_45_278 w_45_279 w_45_280 w_45_281 w_45_282 w_45_283 w_45_284 w_45_285 w_45_286 w_45_287 w_45_288 w_45_289 w_45_290 w_45_291 w_45_292 w_45_293 w_45_294 w_45_295 w_45_296 w_45_297 w_45_298 w_45_299 w_45_300 w_45_301 w_45_302 w_45_303 w_45_304 w_45_305 w_45_306 w_45_307 w_45_308 w_45_309 w_45_310 w_45_311 w_45_312 w_45_313 w_45_314 w_45_315 w_45_316 w_45_317 w_45_318 w_45_319 w_45_320 w_45_321 w_45_322 w_45_323 w_45_324 w_45_325 w_45_326 w_45_327 w_45_328 w_45_329 w_45_330 w_45_331 w_45_332 w_45_333 w_45_334 w_45_335 w_45_336 w_45_337 w_45_338 w_45_339 w_45_340 w_45_341 w_45_342 w_45_343 w_45_344 w_45_345 w_45_346 w_45_347 w_45_348 w_45_349 w_45_350 w_45_351 w_45_352 w_45_353 w_45_354 w_45_355 w_45_356 w_45_357 w_45_358 w_45_359 w_45_360 w_45_361 w_45_362 w_45_363 w_45_364 w_45_365 w_45_366 w_45_367 w_45_368 w_45_369 w_45_370 w_45_371 w_45_372 w_45_373 w_45_374 w_45_375 w_45_376 w_45_377 w_45_378 w_45_379 w_45_380 w_45_381 w_45_382 w_45_383 w_45_384 w_45_385 w_45_386 w_45_387 w_45_388 w_45_389 w_45_390 w_45_391 w_45_392 w_45_393 w_45_394 w_45_395 w_45_396 w_45_397 w_45_398 w_45_399 w_45_400 w_45_401 w_45_402 w_45_403 w_45_404 w_45_405 outputs) : W_Stage_encrypt.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_45_1 w_45_2 w_45_3 w_45_4 w_45_5 w_45_6 w_45_7 w_45_8 w_45_9 w_45_10 w_45_11 w_45_12 w_45_13 w_45_14 w_45_15 w_45_16 w_45_17 w_45_18 w_45_19 w_45_20 w_45_21 w_45_22 w_45_23 w_45_24 w_45_25 w_45_26 w_45_27 w_45_28 w_45_29 w_45_30 w_45_31 w_45_32 w_45_33 w_45_34 w_45_35 w_45_36 w_45_37 w_45_38 w_45_39 w_45_40 w_45_41 w_45_42 w_45_43 w_45_44 w_45_45 w_45_46 w_45_47 w_45_48 w_45_49 w_45_50 w_45_51 w_45_52 w_45_53 w_45_54 w_45_55 w_45_56 w_45_57 w_45_58 w_45_59 w_45_60 w_45_61 w_45_62 w_45_63 w_45_64 w_45_65 w_45_66 w_45_67 w_45_68 w_45_69 w_45_70 w_45_71 w_45_72 w_45_73 w_45_74 w_45_75 w_45_76 w_45_77 w_45_78 w_45_79 w_45_80 w_45_81 w_45_82 w_45_83 w_45_84 w_45_85 w_45_86 w_45_87 w_45_88 w_45_89 w_45_90 w_45_91 w_45_92 w_45_93 w_45_94 w_45_95 w_45_96 w_45_97 w_45_98 w_45_99 w_45_100 w_45_101 w_45_102 w_45_103 w_45_104 w_45_105 w_45_106 w_45_107 w_45_108 w_45_109 w_45_110 w_45_111 w_45_112 w_45_113 w_45_114 w_45_115 w_45_116 w_45_117 w_45_118 w_45_119 w_45_120 w_45_121 w_45_122 w_45_123 w_45_124 w_45_125 w_45_126 w_45_127 w_45_128 w_45_129 w_45_130 w_45_131 w_45_132 w_45_133 w_45_134 w_45_135 w_45_136 w_45_137 w_45_138 w_45_139 w_45_140 w_45_141 w_45_142 w_45_143 w_45_144 w_45_145 w_45_146 w_45_147 w_45_148 w_45_149 w_45_150 w_45_151 w_45_152 w_45_153 w_45_154 w_45_155 w_45_156 w_45_157 w_45_158 w_45_159 w_45_160 w_45_161 w_45_162 w_45_163 w_45_164 w_45_165 w_45_166 w_45_167 w_45_168 w_45_169 w_45_170 w_45_171 w_45_172 w_45_173 w_45_174 w_45_175 w_45_176 w_45_177 w_45_178 w_45_179 w_45_180 w_45_181 w_45_182 w_45_183 w_45_184 w_45_185 w_45_186 w_45_187 w_45_188 w_45_189 w_45_190 w_45_191 w_45_192 w_45_193 w_45_194 w_45_195 w_45_196 w_45_197 w_45_198 w_45_199 w_45_200 w_45_201 w_45_202 w_45_203 w_45_204 w_45_205 w_45_206 w_45_207 w_45_208 w_45_209 w_45_210 w_45_211 w_45_212 w_45_213 w_45_214 w_45_215 w_45_216 w_45_217 w_45_218 w_45_219 w_45_220 w_45_221 w_45_222 w_45_223 w_45_224 w_45_225 w_45_226 w_45_227 w_45_228 w_45_229 w_45_230 w_45_231 w_45_232 w_45_233 w_45_234 w_45_235 w_45_236 w_45_237 w_45_238 w_45_239 w_45_240 w_45_241 w_45_242 w_45_243 w_45_244 w_45_245 w_45_246 w_45_247 w_45_248 w_45_249 w_45_250 w_45_251 w_45_252 w_45_253 w_45_254 w_45_255 w_45_256 w_45_257 w_45_258 w_45_259 w_45_260 w_45_261 w_45_262 w_45_263 w_45_264 w_45_265 w_45_266 w_45_267 w_45_268 w_45_269 w_45_270 w_45_271 w_45_272 w_45_273 w_45_274 w_45_275 w_45_276 w_45_277 w_45_278 w_45_279 w_45_280 w_45_281 w_45_282 w_45_283 w_45_284 w_45_285 w_45_286 w_45_287 w_45_288 w_45_289 w_45_290 w_45_291 w_45_292 w_45_293 w_45_294 w_45_295 w_45_296 w_45_297 w_45_298 w_45_299 w_45_300 w_45_301 w_45_302 w_45_303 w_45_304 w_45_305 w_45_306 w_45_307 w_45_308 w_45_309 w_45_310 w_45_311 w_45_312 w_45_313 w_45_314 w_45_315 w_45_316 w_45_317 w_45_318 w_45_319 w_45_320 w_45_321 w_45_322 w_45_323 w_45_324 w_45_325 w_45_326 w_45_327 w_45_328 w_45_329 w_45_330 w_45_331 w_45_332 w_45_333 w_45_334 w_45_335 w_45_336 w_45_337 w_45_338 w_45_339 w_45_340 w_45_341 w_45_342 w_45_343 w_45_344 w_45_345 w_45_346 w_45_347 w_45_348 w_45_349 w_45_350 w_45_351 w_45_352 w_45_353 w_45_354 w_45_355 w_45_356 w_45_357 w_45_358 w_45_359 w_45_360 w_45_361 w_45_362 w_45_363 w_45_364 w_45_365 w_45_366 w_45_367 w_45_368 w_45_369 w_45_370 w_45_371 w_45_372 w_45_373 w_45_374 w_45_375 w_45_376 w_45_377 w_45_378 w_45_379 w_45_380 w_45_381 w_45_382 w_45_383 w_45_384 w_45_385 w_45_386 w_45_387 w_45_388 w_45_389 w_45_390 w_45_391 w_45_392 w_45_393 w_45_394 w_45_395 w_45_396 w_45_397 w_45_398 w_45_399 w_45_400 w_45_401 w_45_402 w_45_403 w_45_404 w_45_405 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (fun i ↦ (Stage_encrypt.parallel_generatedRoot_45.bridge (h2 i))), h3⟩)


theorem Stage_encrypt.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_encrypt.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_encrypt.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_encrypt.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_45_0, wit1.w_45_1, wit1.w_45_2, wit1.w_45_3, wit1.w_45_4, wit1.w_45_5, wit1.w_45_6, wit1.w_45_7, wit1.w_45_8, wit1.w_45_9, wit1.w_45_10, wit1.w_45_11, wit1.w_45_12, wit1.w_45_13, wit1.w_45_14, wit1.w_45_15, wit1.w_45_16, wit1.w_45_17, wit1.w_45_18, wit1.w_45_19, wit1.w_45_20, wit1.w_45_21, wit1.w_45_22, wit1.w_45_23, wit1.w_45_24, wit1.w_45_25, wit1.w_45_26, wit1.w_45_27, wit1.w_45_28, wit1.w_45_29, wit1.w_45_30, wit1.w_45_31, wit1.w_45_32, wit1.w_45_33, wit1.w_45_34, wit1.w_45_35, wit1.w_45_36, wit1.w_45_37, wit1.w_45_38, wit1.w_45_39, wit1.w_45_40, wit1.w_45_41, wit1.w_45_42, wit1.w_45_43, wit1.w_45_44, wit1.w_45_45, wit1.w_45_46, wit1.w_45_47, wit1.w_45_48, wit1.w_45_49, wit1.w_45_50, wit1.w_45_51, wit1.w_45_52, wit1.w_45_53, wit1.w_45_54, wit1.w_45_55, wit1.w_45_56, wit1.w_45_57, wit1.w_45_58, wit1.w_45_59, wit1.w_45_60, wit1.w_45_61, wit1.w_45_62, wit1.w_45_63, wit1.w_45_64, wit1.w_45_65, wit1.w_45_66, wit1.w_45_67, wit1.w_45_68, wit1.w_45_69, wit1.w_45_70, wit1.w_45_71, wit1.w_45_72, wit1.w_45_73, wit1.w_45_74, wit1.w_45_75, wit1.w_45_76, wit1.w_45_77, wit1.w_45_78, wit1.w_45_79, wit1.w_45_80, wit1.w_45_81, wit1.w_45_82, wit1.w_45_83, wit1.w_45_84, wit1.w_45_85, wit1.w_45_86, wit1.w_45_87, wit1.w_45_88, wit1.w_45_89, wit1.w_45_90, wit1.w_45_91, wit1.w_45_92, wit1.w_45_93, wit1.w_45_94, wit1.w_45_95, wit1.w_45_96, wit1.w_45_97, wit1.w_45_98, wit1.w_45_99, wit1.w_45_100, wit1.w_45_101, wit1.w_45_102, wit1.w_45_103, wit1.w_45_104, wit1.w_45_105, wit1.w_45_106, wit1.w_45_107, wit1.w_45_108, wit1.w_45_109, wit1.w_45_110, wit1.w_45_111, wit1.w_45_112, wit1.w_45_113, wit1.w_45_114, wit1.w_45_115, wit1.w_45_116, wit1.w_45_117, wit1.w_45_118, wit1.w_45_119, wit1.w_45_120, wit1.w_45_121, wit1.w_45_122, wit1.w_45_123, wit1.w_45_124, wit1.w_45_125, wit1.w_45_126, wit1.w_45_127, wit1.w_45_128, wit1.w_45_129, wit1.w_45_130, wit1.w_45_131, wit1.w_45_132, wit1.w_45_133, wit1.w_45_134, wit1.w_45_135, wit1.w_45_136, wit1.w_45_137, wit1.w_45_138, wit1.w_45_139, wit1.w_45_140, wit1.w_45_141, wit1.w_45_142, wit1.w_45_143, wit1.w_45_144, wit1.w_45_145, wit1.w_45_146, wit1.w_45_147, wit1.w_45_148, wit1.w_45_149, wit1.w_45_150, wit1.w_45_151, wit1.w_45_152, wit1.w_45_153, wit1.w_45_154, wit1.w_45_155, wit1.w_45_156, wit1.w_45_157, wit1.w_45_158, wit1.w_45_159, wit1.w_45_160, wit1.w_45_161, wit1.w_45_162, wit1.w_45_163, wit1.w_45_164, wit1.w_45_165, wit1.w_45_166, wit1.w_45_167, wit1.w_45_168, wit1.w_45_169, wit1.w_45_170, wit1.w_45_171, wit1.w_45_172, wit1.w_45_173, wit1.w_45_174, wit1.w_45_175, wit1.w_45_176, wit1.w_45_177, wit1.w_45_178, wit1.w_45_179, wit1.w_45_180, wit1.w_45_181, wit1.w_45_182, wit1.w_45_183, wit1.w_45_184, wit1.w_45_185, wit1.w_45_186, wit1.w_45_187, wit1.w_45_188, wit1.w_45_189, wit1.w_45_190, wit1.w_45_191, wit1.w_45_192, wit1.w_45_193, wit1.w_45_194, wit1.w_45_195, wit1.w_45_196, wit1.w_45_197, wit1.w_45_198, wit1.w_45_199, wit1.w_45_200, wit1.w_45_201, wit1.w_45_202, wit1.w_45_203, wit1.w_45_204, wit1.w_45_205, wit1.w_45_206, wit1.w_45_207, wit1.w_45_208, wit1.w_45_209, wit1.w_45_210, wit1.w_45_211, wit1.w_45_212, wit1.w_45_213, wit1.w_45_214, wit1.w_45_215, wit1.w_45_216, wit1.w_45_217, wit1.w_45_218, wit1.w_45_219, wit1.w_45_220, wit1.w_45_221, wit1.w_45_222, wit1.w_45_223, wit1.w_45_224, wit1.w_45_225, wit1.w_45_226, wit1.w_45_227, wit1.w_45_228, wit1.w_45_229, wit1.w_45_230, wit1.w_45_231, wit1.w_45_232, wit1.w_45_233, wit1.w_45_234, wit1.w_45_235, wit1.w_45_236, wit1.w_45_237, wit1.w_45_238, wit1.w_45_239, wit1.w_45_240, wit1.w_45_241, wit1.w_45_242, wit1.w_45_243, wit1.w_45_244, wit1.w_45_245, wit1.w_45_246, wit1.w_45_247, wit1.w_45_248, wit1.w_45_249, wit1.w_45_250, wit1.w_45_251, wit1.w_45_252, wit1.w_45_253, wit1.w_45_254, wit1.w_45_255, wit1.w_45_256, wit1.w_45_257, wit1.w_45_258, wit1.w_45_259, wit1.w_45_260, wit1.w_45_261, wit1.w_45_262, wit1.w_45_263, wit1.w_45_264, wit1.w_45_265, wit1.w_45_266, wit1.w_45_267, wit1.w_45_268, wit1.w_45_269, wit1.w_45_270, wit1.w_45_271, wit1.w_45_272, wit1.w_45_273, wit1.w_45_274, wit1.w_45_275, wit1.w_45_276, wit1.w_45_277, wit1.w_45_278, wit1.w_45_279, wit1.w_45_280, wit1.w_45_281, wit1.w_45_282, wit1.w_45_283, wit1.w_45_284, wit1.w_45_285, wit1.w_45_286, wit1.w_45_287, wit1.w_45_288, wit1.w_45_289, wit1.w_45_290, wit1.w_45_291, wit1.w_45_292, wit1.w_45_293, wit1.w_45_294, wit1.w_45_295, wit1.w_45_296, wit1.w_45_297, wit1.w_45_298, wit1.w_45_299, wit1.w_45_300, wit1.w_45_301, wit1.w_45_302, wit1.w_45_303, wit1.w_45_304, wit1.w_45_305, wit1.w_45_306, wit1.w_45_307, wit1.w_45_308, wit1.w_45_309, wit1.w_45_310, wit1.w_45_311, wit1.w_45_312, wit1.w_45_313, wit1.w_45_314, wit1.w_45_315, wit1.w_45_316, wit1.w_45_317, wit1.w_45_318, wit1.w_45_319, wit1.w_45_320, wit1.w_45_321, wit1.w_45_322, wit1.w_45_323, wit1.w_45_324, wit1.w_45_325, wit1.w_45_326, wit1.w_45_327, wit1.w_45_328, wit1.w_45_329, wit1.w_45_330, wit1.w_45_331, wit1.w_45_332, wit1.w_45_333, wit1.w_45_334, wit1.w_45_335, wit1.w_45_336, wit1.w_45_337, wit1.w_45_338, wit1.w_45_339, wit1.w_45_340, wit1.w_45_341, wit1.w_45_342, wit1.w_45_343, wit1.w_45_344, wit1.w_45_345, wit1.w_45_346, wit1.w_45_347, wit1.w_45_348, wit1.w_45_349, wit1.w_45_350, wit1.w_45_351, wit1.w_45_352, wit1.w_45_353, wit1.w_45_354, wit1.w_45_355, wit1.w_45_356, wit1.w_45_357, wit1.w_45_358, wit1.w_45_359, wit1.w_45_360, wit1.w_45_361, wit1.w_45_362, wit1.w_45_363, wit1.w_45_364, wit1.w_45_365, wit1.w_45_366, wit1.w_45_367, wit1.w_45_368, wit1.w_45_369, wit1.w_45_370, wit1.w_45_371, wit1.w_45_372, wit1.w_45_373, wit1.w_45_374, wit1.w_45_375, wit1.w_45_376, wit1.w_45_377, wit1.w_45_378, wit1.w_45_379, wit1.w_45_380, wit1.w_45_381, wit1.w_45_382, wit1.w_45_383, wit1.w_45_384, wit1.w_45_385, wit1.w_45_386, wit1.w_45_387, wit1.w_45_388, wit1.w_45_389, wit1.w_45_390, wit1.w_45_391, wit1.w_45_392, wit1.w_45_393, wit1.w_45_394, wit1.w_45_395, wit1.w_45_396, wit1.w_45_397, wit1.w_45_398, wit1.w_45_399, wit1.w_45_400, wit1.w_45_401, wit1.w_45_402, wit1.w_45_403, wit1.w_45_404, wit1.w_45_405⟩, (Stage_encrypt.generatedRoot.constraints_0.bridge h2)⟩)
