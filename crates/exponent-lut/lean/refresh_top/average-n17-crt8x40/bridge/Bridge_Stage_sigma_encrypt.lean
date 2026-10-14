import Stage_sigma_encrypt
import W_Stage_sigma_encrypt
import ExponentSampling

/-! Every run of the average-case relations of `Stage_sigma_encrypt` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_sigma_encrypt.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39.constraints_0.bridge {backend hashModel tape path i_1 w_1_0 w_2_0 w_5_0 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 w_20_0 w_25_0 w_26_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39.constraints_0 backend hashModel tape path i_1 w_1_0 w_2_0 w_5_0 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 w_20_0 w_25_0 w_26_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39.constraints_0 backend hashModel i_1 w_1_0 w_2_0 w_5_0 w_11_0 w_12_0 w_16_0 w_17_0 w_18_0 w_20_0 w_25_0 w_26_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h; exact ⟨h1, h2, h3, (MxxExponentLut.gaussianSampleAt_sample (by decide) h4), h5, h6, (MxxExponentLut.gaussianSampleAt_sample (by decide) h7), h8⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39
  exact (by obtain ⟨w_2_0, w_5_0, w_12_0, w_16_0, w_18_0, w_20_0, w_25_0, h1⟩ := h; exact ⟨w_2_0, w_5_0, w_12_0, w_16_0, w_18_0, w_20_0, w_25_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41.constraints_0 backend hashModel tape path i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41.constraints_0 backend hashModel i_1 w_0_0 w_1_0 w_5_0 w_2_0 w_3_0 w_6_0 w_10_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_20_0 w_23_0 w_27_0 w_28_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, (MxxExponentLut.gaussianSampleAt_sample (by decide) h10), h11, h12, h13, h14, h15, (MxxExponentLut.gaussianSampleAt_sample (by decide) h16), h17⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41
  exact (by obtain ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, h1⟩ := h; exact ⟨w_3_0, w_6_0, w_10_0, w_12_0, w_13_0, w_16_0, w_18_0, w_20_0, w_23_0, w_27_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99.constraints_0.bridge {backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99.constraints_0 backend hashModel tape path i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99.constraints_0 backend hashModel i_0 i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, (MxxExponentLut.gaussianSampleAt_sample (by decide) h14), h15, h16, h17, h18, h19, (MxxExponentLut.gaussianSampleAt_sample (by decide) h20), h21⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99.bridge {backend hashModel tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99 backend hashModel tape path params i_0 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99 backend hashModel params' i_0 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111.constraints_0.bridge {backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111.constraints_0 backend hashModel tape path i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111.constraints_0 backend hashModel i_1 w_0_0 w_2_0 w_6_0 w_1_0 w_3_0 w_4_0 w_7_0 w_8_0 w_12_0 w_14_0 w_15_0 w_18_0 w_19_0 w_20_0 w_22_0 w_25_0 w_29_0 w_30_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, h14, h15, h16, h17, (MxxExponentLut.gaussianSampleAt_sample (by decide) h18), h19⟩)


theorem Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111
  exact (by obtain ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, h1⟩ := h; exact ⟨w_1_0, w_4_0, w_7_0, w_8_0, w_12_0, w_14_0, w_15_0, w_18_0, w_20_0, w_22_0, w_25_0, w_29_0, (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111.constraints_0.bridge h1)⟩)


theorem Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_4.bridge {backend hashModel tape path params params' i_0 w_2_0 w_1_0 w_6_0 w_24_0 w_36_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1 w_97_0 w_97_1 w_98_0 w_98_1 w_99_0 w_99_1 w_100_0 w_100_1 w_101_0 w_101_1 w_102_0 w_102_1 w_103_0 w_103_1 w_104_0 w_104_1 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_107_1 w_108_0 w_108_1 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_111_1 outputs}
    (h : Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_4 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_24_0 w_36_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1 w_97_0 w_97_1 w_98_0 w_98_1 w_99_0 w_99_1 w_100_0 w_100_1 w_101_0 w_101_1 w_102_0 w_102_1 w_103_0 w_103_1 w_104_0 w_104_1 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_107_1 w_108_0 w_108_1 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_111_1 outputs) : W_Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_4 backend hashModel params' i_0 w_2_0 w_1_0 w_6_0 w_24_0 w_36_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1 w_97_0 w_97_1 w_98_0 w_98_1 w_99_0 w_99_1 w_100_0 w_100_1 w_101_0 w_101_1 w_102_0 w_102_1 w_103_0 w_103_1 w_104_0 w_104_1 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_107_1 w_108_0 w_108_1 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_111_1 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_97.bridge (h2 i))), h3, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_98.bridge (h4 i))), h5, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_99.bridge (h6 i))), h7, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_100.bridge (h8 i))), h9, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_101.bridge (h10 i))), h11, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_102.bridge (h12 i))), h13, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_103.bridge (h14 i))), h15, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_104.bridge (h16 i))), h17, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_105.bridge (h18 i))), h19, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_106.bridge (h20 i))), h21, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_107.bridge (h22 i))), h23, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_108.bridge (h24 i))), h25, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_109.bridge (h26 i))), h27, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_110.bridge (h28 i))), h29, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_111.bridge (h30 i))), h31⟩)


theorem Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_3.bridge {backend hashModel tape path params params' i_0 w_2_0 w_1_0 w_6_0 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1}
    (h : Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_3 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1) : W_Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_3 backend hashModel params' i_0 w_2_0 w_1_0 w_6_0 w_81_0 w_81_1 w_82_0 w_82_1 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_85_1 w_86_0 w_86_1 w_87_0 w_87_1 w_88_0 w_88_1 w_89_0 w_89_1 w_90_0 w_90_1 w_91_0 w_91_1 w_92_0 w_92_1 w_93_0 w_93_1 w_94_0 w_94_1 w_95_0 w_95_1 w_96_0 w_96_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_81.bridge (h2 i))), h3, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_82.bridge (h4 i))), h5, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_83.bridge (h6 i))), h7, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_84.bridge (h8 i))), h9, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_85.bridge (h10 i))), h11, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_86.bridge (h12 i))), h13, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_87.bridge (h14 i))), h15, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_88.bridge (h16 i))), h17, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_89.bridge (h18 i))), h19, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_90.bridge (h20 i))), h21, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_91.bridge (h22 i))), h23, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_92.bridge (h24 i))), h25, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_93.bridge (h26 i))), h27, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_94.bridge (h28 i))), h29, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_95.bridge (h30 i))), h31, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_96.bridge (h32 i)))⟩)


theorem Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_2.bridge {backend hashModel tape path params params' i_0 w_2_0 w_1_0 w_6_0 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1}
    (h : Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_2 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1) : W_Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_2 backend hashModel params' i_0 w_2_0 w_1_0 w_6_0 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_65.bridge (h2 i))), h3, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_66.bridge (h4 i))), h5, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_67.bridge (h6 i))), h7, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_68.bridge (h8 i))), h9, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_69.bridge (h10 i))), h11, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_70.bridge (h12 i))), h13, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_71.bridge (h14 i))), h15, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_72.bridge (h16 i))), h17, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_73.bridge (h18 i))), h19, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_74.bridge (h20 i))), h21, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_75.bridge (h22 i))), h23, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_76.bridge (h24 i))), h25, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_77.bridge (h26 i))), h27, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_78.bridge (h28 i))), h29, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_79.bridge (h30 i))), h31, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_80.bridge (h32 i)))⟩)


theorem Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_1.bridge {backend hashModel tape path params params' i_0 w_2_0 w_1_0 w_6_0 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1}
    (h : Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_1 backend hashModel tape path params i_0 w_2_0 w_1_0 w_6_0 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1) : W_Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_1 backend hashModel params' i_0 w_2_0 w_1_0 w_6_0 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_49.bridge (h2 i))), h3, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_50.bridge (h4 i))), h5, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_51.bridge (h6 i))), h7, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_52.bridge (h8 i))), h9, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_53.bridge (h10 i))), h11, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_54.bridge (h12 i))), h13, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_55.bridge (h14 i))), h15, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_56.bridge (h16 i))), h17, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_57.bridge (h18 i))), h19, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_58.bridge (h20 i))), h21, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_59.bridge (h22 i))), h23, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_60.bridge (h24 i))), h25, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_61.bridge (h26 i))), h27, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_62.bridge (h28 i))), h29, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_63.bridge (h30 i))), h31, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_64.bridge (h32 i)))⟩)


theorem Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_2_0 w_10_0 w_37_0 w_38_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 w_11_0 w_12_0 w_13_0 w_23_0 w_25_0 w_27_0 w_30_0 w_31_0 w_35_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1}
    (h : Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_0 backend hashModel tape path params i_0 w_0_0 w_2_0 w_10_0 w_37_0 w_38_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 w_11_0 w_12_0 w_13_0 w_23_0 w_25_0 w_27_0 w_30_0 w_31_0 w_35_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1) : W_Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_0 backend hashModel params' i_0 w_0_0 w_2_0 w_10_0 w_37_0 w_38_0 w_1_0 w_3_0 w_5_0 w_6_0 w_7_0 w_11_0 w_12_0 w_13_0 w_23_0 w_25_0 w_27_0 w_30_0 w_31_0 w_35_0 w_39_0 w_39_1 w_39_2 w_40_0 w_40_1 w_41_0 w_41_1 w_42_0 w_42_1 w_43_0 w_43_1 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, (MxxExponentLut.gaussianSampleAt_sample (by decide) h8), h9, h10, h11, (MxxExponentLut.gaussianSampleAt_sample (by decide) h12), h13, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_39.bridge (h14 i))), h15, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_40.bridge (h16 i))), h17, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_41.bridge (h18 i))), h19, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_42.bridge (h20 i))), h21, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_43.bridge (h22 i))), h23, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_44.bridge (h24 i))), h25, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_45.bridge (h26 i))), h27, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_46.bridge (h28 i))), h29, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_47.bridge (h30 i))), h31, (fun i ↦ (Stage_sigma_encrypt.parallel_parallel_generatedRoot_7_48.bridge (h32 i)))⟩)


theorem Stage_sigma_encrypt.parallel_generatedRoot_7.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_sigma_encrypt.parallel_generatedRoot_7 backend hashModel tape path params i_0 inputs outputs) : W_Stage_sigma_encrypt.parallel_generatedRoot_7 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_sigma_encrypt.parallel_generatedRoot_7
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_3_0, wit1.w_6_0, wit1.w_7_0, wit1.w_11_0, wit1.w_12_0, wit1.w_13_0, wit1.w_23_0, wit1.w_25_0, wit1.w_27_0, wit1.w_31_0, wit1.w_35_0, wit1.w_39_0, wit1.w_39_1, wit1.w_39_2, wit1.w_40_0, wit1.w_40_1, wit1.w_41_0, wit1.w_41_1, wit1.w_42_0, wit1.w_42_1, wit1.w_43_0, wit1.w_43_1, wit1.w_44_0, wit1.w_44_1, wit1.w_45_0, wit1.w_45_1, wit1.w_46_0, wit1.w_46_1, wit1.w_47_0, wit1.w_47_1, wit1.w_48_0, wit1.w_48_1, wit1.w_49_0, wit1.w_49_1, wit1.w_50_0, wit1.w_50_1, wit1.w_51_0, wit1.w_51_1, wit1.w_52_0, wit1.w_52_1, wit1.w_53_0, wit1.w_53_1, wit1.w_54_0, wit1.w_54_1, wit1.w_55_0, wit1.w_55_1, wit1.w_56_0, wit1.w_56_1, wit1.w_57_0, wit1.w_57_1, wit1.w_58_0, wit1.w_58_1, wit1.w_59_0, wit1.w_59_1, wit1.w_60_0, wit1.w_60_1, wit1.w_61_0, wit1.w_61_1, wit1.w_62_0, wit1.w_62_1, wit1.w_63_0, wit1.w_63_1, wit1.w_64_0, wit1.w_64_1, wit1.w_65_0, wit1.w_65_1, wit1.w_66_0, wit1.w_66_1, wit1.w_67_0, wit1.w_67_1, wit1.w_68_0, wit1.w_68_1, wit1.w_69_0, wit1.w_69_1, wit1.w_70_0, wit1.w_70_1, wit1.w_71_0, wit1.w_71_1, wit1.w_72_0, wit1.w_72_1, wit1.w_73_0, wit1.w_73_1, wit1.w_74_0, wit1.w_74_1, wit1.w_75_0, wit1.w_75_1, wit1.w_76_0, wit1.w_76_1, wit1.w_77_0, wit1.w_77_1, wit1.w_78_0, wit1.w_78_1, wit1.w_79_0, wit1.w_79_1, wit1.w_80_0, wit1.w_80_1, wit1.w_81_0, wit1.w_81_1, wit1.w_82_0, wit1.w_82_1, wit1.w_83_0, wit1.w_83_1, wit1.w_84_0, wit1.w_84_1, wit1.w_85_0, wit1.w_85_1, wit1.w_86_0, wit1.w_86_1, wit1.w_87_0, wit1.w_87_1, wit1.w_88_0, wit1.w_88_1, wit1.w_89_0, wit1.w_89_1, wit1.w_90_0, wit1.w_90_1, wit1.w_91_0, wit1.w_91_1, wit1.w_92_0, wit1.w_92_1, wit1.w_93_0, wit1.w_93_1, wit1.w_94_0, wit1.w_94_1, wit1.w_95_0, wit1.w_95_1, wit1.w_96_0, wit1.w_96_1, wit1.w_97_0, wit1.w_97_1, wit1.w_98_0, wit1.w_98_1, wit1.w_99_0, wit1.w_99_1, wit1.w_100_0, wit1.w_100_1, wit1.w_101_0, wit1.w_101_1, wit1.w_102_0, wit1.w_102_1, wit1.w_103_0, wit1.w_103_1, wit1.w_104_0, wit1.w_104_1, wit1.w_105_0, wit1.w_105_1, wit1.w_106_0, wit1.w_106_1, wit1.w_107_0, wit1.w_107_1, wit1.w_108_0, wit1.w_108_1, wit1.w_109_0, wit1.w_109_1, wit1.w_110_0, wit1.w_110_1, wit1.w_111_0, wit1.w_111_1⟩, (by obtain ⟨h3, h4, h5, h6, h7⟩ := h2; exact ⟨(Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_0.bridge h3), (Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_1.bridge h4), (Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_2.bridge h5), (Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_3.bridge h6), (Stage_sigma_encrypt.parallel_generatedRoot_7.constraints_4.bridge h7)⟩)⟩)


theorem Stage_sigma_encrypt.generatedRoot.constraints_0.bridge {backend hashModel tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_7_1 w_7_2 w_7_3 w_7_4 w_7_5 w_7_6 w_7_7 w_7_8 w_7_9 w_7_10 w_7_11 w_7_12 w_7_13 w_7_14 w_7_15 w_7_16 w_7_17 w_7_18 w_7_19 w_7_20 w_7_21 w_7_22 w_7_23 w_7_24 w_7_25 w_7_26 w_7_27 w_7_28 w_7_29 w_7_30 w_7_31 w_7_32 w_7_33 w_7_34 w_7_35 w_7_36 w_7_37 w_7_38 w_7_39 w_7_40 w_7_41 w_7_42 w_7_43 w_7_44 w_7_45 w_7_46 w_7_47 w_7_48 w_7_49 w_7_50 w_7_51 w_7_52 w_7_53 w_7_54 w_7_55 w_7_56 w_7_57 w_7_58 w_7_59 w_7_60 w_7_61 w_7_62 w_7_63 w_7_64 w_7_65 w_7_66 w_7_67 w_7_68 w_7_69 w_7_70 w_7_71 w_7_72 w_7_73 w_7_74 w_7_75 w_7_76 w_7_77 w_7_78 w_7_79 w_7_80 w_7_81 w_7_82 w_7_83 w_7_84 w_7_85 w_7_86 w_7_87 w_7_88 w_7_89 w_7_90 w_7_91 w_7_92 w_7_93 w_7_94 w_7_95 w_7_96 w_7_97 w_7_98 w_7_99 w_7_100 w_7_101 w_7_102 w_7_103 w_7_104 w_7_105 w_7_106 w_7_107 w_7_108 w_7_109 w_7_110 w_7_111 w_7_112 w_7_113 w_7_114 w_7_115 w_7_116 w_7_117 w_7_118 w_7_119 w_7_120 w_7_121 w_7_122 w_7_123 w_7_124 w_7_125 w_7_126 w_7_127 w_7_128 w_7_129 w_7_130 w_7_131 w_7_132 w_7_133 w_7_134 w_7_135 w_7_136 w_7_137 w_7_138 w_7_139 w_7_140 w_7_141 w_7_142 w_7_143 w_7_144 w_7_145 w_7_146 w_7_147 w_7_148 outputs}
    (h : Stage_sigma_encrypt.generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_7_1 w_7_2 w_7_3 w_7_4 w_7_5 w_7_6 w_7_7 w_7_8 w_7_9 w_7_10 w_7_11 w_7_12 w_7_13 w_7_14 w_7_15 w_7_16 w_7_17 w_7_18 w_7_19 w_7_20 w_7_21 w_7_22 w_7_23 w_7_24 w_7_25 w_7_26 w_7_27 w_7_28 w_7_29 w_7_30 w_7_31 w_7_32 w_7_33 w_7_34 w_7_35 w_7_36 w_7_37 w_7_38 w_7_39 w_7_40 w_7_41 w_7_42 w_7_43 w_7_44 w_7_45 w_7_46 w_7_47 w_7_48 w_7_49 w_7_50 w_7_51 w_7_52 w_7_53 w_7_54 w_7_55 w_7_56 w_7_57 w_7_58 w_7_59 w_7_60 w_7_61 w_7_62 w_7_63 w_7_64 w_7_65 w_7_66 w_7_67 w_7_68 w_7_69 w_7_70 w_7_71 w_7_72 w_7_73 w_7_74 w_7_75 w_7_76 w_7_77 w_7_78 w_7_79 w_7_80 w_7_81 w_7_82 w_7_83 w_7_84 w_7_85 w_7_86 w_7_87 w_7_88 w_7_89 w_7_90 w_7_91 w_7_92 w_7_93 w_7_94 w_7_95 w_7_96 w_7_97 w_7_98 w_7_99 w_7_100 w_7_101 w_7_102 w_7_103 w_7_104 w_7_105 w_7_106 w_7_107 w_7_108 w_7_109 w_7_110 w_7_111 w_7_112 w_7_113 w_7_114 w_7_115 w_7_116 w_7_117 w_7_118 w_7_119 w_7_120 w_7_121 w_7_122 w_7_123 w_7_124 w_7_125 w_7_126 w_7_127 w_7_128 w_7_129 w_7_130 w_7_131 w_7_132 w_7_133 w_7_134 w_7_135 w_7_136 w_7_137 w_7_138 w_7_139 w_7_140 w_7_141 w_7_142 w_7_143 w_7_144 w_7_145 w_7_146 w_7_147 w_7_148 outputs) : W_Stage_sigma_encrypt.generatedRoot.constraints_0 backend hashModel params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_7_1 w_7_2 w_7_3 w_7_4 w_7_5 w_7_6 w_7_7 w_7_8 w_7_9 w_7_10 w_7_11 w_7_12 w_7_13 w_7_14 w_7_15 w_7_16 w_7_17 w_7_18 w_7_19 w_7_20 w_7_21 w_7_22 w_7_23 w_7_24 w_7_25 w_7_26 w_7_27 w_7_28 w_7_29 w_7_30 w_7_31 w_7_32 w_7_33 w_7_34 w_7_35 w_7_36 w_7_37 w_7_38 w_7_39 w_7_40 w_7_41 w_7_42 w_7_43 w_7_44 w_7_45 w_7_46 w_7_47 w_7_48 w_7_49 w_7_50 w_7_51 w_7_52 w_7_53 w_7_54 w_7_55 w_7_56 w_7_57 w_7_58 w_7_59 w_7_60 w_7_61 w_7_62 w_7_63 w_7_64 w_7_65 w_7_66 w_7_67 w_7_68 w_7_69 w_7_70 w_7_71 w_7_72 w_7_73 w_7_74 w_7_75 w_7_76 w_7_77 w_7_78 w_7_79 w_7_80 w_7_81 w_7_82 w_7_83 w_7_84 w_7_85 w_7_86 w_7_87 w_7_88 w_7_89 w_7_90 w_7_91 w_7_92 w_7_93 w_7_94 w_7_95 w_7_96 w_7_97 w_7_98 w_7_99 w_7_100 w_7_101 w_7_102 w_7_103 w_7_104 w_7_105 w_7_106 w_7_107 w_7_108 w_7_109 w_7_110 w_7_111 w_7_112 w_7_113 w_7_114 w_7_115 w_7_116 w_7_117 w_7_118 w_7_119 w_7_120 w_7_121 w_7_122 w_7_123 w_7_124 w_7_125 w_7_126 w_7_127 w_7_128 w_7_129 w_7_130 w_7_131 w_7_132 w_7_133 w_7_134 w_7_135 w_7_136 w_7_137 w_7_138 w_7_139 w_7_140 w_7_141 w_7_142 w_7_143 w_7_144 w_7_145 w_7_146 w_7_147 w_7_148 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (fun i ↦ (Stage_sigma_encrypt.parallel_generatedRoot_7.bridge (h2 i))), h3⟩)


theorem Stage_sigma_encrypt.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_sigma_encrypt.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_sigma_encrypt.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_sigma_encrypt.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_7_0, wit1.w_7_1, wit1.w_7_2, wit1.w_7_3, wit1.w_7_4, wit1.w_7_5, wit1.w_7_6, wit1.w_7_7, wit1.w_7_8, wit1.w_7_9, wit1.w_7_10, wit1.w_7_11, wit1.w_7_12, wit1.w_7_13, wit1.w_7_14, wit1.w_7_15, wit1.w_7_16, wit1.w_7_17, wit1.w_7_18, wit1.w_7_19, wit1.w_7_20, wit1.w_7_21, wit1.w_7_22, wit1.w_7_23, wit1.w_7_24, wit1.w_7_25, wit1.w_7_26, wit1.w_7_27, wit1.w_7_28, wit1.w_7_29, wit1.w_7_30, wit1.w_7_31, wit1.w_7_32, wit1.w_7_33, wit1.w_7_34, wit1.w_7_35, wit1.w_7_36, wit1.w_7_37, wit1.w_7_38, wit1.w_7_39, wit1.w_7_40, wit1.w_7_41, wit1.w_7_42, wit1.w_7_43, wit1.w_7_44, wit1.w_7_45, wit1.w_7_46, wit1.w_7_47, wit1.w_7_48, wit1.w_7_49, wit1.w_7_50, wit1.w_7_51, wit1.w_7_52, wit1.w_7_53, wit1.w_7_54, wit1.w_7_55, wit1.w_7_56, wit1.w_7_57, wit1.w_7_58, wit1.w_7_59, wit1.w_7_60, wit1.w_7_61, wit1.w_7_62, wit1.w_7_63, wit1.w_7_64, wit1.w_7_65, wit1.w_7_66, wit1.w_7_67, wit1.w_7_68, wit1.w_7_69, wit1.w_7_70, wit1.w_7_71, wit1.w_7_72, wit1.w_7_73, wit1.w_7_74, wit1.w_7_75, wit1.w_7_76, wit1.w_7_77, wit1.w_7_78, wit1.w_7_79, wit1.w_7_80, wit1.w_7_81, wit1.w_7_82, wit1.w_7_83, wit1.w_7_84, wit1.w_7_85, wit1.w_7_86, wit1.w_7_87, wit1.w_7_88, wit1.w_7_89, wit1.w_7_90, wit1.w_7_91, wit1.w_7_92, wit1.w_7_93, wit1.w_7_94, wit1.w_7_95, wit1.w_7_96, wit1.w_7_97, wit1.w_7_98, wit1.w_7_99, wit1.w_7_100, wit1.w_7_101, wit1.w_7_102, wit1.w_7_103, wit1.w_7_104, wit1.w_7_105, wit1.w_7_106, wit1.w_7_107, wit1.w_7_108, wit1.w_7_109, wit1.w_7_110, wit1.w_7_111, wit1.w_7_112, wit1.w_7_113, wit1.w_7_114, wit1.w_7_115, wit1.w_7_116, wit1.w_7_117, wit1.w_7_118, wit1.w_7_119, wit1.w_7_120, wit1.w_7_121, wit1.w_7_122, wit1.w_7_123, wit1.w_7_124, wit1.w_7_125, wit1.w_7_126, wit1.w_7_127, wit1.w_7_128, wit1.w_7_129, wit1.w_7_130, wit1.w_7_131, wit1.w_7_132, wit1.w_7_133, wit1.w_7_134, wit1.w_7_135, wit1.w_7_136, wit1.w_7_137, wit1.w_7_138, wit1.w_7_139, wit1.w_7_140, wit1.w_7_141, wit1.w_7_142, wit1.w_7_143, wit1.w_7_144, wit1.w_7_145, wit1.w_7_146, wit1.w_7_147, wit1.w_7_148⟩, (Stage_sigma_encrypt.generatedRoot.constraints_0.bridge h2)⟩)
