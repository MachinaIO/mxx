import Stage_online_1
import W_Stage_online_1
import ExponentSampling

/-! Every run of the average-case relations of `Stage_online_1` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_online_1.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_online_1.sequential_generatedRoot_91.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.sequential_generatedRoot_91 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_1.sequential_generatedRoot_91 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_1.sequential_generatedRoot_91
  exact h


theorem Stage_online_1.parallel_generatedRoot_149.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_149 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_149 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_149
  exact h


theorem Stage_online_1.parallel_parallel_generatedRoot_150_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_1.parallel_parallel_generatedRoot_150_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_1.parallel_parallel_generatedRoot_150_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_1.parallel_parallel_generatedRoot_150_6
  exact h


theorem Stage_online_1.sequential_parallel_generatedRoot_150_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_1.sequential_parallel_generatedRoot_150_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_1.sequential_parallel_generatedRoot_150_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_1.sequential_parallel_generatedRoot_150_25
  exact h


theorem Stage_online_1.parallel_generatedRoot_150.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_1.parallel_generatedRoot_150.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_1.parallel_generatedRoot_150.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_1.parallel_parallel_generatedRoot_150_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_1.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_1.sequential_parallel_generatedRoot_150_25.bridge hs18))), h17⟩)


theorem Stage_online_1.parallel_generatedRoot_150.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_150 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_150 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_150
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_1.parallel_generatedRoot_150.constraints_0.bridge h2)⟩)


theorem Stage_online_1.parallel_generatedRoot_153.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_153 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_153 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_153
  exact h


theorem Stage_online_1.parallel_parallel_generatedRoot_154_6.bridge {backend x1 tape path params params' i_0 i_1 inputs outputs}
    (h : Stage_online_1.parallel_parallel_generatedRoot_154_6 backend x1 tape path params i_0 i_1 inputs outputs) : W_Stage_online_1.parallel_parallel_generatedRoot_154_6 backend x1 params' i_0 i_1 inputs outputs := by
  delta W_Stage_online_1.parallel_parallel_generatedRoot_154_6
  exact h


theorem Stage_online_1.sequential_parallel_generatedRoot_154_25.bridge {backend x1 tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_1.sequential_parallel_generatedRoot_154_25 backend x1 tape path params x5 i_1 inputs outputs) : W_Stage_online_1.sequential_parallel_generatedRoot_154_25 backend x1 params' x5 i_1 inputs outputs := by
  delta W_Stage_online_1.sequential_parallel_generatedRoot_154_25
  exact h


theorem Stage_online_1.parallel_generatedRoot_154.constraints_0.bridge {backend hashModel tape path params params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs}
    (h : Stage_online_1.parallel_generatedRoot_154.constraints_0 backend hashModel tape path params i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs) : W_Stage_online_1.parallel_generatedRoot_154.constraints_0 backend hashModel params' i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_6_1 w_6_2 w_6_3 w_6_4 w_6_5 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_25_1 w_25_2 w_25_3 w_25_4 w_25_5 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17⟩ := h; exact ⟨h1, (fun i ↦ (Stage_online_1.parallel_parallel_generatedRoot_154_6.bridge (h2 i))), h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, (Stage_online_1.iterRuns_mono h16 (fun _ _ _ hs18 ↦ (Stage_online_1.sequential_parallel_generatedRoot_154_25.bridge hs18))), h17⟩)


theorem Stage_online_1.parallel_generatedRoot_154.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_154 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_154 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_154
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_6_3, wit1.w_6_4, wit1.w_6_5, wit1.w_7_0, wit1.w_8_0, wit1.w_10_0, wit1.w_11_0, wit1.w_13_0, wit1.w_14_0, wit1.w_16_0, wit1.w_17_0, wit1.w_19_0, wit1.w_20_0, wit1.w_22_0, wit1.w_23_0, wit1.w_25_0, wit1.w_25_1, wit1.w_25_2, wit1.w_25_3, wit1.w_25_4, wit1.w_25_5⟩, (Stage_online_1.parallel_generatedRoot_154.constraints_0.bridge h2)⟩)


theorem Stage_online_1.sequential_parallel_generatedRoot_155_18.bridge {backend hashModel tape path params params' x5 i_1 inputs outputs}
    (h : Stage_online_1.sequential_parallel_generatedRoot_155_18 backend hashModel tape path params x5 i_1 inputs outputs) : W_Stage_online_1.sequential_parallel_generatedRoot_155_18 backend hashModel params' x5 i_1 inputs outputs := by
  delta W_Stage_online_1.sequential_parallel_generatedRoot_155_18
  exact h


theorem Stage_online_1.parallel_generatedRoot_155.constraints_0.bridge {backend hashModel tape path params params' i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1}
    (h : Stage_online_1.parallel_generatedRoot_155.constraints_0 backend hashModel tape path params i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1) : W_Stage_online_1.parallel_generatedRoot_155.constraints_0 backend hashModel params' i_0 w_1_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_0_0 w_2_0 w_4_0 w_6_0 w_8_0 w_18_0 w_18_1 w_18_2 w_18_3 w_18_4 w_18_5 w_18_6 w_18_7 w_18_8 w_18_9 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_concat_1 w_24_0 w_25_0 w_26_0 w_27_0 w_27_concat_1 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_32_concat_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, (Stage_online_1.iterRuns_mono h6 (fun _ _ _ hs33 ↦ (Stage_online_1.sequential_parallel_generatedRoot_155_18.bridge hs33))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_online_1.parallel_generatedRoot_155.bridge {backend hashModel tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_155 backend hashModel tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_155 backend hashModel params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_155
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_2_0, wit1.w_4_0, wit1.w_6_0, wit1.w_8_0, wit1.w_18_0, wit1.w_18_1, wit1.w_18_2, wit1.w_18_3, wit1.w_18_4, wit1.w_18_5, wit1.w_18_6, wit1.w_18_7, wit1.w_18_8, wit1.w_18_9, wit1.w_19_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_23_concat_1, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_27_concat_1, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_32_concat_1, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_36_concat_1, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_48_0, wit1.w_50_0, wit1.w_51_0, wit1.w_52_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_60_0, wit1.w_62_0, wit1.w_64_0, wit1.w_66_0, wit1.w_67_0, wit1.w_70_0, wit1.w_71_0⟩, (by obtain ⟨h3, h4, h5, h6, h7⟩ := h2; exact ⟨(Stage_online_1.parallel_generatedRoot_155.constraints_0.bridge h3), h4, h5, h6, h7⟩)⟩)


theorem Stage_online_1.parallel_generatedRoot_156.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_156 backend x1 tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_156 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_156
  exact h


theorem Stage_online_1.sequential_generatedRoot_175.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.sequential_generatedRoot_175 backend x1 tape path params i_0 inputs outputs) : W_Stage_online_1.sequential_generatedRoot_175 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_online_1.sequential_generatedRoot_175
  exact h


theorem Stage_online_1.parallel_generatedRoot_177.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.parallel_generatedRoot_177 backend x1 tape path params i_0 inputs outputs) : W_Stage_online_1.parallel_generatedRoot_177 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_online_1.parallel_generatedRoot_177
  exact h


theorem Stage_online_1.sequential_generatedRoot_178.bridge {backend x1 tape path params params' i_0 inputs outputs}
    (h : Stage_online_1.sequential_generatedRoot_178 backend x1 tape path params i_0 inputs outputs) : W_Stage_online_1.sequential_generatedRoot_178 backend x1 params' i_0 inputs outputs := by
  delta W_Stage_online_1.sequential_generatedRoot_178
  exact h


theorem Stage_online_1.generatedRoot.constraints_7.bridge {backend hashModel tape path params params' w_91_6 w_91_8 w_132_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_159_0 w_162_0 w_165_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_175_1 w_175_2 w_175_3 w_175_4 w_175_5 w_176_0 w_177_0 w_178_0 w_178_1 w_178_2 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0}
    (h : Stage_online_1.generatedRoot.constraints_7 backend hashModel tape path params w_91_6 w_91_8 w_132_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_159_0 w_162_0 w_165_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_175_1 w_175_2 w_175_3 w_175_4 w_175_5 w_176_0 w_177_0 w_178_0 w_178_1 w_178_2 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0) : W_Stage_online_1.generatedRoot.constraints_7 backend hashModel params' w_91_6 w_91_8 w_132_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_159_0 w_162_0 w_165_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_175_1 w_175_2 w_175_3 w_175_4 w_175_5 w_176_0 w_177_0 w_178_0 w_178_1 w_178_2 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, (Stage_online_1.iterRuns_mono h7 (fun _ _ _ hs33 ↦ (Stage_online_1.sequential_generatedRoot_175.bridge hs33))), h8, (fun i ↦ (Stage_online_1.parallel_generatedRoot_177.bridge (h9 i))), h10, (Stage_online_1.iterRuns_mono h11 (fun _ _ _ hs34 ↦ (Stage_online_1.sequential_generatedRoot_178.bridge hs34))), h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_online_1.generatedRoot.constraints_6.bridge {backend hashModel tape path params params' w_14_0 w_16_0 w_27_0 w_47_0 w_70_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_143_0 w_146_0 w_147_0 w_148_0 w_151_0 w_152_0 w_114_0 w_122_0 w_125_0 w_126_0 w_128_0 w_129_0 w_130_0 w_141_0 w_142_0 w_144_0 w_145_0 w_149_0 w_149_1 w_149_2 w_149_3 w_149_4 w_149_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_153_0 w_153_1 w_153_2 w_153_3 w_153_4 w_153_5 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_155_0 w_155_1 w_155_2 w_155_3 w_155_4 w_155_5 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_157_0 w_158_0 w_160_0 w_161_0 w_163_0 w_164_0 w_166_0}
    (h : Stage_online_1.generatedRoot.constraints_6 backend hashModel tape path params w_14_0 w_16_0 w_27_0 w_47_0 w_70_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_143_0 w_146_0 w_147_0 w_148_0 w_151_0 w_152_0 w_114_0 w_122_0 w_125_0 w_126_0 w_128_0 w_129_0 w_130_0 w_141_0 w_142_0 w_144_0 w_145_0 w_149_0 w_149_1 w_149_2 w_149_3 w_149_4 w_149_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_153_0 w_153_1 w_153_2 w_153_3 w_153_4 w_153_5 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_155_0 w_155_1 w_155_2 w_155_3 w_155_4 w_155_5 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_157_0 w_158_0 w_160_0 w_161_0 w_163_0 w_164_0 w_166_0) : W_Stage_online_1.generatedRoot.constraints_6 backend hashModel params' w_14_0 w_16_0 w_27_0 w_47_0 w_70_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_143_0 w_146_0 w_147_0 w_148_0 w_151_0 w_152_0 w_114_0 w_122_0 w_125_0 w_126_0 w_128_0 w_129_0 w_130_0 w_141_0 w_142_0 w_144_0 w_145_0 w_149_0 w_149_1 w_149_2 w_149_3 w_149_4 w_149_5 w_150_0 w_150_1 w_150_2 w_150_3 w_150_4 w_150_5 w_153_0 w_153_1 w_153_2 w_153_3 w_153_4 w_153_5 w_154_0 w_154_1 w_154_2 w_154_3 w_154_4 w_154_5 w_155_0 w_155_1 w_155_2 w_155_3 w_155_4 w_155_5 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_157_0 w_158_0 w_160_0 w_161_0 w_163_0 w_164_0 w_166_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, (fun i ↦ (by obtain ⟨w_16_0_gathered, w_27_0_gathered, w_47_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42⟩ := (h15 i); exact ⟨w_16_0_gathered, w_27_0_gathered, w_47_0_gathered, h33, h34, h35, h36, h37, h38, h39, h40, h41, (Stage_online_1.parallel_generatedRoot_149.bridge h42)⟩)), h16, (fun i ↦ (Stage_online_1.parallel_generatedRoot_150.bridge (h17 i))), h18, (fun i ↦ (by obtain ⟨w_16_0_gathered, w_27_0_gathered, w_47_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, h52⟩ := (h19 i); exact ⟨w_16_0_gathered, w_27_0_gathered, w_47_0_gathered, h43, h44, h45, h46, h47, h48, h49, h50, h51, (Stage_online_1.parallel_generatedRoot_153.bridge h52)⟩)), h20, (fun i ↦ (Stage_online_1.parallel_generatedRoot_154.bridge (h21 i))), h22, (fun i ↦ (Stage_online_1.parallel_generatedRoot_155.bridge (h23 i))), h24, (fun i ↦ (Stage_online_1.parallel_generatedRoot_156.bridge (h25 i))), h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_online_1.generatedRoot.constraints_3.bridge {backend hashModel tape path params params' w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_70_0 w_82_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_91_1 w_91_2 w_91_3 w_91_4 w_91_5 w_91_6 w_91_7 w_91_8 w_91_9}
    (h : Stage_online_1.generatedRoot.constraints_3 backend hashModel tape path params w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_70_0 w_82_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_91_1 w_91_2 w_91_3 w_91_4 w_91_5 w_91_6 w_91_7 w_91_8 w_91_9) : W_Stage_online_1.generatedRoot.constraints_3 backend hashModel params' w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_70_0 w_82_0 w_4_0 w_6_0 w_8_0 w_10_0 w_12_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_91_1 w_91_2 w_91_3 w_91_4 w_91_5 w_91_6 w_91_7 w_91_8 w_91_9 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, (Stage_online_1.iterRuns_mono h32 (fun _ _ _ hs33 ↦ (Stage_online_1.sequential_generatedRoot_91.bridge hs33)))⟩)


theorem Stage_online_1.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_online_1.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_online_1.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_online_1.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_3_0, wit1.w_6_0, wit1.w_8_0, wit1.w_10_0, wit1.w_12_0, wit1.w_17_0, wit1.w_20_0, wit1.w_23_0, wit1.w_26_0, wit1.w_26_concat_1, wit1.w_28_0, wit1.w_31_0, wit1.w_34_0, wit1.w_37_0, wit1.w_37_concat_1, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_41_0, wit1.w_42_0, wit1.w_43_0, wit1.w_45_0, wit1.w_48_0, wit1.w_53_0, wit1.w_54_0, wit1.w_55_0, wit1.w_55_concat_1, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_59_0, wit1.w_60_0, wit1.w_62_0, wit1.w_66_0, wit1.w_67_0, wit1.w_68_0, wit1.w_68_concat_1, wit1.w_69_0, wit1.w_71_0, wit1.w_73_0, wit1.w_74_0, wit1.w_75_0, wit1.w_76_0, wit1.w_78_0, wit1.w_79_0, wit1.w_80_0, wit1.w_81_0, wit1.w_83_0, wit1.w_84_0, wit1.w_85_0, wit1.w_86_0, wit1.w_88_0, wit1.w_89_0, wit1.w_90_0, wit1.w_91_0, wit1.w_91_1, wit1.w_91_2, wit1.w_91_3, wit1.w_91_4, wit1.w_91_5, wit1.w_91_6, wit1.w_91_7, wit1.w_91_8, wit1.w_91_9, wit1.w_92_0, wit1.w_94_0, wit1.w_95_0, wit1.w_96_0, wit1.w_96_concat_1, wit1.w_97_0, wit1.w_98_0, wit1.w_99_0, wit1.w_100_0, wit1.w_100_concat_1, wit1.w_101_0, wit1.w_102_0, wit1.w_103_0, wit1.w_104_0, wit1.w_105_0, wit1.w_105_concat_1, wit1.w_106_0, wit1.w_107_0, wit1.w_108_0, wit1.w_109_0, wit1.w_109_concat_1, wit1.w_110_0, wit1.w_111_0, wit1.w_112_0, wit1.w_113_0, wit1.w_115_0, wit1.w_116_0, wit1.w_117_0, wit1.w_118_0, wit1.w_119_0, wit1.w_121_0, wit1.w_122_0, wit1.w_126_0, wit1.w_129_0, wit1.w_130_0, wit1.w_141_0, wit1.w_142_0, wit1.w_144_0, wit1.w_145_0, wit1.w_149_0, wit1.w_149_1, wit1.w_149_2, wit1.w_149_3, wit1.w_149_4, wit1.w_149_5, wit1.w_150_0, wit1.w_150_1, wit1.w_150_2, wit1.w_150_3, wit1.w_150_4, wit1.w_150_5, wit1.w_153_0, wit1.w_153_1, wit1.w_153_2, wit1.w_153_3, wit1.w_153_4, wit1.w_153_5, wit1.w_154_0, wit1.w_154_1, wit1.w_154_2, wit1.w_154_3, wit1.w_154_4, wit1.w_154_5, wit1.w_155_0, wit1.w_155_1, wit1.w_155_2, wit1.w_155_3, wit1.w_155_4, wit1.w_155_5, wit1.w_156_0, wit1.w_156_1, wit1.w_156_2, wit1.w_156_3, wit1.w_156_4, wit1.w_156_5, wit1.w_157_0, wit1.w_158_0, wit1.w_160_0, wit1.w_161_0, wit1.w_163_0, wit1.w_164_0, wit1.w_166_0, wit1.w_167_0, wit1.w_169_0, wit1.w_170_0, wit1.w_172_0, wit1.w_173_0, wit1.w_175_0, wit1.w_175_1, wit1.w_175_2, wit1.w_175_3, wit1.w_175_4, wit1.w_175_5, wit1.w_177_0, wit1.w_178_0, wit1.w_178_1, wit1.w_178_2, wit1.w_183_0, wit1.w_184_0, wit1.w_185_0, wit1.w_186_0, wit1.w_188_0, wit1.w_189_0, wit1.w_190_0, wit1.w_191_0, wit1.w_192_0, wit1.w_194_0, wit1.w_196_0, wit1.w_198_0, wit1.w_200_0, wit1.w_202_0, wit1.w_203_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩ := h2; exact ⟨h3, h4, h5, (Stage_online_1.generatedRoot.constraints_3.bridge h6), h7, h8, (Stage_online_1.generatedRoot.constraints_6.bridge h9), (Stage_online_1.generatedRoot.constraints_7.bridge h10), h11, h12⟩)⟩)
