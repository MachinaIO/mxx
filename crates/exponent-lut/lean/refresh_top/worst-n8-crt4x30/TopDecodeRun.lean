import TopDecode

/-! The decode stage of this parameter set (written by `generator/gen_decode.py`). -/

set_option maxRecDepth 65536

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The hash tag of `A'_0`. -/
def topTag0 : Blob := [114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 0]

/-- The hash tag of `A'_1`. -/
def topTag1 : Blob := [114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 1]

/-- The value `y_j + e'_j` whose residue modulo `q₀` the decode stage reads. -/
noncomputable def topValue (model : HashModel) (key : ByteArray) (tag : Blob) (s : Fin 8 → Int)
    (r : ExactMatrix Q0 N 1 1) (c e : Int) (G : ExactMatrix Q1 N 1 6) (E : ErrorMatrix N 1 6) :
    ExactMatrix Q0 N 1 6 :=
  expectedRow (model.sample Q0 N 2 6 key (completeHashTag tag [])) (sigmaPoly s) r c e G +
    reduceMatrix Q0 N 1 6 E

/-- The decode stage outputs zero when each row's residue is that of `y_j + e'_j` for a small
row `e'_j`. -/
theorem decode_zero {model : HashModel} {params : Stage_decode.Params} {inputs} {out : Int}
    (h : Stage_decode.generatedRoot Backend.backend model params inputs out)
    {G : ExactMatrix Q1 N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G)
    (E : Fin 2 → ErrorMatrix N 1 6) (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < (q0 : Int))
    (h0 : Residue inputs.1
      (topValue model inputs.2.2.1 topTag0 inputs.2.1 inputs.2.2.2.1 inputs.2.2.2.2.1 inputs.2.2.2.2.2.1 G (E 0)))
    (h1 : Residue inputs.2.2.2.2.2.2.1
      (topValue model inputs.2.2.1 topTag1 inputs.2.1 1 inputs.2.2.2.2.1 inputs.2.2.2.2.2.1 G (E 1)))
    : out = 0 := by
  obtain ⟨w, h⟩ := h
  unfold Stage_decode.generatedRoot.body at h
  extract_lets -merge _params _backend w_1_0 w_2_0 w_42_0 w_49_0 w_51_0 w_53_0 w_122_0 w_0_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_52_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_63_1 w_63_2 w_63_3 w_63_4 w_63_5 w_63_6 w_63_7 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_73_1 w_73_2 w_73_3 w_73_4 w_73_5 w_73_6 w_73_7 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_93_1 w_93_2 w_93_3 w_93_4 w_93_5 w_93_6 w_93_7 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_103_6 w_103_7 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_113_1 w_113_2 w_113_3 w_113_4 w_113_5 w_113_6 w_113_7 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 at h
  simp only [Stage_decode.generatedRoot.constraints_0, Stage_decode.generatedRoot.constraints_1, Stage_decode.generatedRoot.constraints_2, Stage_decode.generatedRoot.constraints_3, Stage_decode.generatedRoot.constraints_4, Stage_decode.generatedRoot.constraints_5, Stage_decode.generatedRoot.constraints_6, Stage_decode.generatedRoot.constraints_7] at h
  obtain ⟨h3_0, h7_0, h12_0, h17_0, h22_0, h27_0, h32_0, h37_0, h43_0, -, -, -, -, -, -, h44_0, -, -, -, -, -, -, h46_0, h56_0, h57_0, h60_0, -, -, -, -, -, -, h62_0, -, h63_0, h63_1, h63_2, h63_3, h63_4, h63_5, h63_6, h63_7, -, -, -, -, -, -, h72_0, -, h73_0, h73_1, h73_2, h73_3, h73_4, h73_5, h73_6, h73_7, -, -, -, -, -, -, h82_0, -, h83_0, h83_1, h83_2, h83_3, h83_4, h83_5, h83_6, h83_7, -, -, -, -, -, -, h92_0, -, h93_0, h93_1, h93_2, h93_3, h93_4, h93_5, h93_6, h93_7, -, -, -, -, -, -, h102_0, -, h103_0, h103_1, h103_2, h103_3, h103_4, h103_5, h103_6, h103_7, -, -, -, -, -, -, h112_0, -, h113_0, h113_1, h113_2, h113_3, h113_4, h113_5, h113_6, h113_7, h123_0, -, -, -, -, -, -, h124_0, -, -, -, -, -, -, h126_0, h133_0, -, -, -, -, -, -, h135_0, -, h136_0, h136_1, h136_2, h136_3, h136_4, h136_5, h136_6, h136_7, -, -, -, -, -, -, h145_0, -, h146_0, h146_1, h146_2, h146_3, h146_4, h146_5, h146_6, h146_7, -, -, -, -, -, -, h155_0, -, h156_0, h156_1, h156_2, h156_3, h156_4, h156_5, h156_6, h156_7, -, -, -, -, -, -, h165_0, -, h166_0, h166_1, h166_2, h166_3, h166_4, h166_5, h166_6, h166_7, -, -, -, -, -, -, h175_0, -, h176_0, h176_1, h176_2, h176_3, h176_4, h176_5, h176_6, h176_7, -, -, -, -, -, -, h185_0, -, h186_0, h186_1, h186_2, h186_3, h186_4, h186_5, h186_6, h186_7, hout⟩ := h
  have hg : w_56_0 = G := gadgetMatrixRuns_unique h56_0 hG
  have hσ : w_41_0 = sigmaPoly w_2_0 := by
    obtain ⟨_, e0⟩ := familyGet_eq (k := 0) h3_0
    obtain ⟨_, e1⟩ := familyGet_eq (k := 1) h7_0
    obtain ⟨_, e2⟩ := familyGet_eq (k := 2) h12_0
    obtain ⟨_, e3⟩ := familyGet_eq (k := 3) h17_0
    obtain ⟨_, e4⟩ := familyGet_eq (k := 4) h22_0
    obtain ⟨_, e5⟩ := familyGet_eq (k := 5) h27_0
    obtain ⟨_, e6⟩ := familyGet_eq (k := 6) h32_0
    obtain ⟨_, e7⟩ := familyGet_eq (k := 7) h37_0
    simp only [w_41_0, w_40_0, w_39_0, w_38_0, w_36_0, w_35_0, w_34_0, w_33_0, w_31_0, w_30_0, w_29_0, w_28_0, w_26_0, w_25_0, w_24_0, w_23_0, w_21_0, w_20_0, w_19_0, w_18_0, w_16_0, w_15_0, w_14_0, w_13_0, w_11_0, w_10_0, w_9_0, w_8_0, w_6_0, w_5_0, w_4_0]
    rw [e0, e1, e2, e3, e4, e5, e6, e7]
    rfl
  have hA0 := h43_0.2.2
  have hr00 := slice_row h44_0 rfl (by decide)
  have hr01 := slice_row h46_0 rfl (by decide)
  have he0 : w_59_0 = expectedRow (model.sample Q0 N 2 6 w_42_0
      (completeHashTag topTag0 [])) (sigmaPoly w_2_0) w_49_0 w_51_0 w_53_0 G := by
    simp only [w_59_0, w_58_0, w_55_0, w_54_0, w_52_0, w_50_0, w_48_0, w_47_0, w_45_0]
    rw [hσ, hr00, hr01, h57_0.2, hg, hA0]
    rfl
  have hm0 := h60_0
  rw [he0] at hm0
  have hd0 : w_61_0 = reduceMatrix q0 N 1 6 (E 0) := by
    simp only [w_61_0]
    exact residue_sub (E 0) (by unfold topValue at h0; exact h0) hm0
  rw [hd0] at h62_0
  rw [hd0] at h72_0
  rw [hd0] at h82_0
  rw [hd0] at h92_0
  rw [hd0] at h102_0
  rw [hd0] at h112_0
  have hA1 := h123_0.2.2
  have hr10 := slice_row h124_0 rfl (by decide)
  have hr11 := slice_row h126_0 rfl (by decide)
  have he1 : w_132_0 = expectedRow (model.sample Q0 N 2 6 w_42_0
      (completeHashTag topTag1 [])) (sigmaPoly w_2_0) 1 w_51_0 w_53_0 G := by
    simp only [w_132_0, w_131_0, w_130_0, w_54_0, w_52_0, w_129_0, w_128_0, w_48_0, w_127_0, w_125_0]
    rw [hσ, hr10, hr11, h57_0.2, hg, hA1]
    rfl
  have hm1 := h133_0
  rw [he1] at hm1
  have hd1 : w_134_0 = reduceMatrix q0 N 1 6 (E 1) := by
    simp only [w_134_0]
    exact residue_sub (E 1) (by unfold topValue at h1; exact h1) hm1
  rw [hd1] at h135_0
  rw [hd1] at h145_0
  rw [hd1] at h155_0
  rw [hd1] at h165_0
  rw [hd1] at h175_0
  rw [hd1] at h185_0
  have z63_0 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_0
  have z63_1 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_1
  have z63_2 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_2
  have z63_3 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_3
  have z63_4 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_4
  have z63_5 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_5
  have z63_6 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_6
  have z63_7 := column_zero (E 0) (hE 0) h62_0 (by decide) h63_7
  have z73_0 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_0
  have z73_1 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_1
  have z73_2 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_2
  have z73_3 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_3
  have z73_4 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_4
  have z73_5 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_5
  have z73_6 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_6
  have z73_7 := column_zero (E 0) (hE 0) h72_0 (by decide) h73_7
  have z83_0 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_0
  have z83_1 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_1
  have z83_2 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_2
  have z83_3 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_3
  have z83_4 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_4
  have z83_5 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_5
  have z83_6 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_6
  have z83_7 := column_zero (E 0) (hE 0) h82_0 (by decide) h83_7
  have z93_0 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_0
  have z93_1 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_1
  have z93_2 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_2
  have z93_3 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_3
  have z93_4 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_4
  have z93_5 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_5
  have z93_6 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_6
  have z93_7 := column_zero (E 0) (hE 0) h92_0 (by decide) h93_7
  have z103_0 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_0
  have z103_1 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_1
  have z103_2 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_2
  have z103_3 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_3
  have z103_4 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_4
  have z103_5 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_5
  have z103_6 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_6
  have z103_7 := column_zero (E 0) (hE 0) h102_0 (by decide) h103_7
  have z113_0 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_0
  have z113_1 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_1
  have z113_2 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_2
  have z113_3 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_3
  have z113_4 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_4
  have z113_5 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_5
  have z113_6 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_6
  have z113_7 := column_zero (E 0) (hE 0) h112_0 (by decide) h113_7
  have z136_0 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_0
  have z136_1 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_1
  have z136_2 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_2
  have z136_3 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_3
  have z136_4 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_4
  have z136_5 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_5
  have z136_6 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_6
  have z136_7 := column_zero (E 1) (hE 1) h135_0 (by decide) h136_7
  have z146_0 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_0
  have z146_1 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_1
  have z146_2 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_2
  have z146_3 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_3
  have z146_4 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_4
  have z146_5 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_5
  have z146_6 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_6
  have z146_7 := column_zero (E 1) (hE 1) h145_0 (by decide) h146_7
  have z156_0 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_0
  have z156_1 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_1
  have z156_2 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_2
  have z156_3 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_3
  have z156_4 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_4
  have z156_5 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_5
  have z156_6 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_6
  have z156_7 := column_zero (E 1) (hE 1) h155_0 (by decide) h156_7
  have z166_0 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_0
  have z166_1 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_1
  have z166_2 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_2
  have z166_3 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_3
  have z166_4 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_4
  have z166_5 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_5
  have z166_6 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_6
  have z166_7 := column_zero (E 1) (hE 1) h165_0 (by decide) h166_7
  have z176_0 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_0
  have z176_1 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_1
  have z176_2 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_2
  have z176_3 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_3
  have z176_4 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_4
  have z176_5 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_5
  have z176_6 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_6
  have z176_7 := column_zero (E 1) (hE 1) h175_0 (by decide) h176_7
  have z186_0 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_0
  have z186_1 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_1
  have z186_2 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_2
  have z186_3 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_3
  have z186_4 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_4
  have z186_5 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_5
  have z186_6 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_6
  have z186_7 := column_zero (E 1) (hE 1) h185_0 (by decide) h186_7
  rw [hout]
  simp only [w_194_0, w_193_0, w_192_0, w_191_0, w_190_0, w_189_0, w_188_0, w_187_0, w_184_0, w_183_0, w_182_0, w_181_0, w_180_0, w_179_0, w_178_0, w_177_0, w_174_0, w_173_0, w_172_0, w_171_0, w_170_0, w_169_0, w_168_0, w_167_0, w_164_0, w_163_0, w_162_0, w_161_0, w_160_0, w_159_0, w_158_0, w_157_0, w_154_0, w_153_0, w_152_0, w_151_0, w_150_0, w_149_0, w_148_0, w_147_0, w_144_0, w_143_0, w_142_0, w_141_0, w_140_0, w_139_0, w_138_0, w_137_0, w_121_0, w_120_0, w_119_0, w_118_0, w_117_0, w_116_0, w_115_0, w_114_0, w_111_0, w_110_0, w_109_0, w_108_0, w_107_0, w_106_0, w_105_0, w_104_0, w_101_0, w_100_0, w_99_0, w_98_0, w_97_0, w_96_0, w_95_0, w_94_0, w_91_0, w_90_0, w_89_0, w_88_0, w_87_0, w_86_0, w_85_0, w_84_0, w_81_0, w_80_0, w_79_0, w_78_0, w_77_0, w_76_0, w_75_0, w_74_0, w_71_0, w_70_0, w_69_0, w_68_0, w_67_0, w_66_0, w_65_0, w_64_0, w_0_0]
  simp only [z63_0, z63_1, z63_2, z63_3, z63_4, z63_5, z63_6, z63_7, z73_0, z73_1, z73_2, z73_3, z73_4, z73_5, z73_6, z73_7, z83_0, z83_1, z83_2, z83_3, z83_4, z83_5, z83_6, z83_7, z93_0, z93_1, z93_2, z93_3, z93_4, z93_5, z93_6, z93_7, z103_0, z103_1, z103_2, z103_3, z103_4, z103_5, z103_6, z103_7, z113_0, z113_1, z113_2, z113_3, z113_4, z113_5, z113_6, z113_7, z136_0, z136_1, z136_2, z136_3, z136_4, z136_5, z136_6, z136_7, z146_0, z146_1, z146_2, z146_3, z146_4, z146_5, z146_6, z146_7, z156_0, z156_1, z156_2, z156_3, z156_4, z156_5, z156_6, z156_7, z166_0, z166_1, z166_2, z166_3, z166_4, z166_5, z166_6, z166_7, z176_0, z176_1, z176_2, z176_3, z176_4, z176_5, z176_6, z176_7, z186_0, z186_1, z186_2, z186_3, z186_4, z186_5, z186_6, z186_7]
  norm_num

end RefreshTopProof
