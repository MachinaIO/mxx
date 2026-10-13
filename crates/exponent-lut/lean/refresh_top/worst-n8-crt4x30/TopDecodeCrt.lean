import TopCrt
import TopAssembly

/-! The decode stage of the refresh of every CRT slot (written by `generator/gen_decode_crt.py`). -/

set_option maxRecDepth 65536

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- The decode stage outputs zero when, for each row `j`, every slot's residue is that of
`y_j + e'_j` for a small row `e'_j`. -/
theorem decode_zero {model : HashModel} {params : Stage_decode.Params} {inputs} {out : Int}
    (h : Stage_decode.generatedRoot Backend.backend model params inputs out)
    {G : ExactMatrix Q1 N 1 6} (hG : gadgetMatrixRuns Backend.backend 32768 6 G)
    (E : Fin 2 → ErrorMatrix N 1 6) (hE : ∀ j c i, 4 * |(E j 0 c).coeff i| + 4 < (Q0 : Int))
    (h00 : Residue inputs.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 0) inputs.2.2.2.2.1 inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 0)))
    (h01 : Residue inputs.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 0) inputs.2.2.2.2.1 inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 0)))
    (h02 : Residue inputs.2.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 0) inputs.2.2.2.2.1 inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 0)))
    (h03 : Residue inputs.2.2.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 0) inputs.2.2.2.2.1 inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 0)))
    (h10 : Residue inputs.2.2.2.2.2.2.2.2.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 1) inputs.2.2.2.2.1 1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 1)))
    (h11 : Residue inputs.2.2.2.2.2.2.2.2.2.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 1) inputs.2.2.2.2.1 1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 1)))
    (h12 : Residue inputs.2.2.2.2.2.2.2.2.2.2.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 1) inputs.2.2.2.2.1 1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 1)))
    (h13 : Residue inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
      (topValue model inputs.2.2.2.2.2.1 (topTagI 1) inputs.2.2.2.2.1 1 inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 G (E 1)))
    : out = 0 := by
  obtain ⟨w, h⟩ := h
  unfold Stage_decode.generatedRoot.body at h
  extract_lets -merge _params _backend w_1_0 w_5_0 w_10_0 w_15_0 w_20_0 w_60_0 w_67_0 w_69_0 w_71_0 w_139_0 w_143_0 w_148_0 w_153_0 w_0_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_68_0 w_70_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_80_1 w_80_2 w_80_3 w_80_4 w_80_5 w_80_6 w_80_7 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_90_6 w_90_7 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_100_1 w_100_2 w_100_3 w_100_4 w_100_5 w_100_6 w_100_7 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_140_0 w_141_0 w_142_0 w_144_0 w_145_0 w_146_0 w_147_0 w_149_0 w_150_0 w_151_0 w_152_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 at h
  simp only [Stage_decode.generatedRoot.constraints_0, Stage_decode.generatedRoot.constraints_1, Stage_decode.generatedRoot.constraints_2, Stage_decode.generatedRoot.constraints_3, Stage_decode.generatedRoot.constraints_4, Stage_decode.generatedRoot.constraints_5, Stage_decode.generatedRoot.constraints_6, Stage_decode.generatedRoot.constraints_7] at h
  obtain ⟨h2_0, h6_0, h11_0, h16_0, h21_0, h25_0, h30_0, h35_0, h40_0, h45_0, h50_0, h55_0, h61_0, -, -, -, -, -, -, h62_0, -, -, -, -, -, -, h64_0, h74_0, h75_0, -, -, -, -, -, -, h79_0, -, h80_0, h80_1, h80_2, h80_3, h80_4, h80_5, h80_6, h80_7, -, -, -, -, -, -, h89_0, -, h90_0, h90_1, h90_2, h90_3, h90_4, h90_5, h90_6, h90_7, -, -, -, -, -, -, h99_0, -, h100_0, h100_1, h100_2, h100_3, h100_4, h100_5, h100_6, h100_7, -, -, -, -, -, -, h109_0, -, h110_0, h110_1, h110_2, h110_3, h110_4, h110_5, h110_6, h110_7, -, -, -, -, -, -, h119_0, -, h120_0, h120_1, h120_2, h120_3, h120_4, h120_5, h120_6, h120_7, -, -, -, -, -, -, h129_0, -, h130_0, h130_1, h130_2, h130_3, h130_4, h130_5, h130_6, h130_7, h140_0, h144_0, h149_0, h154_0, h158_0, -, -, -, -, -, -, h159_0, -, -, -, -, -, -, h161_0, -, -, -, -, -, -, h169_0, -, h170_0, h170_1, h170_2, h170_3, h170_4, h170_5, h170_6, h170_7, -, -, -, -, -, -, h179_0, -, h180_0, h180_1, h180_2, h180_3, h180_4, h180_5, h180_6, h180_7, -, -, -, -, -, -, h189_0, -, h190_0, h190_1, h190_2, h190_3, h190_4, h190_5, h190_6, h190_7, -, -, -, -, -, -, h199_0, -, h200_0, h200_1, h200_2, h200_3, h200_4, h200_5, h200_6, h200_7, -, -, -, -, -, -, h209_0, -, h210_0, h210_1, h210_2, h210_3, h210_4, h210_5, h210_6, h210_7, -, -, -, -, -, -, h219_0, -, h220_0, h220_1, h220_2, h220_3, h220_4, h220_5, h220_6, h220_7, hout⟩ := h
  have hg : w_74_0 = G := gadgetMatrixRuns_unique h74_0 hG
  have hσ : w_59_0 = sigmaPoly w_20_0 := by
    obtain ⟨_, e0⟩ := familyGet_eq (k := 0) h21_0
    obtain ⟨_, e1⟩ := familyGet_eq (k := 1) h25_0
    obtain ⟨_, e2⟩ := familyGet_eq (k := 2) h30_0
    obtain ⟨_, e3⟩ := familyGet_eq (k := 3) h35_0
    obtain ⟨_, e4⟩ := familyGet_eq (k := 4) h40_0
    obtain ⟨_, e5⟩ := familyGet_eq (k := 5) h45_0
    obtain ⟨_, e6⟩ := familyGet_eq (k := 6) h50_0
    obtain ⟨_, e7⟩ := familyGet_eq (k := 7) h55_0
    simp only [w_59_0, w_58_0, w_57_0, w_56_0, w_54_0, w_53_0, w_52_0, w_51_0, w_49_0, w_48_0, w_47_0, w_46_0, w_44_0, w_43_0, w_42_0, w_41_0, w_39_0, w_38_0, w_37_0, w_36_0, w_34_0, w_33_0, w_32_0, w_31_0, w_29_0, w_28_0, w_27_0, w_26_0, w_24_0, w_23_0, w_22_0]
    rw [e0, e1, e2, e3, e4, e5, e6, e7]
    rfl
  have hA0 := h61_0.2.2
  have hr00 := slice_row h62_0 rfl (by decide)
  have hr01 := slice_row h64_0 rfl (by decide)
  have he0 : w_77_0 = expectedRow (model.sample Q0 N 2 6 w_60_0
      (completeHashTag (topTagI 0) [])) (sigmaPoly w_20_0) w_67_0 w_69_0 w_71_0 G := by
    simp only [w_77_0, w_76_0, w_73_0, w_72_0, w_70_0, w_68_0, w_66_0, w_65_0, w_63_0]
    rw [hσ, hr00, hr01, h75_0.2, hg, hA0, topTagI, completeHashTag_append, List.append_nil]
    rfl
  have hc0 : w_19_0 = topValue model w_60_0 (topTagI 0) w_20_0 w_67_0 w_69_0 w_71_0 G (E 0) := by
    simp only [w_19_0, w_18_0, w_17_0, w_14_0, w_13_0, w_12_0, w_9_0, w_8_0, w_7_0, w_4_0, w_3_0]
    exact crt_matrix h00 h01 h02 h03 h2_0 h6_0 h11_0 h16_0
  have hd0 : w_78_0 = reduceMatrix Q0 N 1 6 (E 0) := by
    simp only [w_78_0]
    rw [hc0, he0, topValue, matrixSub_eq, add_sub_cancel_left]
  rw [hd0] at h79_0
  rw [hd0] at h89_0
  rw [hd0] at h99_0
  rw [hd0] at h109_0
  rw [hd0] at h119_0
  rw [hd0] at h129_0
  have hA1 := h158_0.2.2
  have hr10 := slice_row h159_0 rfl (by decide)
  have hr11 := slice_row h161_0 rfl (by decide)
  have he1 : w_167_0 = expectedRow (model.sample Q0 N 2 6 w_60_0
      (completeHashTag (topTagI 1) [])) (sigmaPoly w_20_0) 1 w_69_0 w_71_0 G := by
    simp only [w_167_0, w_166_0, w_165_0, w_72_0, w_70_0, w_164_0, w_163_0, w_66_0, w_162_0, w_160_0]
    rw [hσ, hr10, hr11, h75_0.2, hg, hA1, topTagI, completeHashTag_append, List.append_nil]
    rfl
  have hc1 : w_157_0 = topValue model w_60_0 (topTagI 1) w_20_0 1 w_69_0 w_71_0 G (E 1) := by
    simp only [w_157_0, w_156_0, w_155_0, w_152_0, w_151_0, w_150_0, w_147_0, w_146_0, w_145_0, w_142_0, w_141_0]
    exact crt_matrix h10 h11 h12 h13 h140_0 h144_0 h149_0 h154_0
  have hd1 : w_168_0 = reduceMatrix Q0 N 1 6 (E 1) := by
    simp only [w_168_0]
    rw [hc1, he1, topValue, matrixSub_eq, add_sub_cancel_left]
  rw [hd1] at h169_0
  rw [hd1] at h179_0
  rw [hd1] at h189_0
  rw [hd1] at h199_0
  rw [hd1] at h209_0
  rw [hd1] at h219_0
  have z80_0 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_0
  have z80_1 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_1
  have z80_2 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_2
  have z80_3 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_3
  have z80_4 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_4
  have z80_5 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_5
  have z80_6 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_6
  have z80_7 := column_zeroQ0 (E 0) (hE 0) h79_0 (by decide) h80_7
  have z90_0 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_0
  have z90_1 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_1
  have z90_2 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_2
  have z90_3 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_3
  have z90_4 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_4
  have z90_5 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_5
  have z90_6 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_6
  have z90_7 := column_zeroQ0 (E 0) (hE 0) h89_0 (by decide) h90_7
  have z100_0 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_0
  have z100_1 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_1
  have z100_2 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_2
  have z100_3 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_3
  have z100_4 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_4
  have z100_5 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_5
  have z100_6 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_6
  have z100_7 := column_zeroQ0 (E 0) (hE 0) h99_0 (by decide) h100_7
  have z110_0 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_0
  have z110_1 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_1
  have z110_2 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_2
  have z110_3 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_3
  have z110_4 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_4
  have z110_5 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_5
  have z110_6 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_6
  have z110_7 := column_zeroQ0 (E 0) (hE 0) h109_0 (by decide) h110_7
  have z120_0 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_0
  have z120_1 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_1
  have z120_2 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_2
  have z120_3 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_3
  have z120_4 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_4
  have z120_5 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_5
  have z120_6 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_6
  have z120_7 := column_zeroQ0 (E 0) (hE 0) h119_0 (by decide) h120_7
  have z130_0 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_0
  have z130_1 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_1
  have z130_2 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_2
  have z130_3 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_3
  have z130_4 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_4
  have z130_5 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_5
  have z130_6 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_6
  have z130_7 := column_zeroQ0 (E 0) (hE 0) h129_0 (by decide) h130_7
  have z170_0 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_0
  have z170_1 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_1
  have z170_2 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_2
  have z170_3 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_3
  have z170_4 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_4
  have z170_5 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_5
  have z170_6 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_6
  have z170_7 := column_zeroQ0 (E 1) (hE 1) h169_0 (by decide) h170_7
  have z180_0 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_0
  have z180_1 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_1
  have z180_2 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_2
  have z180_3 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_3
  have z180_4 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_4
  have z180_5 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_5
  have z180_6 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_6
  have z180_7 := column_zeroQ0 (E 1) (hE 1) h179_0 (by decide) h180_7
  have z190_0 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_0
  have z190_1 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_1
  have z190_2 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_2
  have z190_3 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_3
  have z190_4 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_4
  have z190_5 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_5
  have z190_6 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_6
  have z190_7 := column_zeroQ0 (E 1) (hE 1) h189_0 (by decide) h190_7
  have z200_0 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_0
  have z200_1 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_1
  have z200_2 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_2
  have z200_3 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_3
  have z200_4 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_4
  have z200_5 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_5
  have z200_6 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_6
  have z200_7 := column_zeroQ0 (E 1) (hE 1) h199_0 (by decide) h200_7
  have z210_0 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_0
  have z210_1 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_1
  have z210_2 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_2
  have z210_3 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_3
  have z210_4 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_4
  have z210_5 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_5
  have z210_6 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_6
  have z210_7 := column_zeroQ0 (E 1) (hE 1) h209_0 (by decide) h210_7
  have z220_0 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_0
  have z220_1 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_1
  have z220_2 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_2
  have z220_3 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_3
  have z220_4 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_4
  have z220_5 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_5
  have z220_6 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_6
  have z220_7 := column_zeroQ0 (E 1) (hE 1) h219_0 (by decide) h220_7
  rw [hout]
  simp only [w_228_0, w_227_0, w_226_0, w_225_0, w_224_0, w_223_0, w_222_0, w_221_0, w_218_0, w_217_0, w_216_0, w_215_0, w_214_0, w_213_0, w_212_0, w_211_0, w_208_0, w_207_0, w_206_0, w_205_0, w_204_0, w_203_0, w_202_0, w_201_0, w_198_0, w_197_0, w_196_0, w_195_0, w_194_0, w_193_0, w_192_0, w_191_0, w_188_0, w_187_0, w_186_0, w_185_0, w_184_0, w_183_0, w_182_0, w_181_0, w_178_0, w_177_0, w_176_0, w_175_0, w_174_0, w_173_0, w_172_0, w_171_0, w_138_0, w_137_0, w_136_0, w_135_0, w_134_0, w_133_0, w_132_0, w_131_0, w_128_0, w_127_0, w_126_0, w_125_0, w_124_0, w_123_0, w_122_0, w_121_0, w_118_0, w_117_0, w_116_0, w_115_0, w_114_0, w_113_0, w_112_0, w_111_0, w_108_0, w_107_0, w_106_0, w_105_0, w_104_0, w_103_0, w_102_0, w_101_0, w_98_0, w_97_0, w_96_0, w_95_0, w_94_0, w_93_0, w_92_0, w_91_0, w_88_0, w_87_0, w_86_0, w_85_0, w_84_0, w_83_0, w_82_0, w_81_0, w_0_0]
  simp only [z80_0, z80_1, z80_2, z80_3, z80_4, z80_5, z80_6, z80_7, z90_0, z90_1, z90_2, z90_3, z90_4, z90_5, z90_6, z90_7, z100_0, z100_1, z100_2, z100_3, z100_4, z100_5, z100_6, z100_7, z110_0, z110_1, z110_2, z110_3, z110_4, z110_5, z110_6, z110_7, z120_0, z120_1, z120_2, z120_3, z120_4, z120_5, z120_6, z120_7, z130_0, z130_1, z130_2, z130_3, z130_4, z130_5, z130_6, z130_7, z170_0, z170_1, z170_2, z170_3, z170_4, z170_5, z170_6, z170_7, z180_0, z180_1, z180_2, z180_3, z180_4, z180_5, z180_6, z180_7, z190_0, z190_1, z190_2, z190_3, z190_4, z190_5, z190_6, z190_7, z200_0, z200_1, z200_2, z200_3, z200_4, z200_5, z200_6, z200_7, z210_0, z210_1, z210_2, z210_3, z210_4, z210_5, z210_6, z210_7, z220_0, z220_1, z220_2, z220_3, z220_4, z220_5, z220_6, z220_7]
  norm_num

end RefreshTopProof
