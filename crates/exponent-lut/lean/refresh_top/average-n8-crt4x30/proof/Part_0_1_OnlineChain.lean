import Part_0_1_Online
import Part_0_1_Readout
import Part_0_1_MaskCarrier
import Part_0_1_RootSums

/-! Chain theorems of the online root of part `0_1` (written by `generator/gen_online_chain.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- The bound of the offset of the mask PRF's binary sampling. -/
abbrev offsetBound (Bone BB BB' Bh5 BS Bh0 : Nat) : Nat :=
  ((3 * lutBranchBound252 ((((3 * lutBranchBound186 (((((10 + 10) * N * ((((12 + 12) * N * Bone * 16384 + (N + 2) * (p6 / 2)) / p6 + N * 1 * BB)) * 16384 + (N + 2) * (p5 / 2)) / p5 + N * 1 * BB'))) Bh5 + ((8 + 8) * N * Bh5 * 16384 + (8 + 8) * N * (((((10 + 10) * N * ((((12 + 12) * N * Bone * 16384 + (N + 2) * (p6 / 2)) / p6 + N * 1 * BB)) * 16384 + (N + 2) * (p5 / 2)) / p5 + N * 1 * BB'))) * 16384 + (N * 1 * (p4 / 2) + p4 / 2))) + (N + 2) * (p4 / 2)) / p4 + 4 * (N * 1 * BS))) Bh0 + ((6 + 6) * N * Bh0 * 16384 + (6 + 6) * N * ((((3 * lutBranchBound186 (((((10 + 10) * N * ((((12 + 12) * N * Bone * 16384 + (N + 2) * (p6 / 2)) / p6 + N * 1 * BB)) * 16384 + (N + 2) * (p5 / 2)) / p5 + N * 1 * BB'))) Bh5 + ((8 + 8) * N * Bh5 * 16384 + (8 + 8) * N * (((((10 + 10) * N * ((((12 + 12) * N * Bone * 16384 + (N + 2) * (p6 / 2)) / p6 + N * 1 * BB)) * 16384 + (N + 2) * (p5 / 2)) / p5 + N * 1 * BB'))) * 16384 + (N * 1 * (p4 / 2) + p4 / 2))) + (N + 2) * (p4 / 2)) / p4 + 4 * (N * 1 * BS))) * 16384 + (N * 1 * (q3 / 2) + q3 / 2))))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The offset of the mask PRF's binary sampling: the encoding of one, moved down, read out at
`X^2` and expanded into the LUT branches of `X^2`. -/
theorem offsetChain {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_87_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_88_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_89_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_90_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_91_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_92_0 : ByteArray}
    {w_93_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_94_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_95_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_96_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_97_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_98_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_99_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_106_0 : ExactMatrix Q7 8 1 12}
    {w_107_0 : ExactMatrix Q7 8 1 12}
    {w_108_0 : ExactMatrix Q7 8 1 12}
    {w_110_0 : ExactMatrix Q7 8 1 12}
    {w_111_0 : ExactMatrix Q7 8 1 12}
    {w_112_0 : ExactMatrix Q7 8 1 12}
    {w_136_0 : ExactMatrix Q6 8 1 10}
    {w_149_0 : ExactMatrix Q6 8 1 10}
    {w_163_0 : ExactMatrix Q5 8 1 8}
    {w_175_0 : ExactMatrix Q5 8 1 8}
    {w_122_0 : ExactMatrix Q0 8 1 6}
    {w_123_0 : ExactMatrix Q1 8 1 6}
    {w_124_0 : ExactMatrix Q1 8 1 6}
    {w_125_0 : ExactMatrix Q1 8 1 6}
    {w_126_0 : ExactMatrix Q1 8 1 6}
    {w_127_0 : ExactMatrix Q5 8 1 6}
    {w_128_0 : ExactMatrix Q0 8 1 6}
    {w_129_0 : ExactMatrix Q0 8 1 6}
    {w_130_0 : ExactMatrix Q0 8 1 6}
    {w_131_0 : ExactMatrix Q0 8 1 6}
    {w_132_0 : ExactMatrix Q7 8 3 12}
    {w_132_concat_1 : ExactMatrix Q7 8 2 12}
    {w_133_0 : ExactMatrix Q7 8 3 12}
    {w_133_concat_1 : ExactMatrix Q7 8 2 12}
    {w_134_0 : ExactMatrix Q7 8 3 24}
    {w_135_0 : ExactMatrix Q6 8 1 10}
    {w_137_0 : ExactMatrix Q6 8 2 10}
    {w_138_0 : ExactMatrix Q6 8 24 10}
    {w_139_0 : ExactMatrix Q7 8 24 10}
    {w_140_0 : ExactMatrix Q7 8 3 10}
    {w_141_0 : ExactMatrix Q7 8 1 10}
    {w_142_0 : ExactMatrix Q6 8 1 10}
    {w_143_0 : ExactMatrix Q7 8 1 10}
    {w_144_0 : ExactMatrix Q6 8 1 10}
    {w_145_0 : ExactMatrix Q7 8 1 10}
    {w_146_0 : ExactMatrix Q6 8 1 10}
    {w_147_0 : ExactMatrix Q6 8 3 10}
    {w_147_concat_1 : ExactMatrix Q6 8 2 10}
    {w_148_0 : ExactMatrix Q6 8 1 10}
    {w_150_0 : ExactMatrix Q6 8 2 10}
    {w_151_0 : ExactMatrix Q6 8 24 10}
    {w_152_0 : ExactMatrix Q7 8 24 10}
    {w_153_0 : ExactMatrix Q7 8 3 10}
    {w_154_0 : ExactMatrix Q7 8 1 10}
    {w_155_0 : ExactMatrix Q6 8 1 10}
    {w_156_0 : ExactMatrix Q7 8 1 10}
    {w_157_0 : ExactMatrix Q6 8 1 10}
    {w_158_0 : ExactMatrix Q7 8 1 10}
    {w_159_0 : ExactMatrix Q6 8 1 10}
    {w_160_0 : ExactMatrix Q6 8 3 10}
    {w_160_concat_1 : ExactMatrix Q6 8 2 10}
    {w_161_0 : ExactMatrix Q6 8 3 20}
    {w_162_0 : ExactMatrix Q5 8 1 8}
    {w_164_0 : ExactMatrix Q5 8 2 8}
    {w_165_0 : ExactMatrix Q5 8 20 8}
    {w_166_0 : ExactMatrix Q6 8 20 8}
    {w_167_0 : ExactMatrix Q6 8 3 8}
    {w_168_0 : ExactMatrix Q6 8 1 8}
    {w_169_0 : ExactMatrix Q5 8 1 8}
    {w_170_0 : ExactMatrix Q6 8 1 8}
    {w_171_0 : ExactMatrix Q5 8 1 8}
    {w_172_0 : ExactMatrix Q6 8 1 8}
    {w_173_0 : ExactMatrix Q5 8 1 8}
    {w_174_0 : ExactMatrix Q5 8 1 8}
    {w_176_0 : ExactMatrix Q5 8 2 8}
    {w_177_0 : ExactMatrix Q5 8 20 8}
    {w_178_0 : ExactMatrix Q6 8 20 8}
    {w_179_0 : ExactMatrix Q6 8 3 8}
    {w_180_0 : ExactMatrix Q6 8 1 8}
    {w_181_0 : ExactMatrix Q5 8 1 8}
    {w_182_0 : ExactMatrix Q6 8 1 8}
    {w_183_0 : ExactMatrix Q5 8 1 8}
    {w_184_0 : ExactMatrix Q6 8 1 8}
    {w_185_0 : ExactMatrix Q5 8 1 8}
    {w_186_0 : ExactMatrix Q5 8 1 6}
    {w_186_1 : ExactMatrix Q5 8 1 6}
    {w_186_2 : ExactMatrix Q5 8 1 6}
    {w_186_3 : ExactMatrix Q5 8 1 6}
    {w_186_4 : ExactMatrix Q5 8 1 6}
    {w_186_5 : ExactMatrix Q5 8 1 6}
    {w_186_6 : ExactMatrix Q0 8 1 6}
    {w_186_7 : ExactMatrix Q0 8 1 6}
    {w_186_8 : ExactMatrix Q0 8 1 6}
    {w_186_9 : ExactMatrix Q0 8 1 6}
    {w_187_0 : ExactMatrix Q5 8 1 8}
    {w_188_0 : Int}
    {w_189_0 : ExactMatrix Q5 8 1 8}
    {w_190_0 : ExactMatrix Q5 8 1 8}
    {w_191_0 : ExactMatrix Q5 8 3 8}
    {w_191_concat_1 : ExactMatrix Q5 8 2 8}
    {w_192_0 : ExactMatrix Q5 8 1 8}
    {w_193_0 : ExactMatrix Q5 8 1 8}
    {w_194_0 : ExactMatrix Q5 8 1 8}
    {w_195_0 : ExactMatrix Q5 8 3 8}
    {w_195_concat_1 : ExactMatrix Q5 8 2 8}
    {w_196_0 : ExactMatrix Q5 8 3 16}
    {w_197_0 : ExactMatrix Q5 8 1 8}
    {w_198_0 : ExactMatrix Q5 8 1 8}
    {w_199_0 : ExactMatrix Q5 8 1 8}
    {w_200_0 : ExactMatrix Q5 8 3 8}
    {w_200_concat_1 : ExactMatrix Q5 8 2 8}
    {w_201_0 : ExactMatrix Q5 8 1 8}
    {w_202_0 : ExactMatrix Q5 8 1 8}
    {w_203_0 : ExactMatrix Q5 8 1 8}
    {w_204_0 : ExactMatrix Q5 8 3 8}
    {w_204_concat_1 : ExactMatrix Q5 8 2 8}
    {w_205_0 : ExactMatrix Q5 8 3 16}
    {w_206_0 : ExactMatrix Q0 8 2 6}
    {w_207_0 : ExactMatrix Q0 8 16 6}
    {w_208_0 : ExactMatrix Q5 8 16 6}
    {w_209_0 : ExactMatrix Q5 8 3 6}
    {w_210_0 : ExactMatrix Q5 8 1 6}
    {w_211_0 : ExactMatrix Q0 8 1 6}
    {w_212_0 : ExactMatrix Q5 8 1 6}
    {w_213_0 : ExactMatrix Q0 8 1 6}
    {w_214_0 : ExactMatrix Q0 8 2 6}
    {w_215_0 : ExactMatrix Q0 8 16 6}
    {w_216_0 : ExactMatrix Q5 8 16 6}
    {w_217_0 : ExactMatrix Q5 8 3 6}
    {w_218_0 : ExactMatrix Q5 8 1 6}
    {w_219_0 : ExactMatrix Q5 8 1 6}
    {w_220_0 : ExactMatrix Q5 8 1 6}
    {w_221_0 : ExactMatrix Q5 8 1 6}
    {w_222_0 : ExactMatrix Q0 8 1 6}
    {w_223_0 : ExactMatrix Q5 8 1 6}
    {w_224_0 : ExactMatrix Q5 8 1 6}
    {w_225_0 : ExactMatrix Q0 8 1 6}
    {w_226_0 : ExactMatrix Q5 8 1 6}
    {w_227_0 : ExactMatrix Q5 8 1 6}
    {w_228_0 : ExactMatrix Q0 8 1 6}
    {w_229_0 : ExactMatrix Q0 8 2 6}
    {w_230_0 : ExactMatrix Q0 8 16 6}
    {w_231_0 : ExactMatrix Q5 8 16 6}
    {w_232_0 : ExactMatrix Q5 8 3 6}
    {w_233_0 : ExactMatrix Q5 8 1 6}
    {w_234_0 : ExactMatrix Q0 8 1 6}
    {w_235_0 : ExactMatrix Q5 8 1 6}
    {w_236_0 : ExactMatrix Q0 8 1 6}
    {w_237_0 : ExactMatrix Q0 8 2 6}
    {w_238_0 : ExactMatrix Q0 8 16 6}
    {w_239_0 : ExactMatrix Q5 8 16 6}
    {w_240_0 : ExactMatrix Q5 8 3 6}
    {w_241_0 : ExactMatrix Q5 8 1 6}
    {w_242_0 : ExactMatrix Q5 8 1 6}
    {w_243_0 : ExactMatrix Q5 8 1 6}
    {w_244_0 : ExactMatrix Q5 8 1 6}
    {w_245_0 : ExactMatrix Q0 8 1 6}
    {w_246_0 : ExactMatrix Q5 8 1 6}
    {w_247_0 : ExactMatrix Q5 8 1 6}
    {w_248_0 : ExactMatrix Q0 8 1 6}
    {w_249_0 : ExactMatrix Q5 8 1 6}
    {w_250_0 : ExactMatrix Q5 8 1 6}
    {w_251_0 : ExactMatrix Q0 8 1 6}
    {w_252_0 : ExactMatrix Q0 8 1 6}
    {w_252_1 : ExactMatrix Q0 8 1 6}
    {w_252_2 : ExactMatrix Q0 8 1 6}
    {w_252_3 : ExactMatrix Q0 8 1 6}
    {w_252_4 : ExactMatrix Q0 8 1 6}
    {w_252_5 : ExactMatrix Q0 8 1 6}
    {w_252_6 : ExactMatrix Q1 8 1 6}
    {w_252_7 : ExactMatrix Q1 8 1 6}
    {w_252_8 : ExactMatrix Q1 8 1 6}
    {w_252_9 : ExactMatrix Q1 8 1 6}
    {w_253_0 : ExactMatrix Q0 8 1 6}
    {w_254_0 : Int}
    {w_255_0 : ExactMatrix Q0 8 1 6}
    {w_256_0 : ExactMatrix Q0 8 1 6}
    {w_257_0 : ExactMatrix Q0 8 3 6}
    {w_257_concat_1 : ExactMatrix Q0 8 2 6}
    {w_258_0 : ExactMatrix Q0 8 1 6}
    {w_259_0 : ExactMatrix Q0 8 1 6}
    {w_260_0 : ExactMatrix Q0 8 1 6}
    {w_261_0 : ExactMatrix Q0 8 3 6}
    {w_261_concat_1 : ExactMatrix Q0 8 2 6}
    {w_262_0 : ExactMatrix Q0 8 3 12}
    {w_263_0 : ExactMatrix Q0 8 1 6}
    {w_264_0 : ExactMatrix Q0 8 1 6}
    {w_265_0 : ExactMatrix Q0 8 1 6}
    {w_266_0 : ExactMatrix Q0 8 3 6}
    {w_266_concat_1 : ExactMatrix Q0 8 2 6}
    {w_267_0 : ExactMatrix Q0 8 1 6}
    {w_268_0 : ExactMatrix Q0 8 1 6}
    {w_269_0 : ExactMatrix Q0 8 1 6}
    {w_270_0 : ExactMatrix Q0 8 3 6}
    {w_270_concat_1 : ExactMatrix Q0 8 2 6}
    {w_271_0 : ExactMatrix Q0 8 3 12}
    {w_272_0 : ExactMatrix Q1 8 2 6}
    {w_273_0 : ExactMatrix Q1 8 12 6}
    {w_274_0 : ExactMatrix Q0 8 12 6}
    {w_275_0 : ExactMatrix Q0 8 3 6}
    {w_276_0 : ExactMatrix Q0 8 1 6}
    {w_277_0 : ExactMatrix Q1 8 1 6}
    {w_278_0 : ExactMatrix Q0 8 1 6}
    {w_279_0 : ExactMatrix Q1 8 1 6}
    {w_280_0 : ExactMatrix Q1 8 2 6}
    {w_281_0 : ExactMatrix Q1 8 12 6}
    {w_282_0 : ExactMatrix Q0 8 12 6}
    {w_283_0 : ExactMatrix Q0 8 3 6}
    {w_284_0 : ExactMatrix Q0 8 1 6}
    {w_285_0 : ExactMatrix Q0 8 1 6}
    {w_286_0 : ExactMatrix Q1 8 2 6}
    {w_287_0 : ExactMatrix Q1 8 12 6}
    {w_288_0 : ExactMatrix Q0 8 12 6}
    {w_289_0 : ExactMatrix Q0 8 3 6}
    {w_290_0 : ExactMatrix Q0 8 1 6}
    {w_291_0 : ExactMatrix Q1 8 1 6}
    {w_292_0 : ExactMatrix Q0 8 1 6}
    {w_293_0 : ExactMatrix Q1 8 1 6}
    {w_294_0 : ExactMatrix Q1 8 2 6}
    {w_295_0 : ExactMatrix Q1 8 12 6}
    {w_296_0 : ExactMatrix Q0 8 12 6}
    {w_297_0 : ExactMatrix Q0 8 3 6}
    {w_298_0 : ExactMatrix Q0 8 1 6}
    {w_299_0 : ExactMatrix Q0 8 1 6}
    {w_300_0 : ExactMatrix Q0 8 1 6}
    {w_301_0 : ExactMatrix Q0 8 1 6}
    {w_302_0 : ExactMatrix Q0 8 1 6}
    {w_303_0 : ExactMatrix Q0 8 1 6}
    {w_304_0 : ExactMatrix Q0 8 1 6}
    {w_305_0 : ExactMatrix Q0 8 1 6}
    {w_306_0 : ExactMatrix Q0 8 1 6}
    {w_307_0 : ExactMatrix Q0 8 1 6}
    {w_308_0 : ExactMatrix Q0 8 1 6}
    {w_309_0 : ExactMatrix Q0 8 1 6}
    {w_310_0 : ExactMatrix Q0 8 1 6}
    {w_311_0 : ExactMatrix Q0 8 1 6}
    (e_122_0 : w_122_0 = (0 : ExactMatrix Q0 8 1 6))
    (e_127_0 : w_127_0 = (0 : ExactMatrix Q5 8 1 6))
    (e_140_0 : w_140_0 = matrixMul w_134_0 w_139_0)
    (e_153_0 : w_153_0 = matrixMul w_134_0 w_152_0)
    (e_167_0 : w_167_0 = matrixMul w_161_0 w_166_0)
    (e_179_0 : w_179_0 = matrixMul w_161_0 w_178_0)
    (e_188_0 : w_188_0 = MxxIR.selectInt (3) [(1), (3), (5), (15)])
    (e_209_0 : w_209_0 = matrixMul w_205_0 w_208_0)
    (e_217_0 : w_217_0 = matrixMul w_196_0 w_216_0)
    (e_220_0 : w_220_0 = matrixAdd w_218_0 w_219_0)
    (e_221_0 : w_221_0 = matrixAdd w_186_0 w_220_0)
    (e_224_0 : w_224_0 = matrixAdd w_186_2 w_223_0)
    (e_227_0 : w_227_0 = matrixAdd w_186_4 w_226_0)
    (e_232_0 : w_232_0 = matrixMul w_205_0 w_231_0)
    (e_240_0 : w_240_0 = matrixMul w_196_0 w_239_0)
    (e_243_0 : w_243_0 = matrixAdd w_241_0 w_242_0)
    (e_244_0 : w_244_0 = matrixAdd w_186_1 w_243_0)
    (e_247_0 : w_247_0 = matrixAdd w_186_3 w_246_0)
    (e_250_0 : w_250_0 = matrixAdd w_186_5 w_249_0)
    (e_254_0 : w_254_0 = MxxIR.selectInt (3) [(1), (3), (5), (15)])
    (e_275_0 : w_275_0 = matrixMul w_271_0 w_274_0)
    (e_283_0 : w_283_0 = matrixMul w_262_0 w_282_0)
    (e_285_0 : w_285_0 = matrixAdd w_252_2 w_284_0)
    (e_289_0 : w_289_0 = matrixMul w_271_0 w_288_0)
    (e_297_0 : w_297_0 = matrixMul w_262_0 w_296_0)
    (e_299_0 : w_299_0 = matrixAdd w_252_3 w_298_0)
    (e_301_0 : w_301_0 = matrixAdd w_252_4 w_300_0)
    (e_303_0 : w_303_0 = matrixAdd w_252_5 w_302_0)
    (e_306_0 : w_306_0 = matrixAdd w_304_0 w_305_0)
    (e_307_0 : w_307_0 = matrixAdd w_252_0 w_306_0)
    (e_310_0 : w_310_0 = matrixAdd w_308_0 w_309_0)
    (e_311_0 : w_311_0 = matrixAdd w_252_1 w_310_0)
    (h106_0 : concatRows w_108_0 w_106_0 w_132_concat_1)
    (h132_0 : concatRows w_132_concat_1 w_107_0 w_132_0)
    (h110_0 : concatRows w_112_0 w_110_0 w_133_concat_1)
    (h133_0 : concatRows w_133_concat_1 w_111_0 w_133_0)
    (h134_0 : concatColumns w_132_0 w_133_0 w_134_0)
    (h137_0 : concatRows w_135_0 w_136_0 w_137_0)
    (h138_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_137_0 w_138_0)
    (h139_0 : compactCenteredRebaseRuns 16384 w_138_0 w_139_0)
    (h141_0 : sliceMatrix w_140_0 0 1 0 10 w_141_0)
    (h142_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_141_0 w_142_0)
    (h143_0 : sliceMatrix w_140_0 1 2 0 10 w_143_0)
    (h144_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_143_0 w_144_0)
    (h145_0 : sliceMatrix w_140_0 2 3 0 10 w_145_0)
    (h146_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_145_0 w_146_0)
    (h144_0_1 : concatRows w_142_0 w_144_0 w_147_concat_1)
    (h147_0 : concatRows w_147_concat_1 w_146_0 w_147_0)
    (h150_0 : concatRows w_148_0 w_149_0 w_150_0)
    (h151_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_150_0 w_151_0)
    (h152_0 : compactCenteredRebaseRuns 16384 w_151_0 w_152_0)
    (h154_0 : sliceMatrix w_153_0 0 1 0 10 w_154_0)
    (h155_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_154_0 w_155_0)
    (h156_0 : sliceMatrix w_153_0 1 2 0 10 w_156_0)
    (h157_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_156_0 w_157_0)
    (h158_0 : sliceMatrix w_153_0 2 3 0 10 w_158_0)
    (h159_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_158_0 w_159_0)
    (h157_0_1 : concatRows w_155_0 w_157_0 w_160_concat_1)
    (h160_0 : concatRows w_160_concat_1 w_159_0 w_160_0)
    (h161_0 : concatColumns w_147_0 w_160_0 w_161_0)
    (h164_0 : concatRows w_162_0 w_163_0 w_164_0)
    (h165_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_164_0 w_165_0)
    (h166_0 : compactCenteredRebaseRuns 16384 w_165_0 w_166_0)
    (h168_0 : sliceMatrix w_167_0 0 1 0 8 w_168_0)
    (h169_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_168_0 w_169_0)
    (h170_0 : sliceMatrix w_167_0 1 2 0 8 w_170_0)
    (h171_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_170_0 w_171_0)
    (h172_0 : sliceMatrix w_167_0 2 3 0 8 w_172_0)
    (h173_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_172_0 w_173_0)
    (h176_0 : concatRows w_174_0 w_175_0 w_176_0)
    (h177_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_176_0 w_177_0)
    (h178_0 : compactCenteredRebaseRuns 16384 w_177_0 w_178_0)
    (h180_0 : sliceMatrix w_179_0 0 1 0 8 w_180_0)
    (h181_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_180_0 w_181_0)
    (h182_0 : sliceMatrix w_179_0 1 2 0 8 w_182_0)
    (h183_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_182_0 w_183_0)
    (h184_0 : sliceMatrix w_179_0 2 3 0 8 w_184_0)
    (h185_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_184_0 w_185_0)
    (h128_0 : familyGetStatic w_94_0 (0) w_128_0)
    (h129_0 : familyGetStatic w_95_0 (0) w_129_0)
    (h130_0 : familyGetStatic w_96_0 (0) w_130_0)
    (h131_0 : familyGetStatic w_97_0 (0) w_131_0)
    (h186_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_186 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_98_0, (w_92_0, (w_99_0, (w_169_0, (w_171_0, (w_173_0, (w_181_0, (w_183_0, (w_185_0, (w_94_0, (w_95_0, (w_96_0, (w_97_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, ())))))))))) (w_186_0, (w_186_1, (w_186_2, (w_186_3, (w_186_4, (w_186_5, (w_186_6, (w_186_7, (w_186_8, (w_186_9, ())))))))))))
    (h187_0 : familyGetStatic w_98_0 (3) w_187_0)
    (h189_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_92_0) w_189_0)
    (h190_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_92_0) w_190_0)
    (h189_0_1 : concatRows w_187_0 w_189_0 w_191_concat_1)
    (h191_0 : concatRows w_191_concat_1 w_190_0 w_191_0)
    (h192_0 : familyGetStatic w_99_0 (3) w_192_0)
    (h193_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_92_0) w_193_0)
    (h194_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_92_0) w_194_0)
    (h193_0_1 : concatRows w_192_0 w_193_0 w_195_concat_1)
    (h195_0 : concatRows w_195_concat_1 w_194_0 w_195_0)
    (h196_0 : concatColumns w_191_0 w_195_0 w_196_0)
    (h197_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_169_0 w_197_0)
    (h198_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_171_0 w_198_0)
    (h199_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_173_0 w_199_0)
    (h198_0_1 : concatRows w_197_0 w_198_0 w_200_concat_1)
    (h200_0 : concatRows w_200_concat_1 w_199_0 w_200_0)
    (h201_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_181_0 w_201_0)
    (h202_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_183_0 w_202_0)
    (h203_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_185_0 w_203_0)
    (h202_0_1 : concatRows w_201_0 w_202_0 w_204_concat_1)
    (h204_0 : concatRows w_204_concat_1 w_203_0 w_204_0)
    (h205_0 : concatColumns w_200_0 w_204_0 w_205_0)
    (h206_0 : concatRows w_186_6 w_186_8 w_206_0)
    (h207_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_206_0 w_207_0)
    (h208_0 : compactCenteredRebaseRuns 16384 w_207_0 w_208_0)
    (h210_0 : sliceMatrix w_209_0 1 2 0 6 w_210_0)
    (h211_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_210_0 w_211_0)
    (h212_0 : sliceMatrix w_209_0 2 3 0 6 w_212_0)
    (h213_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_212_0 w_213_0)
    (h214_0 : concatRows w_211_0 w_213_0 w_214_0)
    (h215_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_214_0 w_215_0)
    (h216_0 : compactCenteredRebaseRuns 16384 w_215_0 w_216_0)
    (h218_0 : sliceMatrix w_217_0 0 1 0 6 w_218_0)
    (h219_0 : sliceMatrix w_209_0 0 1 0 6 w_219_0)
    (h222_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_221_0 w_222_0)
    (h223_0 : sliceMatrix w_217_0 1 2 0 6 w_223_0)
    (h225_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_224_0 w_225_0)
    (h226_0 : sliceMatrix w_217_0 2 3 0 6 w_226_0)
    (h228_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_227_0 w_228_0)
    (h229_0 : concatRows w_186_7 w_186_9 w_229_0)
    (h230_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_229_0 w_230_0)
    (h231_0 : compactCenteredRebaseRuns 16384 w_230_0 w_231_0)
    (h233_0 : sliceMatrix w_232_0 1 2 0 6 w_233_0)
    (h234_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_233_0 w_234_0)
    (h235_0 : sliceMatrix w_232_0 2 3 0 6 w_235_0)
    (h236_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_235_0 w_236_0)
    (h237_0 : concatRows w_234_0 w_236_0 w_237_0)
    (h238_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_237_0 w_238_0)
    (h239_0 : compactCenteredRebaseRuns 16384 w_238_0 w_239_0)
    (h241_0 : sliceMatrix w_240_0 0 1 0 6 w_241_0)
    (h242_0 : sliceMatrix w_232_0 0 1 0 6 w_242_0)
    (h245_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_244_0 w_245_0)
    (h246_0 : sliceMatrix w_240_0 1 2 0 6 w_246_0)
    (h248_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_247_0 w_248_0)
    (h249_0 : sliceMatrix w_240_0 2 3 0 6 w_249_0)
    (h251_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_250_0 w_251_0)
    (h123_0 : familyGetStatic w_87_0 (0) w_123_0)
    (h124_0 : familyGetStatic w_88_0 (0) w_124_0)
    (h125_0 : familyGetStatic w_89_0 (0) w_125_0)
    (h126_0 : familyGetStatic w_90_0 (0) w_126_0)
    (h252_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_252 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_91_0, (w_92_0, (w_93_0, (w_222_0, (w_225_0, (w_228_0, (w_245_0, (w_248_0, (w_251_0, (w_87_0, (w_88_0, (w_89_0, (w_90_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_123_0, (w_124_0, (w_125_0, (w_126_0, ())))))))))) (w_252_0, (w_252_1, (w_252_2, (w_252_3, (w_252_4, (w_252_5, (w_252_6, (w_252_7, (w_252_8, (w_252_9, ())))))))))))
    (h253_0 : familyGetStatic w_91_0 (3) w_253_0)
    (h255_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_92_0) w_255_0)
    (h256_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_92_0) w_256_0)
    (h255_0_1 : concatRows w_253_0 w_255_0 w_257_concat_1)
    (h257_0 : concatRows w_257_concat_1 w_256_0 w_257_0)
    (h258_0 : familyGetStatic w_93_0 (3) w_258_0)
    (h259_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_92_0) w_259_0)
    (h260_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_92_0) w_260_0)
    (h259_0_1 : concatRows w_258_0 w_259_0 w_261_concat_1)
    (h261_0 : concatRows w_261_concat_1 w_260_0 w_261_0)
    (h262_0 : concatColumns w_257_0 w_261_0 w_262_0)
    (h263_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_222_0 w_263_0)
    (h264_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_225_0 w_264_0)
    (h265_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_228_0 w_265_0)
    (h264_0_1 : concatRows w_263_0 w_264_0 w_266_concat_1)
    (h266_0 : concatRows w_266_concat_1 w_265_0 w_266_0)
    (h267_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_245_0 w_267_0)
    (h268_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_248_0 w_268_0)
    (h269_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_251_0 w_269_0)
    (h268_0_1 : concatRows w_267_0 w_268_0 w_270_concat_1)
    (h270_0 : concatRows w_270_concat_1 w_269_0 w_270_0)
    (h271_0 : concatColumns w_266_0 w_270_0 w_271_0)
    (h272_0 : concatRows w_252_6 w_252_8 w_272_0)
    (h273_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_272_0 w_273_0)
    (h274_0 : compactCenteredRebaseRuns 16384 w_273_0 w_274_0)
    (h276_0 : sliceMatrix w_275_0 1 2 0 6 w_276_0)
    (h277_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_276_0 w_277_0)
    (h278_0 : sliceMatrix w_275_0 2 3 0 6 w_278_0)
    (h279_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_278_0 w_279_0)
    (h280_0 : concatRows w_277_0 w_279_0 w_280_0)
    (h281_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_280_0 w_281_0)
    (h282_0 : compactCenteredRebaseRuns 16384 w_281_0 w_282_0)
    (h284_0 : sliceMatrix w_283_0 1 2 0 6 w_284_0)
    (h286_0 : concatRows w_252_7 w_252_9 w_286_0)
    (h287_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_286_0 w_287_0)
    (h288_0 : compactCenteredRebaseRuns 16384 w_287_0 w_288_0)
    (h290_0 : sliceMatrix w_289_0 1 2 0 6 w_290_0)
    (h291_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_290_0 w_291_0)
    (h292_0 : sliceMatrix w_289_0 2 3 0 6 w_292_0)
    (h293_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_292_0 w_293_0)
    (h294_0 : concatRows w_291_0 w_293_0 w_294_0)
    (h295_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_294_0 w_295_0)
    (h296_0 : compactCenteredRebaseRuns 16384 w_295_0 w_296_0)
    (h298_0 : sliceMatrix w_297_0 1 2 0 6 w_298_0)
    (h300_0 : sliceMatrix w_283_0 2 3 0 6 w_300_0)
    (h302_0 : sliceMatrix w_297_0 2 3 0 6 w_302_0)
    (h304_0 : sliceMatrix w_283_0 0 1 0 6 w_304_0)
    (h305_0 : sliceMatrix w_275_0 0 1 0 6 w_305_0)
    (h308_0 : sliceMatrix w_297_0 0 1 0 6 w_308_0)
    (h309_0 : sliceMatrix w_289_0 0 1 0 6 w_309_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {Bone : Nat}
    (hone : Rel sm ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_108_0, w_112_0] Bone)
    {g5' : ExactMatrix Q5 N 1 10} {g6 : ExactMatrix Q6 N 1 10}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hg6 : centeredRebaseRuns g5' g6)
    {eB : Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q6 N sp • ![w_135_0, w_148_0] j + ![w_136_0, w_149_0] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6 + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {g0' : ExactMatrix Q0 N 1 8} {h5 : ExactMatrix Q5 N 1 8}
    (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0') (hh5 : centeredRebaseRuns g0' h5)
    {eB' : Fin 2 → ErrorMatrix N 1 8} {BB' : Nat}
    (hrhs' : ∀ j, reducePoly Q5 N sp • ![w_162_0, w_174_0] j + ![w_163_0, w_175_0] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' j))
    (heB' : ∀ j, CoeffBound (eB' j) BB')
    {Bh5 : Nat} (hhelp5 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_98_0 i, w_99_0 i] Bh5)
    {g1' : ExactMatrix Q1 N 1 6} {g0 : ExactMatrix Q0 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1') (hg0 : centeredRebaseRuns g1' g0)
    {inverse : Nat} (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 0 j i + ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • g0 +
            reduceMatrix Q0 N 1 6 (e j))
    {Bh0 : Nat} (hhelp0 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g0) ![w_91_0 i, w_93_0 i] Bh0) :
    Rel sm ![![w_285_0, w_299_0], ![w_301_0, w_303_0]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 q3
        (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * 1))) •
          (reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 0 j i + ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 1 j i)))
      ![w_307_0, w_311_0] (offsetBound Bone BB BB' Bh5 BS Bh0) := by
  have h1 := rootOffset (model := model) (e_140_0 := e_140_0) (e_153_0 := e_153_0) (e_167_0 := e_167_0) (e_179_0 := e_179_0) (h106_0 := h106_0) (h132_0 := h132_0) (h110_0 := h110_0) (h133_0 := h133_0) (h134_0 := h134_0) (h137_0 := h137_0) (h138_0 := h138_0) (h139_0 := h139_0) (h141_0 := h141_0) (h142_0 := h142_0) (h143_0 := h143_0) (h144_0 := h144_0) (h145_0 := h145_0) (h146_0 := h146_0) (h144_0_1 := h144_0_1) (h147_0 := h147_0) (h150_0 := h150_0) (h151_0 := h151_0) (h152_0 := h152_0) (h154_0 := h154_0) (h155_0 := h155_0) (h156_0 := h156_0) (h157_0 := h157_0) (h158_0 := h158_0) (h159_0 := h159_0) (h157_0_1 := h157_0_1) (h160_0 := h160_0) (h161_0 := h161_0) (h164_0 := h164_0) (h165_0 := h165_0) (h166_0 := h166_0) (h168_0 := h168_0) (h169_0 := h169_0) (h170_0 := h170_0) (h171_0 := h171_0) (h172_0 := h172_0) (h173_0 := h173_0) (h176_0 := h176_0) (h177_0 := h177_0) (h178_0 := h178_0) (h180_0 := h180_0) (h181_0 := h181_0) (h182_0 := h182_0) (h183_0 := h183_0) (h184_0 := h184_0) (h185_0 := h185_0) (hsm := hsm) (hg6' := hg6') (hg7 := hg7)
    (hinput := hone) (hg5' := hg5') (hg6 := hg6) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs')
    (heB' := heB')
  have h2 := rootLut186 (model := model) (e_127_0 := e_127_0) (e_188_0 := e_188_0) (e_209_0 := e_209_0) (e_217_0 := e_217_0) (e_220_0 := e_220_0) (e_221_0 := e_221_0) (e_224_0 := e_224_0) (e_227_0 := e_227_0) (e_232_0 := e_232_0) (e_240_0 := e_240_0) (e_243_0 := e_243_0) (e_244_0 := e_244_0) (e_247_0 := e_247_0) (e_250_0 := e_250_0) (h128_0 := h128_0) (h129_0 := h129_0) (h130_0 := h130_0) (h131_0 := h131_0) (h186_9 := h186_9) (h187_0 := h187_0) (h189_0 := h189_0) (h190_0 := h190_0) (h189_0_1 := h189_0_1) (h191_0 := h191_0) (h192_0 := h192_0) (h193_0 := h193_0) (h194_0 := h194_0) (h193_0_1 := h193_0_1) (h195_0 := h195_0) (h196_0 := h196_0) (h197_0 := h197_0) (h198_0 := h198_0) (h199_0 := h199_0) (h198_0_1 := h198_0_1) (h200_0 := h200_0) (h201_0 := h201_0) (h202_0 := h202_0) (h203_0 := h203_0) (h202_0_1 := h202_0_1) (h204_0 := h204_0) (h205_0 := h205_0) (h206_0 := h206_0) (h207_0 := h207_0) (h208_0 := h208_0) (h210_0 := h210_0) (h211_0 := h211_0) (h212_0 := h212_0) (h213_0 := h213_0) (h214_0 := h214_0) (h215_0 := h215_0) (h216_0 := h216_0) (h218_0 := h218_0) (h219_0 := h219_0) (h222_0 := h222_0) (h223_0 := h223_0) (h225_0 := h225_0) (h226_0 := h226_0) (h228_0 := h228_0) (h229_0 := h229_0) (h230_0 := h230_0) (h231_0 := h231_0) (h233_0 := h233_0) (h234_0 := h234_0) (h235_0 := h235_0) (h236_0 := h236_0) (h237_0 := h237_0) (h238_0 := h238_0) (h239_0 := h239_0) (h241_0 := h241_0) (h242_0 := h242_0) (h245_0 := h245_0) (h246_0 := h246_0) (h248_0 := h248_0) (h249_0 := h249_0) (h251_0 := h251_0) (z := 0) (hsm := hsm) (hg' := hg0') (hg := hh5)
    (hinput := h1) (hhelp := hhelp5) (hinverse := hinverse) (hswitch := hswitch)
  have e : ((((-1 : Int) ^ (0 / 4) : Int) : ErrorPoly N) *
      readoutTable ⟨0 % 4, Nat.mod_lt _ (by norm_num)⟩) = Xe (n := N) (2 * 1) := by
    simp [readoutTable]
  rw [e] at h2
  exact rootLut252 (model := model) (e_122_0 := e_122_0) (e_254_0 := e_254_0) (e_275_0 := e_275_0) (e_283_0 := e_283_0) (e_285_0 := e_285_0) (e_289_0 := e_289_0) (e_297_0 := e_297_0) (e_299_0 := e_299_0) (e_301_0 := e_301_0) (e_303_0 := e_303_0) (e_306_0 := e_306_0) (e_307_0 := e_307_0) (e_310_0 := e_310_0) (e_311_0 := e_311_0) (h123_0 := h123_0) (h124_0 := h124_0) (h125_0 := h125_0) (h126_0 := h126_0) (h252_9 := h252_9) (h253_0 := h253_0) (h255_0 := h255_0) (h256_0 := h256_0) (h255_0_1 := h255_0_1) (h257_0 := h257_0) (h258_0 := h258_0) (h259_0 := h259_0) (h260_0 := h260_0) (h259_0_1 := h259_0_1) (h261_0 := h261_0) (h262_0 := h262_0) (h263_0 := h263_0) (h264_0 := h264_0) (h265_0 := h265_0) (h264_0_1 := h264_0_1) (h266_0 := h266_0) (h267_0 := h267_0) (h268_0 := h268_0) (h269_0 := h269_0) (h268_0_1 := h268_0_1) (h270_0 := h270_0) (h271_0 := h271_0) (h272_0 := h272_0) (h273_0 := h273_0) (h274_0 := h274_0) (h276_0 := h276_0) (h277_0 := h277_0) (h278_0 := h278_0) (h279_0 := h279_0) (h280_0 := h280_0) (h281_0 := h281_0) (h282_0 := h282_0) (h284_0 := h284_0) (h286_0 := h286_0) (h287_0 := h287_0) (h288_0 := h288_0) (h290_0 := h290_0) (h291_0 := h291_0) (h292_0 := h292_0) (h293_0 := h293_0) (h294_0 := h294_0) (h295_0 := h295_0) (h296_0 := h296_0) (h298_0 := h298_0) (h300_0 := h300_0) (h302_0 := h302_0) (h304_0 := h304_0) (h305_0 := h305_0) (h308_0 := h308_0) (h309_0 := h309_0) (z := 1) (hsm := hsm) (hg' := hg1') (hg := hg0)
    (hinput := h2) (hhelp := hhelp0)



/-- The bound of the mask carrier. -/
abbrev maskBound (Ba Bs BBm BBm' BB BB' Bh5 BS Bh0 BS1 : Nat) : Nat :=
  ((8 * ((3 * lutBranchBound312 (maskReadoutBound Ba Bs BBm BBm' Bh5 BS) Bh0 + ((6 + 6) * N * Bh0 * 16384 + (6 + 6) * N * (maskReadoutBound Ba Bs BBm BBm' Bh5 BS) * 16384 + (N * 1 * (q3 / 2) + q3 / 2))) + (offsetBound Bs BB BB' Bh5 BS Bh0)) + (N + 2) * (q3 / 2)) / q3 + 64 * (N * 1 * BS1))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The mask carrier: the mask PRF's members read out, sampled with the offset, placed, summed and
moved down to `Q₁`. -/
theorem maskChain {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_87_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_88_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_89_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_90_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_91_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_92_0 : ByteArray}
    {w_93_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_94_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_95_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_96_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_97_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_98_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_99_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_100_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_101_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_102_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_103_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_104_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_105_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_106_0 : ExactMatrix Q7 8 1 12}
    {w_107_0 : ExactMatrix Q7 8 1 12}
    {w_108_0 : ExactMatrix Q7 8 1 12}
    {w_110_0 : ExactMatrix Q7 8 1 12}
    {w_111_0 : ExactMatrix Q7 8 1 12}
    {w_112_0 : ExactMatrix Q7 8 1 12}
    {w_113_0 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_114_0 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_117_0 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_118_0 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_136_0 : ExactMatrix Q6 8 1 10}
    {w_149_0 : ExactMatrix Q6 8 1 10}
    {w_163_0 : ExactMatrix Q5 8 1 8}
    {w_175_0 : ExactMatrix Q5 8 1 8}
    {w_109_0 : Fin 5 → ExactMatrix Q7 8 1 1}
    {w_115_0 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_115_1 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_115_2 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_115_3 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_115_4 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_115_5 : Fin 40 → ExactMatrix Q6 8 1 10}
    {w_116_0 : Fin 8 → ExactMatrix Q6 8 1 10}
    {w_116_1 : Fin 8 → ExactMatrix Q6 8 1 10}
    {w_116_2 : Fin 8 → ExactMatrix Q6 8 1 10}
    {w_116_3 : Fin 8 → ExactMatrix Q6 8 1 10}
    {w_116_4 : Fin 8 → ExactMatrix Q6 8 1 10}
    {w_116_5 : Fin 8 → ExactMatrix Q6 8 1 10}
    {w_119_0 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_119_1 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_119_2 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_119_3 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_119_4 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_119_5 : Fin 40 → ExactMatrix Q5 8 1 8}
    {w_120_0 : Fin 8 → ExactMatrix Q5 8 1 8}
    {w_120_1 : Fin 8 → ExactMatrix Q5 8 1 8}
    {w_120_2 : Fin 8 → ExactMatrix Q5 8 1 8}
    {w_120_3 : Fin 8 → ExactMatrix Q5 8 1 8}
    {w_120_4 : Fin 8 → ExactMatrix Q5 8 1 8}
    {w_120_5 : Fin 8 → ExactMatrix Q5 8 1 8}
    {w_121_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_121_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_121_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_121_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_121_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_121_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_122_0 : ExactMatrix Q0 8 1 6}
    {w_123_0 : ExactMatrix Q1 8 1 6}
    {w_124_0 : ExactMatrix Q1 8 1 6}
    {w_125_0 : ExactMatrix Q1 8 1 6}
    {w_126_0 : ExactMatrix Q1 8 1 6}
    {w_127_0 : ExactMatrix Q5 8 1 6}
    {w_128_0 : ExactMatrix Q0 8 1 6}
    {w_129_0 : ExactMatrix Q0 8 1 6}
    {w_130_0 : ExactMatrix Q0 8 1 6}
    {w_131_0 : ExactMatrix Q0 8 1 6}
    {w_132_0 : ExactMatrix Q7 8 3 12}
    {w_132_concat_1 : ExactMatrix Q7 8 2 12}
    {w_133_0 : ExactMatrix Q7 8 3 12}
    {w_133_concat_1 : ExactMatrix Q7 8 2 12}
    {w_134_0 : ExactMatrix Q7 8 3 24}
    {w_135_0 : ExactMatrix Q6 8 1 10}
    {w_137_0 : ExactMatrix Q6 8 2 10}
    {w_138_0 : ExactMatrix Q6 8 24 10}
    {w_139_0 : ExactMatrix Q7 8 24 10}
    {w_140_0 : ExactMatrix Q7 8 3 10}
    {w_141_0 : ExactMatrix Q7 8 1 10}
    {w_142_0 : ExactMatrix Q6 8 1 10}
    {w_143_0 : ExactMatrix Q7 8 1 10}
    {w_144_0 : ExactMatrix Q6 8 1 10}
    {w_145_0 : ExactMatrix Q7 8 1 10}
    {w_146_0 : ExactMatrix Q6 8 1 10}
    {w_147_0 : ExactMatrix Q6 8 3 10}
    {w_147_concat_1 : ExactMatrix Q6 8 2 10}
    {w_148_0 : ExactMatrix Q6 8 1 10}
    {w_150_0 : ExactMatrix Q6 8 2 10}
    {w_151_0 : ExactMatrix Q6 8 24 10}
    {w_152_0 : ExactMatrix Q7 8 24 10}
    {w_153_0 : ExactMatrix Q7 8 3 10}
    {w_154_0 : ExactMatrix Q7 8 1 10}
    {w_155_0 : ExactMatrix Q6 8 1 10}
    {w_156_0 : ExactMatrix Q7 8 1 10}
    {w_157_0 : ExactMatrix Q6 8 1 10}
    {w_158_0 : ExactMatrix Q7 8 1 10}
    {w_159_0 : ExactMatrix Q6 8 1 10}
    {w_160_0 : ExactMatrix Q6 8 3 10}
    {w_160_concat_1 : ExactMatrix Q6 8 2 10}
    {w_161_0 : ExactMatrix Q6 8 3 20}
    {w_162_0 : ExactMatrix Q5 8 1 8}
    {w_164_0 : ExactMatrix Q5 8 2 8}
    {w_165_0 : ExactMatrix Q5 8 20 8}
    {w_166_0 : ExactMatrix Q6 8 20 8}
    {w_167_0 : ExactMatrix Q6 8 3 8}
    {w_168_0 : ExactMatrix Q6 8 1 8}
    {w_169_0 : ExactMatrix Q5 8 1 8}
    {w_170_0 : ExactMatrix Q6 8 1 8}
    {w_171_0 : ExactMatrix Q5 8 1 8}
    {w_172_0 : ExactMatrix Q6 8 1 8}
    {w_173_0 : ExactMatrix Q5 8 1 8}
    {w_174_0 : ExactMatrix Q5 8 1 8}
    {w_176_0 : ExactMatrix Q5 8 2 8}
    {w_177_0 : ExactMatrix Q5 8 20 8}
    {w_178_0 : ExactMatrix Q6 8 20 8}
    {w_179_0 : ExactMatrix Q6 8 3 8}
    {w_180_0 : ExactMatrix Q6 8 1 8}
    {w_181_0 : ExactMatrix Q5 8 1 8}
    {w_182_0 : ExactMatrix Q6 8 1 8}
    {w_183_0 : ExactMatrix Q5 8 1 8}
    {w_184_0 : ExactMatrix Q6 8 1 8}
    {w_185_0 : ExactMatrix Q5 8 1 8}
    {w_186_0 : ExactMatrix Q5 8 1 6}
    {w_186_1 : ExactMatrix Q5 8 1 6}
    {w_186_2 : ExactMatrix Q5 8 1 6}
    {w_186_3 : ExactMatrix Q5 8 1 6}
    {w_186_4 : ExactMatrix Q5 8 1 6}
    {w_186_5 : ExactMatrix Q5 8 1 6}
    {w_186_6 : ExactMatrix Q0 8 1 6}
    {w_186_7 : ExactMatrix Q0 8 1 6}
    {w_186_8 : ExactMatrix Q0 8 1 6}
    {w_186_9 : ExactMatrix Q0 8 1 6}
    {w_187_0 : ExactMatrix Q5 8 1 8}
    {w_188_0 : Int}
    {w_189_0 : ExactMatrix Q5 8 1 8}
    {w_190_0 : ExactMatrix Q5 8 1 8}
    {w_191_0 : ExactMatrix Q5 8 3 8}
    {w_191_concat_1 : ExactMatrix Q5 8 2 8}
    {w_192_0 : ExactMatrix Q5 8 1 8}
    {w_193_0 : ExactMatrix Q5 8 1 8}
    {w_194_0 : ExactMatrix Q5 8 1 8}
    {w_195_0 : ExactMatrix Q5 8 3 8}
    {w_195_concat_1 : ExactMatrix Q5 8 2 8}
    {w_196_0 : ExactMatrix Q5 8 3 16}
    {w_197_0 : ExactMatrix Q5 8 1 8}
    {w_198_0 : ExactMatrix Q5 8 1 8}
    {w_199_0 : ExactMatrix Q5 8 1 8}
    {w_200_0 : ExactMatrix Q5 8 3 8}
    {w_200_concat_1 : ExactMatrix Q5 8 2 8}
    {w_201_0 : ExactMatrix Q5 8 1 8}
    {w_202_0 : ExactMatrix Q5 8 1 8}
    {w_203_0 : ExactMatrix Q5 8 1 8}
    {w_204_0 : ExactMatrix Q5 8 3 8}
    {w_204_concat_1 : ExactMatrix Q5 8 2 8}
    {w_205_0 : ExactMatrix Q5 8 3 16}
    {w_206_0 : ExactMatrix Q0 8 2 6}
    {w_207_0 : ExactMatrix Q0 8 16 6}
    {w_208_0 : ExactMatrix Q5 8 16 6}
    {w_209_0 : ExactMatrix Q5 8 3 6}
    {w_210_0 : ExactMatrix Q5 8 1 6}
    {w_211_0 : ExactMatrix Q0 8 1 6}
    {w_212_0 : ExactMatrix Q5 8 1 6}
    {w_213_0 : ExactMatrix Q0 8 1 6}
    {w_214_0 : ExactMatrix Q0 8 2 6}
    {w_215_0 : ExactMatrix Q0 8 16 6}
    {w_216_0 : ExactMatrix Q5 8 16 6}
    {w_217_0 : ExactMatrix Q5 8 3 6}
    {w_218_0 : ExactMatrix Q5 8 1 6}
    {w_219_0 : ExactMatrix Q5 8 1 6}
    {w_220_0 : ExactMatrix Q5 8 1 6}
    {w_221_0 : ExactMatrix Q5 8 1 6}
    {w_222_0 : ExactMatrix Q0 8 1 6}
    {w_223_0 : ExactMatrix Q5 8 1 6}
    {w_224_0 : ExactMatrix Q5 8 1 6}
    {w_225_0 : ExactMatrix Q0 8 1 6}
    {w_226_0 : ExactMatrix Q5 8 1 6}
    {w_227_0 : ExactMatrix Q5 8 1 6}
    {w_228_0 : ExactMatrix Q0 8 1 6}
    {w_229_0 : ExactMatrix Q0 8 2 6}
    {w_230_0 : ExactMatrix Q0 8 16 6}
    {w_231_0 : ExactMatrix Q5 8 16 6}
    {w_232_0 : ExactMatrix Q5 8 3 6}
    {w_233_0 : ExactMatrix Q5 8 1 6}
    {w_234_0 : ExactMatrix Q0 8 1 6}
    {w_235_0 : ExactMatrix Q5 8 1 6}
    {w_236_0 : ExactMatrix Q0 8 1 6}
    {w_237_0 : ExactMatrix Q0 8 2 6}
    {w_238_0 : ExactMatrix Q0 8 16 6}
    {w_239_0 : ExactMatrix Q5 8 16 6}
    {w_240_0 : ExactMatrix Q5 8 3 6}
    {w_241_0 : ExactMatrix Q5 8 1 6}
    {w_242_0 : ExactMatrix Q5 8 1 6}
    {w_243_0 : ExactMatrix Q5 8 1 6}
    {w_244_0 : ExactMatrix Q5 8 1 6}
    {w_245_0 : ExactMatrix Q0 8 1 6}
    {w_246_0 : ExactMatrix Q5 8 1 6}
    {w_247_0 : ExactMatrix Q5 8 1 6}
    {w_248_0 : ExactMatrix Q0 8 1 6}
    {w_249_0 : ExactMatrix Q5 8 1 6}
    {w_250_0 : ExactMatrix Q5 8 1 6}
    {w_251_0 : ExactMatrix Q0 8 1 6}
    {w_252_0 : ExactMatrix Q0 8 1 6}
    {w_252_1 : ExactMatrix Q0 8 1 6}
    {w_252_2 : ExactMatrix Q0 8 1 6}
    {w_252_3 : ExactMatrix Q0 8 1 6}
    {w_252_4 : ExactMatrix Q0 8 1 6}
    {w_252_5 : ExactMatrix Q0 8 1 6}
    {w_252_6 : ExactMatrix Q1 8 1 6}
    {w_252_7 : ExactMatrix Q1 8 1 6}
    {w_252_8 : ExactMatrix Q1 8 1 6}
    {w_252_9 : ExactMatrix Q1 8 1 6}
    {w_253_0 : ExactMatrix Q0 8 1 6}
    {w_254_0 : Int}
    {w_255_0 : ExactMatrix Q0 8 1 6}
    {w_256_0 : ExactMatrix Q0 8 1 6}
    {w_257_0 : ExactMatrix Q0 8 3 6}
    {w_257_concat_1 : ExactMatrix Q0 8 2 6}
    {w_258_0 : ExactMatrix Q0 8 1 6}
    {w_259_0 : ExactMatrix Q0 8 1 6}
    {w_260_0 : ExactMatrix Q0 8 1 6}
    {w_261_0 : ExactMatrix Q0 8 3 6}
    {w_261_concat_1 : ExactMatrix Q0 8 2 6}
    {w_262_0 : ExactMatrix Q0 8 3 12}
    {w_263_0 : ExactMatrix Q0 8 1 6}
    {w_264_0 : ExactMatrix Q0 8 1 6}
    {w_265_0 : ExactMatrix Q0 8 1 6}
    {w_266_0 : ExactMatrix Q0 8 3 6}
    {w_266_concat_1 : ExactMatrix Q0 8 2 6}
    {w_267_0 : ExactMatrix Q0 8 1 6}
    {w_268_0 : ExactMatrix Q0 8 1 6}
    {w_269_0 : ExactMatrix Q0 8 1 6}
    {w_270_0 : ExactMatrix Q0 8 3 6}
    {w_270_concat_1 : ExactMatrix Q0 8 2 6}
    {w_271_0 : ExactMatrix Q0 8 3 12}
    {w_272_0 : ExactMatrix Q1 8 2 6}
    {w_273_0 : ExactMatrix Q1 8 12 6}
    {w_274_0 : ExactMatrix Q0 8 12 6}
    {w_275_0 : ExactMatrix Q0 8 3 6}
    {w_276_0 : ExactMatrix Q0 8 1 6}
    {w_277_0 : ExactMatrix Q1 8 1 6}
    {w_278_0 : ExactMatrix Q0 8 1 6}
    {w_279_0 : ExactMatrix Q1 8 1 6}
    {w_280_0 : ExactMatrix Q1 8 2 6}
    {w_281_0 : ExactMatrix Q1 8 12 6}
    {w_282_0 : ExactMatrix Q0 8 12 6}
    {w_283_0 : ExactMatrix Q0 8 3 6}
    {w_284_0 : ExactMatrix Q0 8 1 6}
    {w_285_0 : ExactMatrix Q0 8 1 6}
    {w_286_0 : ExactMatrix Q1 8 2 6}
    {w_287_0 : ExactMatrix Q1 8 12 6}
    {w_288_0 : ExactMatrix Q0 8 12 6}
    {w_289_0 : ExactMatrix Q0 8 3 6}
    {w_290_0 : ExactMatrix Q0 8 1 6}
    {w_291_0 : ExactMatrix Q1 8 1 6}
    {w_292_0 : ExactMatrix Q0 8 1 6}
    {w_293_0 : ExactMatrix Q1 8 1 6}
    {w_294_0 : ExactMatrix Q1 8 2 6}
    {w_295_0 : ExactMatrix Q1 8 12 6}
    {w_296_0 : ExactMatrix Q0 8 12 6}
    {w_297_0 : ExactMatrix Q0 8 3 6}
    {w_298_0 : ExactMatrix Q0 8 1 6}
    {w_299_0 : ExactMatrix Q0 8 1 6}
    {w_300_0 : ExactMatrix Q0 8 1 6}
    {w_301_0 : ExactMatrix Q0 8 1 6}
    {w_302_0 : ExactMatrix Q0 8 1 6}
    {w_303_0 : ExactMatrix Q0 8 1 6}
    {w_304_0 : ExactMatrix Q0 8 1 6}
    {w_305_0 : ExactMatrix Q0 8 1 6}
    {w_306_0 : ExactMatrix Q0 8 1 6}
    {w_307_0 : ExactMatrix Q0 8 1 6}
    {w_308_0 : ExactMatrix Q0 8 1 6}
    {w_309_0 : ExactMatrix Q0 8 1 6}
    {w_310_0 : ExactMatrix Q0 8 1 6}
    {w_311_0 : ExactMatrix Q0 8 1 6}
    {w_312_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_312_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_312_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_312_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_312_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_312_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_313_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_313_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_313_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_313_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_313_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_313_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_314_0 : ExactMatrix Q0 8 1 6}
    {w_315_0 : ExactMatrix Q0 8 1 6}
    {w_316_0 : ExactMatrix Q0 8 1 6}
    {w_317_0 : ExactMatrix Q0 8 1 6}
    {w_318_0 : ExactMatrix Q0 8 1 6}
    {w_319_0 : ExactMatrix Q0 8 1 6}
    {w_320_0 : ExactMatrix Q0 8 1 6}
    {w_321_0 : ExactMatrix Q0 8 1 6}
    {w_322_0 : ExactMatrix Q0 8 1 6}
    {w_323_0 : ExactMatrix Q0 8 1 6}
    {w_324_0 : ExactMatrix Q0 8 1 6}
    {w_325_0 : ExactMatrix Q0 8 1 6}
    {w_326_0 : ExactMatrix Q0 8 1 6}
    {w_327_0 : ExactMatrix Q0 8 1 6}
    {w_328_0 : ExactMatrix Q0 8 1 6}
    {w_329_0 : ExactMatrix Q0 8 1 6}
    {w_330_0 : ExactMatrix Q0 8 1 6}
    {w_331_0 : ExactMatrix Q0 8 1 6}
    {w_332_0 : ExactMatrix Q0 8 1 6}
    {w_332_1 : ExactMatrix Q0 8 1 6}
    {w_332_2 : ExactMatrix Q0 8 1 6}
    {w_332_3 : ExactMatrix Q0 8 1 6}
    {w_332_4 : ExactMatrix Q0 8 1 6}
    {w_332_5 : ExactMatrix Q0 8 1 6}
    {w_333_0 : ExactMatrix Q1 8 1 6}
    (e_122_0 : w_122_0 = (0 : ExactMatrix Q0 8 1 6))
    (e_127_0 : w_127_0 = (0 : ExactMatrix Q5 8 1 6))
    (e_140_0 : w_140_0 = matrixMul w_134_0 w_139_0)
    (e_153_0 : w_153_0 = matrixMul w_134_0 w_152_0)
    (e_167_0 : w_167_0 = matrixMul w_161_0 w_166_0)
    (e_179_0 : w_179_0 = matrixMul w_161_0 w_178_0)
    (e_188_0 : w_188_0 = MxxIR.selectInt (3) [(1), (3), (5), (15)])
    (e_209_0 : w_209_0 = matrixMul w_205_0 w_208_0)
    (e_217_0 : w_217_0 = matrixMul w_196_0 w_216_0)
    (e_220_0 : w_220_0 = matrixAdd w_218_0 w_219_0)
    (e_221_0 : w_221_0 = matrixAdd w_186_0 w_220_0)
    (e_224_0 : w_224_0 = matrixAdd w_186_2 w_223_0)
    (e_227_0 : w_227_0 = matrixAdd w_186_4 w_226_0)
    (e_232_0 : w_232_0 = matrixMul w_205_0 w_231_0)
    (e_240_0 : w_240_0 = matrixMul w_196_0 w_239_0)
    (e_243_0 : w_243_0 = matrixAdd w_241_0 w_242_0)
    (e_244_0 : w_244_0 = matrixAdd w_186_1 w_243_0)
    (e_247_0 : w_247_0 = matrixAdd w_186_3 w_246_0)
    (e_250_0 : w_250_0 = matrixAdd w_186_5 w_249_0)
    (e_254_0 : w_254_0 = MxxIR.selectInt (3) [(1), (3), (5), (15)])
    (e_275_0 : w_275_0 = matrixMul w_271_0 w_274_0)
    (e_283_0 : w_283_0 = matrixMul w_262_0 w_282_0)
    (e_285_0 : w_285_0 = matrixAdd w_252_2 w_284_0)
    (e_289_0 : w_289_0 = matrixMul w_271_0 w_288_0)
    (e_297_0 : w_297_0 = matrixMul w_262_0 w_296_0)
    (e_299_0 : w_299_0 = matrixAdd w_252_3 w_298_0)
    (e_301_0 : w_301_0 = matrixAdd w_252_4 w_300_0)
    (e_303_0 : w_303_0 = matrixAdd w_252_5 w_302_0)
    (e_306_0 : w_306_0 = matrixAdd w_304_0 w_305_0)
    (e_307_0 : w_307_0 = matrixAdd w_252_0 w_306_0)
    (e_310_0 : w_310_0 = matrixAdd w_308_0 w_309_0)
    (e_311_0 : w_311_0 = matrixAdd w_252_1 w_310_0)
    (e_316_0 : w_316_0 = matrixAdd w_314_0 w_315_0)
    (e_319_0 : w_319_0 = matrixAdd w_317_0 w_318_0)
    (e_322_0 : w_322_0 = matrixAdd w_320_0 w_321_0)
    (e_325_0 : w_325_0 = matrixAdd w_323_0 w_324_0)
    (e_328_0 : w_328_0 = matrixAdd w_326_0 w_327_0)
    (e_331_0 : w_331_0 = matrixAdd w_329_0 w_330_0)
    (h115_5 : (∀ i : Fin 40, W_Stage_online_eval_0_1.parallel_generatedRoot_115 Backend.backend model params i (w_100_0, (w_101_0, (w_102_0, (w_103_0, (w_104_0, (w_105_0, (w_106_0, (w_107_0, (w_108_0, (w_109_0, (w_110_0, (w_111_0, (w_112_0, (w_92_0, ((w_113_0 i), ((w_114_0 i), ())))))))))))))))) ((w_115_0 i), ((w_115_1 i), ((w_115_2 i), ((w_115_3 i), ((w_115_4 i), ((w_115_5 i), ()))))))))
    (h116_5 : (∀ i : Fin 8, W_Stage_online_eval_0_1.parallel_generatedRoot_116 Backend.backend model params i (w_115_0, (w_115_1, (w_115_2, (w_115_3, (w_115_4, (w_115_5, ())))))) ((w_116_0 i), ((w_116_1 i), ((w_116_2 i), ((w_116_3 i), ((w_116_4 i), ((w_116_5 i), ()))))))))
    (h119_5 : (∀ i : Fin 40, W_Stage_online_eval_0_1.parallel_generatedRoot_119 Backend.backend model params i (w_100_0, (w_101_0, (w_102_0, (w_103_0, (w_104_0, (w_105_0, (w_116_0, (w_116_2, (w_116_4, (w_109_0, (w_116_1, (w_116_3, (w_116_5, (w_92_0, ((w_117_0 i), ((w_118_0 i), ())))))))))))))))) ((w_119_0 i), ((w_119_1 i), ((w_119_2 i), ((w_119_3 i), ((w_119_4 i), ((w_119_5 i), ()))))))))
    (h120_5 : (∀ i : Fin 8, W_Stage_online_eval_0_1.parallel_generatedRoot_120 Backend.backend model params i (w_119_0, (w_119_1, (w_119_2, (w_119_3, (w_119_4, (w_119_5, ())))))) ((w_120_0 i), ((w_120_1 i), ((w_120_2 i), ((w_120_3 i), ((w_120_4 i), ((w_120_5 i), ()))))))))
    (h121_5 : (∀ i : Fin 8, W_Stage_online_eval_0_1.parallel_generatedRoot_121 Backend.backend model params i (w_94_0, (w_95_0, (w_96_0, (w_97_0, (w_98_0, (w_92_0, (w_99_0, ((w_120_4 i), ((w_120_0 i), ((w_120_2 i), ((w_120_5 i), ((w_120_1 i), ((w_120_3 i), ()))))))))))))) ((w_121_0 i), ((w_121_1 i), ((w_121_2 i), ((w_121_3 i), ((w_121_4 i), ((w_121_5 i), ()))))))))
    (h312_5 : (∀ i : Fin 8, W_Stage_online_eval_0_1.parallel_generatedRoot_312 Backend.backend model params i (w_87_0, (w_88_0, (w_89_0, (w_90_0, (w_91_0, (w_92_0, (w_93_0, ((w_121_4 i), ((w_121_0 i), ((w_121_2 i), ((w_121_5 i), ((w_121_1 i), ((w_121_3 i), (w_285_0, (w_299_0, (w_301_0, (w_303_0, (w_307_0, (w_311_0, ()))))))))))))))))))) ((w_312_0 i), ((w_312_1 i), ((w_312_2 i), ((w_312_3 i), ((w_312_4 i), ((w_312_5 i), ()))))))))
    (h313_5 : (∀ i : Fin 8, W_Stage_online_eval_0_1.parallel_generatedRoot_313 Backend.backend model params i ((w_312_0 i), ((w_312_1 i), ((w_312_2 i), ((w_312_3 i), ((w_312_4 i), ((w_312_5 i), ())))))) ((w_313_0 i), ((w_313_1 i), ((w_313_2 i), ((w_313_3 i), ((w_313_4 i), ((w_313_5 i), ()))))))))
    (h314_0 : familyGetStatic w_313_0 (0) w_314_0)
    (h315_0 : familyGetStatic w_313_0 (1) w_315_0)
    (h317_0 : familyGetStatic w_313_1 (0) w_317_0)
    (h318_0 : familyGetStatic w_313_1 (1) w_318_0)
    (h320_0 : familyGetStatic w_313_2 (0) w_320_0)
    (h321_0 : familyGetStatic w_313_2 (1) w_321_0)
    (h323_0 : familyGetStatic w_313_3 (0) w_323_0)
    (h324_0 : familyGetStatic w_313_3 (1) w_324_0)
    (h326_0 : familyGetStatic w_313_4 (0) w_326_0)
    (h327_0 : familyGetStatic w_313_4 (1) w_327_0)
    (h329_0 : familyGetStatic w_313_5 (0) w_329_0)
    (h330_0 : familyGetStatic w_313_5 (1) w_330_0)
    (h332_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_332 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_313_0, (w_313_1, (w_313_2, (w_313_3, (w_313_4, (w_313_5, ())))))))))))) next) (Int.toNat (6)) (w_316_0, (w_319_0, (w_322_0, (w_325_0, (w_328_0, (w_331_0, ())))))) (w_332_0, (w_332_1, (w_332_2, (w_332_3, (w_332_4, (w_332_5, ())))))))
    (h333_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_332_5 w_333_0)
    (h106_0 : concatRows w_108_0 w_106_0 w_132_concat_1)
    (h132_0 : concatRows w_132_concat_1 w_107_0 w_132_0)
    (h110_0 : concatRows w_112_0 w_110_0 w_133_concat_1)
    (h133_0 : concatRows w_133_concat_1 w_111_0 w_133_0)
    (h134_0 : concatColumns w_132_0 w_133_0 w_134_0)
    (h137_0 : concatRows w_135_0 w_136_0 w_137_0)
    (h138_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_137_0 w_138_0)
    (h139_0 : compactCenteredRebaseRuns 16384 w_138_0 w_139_0)
    (h141_0 : sliceMatrix w_140_0 0 1 0 10 w_141_0)
    (h142_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_141_0 w_142_0)
    (h143_0 : sliceMatrix w_140_0 1 2 0 10 w_143_0)
    (h144_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_143_0 w_144_0)
    (h145_0 : sliceMatrix w_140_0 2 3 0 10 w_145_0)
    (h146_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_145_0 w_146_0)
    (h144_0_1 : concatRows w_142_0 w_144_0 w_147_concat_1)
    (h147_0 : concatRows w_147_concat_1 w_146_0 w_147_0)
    (h150_0 : concatRows w_148_0 w_149_0 w_150_0)
    (h151_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_150_0 w_151_0)
    (h152_0 : compactCenteredRebaseRuns 16384 w_151_0 w_152_0)
    (h154_0 : sliceMatrix w_153_0 0 1 0 10 w_154_0)
    (h155_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_154_0 w_155_0)
    (h156_0 : sliceMatrix w_153_0 1 2 0 10 w_156_0)
    (h157_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_156_0 w_157_0)
    (h158_0 : sliceMatrix w_153_0 2 3 0 10 w_158_0)
    (h159_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_158_0 w_159_0)
    (h157_0_1 : concatRows w_155_0 w_157_0 w_160_concat_1)
    (h160_0 : concatRows w_160_concat_1 w_159_0 w_160_0)
    (h161_0 : concatColumns w_147_0 w_160_0 w_161_0)
    (h164_0 : concatRows w_162_0 w_163_0 w_164_0)
    (h165_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_164_0 w_165_0)
    (h166_0 : compactCenteredRebaseRuns 16384 w_165_0 w_166_0)
    (h168_0 : sliceMatrix w_167_0 0 1 0 8 w_168_0)
    (h169_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_168_0 w_169_0)
    (h170_0 : sliceMatrix w_167_0 1 2 0 8 w_170_0)
    (h171_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_170_0 w_171_0)
    (h172_0 : sliceMatrix w_167_0 2 3 0 8 w_172_0)
    (h173_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_172_0 w_173_0)
    (h176_0 : concatRows w_174_0 w_175_0 w_176_0)
    (h177_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_176_0 w_177_0)
    (h178_0 : compactCenteredRebaseRuns 16384 w_177_0 w_178_0)
    (h180_0 : sliceMatrix w_179_0 0 1 0 8 w_180_0)
    (h181_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_180_0 w_181_0)
    (h182_0 : sliceMatrix w_179_0 1 2 0 8 w_182_0)
    (h183_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_182_0 w_183_0)
    (h184_0 : sliceMatrix w_179_0 2 3 0 8 w_184_0)
    (h185_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_184_0 w_185_0)
    (h128_0 : familyGetStatic w_94_0 (0) w_128_0)
    (h129_0 : familyGetStatic w_95_0 (0) w_129_0)
    (h130_0 : familyGetStatic w_96_0 (0) w_130_0)
    (h131_0 : familyGetStatic w_97_0 (0) w_131_0)
    (h186_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_186 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_98_0, (w_92_0, (w_99_0, (w_169_0, (w_171_0, (w_173_0, (w_181_0, (w_183_0, (w_185_0, (w_94_0, (w_95_0, (w_96_0, (w_97_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, ())))))))))) (w_186_0, (w_186_1, (w_186_2, (w_186_3, (w_186_4, (w_186_5, (w_186_6, (w_186_7, (w_186_8, (w_186_9, ())))))))))))
    (h187_0 : familyGetStatic w_98_0 (3) w_187_0)
    (h189_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_92_0) w_189_0)
    (h190_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_92_0) w_190_0)
    (h189_0_1 : concatRows w_187_0 w_189_0 w_191_concat_1)
    (h191_0 : concatRows w_191_concat_1 w_190_0 w_191_0)
    (h192_0 : familyGetStatic w_99_0 (3) w_192_0)
    (h193_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_92_0) w_193_0)
    (h194_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (2), .integer w_188_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_92_0) w_194_0)
    (h193_0_1 : concatRows w_192_0 w_193_0 w_195_concat_1)
    (h195_0 : concatRows w_195_concat_1 w_194_0 w_195_0)
    (h196_0 : concatColumns w_191_0 w_195_0 w_196_0)
    (h197_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_169_0 w_197_0)
    (h198_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_171_0 w_198_0)
    (h199_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_173_0 w_199_0)
    (h198_0_1 : concatRows w_197_0 w_198_0 w_200_concat_1)
    (h200_0 : concatRows w_200_concat_1 w_199_0 w_200_0)
    (h201_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_181_0 w_201_0)
    (h202_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_183_0 w_202_0)
    (h203_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_185_0 w_203_0)
    (h202_0_1 : concatRows w_201_0 w_202_0 w_204_concat_1)
    (h204_0 : concatRows w_204_concat_1 w_203_0 w_204_0)
    (h205_0 : concatColumns w_200_0 w_204_0 w_205_0)
    (h206_0 : concatRows w_186_6 w_186_8 w_206_0)
    (h207_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_206_0 w_207_0)
    (h208_0 : compactCenteredRebaseRuns 16384 w_207_0 w_208_0)
    (h210_0 : sliceMatrix w_209_0 1 2 0 6 w_210_0)
    (h211_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_210_0 w_211_0)
    (h212_0 : sliceMatrix w_209_0 2 3 0 6 w_212_0)
    (h213_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_212_0 w_213_0)
    (h214_0 : concatRows w_211_0 w_213_0 w_214_0)
    (h215_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_214_0 w_215_0)
    (h216_0 : compactCenteredRebaseRuns 16384 w_215_0 w_216_0)
    (h218_0 : sliceMatrix w_217_0 0 1 0 6 w_218_0)
    (h219_0 : sliceMatrix w_209_0 0 1 0 6 w_219_0)
    (h222_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_221_0 w_222_0)
    (h223_0 : sliceMatrix w_217_0 1 2 0 6 w_223_0)
    (h225_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_224_0 w_225_0)
    (h226_0 : sliceMatrix w_217_0 2 3 0 6 w_226_0)
    (h228_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_227_0 w_228_0)
    (h229_0 : concatRows w_186_7 w_186_9 w_229_0)
    (h230_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_229_0 w_230_0)
    (h231_0 : compactCenteredRebaseRuns 16384 w_230_0 w_231_0)
    (h233_0 : sliceMatrix w_232_0 1 2 0 6 w_233_0)
    (h234_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_233_0 w_234_0)
    (h235_0 : sliceMatrix w_232_0 2 3 0 6 w_235_0)
    (h236_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_235_0 w_236_0)
    (h237_0 : concatRows w_234_0 w_236_0 w_237_0)
    (h238_0 : gadgetDecomposeRuns Backend.backend 32768 8 w_237_0 w_238_0)
    (h239_0 : compactCenteredRebaseRuns 16384 w_238_0 w_239_0)
    (h241_0 : sliceMatrix w_240_0 0 1 0 6 w_241_0)
    (h242_0 : sliceMatrix w_232_0 0 1 0 6 w_242_0)
    (h245_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_244_0 w_245_0)
    (h246_0 : sliceMatrix w_240_0 1 2 0 6 w_246_0)
    (h248_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_247_0 w_248_0)
    (h249_0 : sliceMatrix w_240_0 2 3 0 6 w_249_0)
    (h251_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) w_250_0 w_251_0)
    (h123_0 : familyGetStatic w_87_0 (0) w_123_0)
    (h124_0 : familyGetStatic w_88_0 (0) w_124_0)
    (h125_0 : familyGetStatic w_89_0 (0) w_125_0)
    (h126_0 : familyGetStatic w_90_0 (0) w_126_0)
    (h252_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_252 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_91_0, (w_92_0, (w_93_0, (w_222_0, (w_225_0, (w_228_0, (w_245_0, (w_248_0, (w_251_0, (w_87_0, (w_88_0, (w_89_0, (w_90_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_123_0, (w_124_0, (w_125_0, (w_126_0, ())))))))))) (w_252_0, (w_252_1, (w_252_2, (w_252_3, (w_252_4, (w_252_5, (w_252_6, (w_252_7, (w_252_8, (w_252_9, ())))))))))))
    (h253_0 : familyGetStatic w_91_0 (3) w_253_0)
    (h255_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_92_0) w_255_0)
    (h256_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_92_0) w_256_0)
    (h255_0_1 : concatRows w_253_0 w_255_0 w_257_concat_1)
    (h257_0 : concatRows w_257_concat_1 w_256_0 w_257_0)
    (h258_0 : familyGetStatic w_93_0 (3) w_258_0)
    (h259_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_92_0) w_259_0)
    (h260_0 : hashSample (model) ([114, 110, 115, 45, 98, 103, 103, 47, 108, 117, 116, 45, 97]) ([.u64Le (3), .integer w_254_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_92_0) w_260_0)
    (h259_0_1 : concatRows w_258_0 w_259_0 w_261_concat_1)
    (h261_0 : concatRows w_261_concat_1 w_260_0 w_261_0)
    (h262_0 : concatColumns w_257_0 w_261_0 w_262_0)
    (h263_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_222_0 w_263_0)
    (h264_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_225_0 w_264_0)
    (h265_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_228_0 w_265_0)
    (h264_0_1 : concatRows w_263_0 w_264_0 w_266_concat_1)
    (h266_0 : concatRows w_266_concat_1 w_265_0 w_266_0)
    (h267_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_245_0 w_267_0)
    (h268_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_248_0 w_268_0)
    (h269_0 : ringAutomorphismRuns (MxxIR.selectInt (3) [(1), (3), (5), (15)]) w_251_0 w_269_0)
    (h268_0_1 : concatRows w_267_0 w_268_0 w_270_concat_1)
    (h270_0 : concatRows w_270_concat_1 w_269_0 w_270_0)
    (h271_0 : concatColumns w_266_0 w_270_0 w_271_0)
    (h272_0 : concatRows w_252_6 w_252_8 w_272_0)
    (h273_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_272_0 w_273_0)
    (h274_0 : compactCenteredRebaseRuns 16384 w_273_0 w_274_0)
    (h276_0 : sliceMatrix w_275_0 1 2 0 6 w_276_0)
    (h277_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_276_0 w_277_0)
    (h278_0 : sliceMatrix w_275_0 2 3 0 6 w_278_0)
    (h279_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_278_0 w_279_0)
    (h280_0 : concatRows w_277_0 w_279_0 w_280_0)
    (h281_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_280_0 w_281_0)
    (h282_0 : compactCenteredRebaseRuns 16384 w_281_0 w_282_0)
    (h284_0 : sliceMatrix w_283_0 1 2 0 6 w_284_0)
    (h286_0 : concatRows w_252_7 w_252_9 w_286_0)
    (h287_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_286_0 w_287_0)
    (h288_0 : compactCenteredRebaseRuns 16384 w_287_0 w_288_0)
    (h290_0 : sliceMatrix w_289_0 1 2 0 6 w_290_0)
    (h291_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_290_0 w_291_0)
    (h292_0 : sliceMatrix w_289_0 2 3 0 6 w_292_0)
    (h293_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_292_0 w_293_0)
    (h294_0 : concatRows w_291_0 w_293_0 w_294_0)
    (h295_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_294_0 w_295_0)
    (h296_0 : compactCenteredRebaseRuns 16384 w_295_0 w_296_0)
    (h298_0 : sliceMatrix w_297_0 1 2 0 6 w_298_0)
    (h300_0 : sliceMatrix w_283_0 2 3 0 6 w_300_0)
    (h302_0 : sliceMatrix w_297_0 2 3 0 6 w_302_0)
    (h304_0 : sliceMatrix w_283_0 0 1 0 6 w_304_0)
    (h305_0 : sliceMatrix w_275_0 0 1 0 6 w_305_0)
    (h308_0 : sliceMatrix w_297_0 0 1 0 6 w_308_0)
    (h309_0 : sliceMatrix w_289_0 0 1 0 6 w_309_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {Bs : Nat}
    (hone : Rel sm ![![w_106_0, w_110_0], ![w_107_0, w_111_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_108_0, w_112_0] Bs)
    {g5' : ExactMatrix Q5 N 1 10} {g6 : ExactMatrix Q6 N 1 10}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hg6 : centeredRebaseRuns g5' g6)
    {eB : Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ j, reducePoly Q6 N sp • ![w_135_0, w_148_0] j + ![w_136_0, w_149_0] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g6 + reduceMatrix Q6 N 1 10 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {g0' : ExactMatrix Q0 N 1 8} {h5 : ExactMatrix Q5 N 1 8}
    (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0') (hh5 : centeredRebaseRuns g0' h5)
    {eB' : Fin 2 → ErrorMatrix N 1 8} {BB' : Nat}
    (hrhs' : ∀ j, reducePoly Q5 N sp • ![w_162_0, w_174_0] j + ![w_163_0, w_175_0] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * Xe (n := N) (2 * 0) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' j))
    (heB' : ∀ j, CoeffBound (eB' j) BB')
    {Bh5 : Nat} (hhelp5 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_98_0 i, w_99_0 i] Bh5)
    {g1' : ExactMatrix Q1 N 1 6} {g0 : ExactMatrix Q0 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1') (hg0 : centeredRebaseRuns g1' g0)
    {inverse : Nat} (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 0 j i + ![![w_94_0, w_95_0], ![w_96_0, w_97_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N ((((q3 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • g0 +
            reduceMatrix Q0 N 1 6 (e j))
    {Bh0 : Nat} (hhelp0 : ∀ i : Fin 4, Rel sm (lutA model w_92_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g0) ![w_91_0 i, w_93_0 i] Bh0)
    {a : Fin 5 → Nat} {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_101_0 k, w_104_0 k], ![w_102_0 k, w_105_0 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_100_0 k, w_103_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_109_0 k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {h6 : ExactMatrix Q6 N 1 10} (hh6 : centeredRebaseRuns g5' h6)
    {s : Fin 40 → Int} {eBm : Fin 40 → Fin 2 → ErrorMatrix N 1 10} {BBm : Nat}
    (hrhsm : ∀ (i : Fin 40) j, reducePoly Q6 N sp • rhsTop model w_92_0 Q6 10 maskRhs0 i j +
        ![w_113_0 i, w_114_0 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eBm i j))
    (heBm : ∀ i j, CoeffBound (eBm i j) BBm)
    {s' : Fin 40 → Int} {eBm' : Fin 40 → Fin 2 → ErrorMatrix N 1 8} {BBm' : Nat}
    (hrhsm' : ∀ (i : Fin 40) j, reducePoly Q5 N sp • rhsTop model w_92_0 Q5 8 maskRhs1 i j +
        ![w_117_0 i, w_118_0 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eBm' i j))
    (heBm' : ∀ i j, CoeffBound (eBm' i j) BBm')
    {sel sel' : Fin 8 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix Q1 N 1 6} {inverse' : Nat}
    (hinverse' : (inverse' : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1) {BS1 : Nat}
    (hswitch1 : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS1) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 0 j i + ![![w_87_0, w_88_0], ![w_89_0, w_90_0]] 1 j i =
        (lutCoefficient Q1 inverse' (fun k ↦ reducePoly Q1 N (T k)) (sigma i).toNat *
          reducePoly Q1 N (Δ' * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm (fun k j ↦ blockModSwitch [q0, q1, q2, q3] 1 (![![w_332_0, w_332_1], ![w_332_2, w_332_3]] k j))
      (fun j ↦ reducePoly Q1 N (Δ' * (∑ o : Fin 8, Xe (n := N) o *
        (sampled T (readoutExp (a (sel' o) + a (sel o))) + sampled T 1)) * unitPair sp j) • gq)
      ![blockModSwitch [q0, q1, q2, q3] 1 w_332_4, w_333_0]
      (maskBound Ba Bs BBm BBm' BB BB' Bh5 BS Bh0 BS1) := by
  have hoff := offsetChain (model := model) (e_122_0 := e_122_0) (e_127_0 := e_127_0) (e_140_0 := e_140_0) (e_153_0 := e_153_0) (e_167_0 := e_167_0) (e_179_0 := e_179_0) (e_188_0 := e_188_0) (e_209_0 := e_209_0) (e_217_0 := e_217_0) (e_220_0 := e_220_0) (e_221_0 := e_221_0) (e_224_0 := e_224_0) (e_227_0 := e_227_0) (e_232_0 := e_232_0) (e_240_0 := e_240_0) (e_243_0 := e_243_0) (e_244_0 := e_244_0) (e_247_0 := e_247_0) (e_250_0 := e_250_0) (e_254_0 := e_254_0) (e_275_0 := e_275_0) (e_283_0 := e_283_0) (e_285_0 := e_285_0) (e_289_0 := e_289_0) (e_297_0 := e_297_0) (e_299_0 := e_299_0) (e_301_0 := e_301_0) (e_303_0 := e_303_0) (e_306_0 := e_306_0) (e_307_0 := e_307_0) (e_310_0 := e_310_0) (e_311_0 := e_311_0) (h106_0 := h106_0) (h132_0 := h132_0) (h110_0 := h110_0) (h133_0 := h133_0) (h134_0 := h134_0) (h137_0 := h137_0) (h138_0 := h138_0) (h139_0 := h139_0) (h141_0 := h141_0) (h142_0 := h142_0) (h143_0 := h143_0) (h144_0 := h144_0) (h145_0 := h145_0) (h146_0 := h146_0) (h144_0_1 := h144_0_1) (h147_0 := h147_0) (h150_0 := h150_0) (h151_0 := h151_0) (h152_0 := h152_0) (h154_0 := h154_0) (h155_0 := h155_0) (h156_0 := h156_0) (h157_0 := h157_0) (h158_0 := h158_0) (h159_0 := h159_0) (h157_0_1 := h157_0_1) (h160_0 := h160_0) (h161_0 := h161_0) (h164_0 := h164_0) (h165_0 := h165_0) (h166_0 := h166_0) (h168_0 := h168_0) (h169_0 := h169_0) (h170_0 := h170_0) (h171_0 := h171_0) (h172_0 := h172_0) (h173_0 := h173_0) (h176_0 := h176_0) (h177_0 := h177_0) (h178_0 := h178_0) (h180_0 := h180_0) (h181_0 := h181_0) (h182_0 := h182_0) (h183_0 := h183_0) (h184_0 := h184_0) (h185_0 := h185_0) (h128_0 := h128_0) (h129_0 := h129_0) (h130_0 := h130_0) (h131_0 := h131_0) (h186_9 := h186_9) (h187_0 := h187_0) (h189_0 := h189_0) (h190_0 := h190_0) (h189_0_1 := h189_0_1) (h191_0 := h191_0) (h192_0 := h192_0) (h193_0 := h193_0) (h194_0 := h194_0) (h193_0_1 := h193_0_1) (h195_0 := h195_0) (h196_0 := h196_0) (h197_0 := h197_0) (h198_0 := h198_0) (h199_0 := h199_0) (h198_0_1 := h198_0_1) (h200_0 := h200_0) (h201_0 := h201_0) (h202_0 := h202_0) (h203_0 := h203_0) (h202_0_1 := h202_0_1) (h204_0 := h204_0) (h205_0 := h205_0) (h206_0 := h206_0) (h207_0 := h207_0) (h208_0 := h208_0) (h210_0 := h210_0) (h211_0 := h211_0) (h212_0 := h212_0) (h213_0 := h213_0) (h214_0 := h214_0) (h215_0 := h215_0) (h216_0 := h216_0) (h218_0 := h218_0) (h219_0 := h219_0) (h222_0 := h222_0) (h223_0 := h223_0) (h225_0 := h225_0) (h226_0 := h226_0) (h228_0 := h228_0) (h229_0 := h229_0) (h230_0 := h230_0) (h231_0 := h231_0) (h233_0 := h233_0) (h234_0 := h234_0) (h235_0 := h235_0) (h236_0 := h236_0) (h237_0 := h237_0) (h238_0 := h238_0) (h239_0 := h239_0) (h241_0 := h241_0) (h242_0 := h242_0) (h245_0 := h245_0) (h246_0 := h246_0) (h248_0 := h248_0) (h249_0 := h249_0) (h251_0 := h251_0) (h123_0 := h123_0) (h124_0 := h124_0) (h125_0 := h125_0) (h126_0 := h126_0) (h252_9 := h252_9) (h253_0 := h253_0) (h255_0 := h255_0) (h256_0 := h256_0) (h255_0_1 := h255_0_1) (h257_0 := h257_0) (h258_0 := h258_0) (h259_0 := h259_0) (h260_0 := h260_0) (h259_0_1 := h259_0_1) (h261_0 := h261_0) (h262_0 := h262_0) (h263_0 := h263_0) (h264_0 := h264_0) (h265_0 := h265_0) (h264_0_1 := h264_0_1) (h266_0 := h266_0) (h267_0 := h267_0) (h268_0 := h268_0) (h269_0 := h269_0) (h268_0_1 := h268_0_1) (h270_0 := h270_0) (h271_0 := h271_0) (h272_0 := h272_0) (h273_0 := h273_0) (h274_0 := h274_0) (h276_0 := h276_0) (h277_0 := h277_0) (h278_0 := h278_0) (h279_0 := h279_0) (h280_0 := h280_0) (h281_0 := h281_0) (h282_0 := h282_0) (h284_0 := h284_0) (h286_0 := h286_0) (h287_0 := h287_0) (h288_0 := h288_0) (h290_0 := h290_0) (h291_0 := h291_0) (h292_0 := h292_0) (h293_0 := h293_0) (h294_0 := h294_0) (h295_0 := h295_0) (h296_0 := h296_0) (h298_0 := h298_0) (h300_0 := h300_0) (h302_0 := h302_0) (h304_0 := h304_0) (h305_0 := h305_0) (h308_0 := h308_0) (h309_0 := h309_0) (hsm := hsm) (hg6' := hg6') (hg7 := hg7)
    (hone := hone) (hg5' := hg5') (hg6 := hg6) (hrhs := hrhs) (heB := heB) (hg0' := hg0')
    (hh5 := hh5) (hrhs' := hrhs') (heB' := heB') (hhelp5 := hhelp5) (hg1' := hg1') (hg0 := hg0)
    (hinverse := hinverse) (hswitch := hswitch) (hhelp0 := hhelp0)
  have hread := fun o ↦ maskReadout (model := model) (h115 := h115_5) (h116 := h116_5)
    (h119 := h119_5) (h120 := h120_5) (h121 := h121_5) hsm hG hg6' hg7 hg5' hh6 ha hpay hone hrhsm
    heBm hrhsm' heBm' hsel hsel' hg0' hh5 hhelp5 hinverse hswitch o
  subst e_316_0 e_319_0 e_322_0 e_325_0 e_328_0 e_331_0
  have hs := rootSum332 h314_0 h315_0 h317_0 h318_0 h320_0 h321_0 h323_0 h324_0 h326_0 h327_0
    h329_0 h330_0 h332_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4]
  exact maskCarrier (model := model) (off := (w_285_0, w_299_0, w_301_0, w_303_0, w_307_0, w_311_0, ()))
    (h312 := h312_5) (h313 := h313_5) hsm hg1' hg0 (ζ := fun o ↦ readoutExp (a (sel' o) + a (sel o)))
    hread hhelp0 hoff hinverse' hswitch1 (by rw [← s5]; exact h333_0)

end RefreshTopProof.Part_0_1
