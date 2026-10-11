import Part_0_1_KeysRoot
import Part_0_1_KeysLuts
import Part_0_1_KeysCells
import Part_0_1_Sums
import Part_0_1_PreSums
import Part_0_1_PrePlace
import Part_0_1_PreRootSums
import Part_0_1_RootSums
import Part_0_1_MaskCarrier

/-! The mask keys of the online root of part `0_1` are those the preprocessing computes (written
by `generator/gen_key_links.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The mask carrier's keys: the online root's are the preprocessing's. -/
theorem maskKeyLink {model : HashModel} {params : Stage_online_eval_0_1.Params} {params' : Stage_preprocess_0_1.Params}
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
    (h115_5 : (∀ i : Fin 40, Stage_online_eval_0_1.parallel_generatedRoot_115 Backend.backend model params i (w_100_0, (w_101_0, (w_102_0, (w_103_0, (w_104_0, (w_105_0, (w_106_0, (w_107_0, (w_108_0, (w_109_0, (w_110_0, (w_111_0, (w_112_0, (w_92_0, ((w_113_0 i), ((w_114_0 i), ())))))))))))))))) ((w_115_0 i), ((w_115_1 i), ((w_115_2 i), ((w_115_3 i), ((w_115_4 i), ((w_115_5 i), ()))))))))
    (h116_5 : (∀ i : Fin 8, Stage_online_eval_0_1.parallel_generatedRoot_116 Backend.backend model params i (w_115_0, (w_115_1, (w_115_2, (w_115_3, (w_115_4, (w_115_5, ())))))) ((w_116_0 i), ((w_116_1 i), ((w_116_2 i), ((w_116_3 i), ((w_116_4 i), ((w_116_5 i), ()))))))))
    (h119_5 : (∀ i : Fin 40, Stage_online_eval_0_1.parallel_generatedRoot_119 Backend.backend model params i (w_100_0, (w_101_0, (w_102_0, (w_103_0, (w_104_0, (w_105_0, (w_116_0, (w_116_2, (w_116_4, (w_109_0, (w_116_1, (w_116_3, (w_116_5, (w_92_0, ((w_117_0 i), ((w_118_0 i), ())))))))))))))))) ((w_119_0 i), ((w_119_1 i), ((w_119_2 i), ((w_119_3 i), ((w_119_4 i), ((w_119_5 i), ()))))))))
    (h120_5 : (∀ i : Fin 8, Stage_online_eval_0_1.parallel_generatedRoot_120 Backend.backend model params i (w_119_0, (w_119_1, (w_119_2, (w_119_3, (w_119_4, (w_119_5, ())))))) ((w_120_0 i), ((w_120_1 i), ((w_120_2 i), ((w_120_3 i), ((w_120_4 i), ((w_120_5 i), ()))))))))
    (h121_5 : (∀ i : Fin 8, Stage_online_eval_0_1.parallel_generatedRoot_121 Backend.backend model params i (w_94_0, (w_95_0, (w_96_0, (w_97_0, (w_98_0, (w_92_0, (w_99_0, ((w_120_4 i), ((w_120_0 i), ((w_120_2 i), ((w_120_5 i), ((w_120_1 i), ((w_120_3 i), ()))))))))))))) ((w_121_0 i), ((w_121_1 i), ((w_121_2 i), ((w_121_3 i), ((w_121_4 i), ((w_121_5 i), ()))))))))
    (h312_5 : (∀ i : Fin 8, Stage_online_eval_0_1.parallel_generatedRoot_312 Backend.backend model params i (w_87_0, (w_88_0, (w_89_0, (w_90_0, (w_91_0, (w_92_0, (w_93_0, ((w_121_4 i), ((w_121_0 i), ((w_121_2 i), ((w_121_5 i), ((w_121_1 i), ((w_121_3 i), (w_285_0, (w_299_0, (w_301_0, (w_303_0, (w_307_0, (w_311_0, ()))))))))))))))))))) ((w_312_0 i), ((w_312_1 i), ((w_312_2 i), ((w_312_3 i), ((w_312_4 i), ((w_312_5 i), ()))))))))
    (h313_5 : (∀ i : Fin 8, Stage_online_eval_0_1.parallel_generatedRoot_313 Backend.backend model params i ((w_312_0 i), ((w_312_1 i), ((w_312_2 i), ((w_312_3 i), ((w_312_4 i), ((w_312_5 i), ())))))) ((w_313_0 i), ((w_313_1 i), ((w_313_2 i), ((w_313_3 i), ((w_313_4 i), ((w_313_5 i), ()))))))))
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
    (h332_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_1.sequential_generatedRoot_332 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_313_0, (w_313_1, (w_313_2, (w_313_3, (w_313_4, (w_313_5, ())))))))))))) next) (Int.toNat (6)) (w_316_0, (w_319_0, (w_322_0, (w_325_0, (w_328_0, (w_331_0, ())))))) (w_332_0, (w_332_1, (w_332_2, (w_332_3, (w_332_4, (w_332_5, ())))))))
    (h106_0 : concatRows w_108_0 w_106_0 w_132_concat_1)
    (h132_0 : concatRows w_132_concat_1 w_107_0 w_132_0)
    (h110_0 : concatRows w_112_0 w_110_0 w_133_concat_1)
    (h133_0 : concatRows w_133_concat_1 w_111_0 w_133_0)
    (h134_0 : concatColumns w_132_0 w_133_0 w_134_0)
    (h137_0 : concatRows w_135_0 w_136_0 w_137_0)
    (h138_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_137_0 w_138_0)
    (h139_0 : compactCenteredRebaseRuns 16384 w_138_0 w_139_0)
    (h143_0 : sliceMatrix w_140_0 1 2 0 10 w_143_0)
    (h144_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_143_0 w_144_0)
    (h145_0 : sliceMatrix w_140_0 2 3 0 10 w_145_0)
    (h146_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) w_145_0 w_146_0)
    (h144_0_1 : concatRows w_142_0 w_144_0 w_147_concat_1)
    (h147_0 : concatRows w_147_concat_1 w_146_0 w_147_0)
    (h150_0 : concatRows w_148_0 w_149_0 w_150_0)
    (h151_0 : gadgetDecomposeRuns Backend.backend 32768 12 w_150_0 w_151_0)
    (h152_0 : compactCenteredRebaseRuns 16384 w_151_0 w_152_0)
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
    (h170_0 : sliceMatrix w_167_0 1 2 0 8 w_170_0)
    (h171_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_170_0 w_171_0)
    (h172_0 : sliceMatrix w_167_0 2 3 0 8 w_172_0)
    (h173_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_172_0 w_173_0)
    (h176_0 : concatRows w_174_0 w_175_0 w_176_0)
    (h177_0 : gadgetDecomposeRuns Backend.backend 32768 10 w_176_0 w_177_0)
    (h178_0 : compactCenteredRebaseRuns 16384 w_177_0 w_178_0)
    (h182_0 : sliceMatrix w_179_0 1 2 0 8 w_182_0)
    (h183_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_182_0 w_183_0)
    (h184_0 : sliceMatrix w_179_0 2 3 0 8 w_184_0)
    (h185_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) w_184_0 w_185_0)
    (h128_0 : familyGetStatic w_94_0 (0) w_128_0)
    (h129_0 : familyGetStatic w_95_0 (0) w_129_0)
    (h130_0 : familyGetStatic w_96_0 (0) w_130_0)
    (h131_0 : familyGetStatic w_97_0 (0) w_131_0)
    (h186_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_1.sequential_generatedRoot_186 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_98_0, (w_92_0, (w_99_0, (w_169_0, (w_171_0, (w_173_0, (w_181_0, (w_183_0, (w_185_0, (w_94_0, (w_95_0, (w_96_0, (w_97_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, ())))))))))) (w_186_0, (w_186_1, (w_186_2, (w_186_3, (w_186_4, (w_186_5, (w_186_6, (w_186_7, (w_186_8, (w_186_9, ())))))))))))
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
    (h252_9 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × ExactMatrix Q1 8 1 6 × Unit) => Stage_online_eval_0_1.sequential_generatedRoot_252 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (w_91_0, (w_92_0, (w_93_0, (w_222_0, (w_225_0, (w_228_0, (w_245_0, (w_248_0, (w_251_0, (w_87_0, (w_88_0, (w_89_0, (w_90_0, ()))))))))))))))))))))))) next) (Int.toNat (3)) (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_122_0, (w_123_0, (w_124_0, (w_125_0, (w_126_0, ())))))))))) (w_252_0, (w_252_1, (w_252_2, (w_252_3, (w_252_4, (w_252_5, (w_252_6, (w_252_7, (w_252_8, (w_252_9, ())))))))))))
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
    {pw_26_0 : ByteArray}
    {pw_32_0 : Fin 40 → Int}
    {pw_33_0 : Fin 40 → Int}
    {pw_37_0 : Fin 40 → Int}
    {pw_38_0 : Fin 40 → Int}
    {pw_27_0 : ExactMatrix Q7 8 1 12}
    {pw_28_0 : ExactMatrix Q7 8 1 12}
    {pw_29_0 : ExactMatrix Q7 8 1 12}
    {pw_30_0 : ExactMatrix Q7 8 1 12}
    {pw_31_0 : ExactMatrix Q7 8 1 1}
    {pw_34_0 : ExactMatrix Q7 8 1 1}
    {pw_35_0 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_1 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_2 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_3 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_4 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_5 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_6 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_35_7 : Fin 40 → ExactMatrix Q6 8 1 10}
    {pw_36_0 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_36_1 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_36_2 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_36_3 : Fin 8 → ExactMatrix Q6 8 1 10}
    {pw_39_0 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_1 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_2 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_3 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_4 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_5 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_6 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_39_7 : Fin 40 → ExactMatrix Q5 8 1 8}
    {pw_40_0 : Fin 8 → ExactMatrix Q5 8 1 8}
    {pw_40_1 : Fin 8 → ExactMatrix Q5 8 1 8}
    {pw_40_2 : Fin 8 → ExactMatrix Q5 8 1 8}
    {pw_40_3 : Fin 8 → ExactMatrix Q5 8 1 8}
    {pw_41_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_41_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_41_2 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_41_3 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_42_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_42_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_42_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_42_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_43_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_43_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_43_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_43_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_44_0 : ExactMatrix Q0 8 1 6}
    {pw_45_0 : ExactMatrix Q5 8 1 6}
    {pw_46_0 : ExactMatrix Q7 8 2 12}
    {pw_47_0 : ExactMatrix Q6 8 1 10}
    {pw_48_0 : ExactMatrix Q6 8 12 10}
    {pw_49_0 : ExactMatrix Q7 8 12 10}
    {pw_50_0 : ExactMatrix Q7 8 2 10}
    {pw_51_0 : ExactMatrix Q7 8 2 12}
    {pw_52_0 : ExactMatrix Q6 8 1 1}
    {pw_54_0 : ExactMatrix Q6 8 1 10}
    {pw_55_0 : ExactMatrix Q6 8 1 10}
    {pw_56_0 : ExactMatrix Q6 8 1 1}
    {pw_57_0 : ExactMatrix Q6 8 1 10}
    {pw_62_0 : ExactMatrix Q6 8 1 1}
    {pw_63_0 : ExactMatrix Q6 8 1 10}
    {pw_64_0 : ExactMatrix Q6 8 1 10}
    {pw_65_0 : ExactMatrix Q6 8 1 10}
    {pw_66_0 : ExactMatrix Q6 8 1 10}
    {pw_67_0 : ExactMatrix Q6 8 1 10}
    {pw_68_0 : ExactMatrix Q6 8 12 10}
    {pw_69_0 : ExactMatrix Q7 8 12 10}
    {pw_70_0 : ExactMatrix Q7 8 2 10}
    {pw_71_0 : ExactMatrix Q7 8 2 10}
    {pw_72_0 : ExactMatrix Q7 8 1 10}
    {pw_73_0 : ExactMatrix Q6 8 1 10}
    {pw_74_0 : ExactMatrix Q7 8 1 10}
    {pw_75_0 : ExactMatrix Q6 8 1 10}
    {pw_76_0 : ExactMatrix Q6 8 2 10}
    {pw_77_0 : ExactMatrix Q5 8 1 8}
    {pw_78_0 : ExactMatrix Q5 8 10 8}
    {pw_79_0 : ExactMatrix Q6 8 10 8}
    {pw_80_0 : ExactMatrix Q6 8 2 8}
    {pw_81_0 : ExactMatrix Q6 8 1 10}
    {pw_82_0 : ExactMatrix Q6 8 12 10}
    {pw_83_0 : ExactMatrix Q7 8 12 10}
    {pw_84_0 : ExactMatrix Q7 8 2 10}
    {pw_85_0 : ExactMatrix Q6 8 1 1}
    {pw_86_0 : ExactMatrix Q6 8 1 10}
    {pw_87_0 : ExactMatrix Q6 8 1 10}
    {pw_88_0 : ExactMatrix Q6 8 1 10}
    {pw_89_0 : ExactMatrix Q6 8 1 10}
    {pw_90_0 : ExactMatrix Q6 8 1 10}
    {pw_91_0 : ExactMatrix Q6 8 1 10}
    {pw_92_0 : ExactMatrix Q6 8 1 10}
    {pw_93_0 : ExactMatrix Q6 8 12 10}
    {pw_94_0 : ExactMatrix Q7 8 12 10}
    {pw_95_0 : ExactMatrix Q7 8 2 10}
    {pw_96_0 : ExactMatrix Q7 8 2 10}
    {pw_97_0 : ExactMatrix Q7 8 1 10}
    {pw_98_0 : ExactMatrix Q6 8 1 10}
    {pw_99_0 : ExactMatrix Q7 8 1 10}
    {pw_100_0 : ExactMatrix Q6 8 1 10}
    {pw_101_0 : ExactMatrix Q6 8 2 10}
    {pw_102_0 : ExactMatrix Q5 8 1 1}
    {pw_104_0 : ExactMatrix Q5 8 1 8}
    {pw_105_0 : ExactMatrix Q5 8 1 8}
    {pw_106_0 : ExactMatrix Q5 8 1 1}
    {pw_107_0 : ExactMatrix Q5 8 1 8}
    {pw_112_0 : ExactMatrix Q5 8 1 1}
    {pw_113_0 : ExactMatrix Q5 8 1 8}
    {pw_114_0 : ExactMatrix Q5 8 1 8}
    {pw_115_0 : ExactMatrix Q5 8 1 8}
    {pw_116_0 : ExactMatrix Q5 8 1 8}
    {pw_117_0 : ExactMatrix Q5 8 1 8}
    {pw_118_0 : ExactMatrix Q5 8 10 8}
    {pw_119_0 : ExactMatrix Q6 8 10 8}
    {pw_120_0 : ExactMatrix Q6 8 2 8}
    {pw_121_0 : ExactMatrix Q6 8 2 8}
    {pw_122_0 : ExactMatrix Q6 8 1 8}
    {pw_123_0 : ExactMatrix Q5 8 1 8}
    {pw_124_0 : ExactMatrix Q6 8 1 8}
    {pw_125_0 : ExactMatrix Q5 8 1 8}
    {pw_126_0 : ExactMatrix Q5 8 1 8}
    {pw_127_0 : ExactMatrix Q5 8 10 8}
    {pw_128_0 : ExactMatrix Q6 8 10 8}
    {pw_129_0 : ExactMatrix Q6 8 2 8}
    {pw_130_0 : ExactMatrix Q5 8 1 1}
    {pw_131_0 : ExactMatrix Q5 8 1 8}
    {pw_132_0 : ExactMatrix Q5 8 1 8}
    {pw_133_0 : ExactMatrix Q5 8 1 8}
    {pw_134_0 : ExactMatrix Q5 8 1 8}
    {pw_135_0 : ExactMatrix Q5 8 1 8}
    {pw_136_0 : ExactMatrix Q5 8 1 8}
    {pw_137_0 : ExactMatrix Q5 8 1 8}
    {pw_138_0 : ExactMatrix Q5 8 10 8}
    {pw_139_0 : ExactMatrix Q6 8 10 8}
    {pw_140_0 : ExactMatrix Q6 8 2 8}
    {pw_141_0 : ExactMatrix Q6 8 2 8}
    {pw_142_0 : ExactMatrix Q6 8 1 8}
    {pw_143_0 : ExactMatrix Q5 8 1 8}
    {pw_144_0 : ExactMatrix Q6 8 1 8}
    {pw_145_0 : ExactMatrix Q5 8 1 8}
    {pw_146_0 : ExactMatrix Q5 8 1 6}
    {pw_146_1 : ExactMatrix Q5 8 1 6}
    {pw_146_2 : ExactMatrix Q5 8 1 6}
    {pw_146_3 : ExactMatrix Q5 8 1 6}
    {pw_147_0 : ExactMatrix Q0 8 1 6}
    {pw_148_0 : ExactMatrix Q0 8 1 6}
    {pw_149_0 : ExactMatrix Q0 8 1 6}
    {pw_150_0 : ExactMatrix Q0 8 1 6}
    {pw_151_0 : ExactMatrix Q0 8 1 6}
    {pw_151_1 : ExactMatrix Q0 8 1 6}
    {pw_151_2 : ExactMatrix Q0 8 1 6}
    {pw_151_3 : ExactMatrix Q0 8 1 6}
    {pw_152_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_152_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_152_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_152_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_153_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_153_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_153_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_153_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_154_0 : ExactMatrix Q0 8 1 6}
    {pw_155_0 : ExactMatrix Q0 8 1 6}
    {pw_156_0 : ExactMatrix Q0 8 1 6}
    {pw_157_0 : ExactMatrix Q0 8 1 6}
    {pw_158_0 : ExactMatrix Q0 8 1 6}
    {pw_159_0 : ExactMatrix Q0 8 1 6}
    {pw_160_0 : ExactMatrix Q0 8 1 6}
    {pw_161_0 : ExactMatrix Q0 8 1 6}
    {pw_162_0 : ExactMatrix Q0 8 1 6}
    {pw_163_0 : ExactMatrix Q0 8 1 6}
    {pw_164_0 : ExactMatrix Q0 8 1 6}
    {pw_165_0 : ExactMatrix Q0 8 1 6}
    {pw_166_0 : ExactMatrix Q0 8 1 6}
    {pw_166_1 : ExactMatrix Q0 8 1 6}
    {pw_166_2 : ExactMatrix Q0 8 1 6}
    {pw_166_3 : ExactMatrix Q0 8 1 6}
    {pw_167_0 : ExactMatrix Q1 8 1 6}
    {pw_338_0 : ExactMatrix Q1 8 1 6}
    (pe_44_0 : pw_44_0 = (0 : ExactMatrix Q0 8 1 6))
    (pe_45_0 : pw_45_0 = (0 : ExactMatrix Q5 8 1 6))
    (pe_50_0 : pw_50_0 = matrixMul pw_46_0 pw_49_0)
    (pe_55_0 : pw_55_0 = matrixMulScalarLeft pw_52_0 pw_54_0)
    (pe_56_0 : pw_56_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (pe_57_0 : pw_57_0 = matrixMulScalarRight pw_55_0 pw_56_0)
    (pe_63_0 : pw_63_0 = matrixMulScalarRight pw_57_0 pw_62_0)
    (pe_64_0 : pw_64_0 = matrixMulScalarLeft pw_52_0 pw_47_0)
    (pe_65_0 : pw_65_0 = matrixSub pw_63_0 pw_64_0)
    (pe_67_0 : pw_67_0 = matrixAdd pw_65_0 pw_66_0)
    (pe_70_0 : pw_70_0 = matrixMul pw_51_0 pw_69_0)
    (pe_71_0 : pw_71_0 = matrixAdd pw_50_0 pw_70_0)
    (pe_80_0 : pw_80_0 = matrixMul pw_76_0 pw_79_0)
    (pe_84_0 : pw_84_0 = matrixMul pw_46_0 pw_83_0)
    (pe_86_0 : pw_86_0 = matrixMulScalarLeft pw_85_0 pw_54_0)
    (pe_87_0 : pw_87_0 = matrixMulScalarRight pw_86_0 pw_56_0)
    (pe_88_0 : pw_88_0 = matrixMulScalarRight pw_87_0 pw_62_0)
    (pe_89_0 : pw_89_0 = matrixMulScalarLeft pw_52_0 pw_81_0)
    (pe_90_0 : pw_90_0 = matrixSub pw_88_0 pw_89_0)
    (pe_92_0 : pw_92_0 = matrixAdd pw_90_0 pw_91_0)
    (pe_95_0 : pw_95_0 = matrixMul pw_51_0 pw_94_0)
    (pe_96_0 : pw_96_0 = matrixAdd pw_84_0 pw_95_0)
    (pe_105_0 : pw_105_0 = matrixMulScalarLeft pw_102_0 pw_104_0)
    (pe_106_0 : pw_106_0 = (matrixPolynomial [(p4)] : ExactMatrix Q5 8 1 1))
    (pe_107_0 : pw_107_0 = matrixMulScalarRight pw_105_0 pw_106_0)
    (pe_113_0 : pw_113_0 = matrixMulScalarRight pw_107_0 pw_112_0)
    (pe_114_0 : pw_114_0 = matrixMulScalarLeft pw_102_0 pw_77_0)
    (pe_115_0 : pw_115_0 = matrixSub pw_113_0 pw_114_0)
    (pe_117_0 : pw_117_0 = matrixAdd pw_115_0 pw_116_0)
    (pe_120_0 : pw_120_0 = matrixMul pw_101_0 pw_119_0)
    (pe_121_0 : pw_121_0 = matrixAdd pw_80_0 pw_120_0)
    (pe_129_0 : pw_129_0 = matrixMul pw_76_0 pw_128_0)
    (pe_131_0 : pw_131_0 = matrixMulScalarLeft pw_130_0 pw_104_0)
    (pe_132_0 : pw_132_0 = matrixMulScalarRight pw_131_0 pw_106_0)
    (pe_133_0 : pw_133_0 = matrixMulScalarRight pw_132_0 pw_112_0)
    (pe_134_0 : pw_134_0 = matrixMulScalarLeft pw_102_0 pw_126_0)
    (pe_135_0 : pw_135_0 = matrixSub pw_133_0 pw_134_0)
    (pe_137_0 : pw_137_0 = matrixAdd pw_135_0 pw_136_0)
    (pe_140_0 : pw_140_0 = matrixMul pw_101_0 pw_139_0)
    (pe_141_0 : pw_141_0 = matrixAdd pw_129_0 pw_140_0)
    (pe_156_0 : pw_156_0 = matrixAdd pw_154_0 pw_155_0)
    (pe_159_0 : pw_159_0 = matrixAdd pw_157_0 pw_158_0)
    (pe_162_0 : pw_162_0 = matrixAdd pw_160_0 pw_161_0)
    (pe_165_0 : pw_165_0 = matrixAdd pw_163_0 pw_164_0)
    (ph35_7 : (∀ i : Fin 40, Stage_preprocess_0_1.parallel_generatedRoot_35 Backend.backend model params' i (pw_26_0, (pw_27_0, (pw_28_0, (pw_29_0, (pw_30_0, (pw_31_0, ((pw_32_0 i), ((pw_33_0 i), (pw_34_0, ()))))))))) ((pw_35_0 i), ((pw_35_1 i), ((pw_35_2 i), ((pw_35_3 i), ((pw_35_4 i), ((pw_35_5 i), ((pw_35_6 i), ((pw_35_7 i), ()))))))))))
    (ph36_3 : (∀ i : Fin 8, Stage_preprocess_0_1.parallel_generatedRoot_36 Backend.backend model params' i (pw_35_0, (pw_35_1, (pw_35_2, (pw_35_3, ())))) ((pw_36_0 i), ((pw_36_1 i), ((pw_36_2 i), ((pw_36_3 i), ()))))))
    (ph39_7 : (∀ i : Fin 40, Stage_preprocess_0_1.parallel_generatedRoot_39 Backend.backend model params' i (pw_26_0, (pw_36_0, (pw_36_2, (pw_36_1, (pw_36_3, (pw_31_0, ((pw_37_0 i), ((pw_38_0 i), (pw_34_0, ()))))))))) ((pw_39_0 i), ((pw_39_1 i), ((pw_39_2 i), ((pw_39_3 i), ((pw_39_4 i), ((pw_39_5 i), ((pw_39_6 i), ((pw_39_7 i), ()))))))))))
    (ph40_3 : (∀ i : Fin 8, Stage_preprocess_0_1.parallel_generatedRoot_40 Backend.backend model params' i (pw_39_0, (pw_39_1, (pw_39_2, (pw_39_3, ())))) ((pw_40_0 i), ((pw_40_1 i), ((pw_40_2 i), ((pw_40_3 i), ()))))))
    (ph42_3 : (∀ i : Fin 8, Stage_preprocess_0_1.parallel_generatedRoot_42 Backend.backend model params' i (pw_26_0, ((pw_40_0 i), ((pw_40_2 i), (pw_41_0, ((pw_40_1 i), ((pw_40_3 i), (pw_41_2, (pw_41_1, (pw_41_3, ()))))))))) ((pw_42_0 i), ((pw_42_1 i), ((pw_42_2 i), ((pw_42_3 i), ()))))))
    (ph146_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × ExactMatrix Q5 8 1 6 × Unit) => Stage_preprocess_0_1.sequential_generatedRoot_146 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_26_0, (pw_123_0, (pw_125_0, (pw_41_0, (pw_143_0, (pw_145_0, (pw_41_2, (pw_41_1, (pw_41_3, ()))))))))))))) next) (Int.toNat (4)) (pw_45_0, (pw_45_0, (pw_45_0, (pw_45_0, ())))) (pw_146_0, (pw_146_1, (pw_146_2, (pw_146_3, ())))))
    (ph147_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) pw_146_0 pw_147_0)
    (ph148_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) pw_146_2 pw_148_0)
    (ph149_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) pw_146_1 pw_149_0)
    (ph150_0 : blockModSwitchRuns [q0, q1, q2, q3, p4] (1) pw_146_3 pw_150_0)
    (ph151_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_1.sequential_generatedRoot_151 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_26_0, (pw_147_0, (pw_148_0, (pw_43_0, (pw_149_0, (pw_150_0, (pw_43_2, (pw_43_1, (pw_43_3, ()))))))))))))) next) (Int.toNat (4)) (pw_44_0, (pw_44_0, (pw_44_0, (pw_44_0, ())))) (pw_151_0, (pw_151_1, (pw_151_2, (pw_151_3, ())))))
    (ph152_3 : (∀ i : Fin 8, Stage_preprocess_0_1.parallel_generatedRoot_152 Backend.backend model params' i (pw_26_0, ((pw_42_0 i), ((pw_42_2 i), (pw_43_0, ((pw_42_1 i), ((pw_42_3 i), (pw_43_2, (pw_43_1, (pw_43_3, (pw_151_0, (pw_151_1, (pw_151_2, (pw_151_3, ()))))))))))))) ((pw_152_0 i), ((pw_152_1 i), ((pw_152_2 i), ((pw_152_3 i), ()))))))
    (ph153_3 : (∀ i : Fin 8, Stage_preprocess_0_1.parallel_generatedRoot_153 Backend.backend model params' i ((pw_152_0 i), ((pw_152_1 i), ((pw_152_2 i), ((pw_152_3 i), ())))) ((pw_153_0 i), ((pw_153_1 i), ((pw_153_2 i), ((pw_153_3 i), ()))))))
    (ph154_0 : familyGetStatic pw_153_0 (0) pw_154_0)
    (ph155_0 : familyGetStatic pw_153_0 (1) pw_155_0)
    (ph157_0 : familyGetStatic pw_153_1 (0) pw_157_0)
    (ph158_0 : familyGetStatic pw_153_1 (1) pw_158_0)
    (ph160_0 : familyGetStatic pw_153_2 (0) pw_160_0)
    (ph161_0 : familyGetStatic pw_153_2 (1) pw_161_0)
    (ph163_0 : familyGetStatic pw_153_3 (0) pw_163_0)
    (ph164_0 : familyGetStatic pw_153_3 (1) pw_164_0)
    (ph166_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_1.sequential_generatedRoot_166 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_153_0, (pw_153_1, (pw_153_2, (pw_153_3, ())))))))) next) (Int.toNat (6)) (pw_156_0, (pw_159_0, (pw_162_0, (pw_165_0, ())))) (pw_166_0, (pw_166_1, (pw_166_2, (pw_166_3, ())))))
    (ph167_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) pw_166_1 pw_167_0)
    (ph338_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) pw_166_3 pw_338_0)
    (ph46_0 : concatRows pw_27_0 pw_28_0 pw_46_0)
    (ph48_0 : gadgetDecomposeRuns Backend.backend 32768 12 pw_47_0 pw_48_0)
    (ph49_0 : compactCenteredRebaseRuns 16384 pw_48_0 pw_49_0)
    (ph51_0 : concatRows pw_29_0 pw_30_0 pw_51_0)
    (ph68_0 : gadgetDecomposeRuns Backend.backend 32768 12 pw_67_0 pw_68_0)
    (ph69_0 : compactCenteredRebaseRuns 16384 pw_68_0 pw_69_0)
    (ph72_0 : sliceMatrix pw_71_0 0 1 0 10 pw_72_0)
    (ph73_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) pw_72_0 pw_73_0)
    (ph74_0 : sliceMatrix pw_71_0 1 2 0 10 pw_74_0)
    (ph75_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) pw_74_0 pw_75_0)
    (ph76_0 : concatRows pw_73_0 pw_75_0 pw_76_0)
    (ph78_0 : gadgetDecomposeRuns Backend.backend 32768 10 pw_77_0 pw_78_0)
    (ph79_0 : compactCenteredRebaseRuns 16384 pw_78_0 pw_79_0)
    (ph82_0 : gadgetDecomposeRuns Backend.backend 32768 12 pw_81_0 pw_82_0)
    (ph83_0 : compactCenteredRebaseRuns 16384 pw_82_0 pw_83_0)
    (ph93_0 : gadgetDecomposeRuns Backend.backend 32768 12 pw_92_0 pw_93_0)
    (ph94_0 : compactCenteredRebaseRuns 16384 pw_93_0 pw_94_0)
    (ph97_0 : sliceMatrix pw_96_0 0 1 0 10 pw_97_0)
    (ph98_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) pw_97_0 pw_98_0)
    (ph99_0 : sliceMatrix pw_96_0 1 2 0 10 pw_99_0)
    (ph100_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5, p6] (1) pw_99_0 pw_100_0)
    (ph101_0 : concatRows pw_98_0 pw_100_0 pw_101_0)
    (ph118_0 : gadgetDecomposeRuns Backend.backend 32768 10 pw_117_0 pw_118_0)
    (ph119_0 : compactCenteredRebaseRuns 16384 pw_118_0 pw_119_0)
    (ph122_0 : sliceMatrix pw_121_0 0 1 0 8 pw_122_0)
    (ph123_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) pw_122_0 pw_123_0)
    (ph124_0 : sliceMatrix pw_121_0 1 2 0 8 pw_124_0)
    (ph125_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) pw_124_0 pw_125_0)
    (ph127_0 : gadgetDecomposeRuns Backend.backend 32768 10 pw_126_0 pw_127_0)
    (ph128_0 : compactCenteredRebaseRuns 16384 pw_127_0 pw_128_0)
    (ph138_0 : gadgetDecomposeRuns Backend.backend 32768 10 pw_137_0 pw_138_0)
    (ph139_0 : compactCenteredRebaseRuns 16384 pw_138_0 pw_139_0)
    (ph142_0 : sliceMatrix pw_141_0 0 1 0 8 pw_142_0)
    (ph143_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) pw_142_0 pw_143_0)
    (ph144_0 : sliceMatrix pw_141_0 1 2 0 8 pw_144_0)
    (ph145_0 : blockModSwitchRuns [q0, q1, q2, q3, p4, p5] (1) pw_144_0 pw_145_0)
    (hk : w_92_0 = pw_26_0)
    (ho0 : w_106_0 = pw_27_0) (ho1 : w_107_0 = pw_28_0) (ho2 : w_110_0 = pw_29_0) (ho3 : w_111_0 = pw_30_0)
    (ht0 : w_135_0 = pw_47_0) (ht1 : w_148_0 = pw_81_0) (ht2 : w_162_0 = pw_77_0) (ht3 : w_174_0 = pw_126_0)
    (hb0 : w_136_0 = pw_67_0) (hb1 : w_149_0 = pw_92_0) (hb2 : w_163_0 = pw_117_0) (hb3 : w_175_0 = pw_137_0)
    (hf0 : w_94_0 = pw_41_0) (hf1 : w_95_0 = pw_41_1) (hf2 : w_96_0 = pw_41_2) (hf3 : w_97_0 = pw_41_3)
    (hg0 : w_87_0 = pw_43_0) (hg1 : w_88_0 = pw_43_1) (hg2 : w_89_0 = pw_43_2) (hg3 : w_90_0 = pw_43_3)
    (hc0 : w_113_0 = pw_35_6) (hc1 : w_114_0 = pw_35_7) (hc2 : w_117_0 = pw_39_6) (hc3 : w_118_0 = pw_39_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_101_0, w_104_0, w_102_0, w_105_0] c k =
      model.sample Q7 N 1 14 w_92_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    blockModSwitch [q0, q1, q2, q3] 1 w_332_1 = pw_167_0 ∧
      blockModSwitch [q0, q1, q2, q3] 1 w_332_3 = pw_338_0 := by
  have hon := keyRootOffset (model := model) (e_140_0 := e_140_0) (e_153_0 := e_153_0) (e_167_0 := e_167_0) (e_179_0 := e_179_0) (h106_0 := h106_0) (h132_0 := h132_0) (h110_0 := h110_0) (h133_0 := h133_0) (h134_0 := h134_0) (h137_0 := h137_0) (h138_0 := h138_0) (h139_0 := h139_0) (h143_0 := h143_0) (h144_0 := h144_0) (h145_0 := h145_0) (h146_0 := h146_0) (h144_0_1 := h144_0_1) (h147_0 := h147_0) (h150_0 := h150_0) (h151_0 := h151_0) (h152_0 := h152_0) (h156_0 := h156_0) (h157_0 := h157_0) (h158_0 := h158_0) (h159_0 := h159_0) (h157_0_1 := h157_0_1) (h160_0 := h160_0) (h161_0 := h161_0) (h164_0 := h164_0) (h165_0 := h165_0) (h166_0 := h166_0) (h170_0 := h170_0) (h171_0 := h171_0) (h172_0 := h172_0) (h173_0 := h173_0) (h176_0 := h176_0) (h177_0 := h177_0) (h178_0 := h178_0) (h182_0 := h182_0) (h183_0 := h183_0) (h184_0 := h184_0) (h185_0 := h185_0)
  have hpo := preKeyOffset (model := model) (e_50_0 := pe_50_0) (e_55_0 := pe_55_0) (e_56_0 := pe_56_0) (e_57_0 := pe_57_0) (e_63_0 := pe_63_0) (e_64_0 := pe_64_0) (e_65_0 := pe_65_0) (e_67_0 := pe_67_0) (e_70_0 := pe_70_0) (e_71_0 := pe_71_0) (e_80_0 := pe_80_0) (e_84_0 := pe_84_0) (e_86_0 := pe_86_0) (e_87_0 := pe_87_0) (e_88_0 := pe_88_0) (e_89_0 := pe_89_0) (e_90_0 := pe_90_0) (e_92_0 := pe_92_0) (e_95_0 := pe_95_0) (e_96_0 := pe_96_0) (e_105_0 := pe_105_0) (e_106_0 := pe_106_0) (e_107_0 := pe_107_0) (e_113_0 := pe_113_0) (e_114_0 := pe_114_0) (e_115_0 := pe_115_0) (e_117_0 := pe_117_0) (e_120_0 := pe_120_0) (e_121_0 := pe_121_0) (e_129_0 := pe_129_0) (e_131_0 := pe_131_0) (e_132_0 := pe_132_0) (e_133_0 := pe_133_0) (e_134_0 := pe_134_0) (e_135_0 := pe_135_0) (e_137_0 := pe_137_0) (e_140_0 := pe_140_0) (e_141_0 := pe_141_0) (h46_0 := ph46_0) (h48_0 := ph48_0) (h49_0 := ph49_0) (h51_0 := ph51_0) (h68_0 := ph68_0) (h69_0 := ph69_0) (h72_0 := ph72_0) (h73_0 := ph73_0) (h74_0 := ph74_0) (h75_0 := ph75_0) (h76_0 := ph76_0) (h78_0 := ph78_0) (h79_0 := ph79_0) (h82_0 := ph82_0) (h83_0 := ph83_0) (h93_0 := ph93_0) (h94_0 := ph94_0) (h97_0 := ph97_0) (h98_0 := ph98_0) (h99_0 := ph99_0) (h100_0 := ph100_0) (h101_0 := ph101_0) (h118_0 := ph118_0) (h119_0 := ph119_0) (h122_0 := ph122_0) (h123_0 := ph123_0) (h124_0 := ph124_0) (h125_0 := ph125_0) (h127_0 := ph127_0) (h128_0 := ph128_0) (h138_0 := ph138_0) (h139_0 := ph139_0) (h142_0 := ph142_0) (h143_0 := ph143_0) (h144_0 := ph144_0) (h145_0 := ph145_0)
  subst hk ho0 ho1 ho2 ho3 ht0 ht1 ht2 ht3 hb0 hb1 hb2 hb3 hf0 hf1 hf2 hf3 hg0 hg1 hg2 hg3 hc0 hc1 hc2 hc3
  rw [← hpo] at hon
  -- The offset's readout keys.
  have h186 := keyRootLut186 (model := model) (e_127_0 := e_127_0) (e_188_0 := e_188_0) (e_209_0 := e_209_0) (e_217_0 := e_217_0) (e_220_0 := e_220_0) (e_221_0 := e_221_0) (e_224_0 := e_224_0) (e_227_0 := e_227_0) (e_232_0 := e_232_0) (e_240_0 := e_240_0) (e_243_0 := e_243_0) (e_244_0 := e_244_0) (e_247_0 := e_247_0) (e_250_0 := e_250_0) (h128_0 := h128_0) (h129_0 := h129_0) (h130_0 := h130_0) (h131_0 := h131_0) (h186_9 := h186_9) (h187_0 := h187_0) (h189_0 := h189_0) (h190_0 := h190_0) (h189_0_1 := h189_0_1) (h191_0 := h191_0) (h192_0 := h192_0) (h193_0 := h193_0) (h194_0 := h194_0) (h193_0_1 := h193_0_1) (h195_0 := h195_0) (h196_0 := h196_0) (h197_0 := h197_0) (h198_0 := h198_0) (h199_0 := h199_0) (h198_0_1 := h198_0_1) (h200_0 := h200_0) (h201_0 := h201_0) (h202_0 := h202_0) (h203_0 := h203_0) (h202_0_1 := h202_0_1) (h204_0 := h204_0) (h205_0 := h205_0) (h206_0 := h206_0) (h207_0 := h207_0) (h208_0 := h208_0) (h210_0 := h210_0) (h211_0 := h211_0) (h212_0 := h212_0) (h213_0 := h213_0) (h214_0 := h214_0) (h215_0 := h215_0) (h216_0 := h216_0) (h218_0 := h218_0) (h219_0 := h219_0) (h222_0 := h222_0) (h223_0 := h223_0) (h225_0 := h225_0) (h226_0 := h226_0) (h228_0 := h228_0) (h229_0 := h229_0) (h230_0 := h230_0) (h231_0 := h231_0) (h233_0 := h233_0) (h234_0 := h234_0) (h235_0 := h235_0) (h236_0 := h236_0) (h237_0 := h237_0) (h238_0 := h238_0) (h239_0 := h239_0) (h241_0 := h241_0) (h242_0 := h242_0) (h245_0 := h245_0) (h246_0 := h246_0) (h248_0 := h248_0) (h249_0 := h249_0) (h251_0 := h251_0)
  rw [hon] at h186
  have h146 := preKeyLoop_sequential_generatedRoot_146 (by decide) ph146_3
  subst pe_45_0
  rw [blockModSwitchRuns_eq ph147_0, blockModSwitchRuns_eq ph148_0, blockModSwitchRuns_eq ph149_0,
    blockModSwitchRuns_eq ph150_0] at ph151_3
  simp only [keyMap, Prod.mk.injEq] at h186
  obtain ⟨e225, e248, e228, e251, -⟩ := h186
  rw [show ((0 : ExactMatrix Q5 N 1 6), (0 : ExactMatrix Q5 N 1 6), (0 : ExactMatrix Q5 N 1 6),
    (0 : ExactMatrix Q5 N 1 6), ()) = (0 : Keys Q5 6) from rfl, zero_add,
    show Int.toNat 4 = 4 from rfl] at h146
  have f0 := congrArg (fun x : Keys Q5 6 ↦ x.1) h146
  have f1 := congrArg (fun x : Keys Q5 6 ↦ x.2.1) h146
  have f2 := congrArg (fun x : Keys Q5 6 ↦ x.2.2.1) h146
  have f3 := congrArg (fun x : Keys Q5 6 ↦ x.2.2.2.1) h146
  simp only at f0 f1 f2 f3
  rw [← f0] at e225
  rw [← f1] at e248
  rw [← f2] at e228
  rw [← f3] at e251
  -- The offset's branch keys.
  have h252 := keyRootLut252 (model := model) (e_122_0 := e_122_0) (e_254_0 := e_254_0) (e_275_0 := e_275_0) (e_283_0 := e_283_0) (e_285_0 := e_285_0) (e_289_0 := e_289_0) (e_297_0 := e_297_0) (e_299_0 := e_299_0) (e_301_0 := e_301_0) (e_303_0 := e_303_0) (h123_0 := h123_0) (h124_0 := h124_0) (h125_0 := h125_0) (h126_0 := h126_0) (h252_9 := h252_9) (h253_0 := h253_0) (h255_0 := h255_0) (h256_0 := h256_0) (h255_0_1 := h255_0_1) (h257_0 := h257_0) (h258_0 := h258_0) (h259_0 := h259_0) (h260_0 := h260_0) (h259_0_1 := h259_0_1) (h261_0 := h261_0) (h262_0 := h262_0) (h263_0 := h263_0) (h264_0 := h264_0) (h265_0 := h265_0) (h264_0_1 := h264_0_1) (h266_0 := h266_0) (h267_0 := h267_0) (h268_0 := h268_0) (h269_0 := h269_0) (h268_0_1 := h268_0_1) (h270_0 := h270_0) (h271_0 := h271_0) (h272_0 := h272_0) (h273_0 := h273_0) (h274_0 := h274_0) (h276_0 := h276_0) (h277_0 := h277_0) (h278_0 := h278_0) (h279_0 := h279_0) (h280_0 := h280_0) (h281_0 := h281_0) (h282_0 := h282_0) (h284_0 := h284_0) (h286_0 := h286_0) (h287_0 := h287_0) (h288_0 := h288_0) (h290_0 := h290_0) (h291_0 := h291_0) (h292_0 := h292_0) (h293_0 := h293_0) (h294_0 := h294_0) (h295_0 := h295_0) (h296_0 := h296_0) (h298_0 := h298_0) (h300_0 := h300_0) (h302_0 := h302_0) (h304_0 := h304_0) (h305_0 := h305_0) (h308_0 := h308_0) (h309_0 := h309_0)
  rw [e225, e248, e228, e251] at h252
  have h151 := preKeyLoop_sequential_generatedRoot_151 (by decide) ph151_3
  subst pe_44_0
  rw [show ((0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6), (0 : ExactMatrix Q0 N 1 6),
    (0 : ExactMatrix Q0 N 1 6), ()) = (0 : Keys Q0 6) from rfl, zero_add,
    show Int.toNat 4 = 4 from rfl] at h151
  have hoff : (w_285_0, w_299_0, w_301_0, w_303_0, ()) =
      (pw_151_0, pw_151_1, pw_151_2, pw_151_3, ()) := by
    rw [h252, h151]
  -- The cells and members of the PRF.
  have hc115 : ∀ i : Fin 40, w_115_0 i = pw_35_0 i ∧ w_115_1 i = pw_35_1 i ∧ w_115_2 i = pw_35_2 i ∧
      w_115_3 i = pw_35_3 i := fun i ↦
    keysCell115 (hon := h115_5 i) (hpre := ph35_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm116 : w_116_0 = pw_36_0 ∧ w_116_1 = pw_36_1 ∧ w_116_2 = pw_36_2 ∧ w_116_3 = pw_36_3 := by
    have a := fun o ↦ members116 (h116_5 o)
    have b := fun o ↦ premembers36 (ph36_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc115 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc115 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc115 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc115 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm116
  rw [m0, m1, m2, m3] at h119_5
  have hc119 : ∀ i : Fin 40, w_119_0 i = pw_39_0 i ∧ w_119_1 i = pw_39_1 i ∧ w_119_2 i = pw_39_2 i ∧
      w_119_3 i = pw_39_3 i := fun i ↦
    keysCell119 (hon := h119_5 i) (hpre := ph39_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm120 : w_120_0 = pw_40_0 ∧ w_120_1 = pw_40_1 ∧ w_120_2 = pw_40_2 ∧ w_120_3 = pw_40_3 := by
    have a := fun o ↦ members120 (h120_5 o)
    have b := fun o ↦ premembers40 (ph40_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc119 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc119 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc119 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc119 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm120
  rw [n0, n1, n2, n3] at h121_5
  -- The readout.
  have hr : ∀ o : Fin 8, w_121_0 o = pw_42_0 o ∧ w_121_1 o = pw_42_1 o ∧ w_121_2 o = pw_42_2 o ∧
      w_121_3 o = pw_42_3 o := by
    intro o
    have a := keyRound121 (h121_5 o)
    have b := preKeyRound42 (ph42_3 o)
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  -- The sampling with the offset.
  have hs : ∀ o : Fin 8, w_312_0 o = pw_152_0 o ∧ w_312_1 o = pw_152_1 o ∧ w_312_2 o = pw_152_2 o ∧
      w_312_3 o = pw_152_3 o := by
    intro o
    have a := keyRound312 (h := h312_5 o) (off := (w_285_0, w_299_0, w_301_0, w_303_0, w_307_0, w_311_0, ()))
    have b := preKeyRound152 (h := ph152_3 o) (off := (pw_151_0, pw_151_1, pw_151_2, pw_151_3, ()))
    simp only at a b
    obtain ⟨r0, r1, r2, r3⟩ := hr o
    rw [r0, r1, r2, r3, hoff, ← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  -- The placement.
  have hp : ∀ o : Fin 8, w_313_0 o = pw_153_0 o ∧ w_313_1 o = pw_153_1 o ∧ w_313_2 o = pw_153_2 o ∧
      w_313_3 o = pw_153_3 o := by
    intro o
    have a := place313 (h := h313_5 o) (x := (w_312_0 o, w_312_1 o, w_312_2 o, w_312_3 o, w_312_4 o, w_312_5 o, ()))
    have b := prePlace153 (h := ph153_3 o) (x := (pw_152_0 o, pw_152_1 o, pw_152_2 o, pw_152_3 o, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs o
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [a.1, b.1, s0]
    · rw [a.2.1, b.2.1, s1]
    · rw [a.2.2.1, b.2.2.1, s2]
    · rw [a.2.2.2.1, b.2.2.2.1, s3]
  -- The member sums and the move down.
  subst e_316_0 e_319_0 e_322_0 e_325_0 e_328_0 e_331_0 pe_156_0 pe_159_0 pe_162_0 pe_165_0
  have a := rootSum332 h314_0 h315_0 h317_0 h318_0 h320_0 h321_0 h323_0 h324_0 h326_0 h327_0
    h329_0 h330_0 h332_5
  have b := prerootSum166 ph154_0 ph155_0 ph157_0 ph158_0 ph160_0 ph161_0 ph163_0 ph164_0 ph166_3
  simp only [Prod.mk.injEq] at a b
  rw [blockModSwitchRuns_eq ph167_0, blockModSwitchRuns_eq ph338_0, a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨congrArg _ (Finset.sum_congr rfl fun o _ ↦ (hp o).2.1),
    congrArg _ (Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2)⟩

end RefreshTopProof.Part_0_1
