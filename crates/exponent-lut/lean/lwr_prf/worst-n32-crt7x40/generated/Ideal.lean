import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Ideal

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Int
  w_4_0 : Int
  w_7_0 : Int
  w_8_0 : Int
  w_11_0 : Int
  w_12_0 : Int
  w_15_0 : Int
  w_16_0 : Int
  w_19_0 : Int
  w_20_0 : Int
  w_23_0 : Int
  w_24_0 : Int
  w_27_0 : Int
  w_28_0 : Int
  w_31_0 : Int
  w_32_0 : Int
  w_35_0 : Int
  w_36_0 : Int
  w_39_0 : Int
  w_40_0 : Int
  w_43_0 : Int
  w_44_0 : Int
  w_47_0 : Int
  w_48_0 : Int
  w_51_0 : Int
  w_52_0 : Int
  w_55_0 : Int
  w_56_0 : Int
  w_60_0 : Int
  w_61_0 : Int
  w_64_0 : Int
  w_65_0 : Int
  w_68_0 : Int
  w_69_0 : Int
  w_72_0 : Int
  w_73_0 : Int
  w_76_0 : Int
  w_77_0 : Int
  w_80_0 : Int
  w_81_0 : Int
  w_84_0 : Int
  w_85_0 : Int
  w_88_0 : Int
  w_89_0 : Int
  w_92_0 : Int
  w_93_0 : Int
  w_96_0 : Int
  w_97_0 : Int
  w_100_0 : Int
  w_101_0 : Int
  w_104_0 : Int
  w_105_0 : Int
  w_108_0 : Int
  w_109_0 : Int
  w_112_0 : Int
  w_113_0 : Int
  w_117_0 : Int
  w_118_0 : Int
  w_121_0 : Int
  w_122_0 : Int
  w_125_0 : Int
  w_126_0 : Int
  w_129_0 : Int
  w_130_0 : Int
  w_133_0 : Int
  w_134_0 : Int
  w_137_0 : Int
  w_138_0 : Int
  w_141_0 : Int
  w_142_0 : Int
  w_145_0 : Int
  w_146_0 : Int
  w_149_0 : Int
  w_150_0 : Int
  w_153_0 : Int
  w_154_0 : Int
  w_157_0 : Int
  w_158_0 : Int
  w_161_0 : Int
  w_162_0 : Int
  w_165_0 : Int
  w_166_0 : Int
  w_169_0 : Int
  w_170_0 : Int
  w_174_0 : Int
  w_175_0 : Int
  w_178_0 : Int
  w_179_0 : Int
  w_182_0 : Int
  w_183_0 : Int
  w_186_0 : Int
  w_187_0 : Int
  w_190_0 : Int
  w_191_0 : Int
  w_194_0 : Int
  w_195_0 : Int
  w_198_0 : Int
  w_199_0 : Int
  w_202_0 : Int
  w_203_0 : Int
  w_206_0 : Int
  w_207_0 : Int
  w_210_0 : Int
  w_211_0 : Int
  w_214_0 : Int
  w_215_0 : Int
  w_218_0 : Int
  w_219_0 : Int
  w_222_0 : Int
  w_223_0 : Int
  w_226_0 : Int
  w_227_0 : Int
  w_231_0 : Int
  w_232_0 : Int
  w_235_0 : Int
  w_236_0 : Int
  w_239_0 : Int
  w_240_0 : Int
  w_243_0 : Int
  w_244_0 : Int
  w_247_0 : Int
  w_248_0 : Int
  w_251_0 : Int
  w_252_0 : Int
  w_255_0 : Int
  w_256_0 : Int
  w_259_0 : Int
  w_260_0 : Int
  w_263_0 : Int
  w_264_0 : Int
  w_267_0 : Int
  w_268_0 : Int
  w_271_0 : Int
  w_272_0 : Int
  w_275_0 : Int
  w_276_0 : Int
  w_279_0 : Int
  w_280_0 : Int
  w_283_0 : Int
  w_284_0 : Int

abbrev generatedRoot.constraints_4 (w_3_0 : Fin 17 → Int) (w_230_0 : Fin 14 → Int) (w_263_0 : Int) (w_264_0 : Int) (w_267_0 : Int) (w_268_0 : Int) (w_271_0 : Int) (w_272_0 : Int) (w_275_0 : Int) (w_276_0 : Int) (w_279_0 : Int) (w_280_0 : Int) (w_283_0 : Int) (w_284_0 : Int) (w_287_0 : Int) (w_291_0 : Int) (w_295_0 : Int) (w_296_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.familyGetStatic w_230_0 (8) w_263_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_264_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (9) w_267_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_268_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (10) w_271_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_272_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (11) w_275_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_276_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (12) w_279_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_280_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (13) w_283_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_284_0 ∧
  w_287_0 ≠ 0 ∧
  w_291_0 ≠ 0 ∧
  w_295_0 ≠ 0 ∧
  outputs = w_296_0

abbrev generatedRoot.constraints_3 (w_3_0 : Fin 17 → Int) (w_173_0 : Fin 14 → Int) (w_230_0 : Fin 14 → Int) (w_198_0 : Int) (w_199_0 : Int) (w_202_0 : Int) (w_203_0 : Int) (w_206_0 : Int) (w_207_0 : Int) (w_210_0 : Int) (w_211_0 : Int) (w_214_0 : Int) (w_215_0 : Int) (w_218_0 : Int) (w_219_0 : Int) (w_222_0 : Int) (w_223_0 : Int) (w_226_0 : Int) (w_227_0 : Int) (w_231_0 : Int) (w_232_0 : Int) (w_235_0 : Int) (w_236_0 : Int) (w_239_0 : Int) (w_240_0 : Int) (w_243_0 : Int) (w_244_0 : Int) (w_247_0 : Int) (w_248_0 : Int) (w_251_0 : Int) (w_252_0 : Int) (w_255_0 : Int) (w_256_0 : Int) (w_259_0 : Int) (w_260_0 : Int) (w_263_0 : Int) (w_264_0 : Int) (w_267_0 : Int) (w_268_0 : Int) (w_271_0 : Int) (w_272_0 : Int) (w_275_0 : Int) (w_276_0 : Int) (w_279_0 : Int) (w_280_0 : Int) (w_283_0 : Int) (w_284_0 : Int) (w_287_0 : Int) (w_291_0 : Int) (w_295_0 : Int) (w_296_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.familyGetStatic w_173_0 (6) w_198_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (6) w_199_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (7) w_202_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (7) w_203_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (8) w_206_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (8) w_207_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (9) w_210_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (9) w_211_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (10) w_214_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (10) w_215_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (11) w_218_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (11) w_219_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (12) w_222_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (13) w_223_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (13) w_226_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_227_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (0) w_231_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (3) w_232_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (1) w_235_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (6) w_236_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (2) w_239_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (7) w_240_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (3) w_243_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (13) w_244_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (4) w_247_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (14) w_248_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (5) w_251_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_252_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (6) w_255_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_256_0 ∧
  MxxRuntime.familyGetStatic w_230_0 (7) w_259_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_260_0 ∧
  generatedRoot.constraints_4 w_3_0 w_230_0 w_263_0 w_264_0 w_267_0 w_268_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_283_0 w_284_0 w_287_0 w_291_0 w_295_0 w_296_0 outputs

abbrev generatedRoot.constraints_2 (w_3_0 : Fin 17 → Int) (w_116_0 : Fin 14 → Int) (w_173_0 : Fin 14 → Int) (w_230_0 : Fin 14 → Int) (w_133_0 : Int) (w_134_0 : Int) (w_137_0 : Int) (w_138_0 : Int) (w_141_0 : Int) (w_142_0 : Int) (w_145_0 : Int) (w_146_0 : Int) (w_149_0 : Int) (w_150_0 : Int) (w_153_0 : Int) (w_154_0 : Int) (w_157_0 : Int) (w_158_0 : Int) (w_161_0 : Int) (w_162_0 : Int) (w_165_0 : Int) (w_166_0 : Int) (w_169_0 : Int) (w_170_0 : Int) (w_174_0 : Int) (w_175_0 : Int) (w_178_0 : Int) (w_179_0 : Int) (w_182_0 : Int) (w_183_0 : Int) (w_186_0 : Int) (w_187_0 : Int) (w_190_0 : Int) (w_191_0 : Int) (w_194_0 : Int) (w_195_0 : Int) (w_198_0 : Int) (w_199_0 : Int) (w_202_0 : Int) (w_203_0 : Int) (w_206_0 : Int) (w_207_0 : Int) (w_210_0 : Int) (w_211_0 : Int) (w_214_0 : Int) (w_215_0 : Int) (w_218_0 : Int) (w_219_0 : Int) (w_222_0 : Int) (w_223_0 : Int) (w_226_0 : Int) (w_227_0 : Int) (w_231_0 : Int) (w_232_0 : Int) (w_235_0 : Int) (w_236_0 : Int) (w_239_0 : Int) (w_240_0 : Int) (w_243_0 : Int) (w_244_0 : Int) (w_247_0 : Int) (w_248_0 : Int) (w_251_0 : Int) (w_252_0 : Int) (w_255_0 : Int) (w_256_0 : Int) (w_259_0 : Int) (w_260_0 : Int) (w_263_0 : Int) (w_264_0 : Int) (w_267_0 : Int) (w_268_0 : Int) (w_271_0 : Int) (w_272_0 : Int) (w_275_0 : Int) (w_276_0 : Int) (w_279_0 : Int) (w_280_0 : Int) (w_283_0 : Int) (w_284_0 : Int) (w_287_0 : Int) (w_291_0 : Int) (w_295_0 : Int) (w_296_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.familyGetStatic w_116_0 (4) w_133_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (5) w_134_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (5) w_137_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (8) w_138_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (6) w_141_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (9) w_142_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (7) w_145_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (10) w_146_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (8) w_149_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (11) w_150_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (9) w_153_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (12) w_154_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (10) w_157_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (15) w_158_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (11) w_161_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_162_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (12) w_165_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_166_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (13) w_169_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_170_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (0) w_174_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_175_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (1) w_178_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (1) w_179_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (2) w_182_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (2) w_183_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (3) w_186_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (3) w_187_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (4) w_190_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (4) w_191_0 ∧
  MxxRuntime.familyGetStatic w_173_0 (5) w_194_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (5) w_195_0 ∧
  generatedRoot.constraints_3 w_3_0 w_173_0 w_230_0 w_198_0 w_199_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_231_0 w_232_0 w_235_0 w_236_0 w_239_0 w_240_0 w_243_0 w_244_0 w_247_0 w_248_0 w_251_0 w_252_0 w_255_0 w_256_0 w_259_0 w_260_0 w_263_0 w_264_0 w_267_0 w_268_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_283_0 w_284_0 w_287_0 w_291_0 w_295_0 w_296_0 outputs

abbrev generatedRoot.constraints_1 (w_3_0 : Fin 17 → Int) (w_59_0 : Fin 14 → Int) (w_116_0 : Fin 14 → Int) (w_173_0 : Fin 14 → Int) (w_230_0 : Fin 14 → Int) (w_68_0 : Int) (w_69_0 : Int) (w_72_0 : Int) (w_73_0 : Int) (w_76_0 : Int) (w_77_0 : Int) (w_80_0 : Int) (w_81_0 : Int) (w_84_0 : Int) (w_85_0 : Int) (w_88_0 : Int) (w_89_0 : Int) (w_92_0 : Int) (w_93_0 : Int) (w_96_0 : Int) (w_97_0 : Int) (w_100_0 : Int) (w_101_0 : Int) (w_104_0 : Int) (w_105_0 : Int) (w_108_0 : Int) (w_109_0 : Int) (w_112_0 : Int) (w_113_0 : Int) (w_117_0 : Int) (w_118_0 : Int) (w_121_0 : Int) (w_122_0 : Int) (w_125_0 : Int) (w_126_0 : Int) (w_129_0 : Int) (w_130_0 : Int) (w_133_0 : Int) (w_134_0 : Int) (w_137_0 : Int) (w_138_0 : Int) (w_141_0 : Int) (w_142_0 : Int) (w_145_0 : Int) (w_146_0 : Int) (w_149_0 : Int) (w_150_0 : Int) (w_153_0 : Int) (w_154_0 : Int) (w_157_0 : Int) (w_158_0 : Int) (w_161_0 : Int) (w_162_0 : Int) (w_165_0 : Int) (w_166_0 : Int) (w_169_0 : Int) (w_170_0 : Int) (w_174_0 : Int) (w_175_0 : Int) (w_178_0 : Int) (w_179_0 : Int) (w_182_0 : Int) (w_183_0 : Int) (w_186_0 : Int) (w_187_0 : Int) (w_190_0 : Int) (w_191_0 : Int) (w_194_0 : Int) (w_195_0 : Int) (w_198_0 : Int) (w_199_0 : Int) (w_202_0 : Int) (w_203_0 : Int) (w_206_0 : Int) (w_207_0 : Int) (w_210_0 : Int) (w_211_0 : Int) (w_214_0 : Int) (w_215_0 : Int) (w_218_0 : Int) (w_219_0 : Int) (w_222_0 : Int) (w_223_0 : Int) (w_226_0 : Int) (w_227_0 : Int) (w_231_0 : Int) (w_232_0 : Int) (w_235_0 : Int) (w_236_0 : Int) (w_239_0 : Int) (w_240_0 : Int) (w_243_0 : Int) (w_244_0 : Int) (w_247_0 : Int) (w_248_0 : Int) (w_251_0 : Int) (w_252_0 : Int) (w_255_0 : Int) (w_256_0 : Int) (w_259_0 : Int) (w_260_0 : Int) (w_263_0 : Int) (w_264_0 : Int) (w_267_0 : Int) (w_268_0 : Int) (w_271_0 : Int) (w_272_0 : Int) (w_275_0 : Int) (w_276_0 : Int) (w_279_0 : Int) (w_280_0 : Int) (w_283_0 : Int) (w_284_0 : Int) (w_287_0 : Int) (w_291_0 : Int) (w_295_0 : Int) (w_296_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.familyGetStatic w_59_0 (2) w_68_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (2) w_69_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (3) w_72_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (4) w_73_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (4) w_76_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (6) w_77_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (5) w_80_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (7) w_81_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (6) w_84_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (8) w_85_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (7) w_88_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (9) w_89_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (8) w_92_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (12) w_93_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (9) w_96_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (13) w_97_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (10) w_100_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (14) w_101_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (11) w_104_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (15) w_105_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (12) w_108_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_109_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (13) w_112_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_113_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (0) w_117_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_118_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (1) w_121_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (1) w_122_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (2) w_125_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (2) w_126_0 ∧
  MxxRuntime.familyGetStatic w_116_0 (3) w_129_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (3) w_130_0 ∧
  generatedRoot.constraints_2 w_3_0 w_116_0 w_173_0 w_230_0 w_133_0 w_134_0 w_137_0 w_138_0 w_141_0 w_142_0 w_145_0 w_146_0 w_149_0 w_150_0 w_153_0 w_154_0 w_157_0 w_158_0 w_161_0 w_162_0 w_165_0 w_166_0 w_169_0 w_170_0 w_174_0 w_175_0 w_178_0 w_179_0 w_182_0 w_183_0 w_186_0 w_187_0 w_190_0 w_191_0 w_194_0 w_195_0 w_198_0 w_199_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_231_0 w_232_0 w_235_0 w_236_0 w_239_0 w_240_0 w_243_0 w_244_0 w_247_0 w_248_0 w_251_0 w_252_0 w_255_0 w_256_0 w_259_0 w_260_0 w_263_0 w_264_0 w_267_0 w_268_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_283_0 w_284_0 w_287_0 w_291_0 w_295_0 w_296_0 outputs

abbrev generatedRoot.constraints_0 (w_1_0 : Fin 14 → Int) (w_3_0 : Fin 17 → Int) (w_59_0 : Fin 14 → Int) (w_116_0 : Fin 14 → Int) (w_173_0 : Fin 14 → Int) (w_230_0 : Fin 14 → Int) (w_2_0 : Int) (w_4_0 : Int) (w_7_0 : Int) (w_8_0 : Int) (w_11_0 : Int) (w_12_0 : Int) (w_15_0 : Int) (w_16_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_24_0 : Int) (w_27_0 : Int) (w_28_0 : Int) (w_31_0 : Int) (w_32_0 : Int) (w_35_0 : Int) (w_36_0 : Int) (w_39_0 : Int) (w_40_0 : Int) (w_43_0 : Int) (w_44_0 : Int) (w_47_0 : Int) (w_48_0 : Int) (w_51_0 : Int) (w_52_0 : Int) (w_55_0 : Int) (w_56_0 : Int) (w_60_0 : Int) (w_61_0 : Int) (w_64_0 : Int) (w_65_0 : Int) (w_68_0 : Int) (w_69_0 : Int) (w_72_0 : Int) (w_73_0 : Int) (w_76_0 : Int) (w_77_0 : Int) (w_80_0 : Int) (w_81_0 : Int) (w_84_0 : Int) (w_85_0 : Int) (w_88_0 : Int) (w_89_0 : Int) (w_92_0 : Int) (w_93_0 : Int) (w_96_0 : Int) (w_97_0 : Int) (w_100_0 : Int) (w_101_0 : Int) (w_104_0 : Int) (w_105_0 : Int) (w_108_0 : Int) (w_109_0 : Int) (w_112_0 : Int) (w_113_0 : Int) (w_117_0 : Int) (w_118_0 : Int) (w_121_0 : Int) (w_122_0 : Int) (w_125_0 : Int) (w_126_0 : Int) (w_129_0 : Int) (w_130_0 : Int) (w_133_0 : Int) (w_134_0 : Int) (w_137_0 : Int) (w_138_0 : Int) (w_141_0 : Int) (w_142_0 : Int) (w_145_0 : Int) (w_146_0 : Int) (w_149_0 : Int) (w_150_0 : Int) (w_153_0 : Int) (w_154_0 : Int) (w_157_0 : Int) (w_158_0 : Int) (w_161_0 : Int) (w_162_0 : Int) (w_165_0 : Int) (w_166_0 : Int) (w_169_0 : Int) (w_170_0 : Int) (w_174_0 : Int) (w_175_0 : Int) (w_178_0 : Int) (w_179_0 : Int) (w_182_0 : Int) (w_183_0 : Int) (w_186_0 : Int) (w_187_0 : Int) (w_190_0 : Int) (w_191_0 : Int) (w_194_0 : Int) (w_195_0 : Int) (w_198_0 : Int) (w_199_0 : Int) (w_202_0 : Int) (w_203_0 : Int) (w_206_0 : Int) (w_207_0 : Int) (w_210_0 : Int) (w_211_0 : Int) (w_214_0 : Int) (w_215_0 : Int) (w_218_0 : Int) (w_219_0 : Int) (w_222_0 : Int) (w_223_0 : Int) (w_226_0 : Int) (w_227_0 : Int) (w_231_0 : Int) (w_232_0 : Int) (w_235_0 : Int) (w_236_0 : Int) (w_239_0 : Int) (w_240_0 : Int) (w_243_0 : Int) (w_244_0 : Int) (w_247_0 : Int) (w_248_0 : Int) (w_251_0 : Int) (w_252_0 : Int) (w_255_0 : Int) (w_256_0 : Int) (w_259_0 : Int) (w_260_0 : Int) (w_263_0 : Int) (w_264_0 : Int) (w_267_0 : Int) (w_268_0 : Int) (w_271_0 : Int) (w_272_0 : Int) (w_275_0 : Int) (w_276_0 : Int) (w_279_0 : Int) (w_280_0 : Int) (w_283_0 : Int) (w_284_0 : Int) (w_287_0 : Int) (w_291_0 : Int) (w_295_0 : Int) (w_296_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.familyGetStatic w_1_0 (0) w_2_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (4) w_4_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (1) w_7_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (5) w_8_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (2) w_11_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (10) w_12_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (3) w_15_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (11) w_16_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (4) w_19_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (12) w_20_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (5) w_23_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (14) w_24_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (6) w_27_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (15) w_28_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (7) w_31_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_32_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (8) w_35_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_36_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (9) w_39_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_40_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (10) w_43_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_44_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (11) w_47_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_48_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (12) w_51_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_52_0 ∧
  MxxRuntime.familyGetStatic w_1_0 (13) w_55_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (16) w_56_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (0) w_60_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (0) w_61_0 ∧
  MxxRuntime.familyGetStatic w_59_0 (1) w_64_0 ∧
  MxxRuntime.familyGetStatic w_3_0 (1) w_65_0 ∧
  generatedRoot.constraints_1 w_3_0 w_59_0 w_116_0 w_173_0 w_230_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_117_0 w_118_0 w_121_0 w_122_0 w_125_0 w_126_0 w_129_0 w_130_0 w_133_0 w_134_0 w_137_0 w_138_0 w_141_0 w_142_0 w_145_0 w_146_0 w_149_0 w_150_0 w_153_0 w_154_0 w_157_0 w_158_0 w_161_0 w_162_0 w_165_0 w_166_0 w_169_0 w_170_0 w_174_0 w_175_0 w_178_0 w_179_0 w_182_0 w_183_0 w_186_0 w_187_0 w_190_0 w_191_0 w_194_0 w_195_0 w_198_0 w_199_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_231_0 w_232_0 w_235_0 w_236_0 w_239_0 w_240_0 w_243_0 w_244_0 w_247_0 w_248_0 w_251_0 w_252_0 w_255_0 w_256_0 w_259_0 w_260_0 w_263_0 w_264_0 w_267_0 w_268_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_283_0 w_284_0 w_287_0 w_291_0 w_295_0 w_296_0 outputs

abbrev generatedRoot.body (params : Params) (inputs : (Fin 14 → Int) × (Fin 17 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let w_1_0 : Fin 14 → Int := inputs.1
  let w_3_0 : Fin 17 → Int := inputs.2.1
  let w_59_0 : Fin 14 → Int := inputs.2.2.1
  let w_116_0 : Fin 14 → Int := inputs.2.2.2.1
  let w_173_0 : Fin 14 → Int := inputs.2.2.2.2.1
  let w_230_0 : Fin 14 → Int := inputs.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_2_0 := witness.w_2_0
  let w_4_0 := witness.w_4_0
  let w_5_0 : Int := (w_2_0 * w_4_0)
  let w_6_0 : Int := (w_0_0 + w_5_0)
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : Int := (w_7_0 * w_8_0)
  let w_10_0 : Int := (w_6_0 + w_9_0)
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : Int := (w_11_0 * w_12_0)
  let w_14_0 : Int := (w_10_0 + w_13_0)
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Int := (w_15_0 * w_16_0)
  let w_18_0 : Int := (w_14_0 + w_17_0)
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : Int := (w_19_0 * w_20_0)
  let w_22_0 : Int := (w_18_0 + w_21_0)
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_25_0 : Int := (w_23_0 * w_24_0)
  let w_26_0 : Int := (w_22_0 + w_25_0)
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : Int := (w_27_0 * w_28_0)
  let w_30_0 : Int := (w_26_0 + w_29_0)
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : Int := (w_31_0 * w_32_0)
  let w_34_0 : Int := (w_30_0 + w_33_0)
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Int := (w_35_0 * w_36_0)
  let w_38_0 : Int := (w_34_0 + w_37_0)
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : Int := (w_39_0 * w_40_0)
  let w_42_0 : Int := (w_38_0 + w_41_0)
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : Int := (w_43_0 * w_44_0)
  let w_46_0 : Int := (w_42_0 + w_45_0)
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : Int := (w_47_0 * w_48_0)
  let w_50_0 : Int := (w_46_0 + w_49_0)
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 : Int := (w_51_0 * w_52_0)
  let w_54_0 : Int := (w_50_0 + w_53_0)
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : Int := (w_55_0 * w_56_0)
  let w_58_0 : Int := (w_54_0 + w_57_0)
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : Int := (w_60_0 * w_61_0)
  let w_63_0 : Int := (w_58_0 + w_62_0)
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : Int := (w_64_0 * w_65_0)
  let w_67_0 : Int := (w_63_0 + w_66_0)
  let w_68_0 := witness.w_68_0
  let w_69_0 := witness.w_69_0
  let w_70_0 : Int := (w_68_0 * w_69_0)
  let w_71_0 : Int := (w_67_0 + w_70_0)
  let w_72_0 := witness.w_72_0
  let w_73_0 := witness.w_73_0
  let w_74_0 : Int := (w_72_0 * w_73_0)
  let w_75_0 : Int := (w_71_0 + w_74_0)
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : Int := (w_76_0 * w_77_0)
  let w_79_0 : Int := (w_75_0 + w_78_0)
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : Int := (w_80_0 * w_81_0)
  let w_83_0 : Int := (w_79_0 + w_82_0)
  let w_84_0 := witness.w_84_0
  let w_85_0 := witness.w_85_0
  let w_86_0 : Int := (w_84_0 * w_85_0)
  let w_87_0 : Int := (w_83_0 + w_86_0)
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : Int := (w_88_0 * w_89_0)
  let w_91_0 : Int := (w_87_0 + w_90_0)
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : Int := (w_92_0 * w_93_0)
  let w_95_0 : Int := (w_91_0 + w_94_0)
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_98_0 : Int := (w_96_0 * w_97_0)
  let w_99_0 : Int := (w_95_0 + w_98_0)
  let w_100_0 := witness.w_100_0
  let w_101_0 := witness.w_101_0
  let w_102_0 : Int := (w_100_0 * w_101_0)
  let w_103_0 : Int := (w_99_0 + w_102_0)
  let w_104_0 := witness.w_104_0
  let w_105_0 := witness.w_105_0
  let w_106_0 : Int := (w_104_0 * w_105_0)
  let w_107_0 : Int := (w_103_0 + w_106_0)
  let w_108_0 := witness.w_108_0
  let w_109_0 := witness.w_109_0
  let w_110_0 : Int := (w_108_0 * w_109_0)
  let w_111_0 : Int := (w_107_0 + w_110_0)
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_114_0 : Int := (w_112_0 * w_113_0)
  let w_115_0 : Int := (w_111_0 + w_114_0)
  let w_117_0 := witness.w_117_0
  let w_118_0 := witness.w_118_0
  let w_119_0 : Int := (w_117_0 * w_118_0)
  let w_120_0 : Int := (w_115_0 + w_119_0)
  let w_121_0 := witness.w_121_0
  let w_122_0 := witness.w_122_0
  let w_123_0 : Int := (w_121_0 * w_122_0)
  let w_124_0 : Int := (w_120_0 + w_123_0)
  let w_125_0 := witness.w_125_0
  let w_126_0 := witness.w_126_0
  let w_127_0 : Int := (w_125_0 * w_126_0)
  let w_128_0 : Int := (w_124_0 + w_127_0)
  let w_129_0 := witness.w_129_0
  let w_130_0 := witness.w_130_0
  let w_131_0 : Int := (w_129_0 * w_130_0)
  let w_132_0 : Int := (w_128_0 + w_131_0)
  let w_133_0 := witness.w_133_0
  let w_134_0 := witness.w_134_0
  let w_135_0 : Int := (w_133_0 * w_134_0)
  let w_136_0 : Int := (w_132_0 + w_135_0)
  let w_137_0 := witness.w_137_0
  let w_138_0 := witness.w_138_0
  let w_139_0 : Int := (w_137_0 * w_138_0)
  let w_140_0 : Int := (w_136_0 + w_139_0)
  let w_141_0 := witness.w_141_0
  let w_142_0 := witness.w_142_0
  let w_143_0 : Int := (w_141_0 * w_142_0)
  let w_144_0 : Int := (w_140_0 + w_143_0)
  let w_145_0 := witness.w_145_0
  let w_146_0 := witness.w_146_0
  let w_147_0 : Int := (w_145_0 * w_146_0)
  let w_148_0 : Int := (w_144_0 + w_147_0)
  let w_149_0 := witness.w_149_0
  let w_150_0 := witness.w_150_0
  let w_151_0 : Int := (w_149_0 * w_150_0)
  let w_152_0 : Int := (w_148_0 + w_151_0)
  let w_153_0 := witness.w_153_0
  let w_154_0 := witness.w_154_0
  let w_155_0 : Int := (w_153_0 * w_154_0)
  let w_156_0 : Int := (w_152_0 + w_155_0)
  let w_157_0 := witness.w_157_0
  let w_158_0 := witness.w_158_0
  let w_159_0 : Int := (w_157_0 * w_158_0)
  let w_160_0 : Int := (w_156_0 + w_159_0)
  let w_161_0 := witness.w_161_0
  let w_162_0 := witness.w_162_0
  let w_163_0 : Int := (w_161_0 * w_162_0)
  let w_164_0 : Int := (w_160_0 + w_163_0)
  let w_165_0 := witness.w_165_0
  let w_166_0 := witness.w_166_0
  let w_167_0 : Int := (w_165_0 * w_166_0)
  let w_168_0 : Int := (w_164_0 + w_167_0)
  let w_169_0 := witness.w_169_0
  let w_170_0 := witness.w_170_0
  let w_171_0 : Int := (w_169_0 * w_170_0)
  let w_172_0 : Int := (w_168_0 + w_171_0)
  let w_174_0 := witness.w_174_0
  let w_175_0 := witness.w_175_0
  let w_176_0 : Int := (w_174_0 * w_175_0)
  let w_177_0 : Int := (w_172_0 + w_176_0)
  let w_178_0 := witness.w_178_0
  let w_179_0 := witness.w_179_0
  let w_180_0 : Int := (w_178_0 * w_179_0)
  let w_181_0 : Int := (w_177_0 + w_180_0)
  let w_182_0 := witness.w_182_0
  let w_183_0 := witness.w_183_0
  let w_184_0 : Int := (w_182_0 * w_183_0)
  let w_185_0 : Int := (w_181_0 + w_184_0)
  let w_186_0 := witness.w_186_0
  let w_187_0 := witness.w_187_0
  let w_188_0 : Int := (w_186_0 * w_187_0)
  let w_189_0 : Int := (w_185_0 + w_188_0)
  let w_190_0 := witness.w_190_0
  let w_191_0 := witness.w_191_0
  let w_192_0 : Int := (w_190_0 * w_191_0)
  let w_193_0 : Int := (w_189_0 + w_192_0)
  let w_194_0 := witness.w_194_0
  let w_195_0 := witness.w_195_0
  let w_196_0 : Int := (w_194_0 * w_195_0)
  let w_197_0 : Int := (w_193_0 + w_196_0)
  let w_198_0 := witness.w_198_0
  let w_199_0 := witness.w_199_0
  let w_200_0 : Int := (w_198_0 * w_199_0)
  let w_201_0 : Int := (w_197_0 + w_200_0)
  let w_202_0 := witness.w_202_0
  let w_203_0 := witness.w_203_0
  let w_204_0 : Int := (w_202_0 * w_203_0)
  let w_205_0 : Int := (w_201_0 + w_204_0)
  let w_206_0 := witness.w_206_0
  let w_207_0 := witness.w_207_0
  let w_208_0 : Int := (w_206_0 * w_207_0)
  let w_209_0 : Int := (w_205_0 + w_208_0)
  let w_210_0 := witness.w_210_0
  let w_211_0 := witness.w_211_0
  let w_212_0 : Int := (w_210_0 * w_211_0)
  let w_213_0 : Int := (w_209_0 + w_212_0)
  let w_214_0 := witness.w_214_0
  let w_215_0 := witness.w_215_0
  let w_216_0 : Int := (w_214_0 * w_215_0)
  let w_217_0 : Int := (w_213_0 + w_216_0)
  let w_218_0 := witness.w_218_0
  let w_219_0 := witness.w_219_0
  let w_220_0 : Int := (w_218_0 * w_219_0)
  let w_221_0 : Int := (w_217_0 + w_220_0)
  let w_222_0 := witness.w_222_0
  let w_223_0 := witness.w_223_0
  let w_224_0 : Int := (w_222_0 * w_223_0)
  let w_225_0 : Int := (w_221_0 + w_224_0)
  let w_226_0 := witness.w_226_0
  let w_227_0 := witness.w_227_0
  let w_228_0 : Int := (w_226_0 * w_227_0)
  let w_229_0 : Int := (w_225_0 + w_228_0)
  let w_231_0 := witness.w_231_0
  let w_232_0 := witness.w_232_0
  let w_233_0 : Int := (w_231_0 * w_232_0)
  let w_234_0 : Int := (w_229_0 + w_233_0)
  let w_235_0 := witness.w_235_0
  let w_236_0 := witness.w_236_0
  let w_237_0 : Int := (w_235_0 * w_236_0)
  let w_238_0 : Int := (w_234_0 + w_237_0)
  let w_239_0 := witness.w_239_0
  let w_240_0 := witness.w_240_0
  let w_241_0 : Int := (w_239_0 * w_240_0)
  let w_242_0 : Int := (w_238_0 + w_241_0)
  let w_243_0 := witness.w_243_0
  let w_244_0 := witness.w_244_0
  let w_245_0 : Int := (w_243_0 * w_244_0)
  let w_246_0 : Int := (w_242_0 + w_245_0)
  let w_247_0 := witness.w_247_0
  let w_248_0 := witness.w_248_0
  let w_249_0 : Int := (w_247_0 * w_248_0)
  let w_250_0 : Int := (w_246_0 + w_249_0)
  let w_251_0 := witness.w_251_0
  let w_252_0 := witness.w_252_0
  let w_253_0 : Int := (w_251_0 * w_252_0)
  let w_254_0 : Int := (w_250_0 + w_253_0)
  let w_255_0 := witness.w_255_0
  let w_256_0 := witness.w_256_0
  let w_257_0 : Int := (w_255_0 * w_256_0)
  let w_258_0 : Int := (w_254_0 + w_257_0)
  let w_259_0 := witness.w_259_0
  let w_260_0 := witness.w_260_0
  let w_261_0 : Int := (w_259_0 * w_260_0)
  let w_262_0 : Int := (w_258_0 + w_261_0)
  let w_263_0 := witness.w_263_0
  let w_264_0 := witness.w_264_0
  let w_265_0 : Int := (w_263_0 * w_264_0)
  let w_266_0 : Int := (w_262_0 + w_265_0)
  let w_267_0 := witness.w_267_0
  let w_268_0 := witness.w_268_0
  let w_269_0 : Int := (w_267_0 * w_268_0)
  let w_270_0 : Int := (w_266_0 + w_269_0)
  let w_271_0 := witness.w_271_0
  let w_272_0 := witness.w_272_0
  let w_273_0 : Int := (w_271_0 * w_272_0)
  let w_274_0 : Int := (w_270_0 + w_273_0)
  let w_275_0 := witness.w_275_0
  let w_276_0 := witness.w_276_0
  let w_277_0 : Int := (w_275_0 * w_276_0)
  let w_278_0 : Int := (w_274_0 + w_277_0)
  let w_279_0 := witness.w_279_0
  let w_280_0 := witness.w_280_0
  let w_281_0 : Int := (w_279_0 * w_280_0)
  let w_282_0 : Int := (w_278_0 + w_281_0)
  let w_283_0 := witness.w_283_0
  let w_284_0 := witness.w_284_0
  let w_285_0 : Int := (w_283_0 * w_284_0)
  let w_286_0 : Int := (w_282_0 + w_285_0)
  let w_287_0 : Int := 16
  let w_288_0 : Int := (w_286_0 % w_287_0)
  let w_289_0 : Int := 2
  let w_290_0 : Int := (w_288_0 * w_289_0)
  let w_291_0 : Int := 16
  let w_292_0 : Int := (w_290_0 / w_291_0)
  let w_293_0 : Int := 1
  let w_294_0 : Int := (w_292_0 + w_293_0)
  let w_295_0 : Int := 2
  let w_296_0 : Int := (w_294_0 % w_295_0)
  generatedRoot.constraints_0 w_1_0 w_3_0 w_59_0 w_116_0 w_173_0 w_230_0 w_2_0 w_4_0 w_7_0 w_8_0 w_11_0 w_12_0 w_15_0 w_16_0 w_19_0 w_20_0 w_23_0 w_24_0 w_27_0 w_28_0 w_31_0 w_32_0 w_35_0 w_36_0 w_39_0 w_40_0 w_43_0 w_44_0 w_47_0 w_48_0 w_51_0 w_52_0 w_55_0 w_56_0 w_60_0 w_61_0 w_64_0 w_65_0 w_68_0 w_69_0 w_72_0 w_73_0 w_76_0 w_77_0 w_80_0 w_81_0 w_84_0 w_85_0 w_88_0 w_89_0 w_92_0 w_93_0 w_96_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_108_0 w_109_0 w_112_0 w_113_0 w_117_0 w_118_0 w_121_0 w_122_0 w_125_0 w_126_0 w_129_0 w_130_0 w_133_0 w_134_0 w_137_0 w_138_0 w_141_0 w_142_0 w_145_0 w_146_0 w_149_0 w_150_0 w_153_0 w_154_0 w_157_0 w_158_0 w_161_0 w_162_0 w_165_0 w_166_0 w_169_0 w_170_0 w_174_0 w_175_0 w_178_0 w_179_0 w_182_0 w_183_0 w_186_0 w_187_0 w_190_0 w_191_0 w_194_0 w_195_0 w_198_0 w_199_0 w_202_0 w_203_0 w_206_0 w_207_0 w_210_0 w_211_0 w_214_0 w_215_0 w_218_0 w_219_0 w_222_0 w_223_0 w_226_0 w_227_0 w_231_0 w_232_0 w_235_0 w_236_0 w_239_0 w_240_0 w_243_0 w_244_0 w_247_0 w_248_0 w_251_0 w_252_0 w_255_0 w_256_0 w_259_0 w_260_0 w_263_0 w_264_0 w_267_0 w_268_0 w_271_0 w_272_0 w_275_0 w_276_0 w_279_0 w_280_0 w_283_0 w_284_0 w_287_0 w_291_0 w_295_0 w_296_0 outputs

def generatedRoot (params : Params) (inputs : (Fin 14 → Int) × (Fin 17 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × (Fin 14 → Int) × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body params inputs outputs witness


end Ideal
