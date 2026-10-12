import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_decode

structure Params where
  unit : Unit

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_6_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_11_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_16_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_21_0 : Int
  w_25_0 : Int
  w_30_0 : Int
  w_35_0 : Int
  w_40_0 : Int
  w_45_0 : Int
  w_50_0 : Int
  w_55_0 : Int
  w_61_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6
  w_62_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_64_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_74_0 : Mxx.Primitives.ExactMatrix 1237937626222020133447349201 8 1 6
  w_75_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_79_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_80_0 : Int
  w_80_1 : Int
  w_80_2 : Int
  w_80_3 : Int
  w_80_4 : Int
  w_80_5 : Int
  w_80_6 : Int
  w_80_7 : Int
  w_89_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_90_0 : Int
  w_90_1 : Int
  w_90_2 : Int
  w_90_3 : Int
  w_90_4 : Int
  w_90_5 : Int
  w_90_6 : Int
  w_90_7 : Int
  w_99_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_100_0 : Int
  w_100_1 : Int
  w_100_2 : Int
  w_100_3 : Int
  w_100_4 : Int
  w_100_5 : Int
  w_100_6 : Int
  w_100_7 : Int
  w_109_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_110_0 : Int
  w_110_1 : Int
  w_110_2 : Int
  w_110_3 : Int
  w_110_4 : Int
  w_110_5 : Int
  w_110_6 : Int
  w_110_7 : Int
  w_119_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_120_0 : Int
  w_120_1 : Int
  w_120_2 : Int
  w_120_3 : Int
  w_120_4 : Int
  w_120_5 : Int
  w_120_6 : Int
  w_120_7 : Int
  w_129_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_130_0 : Int
  w_130_1 : Int
  w_130_2 : Int
  w_130_3 : Int
  w_130_4 : Int
  w_130_5 : Int
  w_130_6 : Int
  w_130_7 : Int
  w_140_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_144_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_149_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_154_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_158_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6
  w_159_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_161_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_169_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_170_0 : Int
  w_170_1 : Int
  w_170_2 : Int
  w_170_3 : Int
  w_170_4 : Int
  w_170_5 : Int
  w_170_6 : Int
  w_170_7 : Int
  w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_180_0 : Int
  w_180_1 : Int
  w_180_2 : Int
  w_180_3 : Int
  w_180_4 : Int
  w_180_5 : Int
  w_180_6 : Int
  w_180_7 : Int
  w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_190_0 : Int
  w_190_1 : Int
  w_190_2 : Int
  w_190_3 : Int
  w_190_4 : Int
  w_190_5 : Int
  w_190_6 : Int
  w_190_7 : Int
  w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_200_0 : Int
  w_200_1 : Int
  w_200_2 : Int
  w_200_3 : Int
  w_200_4 : Int
  w_200_5 : Int
  w_200_6 : Int
  w_200_7 : Int
  w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_210_0 : Int
  w_210_1 : Int
  w_210_2 : Int
  w_210_3 : Int
  w_210_4 : Int
  w_210_5 : Int
  w_210_6 : Int
  w_210_7 : Int
  w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1
  w_220_0 : Int
  w_220_1 : Int
  w_220_2 : Int
  w_220_3 : Int
  w_220_4 : Int
  w_220_5 : Int
  w_220_6 : Int
  w_220_7 : Int

abbrev generatedRoot.constraints_7 (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 5 ∧
  5 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_168_0 0 1 5 6 w_219_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_219_0 w_220_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_219_0 w_220_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_219_0 w_220_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_219_0 w_220_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_219_0 w_220_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_219_0 w_220_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_219_0 w_220_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_219_0 w_220_7 ∧
  outputs = w_228_0

abbrev generatedRoot.constraints_6 (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_168_0 0 1 3 4 w_199_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_199_0 w_200_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_199_0 w_200_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_199_0 w_200_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_199_0 w_200_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_199_0 w_200_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_199_0 w_200_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_199_0 w_200_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_199_0 w_200_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 4 ∧
  4 < 5 ∧
  5 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_168_0 0 1 4 5 w_209_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_209_0 w_210_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_209_0 w_210_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_209_0 w_210_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_209_0 w_210_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_209_0 w_210_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_209_0 w_210_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_209_0 w_210_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_209_0 w_210_7 ∧
  generatedRoot.constraints_7 w_168_0 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.constraints_5 (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_180_0 : Int) (w_180_1 : Int) (w_180_2 : Int) (w_180_3 : Int) (w_180_4 : Int) (w_180_5 : Int) (w_180_6 : Int) (w_180_7 : Int) (w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_190_0 : Int) (w_190_1 : Int) (w_190_2 : Int) (w_190_3 : Int) (w_190_4 : Int) (w_190_5 : Int) (w_190_6 : Int) (w_190_7 : Int) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_168_0 0 1 1 2 w_179_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_179_0 w_180_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_179_0 w_180_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_179_0 w_180_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_179_0 w_180_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_179_0 w_180_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_179_0 w_180_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_179_0 w_180_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_179_0 w_180_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_168_0 0 1 2 3 w_189_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_189_0 w_190_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_189_0 w_190_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_189_0 w_190_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_189_0 w_190_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_189_0 w_190_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_189_0 w_190_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_189_0 w_190_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_189_0 w_190_7 ∧
  generatedRoot.constraints_6 w_168_0 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.constraints_4 (hashModel : MxxRuntime.HashModel) (w_60_0 : ByteArray) (w_153_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6) (w_154_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_158_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_159_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_161_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_169_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_170_0 : Int) (w_170_1 : Int) (w_170_2 : Int) (w_170_3 : Int) (w_170_4 : Int) (w_170_5 : Int) (w_170_6 : Int) (w_170_7 : Int) (w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_180_0 : Int) (w_180_1 : Int) (w_180_2 : Int) (w_180_3 : Int) (w_180_4 : Int) (w_180_5 : Int) (w_180_6 : Int) (w_180_7 : Int) (w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_190_0 : Int) (w_190_1 : Int) (w_190_2 : Int) (w_190_3 : Int) (w_190_4 : Int) (w_190_5 : Int) (w_190_6 : Int) (w_190_7 : Int) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.centeredRebaseRuns w_153_0 w_154_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_60_0) w_158_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_158_0 0 1 0 6 w_159_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_158_0 1 2 0 6 w_161_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_168_0 0 1 0 1 w_169_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_169_0 w_170_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_169_0 w_170_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_169_0 w_170_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_169_0 w_170_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_169_0 w_170_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_169_0 w_170_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_169_0 w_170_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_169_0 w_170_7 ∧
  generatedRoot.constraints_5 w_168_0 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.constraints_3 (hashModel : MxxRuntime.HashModel) (w_60_0 : ByteArray) (w_139_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_143_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6) (w_148_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6) (w_153_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6) (w_78_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_119_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_120_0 : Int) (w_120_1 : Int) (w_120_2 : Int) (w_120_3 : Int) (w_120_4 : Int) (w_120_5 : Int) (w_120_6 : Int) (w_120_7 : Int) (w_129_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_130_0 : Int) (w_130_1 : Int) (w_130_2 : Int) (w_130_3 : Int) (w_130_4 : Int) (w_130_5 : Int) (w_130_6 : Int) (w_130_7 : Int) (w_140_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_144_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_149_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_154_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_158_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_159_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_161_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_169_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_170_0 : Int) (w_170_1 : Int) (w_170_2 : Int) (w_170_3 : Int) (w_170_4 : Int) (w_170_5 : Int) (w_170_6 : Int) (w_170_7 : Int) (w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_180_0 : Int) (w_180_1 : Int) (w_180_2 : Int) (w_180_3 : Int) (w_180_4 : Int) (w_180_5 : Int) (w_180_6 : Int) (w_180_7 : Int) (w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_190_0 : Int) (w_190_1 : Int) (w_190_2 : Int) (w_190_3 : Int) (w_190_4 : Int) (w_190_5 : Int) (w_190_6 : Int) (w_190_7 : Int) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 4 ∧
  4 < 5 ∧
  5 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 4 5 w_119_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_119_0 w_120_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_119_0 w_120_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_119_0 w_120_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_119_0 w_120_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_119_0 w_120_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_119_0 w_120_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_119_0 w_120_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_119_0 w_120_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 5 ∧
  5 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 5 6 w_129_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_129_0 w_130_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_129_0 w_130_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_129_0 w_130_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_129_0 w_130_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_129_0 w_130_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_129_0 w_130_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_129_0 w_130_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_129_0 w_130_7 ∧
  MxxRuntime.centeredRebaseRuns w_139_0 w_140_0 ∧
  MxxRuntime.centeredRebaseRuns w_143_0 w_144_0 ∧
  MxxRuntime.centeredRebaseRuns w_148_0 w_149_0 ∧
  generatedRoot.constraints_4 hashModel w_60_0 w_153_0 w_154_0 w_158_0 w_159_0 w_161_0 w_168_0 w_169_0 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.constraints_2 (hashModel : MxxRuntime.HashModel) (w_60_0 : ByteArray) (w_139_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_143_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6) (w_148_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6) (w_153_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6) (w_78_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_99_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_100_0 : Int) (w_100_1 : Int) (w_100_2 : Int) (w_100_3 : Int) (w_100_4 : Int) (w_100_5 : Int) (w_100_6 : Int) (w_100_7 : Int) (w_109_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_110_0 : Int) (w_110_1 : Int) (w_110_2 : Int) (w_110_3 : Int) (w_110_4 : Int) (w_110_5 : Int) (w_110_6 : Int) (w_110_7 : Int) (w_119_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_120_0 : Int) (w_120_1 : Int) (w_120_2 : Int) (w_120_3 : Int) (w_120_4 : Int) (w_120_5 : Int) (w_120_6 : Int) (w_120_7 : Int) (w_129_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_130_0 : Int) (w_130_1 : Int) (w_130_2 : Int) (w_130_3 : Int) (w_130_4 : Int) (w_130_5 : Int) (w_130_6 : Int) (w_130_7 : Int) (w_140_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_144_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_149_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_154_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_158_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_159_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_161_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_169_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_170_0 : Int) (w_170_1 : Int) (w_170_2 : Int) (w_170_3 : Int) (w_170_4 : Int) (w_170_5 : Int) (w_170_6 : Int) (w_170_7 : Int) (w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_180_0 : Int) (w_180_1 : Int) (w_180_2 : Int) (w_180_3 : Int) (w_180_4 : Int) (w_180_5 : Int) (w_180_6 : Int) (w_180_7 : Int) (w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_190_0 : Int) (w_190_1 : Int) (w_190_2 : Int) (w_190_3 : Int) (w_190_4 : Int) (w_190_5 : Int) (w_190_6 : Int) (w_190_7 : Int) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 2 3 w_99_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_99_0 w_100_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_99_0 w_100_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_99_0 w_100_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_99_0 w_100_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_99_0 w_100_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_99_0 w_100_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_99_0 w_100_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_99_0 w_100_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 3 4 w_109_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_109_0 w_110_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_109_0 w_110_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_109_0 w_110_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_109_0 w_110_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_109_0 w_110_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_109_0 w_110_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_109_0 w_110_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_109_0 w_110_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  generatedRoot.constraints_3 hashModel w_60_0 w_139_0 w_143_0 w_148_0 w_153_0 w_78_0 w_119_0 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_129_0 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_140_0 w_144_0 w_149_0 w_154_0 w_158_0 w_159_0 w_161_0 w_168_0 w_169_0 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.constraints_1 (hashModel : MxxRuntime.HashModel) (w_60_0 : ByteArray) (w_139_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_143_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6) (w_148_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6) (w_153_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6) (w_78_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_79_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_80_0 : Int) (w_80_1 : Int) (w_80_2 : Int) (w_80_3 : Int) (w_80_4 : Int) (w_80_5 : Int) (w_80_6 : Int) (w_80_7 : Int) (w_89_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_90_0 : Int) (w_90_1 : Int) (w_90_2 : Int) (w_90_3 : Int) (w_90_4 : Int) (w_90_5 : Int) (w_90_6 : Int) (w_90_7 : Int) (w_99_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_100_0 : Int) (w_100_1 : Int) (w_100_2 : Int) (w_100_3 : Int) (w_100_4 : Int) (w_100_5 : Int) (w_100_6 : Int) (w_100_7 : Int) (w_109_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_110_0 : Int) (w_110_1 : Int) (w_110_2 : Int) (w_110_3 : Int) (w_110_4 : Int) (w_110_5 : Int) (w_110_6 : Int) (w_110_7 : Int) (w_119_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_120_0 : Int) (w_120_1 : Int) (w_120_2 : Int) (w_120_3 : Int) (w_120_4 : Int) (w_120_5 : Int) (w_120_6 : Int) (w_120_7 : Int) (w_129_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_130_0 : Int) (w_130_1 : Int) (w_130_2 : Int) (w_130_3 : Int) (w_130_4 : Int) (w_130_5 : Int) (w_130_6 : Int) (w_130_7 : Int) (w_140_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_144_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_149_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_154_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_158_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_159_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_161_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_169_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_170_0 : Int) (w_170_1 : Int) (w_170_2 : Int) (w_170_3 : Int) (w_170_4 : Int) (w_170_5 : Int) (w_170_6 : Int) (w_170_7 : Int) (w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_180_0 : Int) (w_180_1 : Int) (w_180_2 : Int) (w_180_3 : Int) (w_180_4 : Int) (w_180_5 : Int) (w_180_6 : Int) (w_180_7 : Int) (w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_190_0 : Int) (w_190_1 : Int) (w_190_2 : Int) (w_190_3 : Int) (w_190_4 : Int) (w_190_5 : Int) (w_190_6 : Int) (w_190_7 : Int) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 0 1 w_79_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_79_0 w_80_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_79_0 w_80_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_79_0 w_80_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_79_0 w_80_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_79_0 w_80_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_79_0 w_80_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_79_0 w_80_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_79_0 w_80_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_78_0 0 1 1 2 w_89_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_89_0 w_90_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_89_0 w_90_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_89_0 w_90_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_89_0 w_90_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_89_0 w_90_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_89_0 w_90_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_89_0 w_90_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_89_0 w_90_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  generatedRoot.constraints_2 hashModel w_60_0 w_139_0 w_143_0 w_148_0 w_153_0 w_78_0 w_99_0 w_100_0 w_100_1 w_100_2 w_100_3 w_100_4 w_100_5 w_100_6 w_100_7 w_109_0 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_119_0 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_129_0 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_140_0 w_144_0 w_149_0 w_154_0 w_158_0 w_159_0 w_161_0 w_168_0 w_169_0 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_1_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_5_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6) (w_10_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6) (w_15_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6) (w_20_0 : Fin 8 → Int) (w_60_0 : ByteArray) (w_139_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_143_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6) (w_148_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6) (w_153_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6) (w_2_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_6_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_11_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_16_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_21_0 : Int) (w_25_0 : Int) (w_30_0 : Int) (w_35_0 : Int) (w_40_0 : Int) (w_45_0 : Int) (w_50_0 : Int) (w_55_0 : Int) (w_61_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_62_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_64_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_74_0 : Mxx.Primitives.ExactMatrix 1237937626222020133447349201 8 1 6) (w_75_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_78_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_79_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_80_0 : Int) (w_80_1 : Int) (w_80_2 : Int) (w_80_3 : Int) (w_80_4 : Int) (w_80_5 : Int) (w_80_6 : Int) (w_80_7 : Int) (w_89_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_90_0 : Int) (w_90_1 : Int) (w_90_2 : Int) (w_90_3 : Int) (w_90_4 : Int) (w_90_5 : Int) (w_90_6 : Int) (w_90_7 : Int) (w_99_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_100_0 : Int) (w_100_1 : Int) (w_100_2 : Int) (w_100_3 : Int) (w_100_4 : Int) (w_100_5 : Int) (w_100_6 : Int) (w_100_7 : Int) (w_109_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_110_0 : Int) (w_110_1 : Int) (w_110_2 : Int) (w_110_3 : Int) (w_110_4 : Int) (w_110_5 : Int) (w_110_6 : Int) (w_110_7 : Int) (w_119_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_120_0 : Int) (w_120_1 : Int) (w_120_2 : Int) (w_120_3 : Int) (w_120_4 : Int) (w_120_5 : Int) (w_120_6 : Int) (w_120_7 : Int) (w_129_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_130_0 : Int) (w_130_1 : Int) (w_130_2 : Int) (w_130_3 : Int) (w_130_4 : Int) (w_130_5 : Int) (w_130_6 : Int) (w_130_7 : Int) (w_140_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_144_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_149_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_154_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_158_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_159_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_161_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_169_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_170_0 : Int) (w_170_1 : Int) (w_170_2 : Int) (w_170_3 : Int) (w_170_4 : Int) (w_170_5 : Int) (w_170_6 : Int) (w_170_7 : Int) (w_179_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_180_0 : Int) (w_180_1 : Int) (w_180_2 : Int) (w_180_3 : Int) (w_180_4 : Int) (w_180_5 : Int) (w_180_6 : Int) (w_180_7 : Int) (w_189_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_190_0 : Int) (w_190_1 : Int) (w_190_2 : Int) (w_190_3 : Int) (w_190_4 : Int) (w_190_5 : Int) (w_190_6 : Int) (w_190_7 : Int) (w_199_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_200_0 : Int) (w_200_1 : Int) (w_200_2 : Int) (w_200_3 : Int) (w_200_4 : Int) (w_200_5 : Int) (w_200_6 : Int) (w_200_7 : Int) (w_209_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_210_0 : Int) (w_210_1 : Int) (w_210_2 : Int) (w_210_3 : Int) (w_210_4 : Int) (w_210_5 : Int) (w_210_6 : Int) (w_210_7 : Int) (w_219_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1) (w_220_0 : Int) (w_220_1 : Int) (w_220_2 : Int) (w_220_3 : Int) (w_220_4 : Int) (w_220_5 : Int) (w_220_6 : Int) (w_220_7 : Int) (w_228_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.centeredRebaseRuns w_1_0 w_2_0 ∧
  MxxRuntime.centeredRebaseRuns w_5_0 w_6_0 ∧
  MxxRuntime.centeredRebaseRuns w_10_0 w_11_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (0) w_21_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (1) w_25_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (2) w_30_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (3) w_35_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (4) w_40_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (5) w_45_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (6) w_50_0 ∧
  MxxRuntime.familyGetStatic w_20_0 (7) w_55_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_60_0) w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_61_0 0 1 0 6 w_62_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_61_0 1 2 0 6 w_64_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 6 w_74_0 ∧
  MxxRuntime.centeredRebaseRuns w_74_0 w_75_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  generatedRoot.constraints_1 hashModel w_60_0 w_139_0 w_143_0 w_148_0 w_153_0 w_78_0 w_79_0 w_80_0 w_80_1 w_80_2 w_80_3 w_80_4 w_80_5 w_80_6 w_80_7 w_89_0 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_90_6 w_90_7 w_99_0 w_100_0 w_100_1 w_100_2 w_100_3 w_100_4 w_100_5 w_100_6 w_100_7 w_109_0 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_119_0 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_129_0 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_140_0 w_144_0 w_149_0 w_154_0 w_158_0 w_159_0 w_161_0 w_168_0 w_169_0 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Mxx.Primitives.ExactMatrix 1073741329 8 1 6 × Mxx.Primitives.ExactMatrix 1073740609 8 1 6 × Mxx.Primitives.ExactMatrix 1073740529 8 1 6 × (Fin 8 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Mxx.Primitives.ExactMatrix 1073741329 8 1 6 × Mxx.Primitives.ExactMatrix 1073740609 8 1 6 × Mxx.Primitives.ExactMatrix 1073740529 8 1 6 × Unit) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 := inputs.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6 := inputs.2.1
  let w_10_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6 := inputs.2.2.1
  let w_15_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6 := inputs.2.2.2.1
  let w_20_0 : Fin 8 → Int := inputs.2.2.2.2.1
  let w_60_0 : ByteArray := inputs.2.2.2.2.2.1
  let w_67_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := inputs.2.2.2.2.2.2.1
  let w_69_0 : Int := inputs.2.2.2.2.2.2.2.1
  let w_71_0 : Int := inputs.2.2.2.2.2.2.2.2.1
  let w_139_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_143_0 : Mxx.Primitives.ExactMatrix 1073741329 8 1 6 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_148_0 : Mxx.Primitives.ExactMatrix 1073740609 8 1 6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_153_0 : Mxx.Primitives.ExactMatrix 1073740529 8 1 6 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_2_0 := witness.w_2_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(890511653275736298117428745327756462)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_4_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_2_0 w_3_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(673373482644736804462909142109450233)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_8_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_6_0 w_7_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_4_0 w_8_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(676801805491843546832002660143739331)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_13_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_11_0 w_12_0
  let w_14_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_9_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(417760661884955689660474101877988633)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_18_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_16_0 w_17_0
  let w_19_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_14_0 w_18_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_21_0
  let w_23_0 : Int := 0
  let w_24_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_25_0
  let w_27_0 : Int := 1
  let w_28_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_26_0 w_27_0
  let w_29_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_24_0 w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_30_0
  let w_32_0 : Int := 2
  let w_33_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_31_0 w_32_0
  let w_34_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_29_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_35_0
  let w_37_0 : Int := 3
  let w_38_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_36_0 w_37_0
  let w_39_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_34_0 w_38_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_40_0
  let w_42_0 : Int := 4
  let w_43_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_41_0 w_42_0
  let w_44_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_39_0 w_43_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_45_0
  let w_47_0 : Int := 5
  let w_48_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_46_0 w_47_0
  let w_49_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_44_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_50_0
  let w_52_0 : Int := 6
  let w_53_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_51_0 w_52_0
  let w_54_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_49_0 w_53_0
  let w_55_0 := witness.w_55_0
  let w_56_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_55_0
  let w_57_0 : Int := 7
  let w_58_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_56_0 w_57_0
  let w_59_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_54_0 w_58_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_59_0 w_62_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_63_0 w_64_0
  let w_66_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(1073740529)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_68_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_66_0 w_67_0
  let w_70_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_69_0
  let w_72_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_70_0 w_71_0
  let w_73_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_68_0 w_72_0
  let w_74_0 := witness.w_74_0
  let w_75_0 := witness.w_75_0
  let w_76_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_73_0 w_75_0
  let w_77_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixSub w_65_0 w_76_0
  let w_78_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixSub w_19_0 w_77_0
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_80_1 := witness.w_80_1
  let w_80_2 := witness.w_80_2
  let w_80_3 := witness.w_80_3
  let w_80_4 := witness.w_80_4
  let w_80_5 := witness.w_80_5
  let w_80_6 := witness.w_80_6
  let w_80_7 := witness.w_80_7
  let w_81_0 : Int := (w_0_0 + w_80_0)
  let w_82_0 : Int := (w_81_0 + w_80_1)
  let w_83_0 : Int := (w_82_0 + w_80_2)
  let w_84_0 : Int := (w_83_0 + w_80_3)
  let w_85_0 : Int := (w_84_0 + w_80_4)
  let w_86_0 : Int := (w_85_0 + w_80_5)
  let w_87_0 : Int := (w_86_0 + w_80_6)
  let w_88_0 : Int := (w_87_0 + w_80_7)
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_90_1 := witness.w_90_1
  let w_90_2 := witness.w_90_2
  let w_90_3 := witness.w_90_3
  let w_90_4 := witness.w_90_4
  let w_90_5 := witness.w_90_5
  let w_90_6 := witness.w_90_6
  let w_90_7 := witness.w_90_7
  let w_91_0 : Int := (w_88_0 + w_90_0)
  let w_92_0 : Int := (w_91_0 + w_90_1)
  let w_93_0 : Int := (w_92_0 + w_90_2)
  let w_94_0 : Int := (w_93_0 + w_90_3)
  let w_95_0 : Int := (w_94_0 + w_90_4)
  let w_96_0 : Int := (w_95_0 + w_90_5)
  let w_97_0 : Int := (w_96_0 + w_90_6)
  let w_98_0 : Int := (w_97_0 + w_90_7)
  let w_99_0 := witness.w_99_0
  let w_100_0 := witness.w_100_0
  let w_100_1 := witness.w_100_1
  let w_100_2 := witness.w_100_2
  let w_100_3 := witness.w_100_3
  let w_100_4 := witness.w_100_4
  let w_100_5 := witness.w_100_5
  let w_100_6 := witness.w_100_6
  let w_100_7 := witness.w_100_7
  let w_101_0 : Int := (w_98_0 + w_100_0)
  let w_102_0 : Int := (w_101_0 + w_100_1)
  let w_103_0 : Int := (w_102_0 + w_100_2)
  let w_104_0 : Int := (w_103_0 + w_100_3)
  let w_105_0 : Int := (w_104_0 + w_100_4)
  let w_106_0 : Int := (w_105_0 + w_100_5)
  let w_107_0 : Int := (w_106_0 + w_100_6)
  let w_108_0 : Int := (w_107_0 + w_100_7)
  let w_109_0 := witness.w_109_0
  let w_110_0 := witness.w_110_0
  let w_110_1 := witness.w_110_1
  let w_110_2 := witness.w_110_2
  let w_110_3 := witness.w_110_3
  let w_110_4 := witness.w_110_4
  let w_110_5 := witness.w_110_5
  let w_110_6 := witness.w_110_6
  let w_110_7 := witness.w_110_7
  let w_111_0 : Int := (w_108_0 + w_110_0)
  let w_112_0 : Int := (w_111_0 + w_110_1)
  let w_113_0 : Int := (w_112_0 + w_110_2)
  let w_114_0 : Int := (w_113_0 + w_110_3)
  let w_115_0 : Int := (w_114_0 + w_110_4)
  let w_116_0 : Int := (w_115_0 + w_110_5)
  let w_117_0 : Int := (w_116_0 + w_110_6)
  let w_118_0 : Int := (w_117_0 + w_110_7)
  let w_119_0 := witness.w_119_0
  let w_120_0 := witness.w_120_0
  let w_120_1 := witness.w_120_1
  let w_120_2 := witness.w_120_2
  let w_120_3 := witness.w_120_3
  let w_120_4 := witness.w_120_4
  let w_120_5 := witness.w_120_5
  let w_120_6 := witness.w_120_6
  let w_120_7 := witness.w_120_7
  let w_121_0 : Int := (w_118_0 + w_120_0)
  let w_122_0 : Int := (w_121_0 + w_120_1)
  let w_123_0 : Int := (w_122_0 + w_120_2)
  let w_124_0 : Int := (w_123_0 + w_120_3)
  let w_125_0 : Int := (w_124_0 + w_120_4)
  let w_126_0 : Int := (w_125_0 + w_120_5)
  let w_127_0 : Int := (w_126_0 + w_120_6)
  let w_128_0 : Int := (w_127_0 + w_120_7)
  let w_129_0 := witness.w_129_0
  let w_130_0 := witness.w_130_0
  let w_130_1 := witness.w_130_1
  let w_130_2 := witness.w_130_2
  let w_130_3 := witness.w_130_3
  let w_130_4 := witness.w_130_4
  let w_130_5 := witness.w_130_5
  let w_130_6 := witness.w_130_6
  let w_130_7 := witness.w_130_7
  let w_131_0 : Int := (w_128_0 + w_130_0)
  let w_132_0 : Int := (w_131_0 + w_130_1)
  let w_133_0 : Int := (w_132_0 + w_130_2)
  let w_134_0 : Int := (w_133_0 + w_130_3)
  let w_135_0 : Int := (w_134_0 + w_130_4)
  let w_136_0 : Int := (w_135_0 + w_130_5)
  let w_137_0 : Int := (w_136_0 + w_130_6)
  let w_138_0 : Int := (w_137_0 + w_130_7)
  let w_140_0 := witness.w_140_0
  let w_141_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(890511653275736298117428745327756462)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_142_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_140_0 w_141_0
  let w_144_0 := witness.w_144_0
  let w_145_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(673373482644736804462909142109450233)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_146_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_144_0 w_145_0
  let w_147_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_142_0 w_146_0
  let w_149_0 := witness.w_149_0
  let w_150_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(676801805491843546832002660143739331)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_151_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_149_0 w_150_0
  let w_152_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_147_0 w_151_0
  let w_154_0 := witness.w_154_0
  let w_155_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(417760661884955689660474101877988633)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_156_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarRight w_154_0 w_155_0
  let w_157_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_152_0 w_156_0
  let w_158_0 := witness.w_158_0
  let w_159_0 := witness.w_159_0
  let w_160_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_59_0 w_159_0
  let w_161_0 := witness.w_161_0
  let w_162_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_160_0 w_161_0
  let w_163_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (1 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_164_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_66_0 w_163_0
  let w_165_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_164_0 w_72_0
  let w_166_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_165_0 w_75_0
  let w_167_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixSub w_162_0 w_166_0
  let w_168_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixSub w_157_0 w_167_0
  let w_169_0 := witness.w_169_0
  let w_170_0 := witness.w_170_0
  let w_170_1 := witness.w_170_1
  let w_170_2 := witness.w_170_2
  let w_170_3 := witness.w_170_3
  let w_170_4 := witness.w_170_4
  let w_170_5 := witness.w_170_5
  let w_170_6 := witness.w_170_6
  let w_170_7 := witness.w_170_7
  let w_171_0 : Int := (w_138_0 + w_170_0)
  let w_172_0 : Int := (w_171_0 + w_170_1)
  let w_173_0 : Int := (w_172_0 + w_170_2)
  let w_174_0 : Int := (w_173_0 + w_170_3)
  let w_175_0 : Int := (w_174_0 + w_170_4)
  let w_176_0 : Int := (w_175_0 + w_170_5)
  let w_177_0 : Int := (w_176_0 + w_170_6)
  let w_178_0 : Int := (w_177_0 + w_170_7)
  let w_179_0 := witness.w_179_0
  let w_180_0 := witness.w_180_0
  let w_180_1 := witness.w_180_1
  let w_180_2 := witness.w_180_2
  let w_180_3 := witness.w_180_3
  let w_180_4 := witness.w_180_4
  let w_180_5 := witness.w_180_5
  let w_180_6 := witness.w_180_6
  let w_180_7 := witness.w_180_7
  let w_181_0 : Int := (w_178_0 + w_180_0)
  let w_182_0 : Int := (w_181_0 + w_180_1)
  let w_183_0 : Int := (w_182_0 + w_180_2)
  let w_184_0 : Int := (w_183_0 + w_180_3)
  let w_185_0 : Int := (w_184_0 + w_180_4)
  let w_186_0 : Int := (w_185_0 + w_180_5)
  let w_187_0 : Int := (w_186_0 + w_180_6)
  let w_188_0 : Int := (w_187_0 + w_180_7)
  let w_189_0 := witness.w_189_0
  let w_190_0 := witness.w_190_0
  let w_190_1 := witness.w_190_1
  let w_190_2 := witness.w_190_2
  let w_190_3 := witness.w_190_3
  let w_190_4 := witness.w_190_4
  let w_190_5 := witness.w_190_5
  let w_190_6 := witness.w_190_6
  let w_190_7 := witness.w_190_7
  let w_191_0 : Int := (w_188_0 + w_190_0)
  let w_192_0 : Int := (w_191_0 + w_190_1)
  let w_193_0 : Int := (w_192_0 + w_190_2)
  let w_194_0 : Int := (w_193_0 + w_190_3)
  let w_195_0 : Int := (w_194_0 + w_190_4)
  let w_196_0 : Int := (w_195_0 + w_190_5)
  let w_197_0 : Int := (w_196_0 + w_190_6)
  let w_198_0 : Int := (w_197_0 + w_190_7)
  let w_199_0 := witness.w_199_0
  let w_200_0 := witness.w_200_0
  let w_200_1 := witness.w_200_1
  let w_200_2 := witness.w_200_2
  let w_200_3 := witness.w_200_3
  let w_200_4 := witness.w_200_4
  let w_200_5 := witness.w_200_5
  let w_200_6 := witness.w_200_6
  let w_200_7 := witness.w_200_7
  let w_201_0 : Int := (w_198_0 + w_200_0)
  let w_202_0 : Int := (w_201_0 + w_200_1)
  let w_203_0 : Int := (w_202_0 + w_200_2)
  let w_204_0 : Int := (w_203_0 + w_200_3)
  let w_205_0 : Int := (w_204_0 + w_200_4)
  let w_206_0 : Int := (w_205_0 + w_200_5)
  let w_207_0 : Int := (w_206_0 + w_200_6)
  let w_208_0 : Int := (w_207_0 + w_200_7)
  let w_209_0 := witness.w_209_0
  let w_210_0 := witness.w_210_0
  let w_210_1 := witness.w_210_1
  let w_210_2 := witness.w_210_2
  let w_210_3 := witness.w_210_3
  let w_210_4 := witness.w_210_4
  let w_210_5 := witness.w_210_5
  let w_210_6 := witness.w_210_6
  let w_210_7 := witness.w_210_7
  let w_211_0 : Int := (w_208_0 + w_210_0)
  let w_212_0 : Int := (w_211_0 + w_210_1)
  let w_213_0 : Int := (w_212_0 + w_210_2)
  let w_214_0 : Int := (w_213_0 + w_210_3)
  let w_215_0 : Int := (w_214_0 + w_210_4)
  let w_216_0 : Int := (w_215_0 + w_210_5)
  let w_217_0 : Int := (w_216_0 + w_210_6)
  let w_218_0 : Int := (w_217_0 + w_210_7)
  let w_219_0 := witness.w_219_0
  let w_220_0 := witness.w_220_0
  let w_220_1 := witness.w_220_1
  let w_220_2 := witness.w_220_2
  let w_220_3 := witness.w_220_3
  let w_220_4 := witness.w_220_4
  let w_220_5 := witness.w_220_5
  let w_220_6 := witness.w_220_6
  let w_220_7 := witness.w_220_7
  let w_221_0 : Int := (w_218_0 + w_220_0)
  let w_222_0 : Int := (w_221_0 + w_220_1)
  let w_223_0 : Int := (w_222_0 + w_220_2)
  let w_224_0 : Int := (w_223_0 + w_220_3)
  let w_225_0 : Int := (w_224_0 + w_220_4)
  let w_226_0 : Int := (w_225_0 + w_220_5)
  let w_227_0 : Int := (w_226_0 + w_220_6)
  let w_228_0 : Int := (w_227_0 + w_220_7)
  generatedRoot.constraints_0 backend hashModel w_1_0 w_5_0 w_10_0 w_15_0 w_20_0 w_60_0 w_139_0 w_143_0 w_148_0 w_153_0 w_2_0 w_6_0 w_11_0 w_16_0 w_21_0 w_25_0 w_30_0 w_35_0 w_40_0 w_45_0 w_50_0 w_55_0 w_61_0 w_62_0 w_64_0 w_74_0 w_75_0 w_78_0 w_79_0 w_80_0 w_80_1 w_80_2 w_80_3 w_80_4 w_80_5 w_80_6 w_80_7 w_89_0 w_90_0 w_90_1 w_90_2 w_90_3 w_90_4 w_90_5 w_90_6 w_90_7 w_99_0 w_100_0 w_100_1 w_100_2 w_100_3 w_100_4 w_100_5 w_100_6 w_100_7 w_109_0 w_110_0 w_110_1 w_110_2 w_110_3 w_110_4 w_110_5 w_110_6 w_110_7 w_119_0 w_120_0 w_120_1 w_120_2 w_120_3 w_120_4 w_120_5 w_120_6 w_120_7 w_129_0 w_130_0 w_130_1 w_130_2 w_130_3 w_130_4 w_130_5 w_130_6 w_130_7 w_140_0 w_144_0 w_149_0 w_154_0 w_158_0 w_159_0 w_161_0 w_168_0 w_169_0 w_170_0 w_170_1 w_170_2 w_170_3 w_170_4 w_170_5 w_170_6 w_170_7 w_179_0 w_180_0 w_180_1 w_180_2 w_180_3 w_180_4 w_180_5 w_180_6 w_180_7 w_189_0 w_190_0 w_190_1 w_190_2 w_190_3 w_190_4 w_190_5 w_190_6 w_190_7 w_199_0 w_200_0 w_200_1 w_200_2 w_200_3 w_200_4 w_200_5 w_200_6 w_200_7 w_209_0 w_210_0 w_210_1 w_210_2 w_210_3 w_210_4 w_210_5 w_210_6 w_210_7 w_219_0 w_220_0 w_220_1 w_220_2 w_220_3 w_220_4 w_220_5 w_220_6 w_220_7 w_228_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Mxx.Primitives.ExactMatrix 1073741329 8 1 6 × Mxx.Primitives.ExactMatrix 1073740609 8 1 6 × Mxx.Primitives.ExactMatrix 1073740529 8 1 6 × (Fin 8 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Mxx.Primitives.ExactMatrix 1073741329 8 1 6 × Mxx.Primitives.ExactMatrix 1073740609 8 1 6 × Mxx.Primitives.ExactMatrix 1073740529 8 1 6 × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end Stage_decode
