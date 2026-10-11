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
  w_3_0 : Int
  w_7_0 : Int
  w_12_0 : Int
  w_17_0 : Int
  w_22_0 : Int
  w_27_0 : Int
  w_32_0 : Int
  w_37_0 : Int
  w_43_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6
  w_44_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_46_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_56_0 : Mxx.Primitives.ExactMatrix 1237937626222020133447349201 8 1 6
  w_57_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_60_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6
  w_62_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_63_0 : Int
  w_63_1 : Int
  w_63_2 : Int
  w_63_3 : Int
  w_63_4 : Int
  w_63_5 : Int
  w_63_6 : Int
  w_63_7 : Int
  w_72_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_73_0 : Int
  w_73_1 : Int
  w_73_2 : Int
  w_73_3 : Int
  w_73_4 : Int
  w_73_5 : Int
  w_73_6 : Int
  w_73_7 : Int
  w_82_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_83_0 : Int
  w_83_1 : Int
  w_83_2 : Int
  w_83_3 : Int
  w_83_4 : Int
  w_83_5 : Int
  w_83_6 : Int
  w_83_7 : Int
  w_92_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_93_0 : Int
  w_93_1 : Int
  w_93_2 : Int
  w_93_3 : Int
  w_93_4 : Int
  w_93_5 : Int
  w_93_6 : Int
  w_93_7 : Int
  w_102_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_103_0 : Int
  w_103_1 : Int
  w_103_2 : Int
  w_103_3 : Int
  w_103_4 : Int
  w_103_5 : Int
  w_103_6 : Int
  w_103_7 : Int
  w_112_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_113_0 : Int
  w_113_1 : Int
  w_113_2 : Int
  w_113_3 : Int
  w_113_4 : Int
  w_113_5 : Int
  w_113_6 : Int
  w_113_7 : Int
  w_123_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6
  w_124_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_126_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6
  w_133_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6
  w_135_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_136_0 : Int
  w_136_1 : Int
  w_136_2 : Int
  w_136_3 : Int
  w_136_4 : Int
  w_136_5 : Int
  w_136_6 : Int
  w_136_7 : Int
  w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_146_0 : Int
  w_146_1 : Int
  w_146_2 : Int
  w_146_3 : Int
  w_146_4 : Int
  w_146_5 : Int
  w_146_6 : Int
  w_146_7 : Int
  w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_156_0 : Int
  w_156_1 : Int
  w_156_2 : Int
  w_156_3 : Int
  w_156_4 : Int
  w_156_5 : Int
  w_156_6 : Int
  w_156_7 : Int
  w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_166_0 : Int
  w_166_1 : Int
  w_166_2 : Int
  w_166_3 : Int
  w_166_4 : Int
  w_166_5 : Int
  w_166_6 : Int
  w_166_7 : Int
  w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_176_0 : Int
  w_176_1 : Int
  w_176_2 : Int
  w_176_3 : Int
  w_176_4 : Int
  w_176_5 : Int
  w_176_6 : Int
  w_176_7 : Int
  w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1
  w_186_0 : Int
  w_186_1 : Int
  w_186_2 : Int
  w_186_3 : Int
  w_186_4 : Int
  w_186_5 : Int
  w_186_6 : Int
  w_186_7 : Int

abbrev generatedRoot.constraints_7 (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.sliceMatrix w_134_0 0 1 5 6 w_185_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_185_0 w_186_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_185_0 w_186_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_185_0 w_186_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_185_0 w_186_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_185_0 w_186_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_185_0 w_186_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_185_0 w_186_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_185_0 w_186_7 ∧
  outputs = w_194_0

abbrev generatedRoot.constraints_6 (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.sliceMatrix w_134_0 0 1 3 4 w_165_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_165_0 w_166_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_165_0 w_166_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_165_0 w_166_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_165_0 w_166_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_165_0 w_166_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_165_0 w_166_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_165_0 w_166_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_165_0 w_166_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 4 ∧
  4 < 5 ∧
  5 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_134_0 0 1 4 5 w_175_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_175_0 w_176_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_175_0 w_176_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_175_0 w_176_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_175_0 w_176_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_175_0 w_176_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_175_0 w_176_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_175_0 w_176_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_175_0 w_176_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 5 ∧
  5 < 6 ∧
  6 ≤ 6 ∧
  generatedRoot.constraints_7 w_134_0 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.constraints_5 (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_146_0 : Int) (w_146_1 : Int) (w_146_2 : Int) (w_146_3 : Int) (w_146_4 : Int) (w_146_5 : Int) (w_146_6 : Int) (w_146_7 : Int) (w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_156_0 : Int) (w_156_1 : Int) (w_156_2 : Int) (w_156_3 : Int) (w_156_4 : Int) (w_156_5 : Int) (w_156_6 : Int) (w_156_7 : Int) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.sliceMatrix w_134_0 0 1 1 2 w_145_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_145_0 w_146_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_145_0 w_146_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_145_0 w_146_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_145_0 w_146_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_145_0 w_146_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_145_0 w_146_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_145_0 w_146_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_145_0 w_146_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_134_0 0 1 2 3 w_155_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_155_0 w_156_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_155_0 w_156_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_155_0 w_156_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_155_0 w_156_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_155_0 w_156_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_155_0 w_156_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_155_0 w_156_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_155_0 w_156_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 6 ∧
  generatedRoot.constraints_6 w_134_0 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.constraints_4 (w_123_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_124_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_126_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_132_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_133_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_135_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_136_0 : Int) (w_136_1 : Int) (w_136_2 : Int) (w_136_3 : Int) (w_136_4 : Int) (w_136_5 : Int) (w_136_6 : Int) (w_136_7 : Int) (w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_146_0 : Int) (w_146_1 : Int) (w_146_2 : Int) (w_146_3 : Int) (w_146_4 : Int) (w_146_5 : Int) (w_146_6 : Int) (w_146_7 : Int) (w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_156_0 : Int) (w_156_1 : Int) (w_156_2 : Int) (w_156_3 : Int) (w_156_4 : Int) (w_156_5 : Int) (w_156_6 : Int) (w_156_7 : Int) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_123_0 0 1 0 6 w_124_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_123_0 1 2 0 6 w_126_0 ∧
  MxxRuntime.modulusReduceRuns w_132_0 w_133_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_134_0 0 1 0 1 w_135_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_135_0 w_136_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_135_0 w_136_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_135_0 w_136_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_135_0 w_136_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_135_0 w_136_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_135_0 w_136_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_135_0 w_136_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_135_0 w_136_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 6 ∧
  generatedRoot.constraints_5 w_134_0 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.constraints_3 (hashModel : MxxRuntime.HashModel) (w_42_0 : ByteArray) (w_61_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_102_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_103_0 : Int) (w_103_1 : Int) (w_103_2 : Int) (w_103_3 : Int) (w_103_4 : Int) (w_103_5 : Int) (w_103_6 : Int) (w_103_7 : Int) (w_112_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_113_0 : Int) (w_113_1 : Int) (w_113_2 : Int) (w_113_3 : Int) (w_113_4 : Int) (w_113_5 : Int) (w_113_6 : Int) (w_113_7 : Int) (w_123_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_124_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_126_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_132_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_133_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_135_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_136_0 : Int) (w_136_1 : Int) (w_136_2 : Int) (w_136_3 : Int) (w_136_4 : Int) (w_136_5 : Int) (w_136_6 : Int) (w_136_7 : Int) (w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_146_0 : Int) (w_146_1 : Int) (w_146_2 : Int) (w_146_3 : Int) (w_146_4 : Int) (w_146_5 : Int) (w_146_6 : Int) (w_146_7 : Int) (w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_156_0 : Int) (w_156_1 : Int) (w_156_2 : Int) (w_156_3 : Int) (w_156_4 : Int) (w_156_5 : Int) (w_156_6 : Int) (w_156_7 : Int) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.sliceMatrix w_61_0 0 1 4 5 w_102_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_102_0 w_103_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_102_0 w_103_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_102_0 w_103_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_102_0 w_103_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_102_0 w_103_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_102_0 w_103_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_102_0 w_103_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_102_0 w_103_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 5 ∧
  5 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_61_0 0 1 5 6 w_112_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_112_0 w_113_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_112_0 w_113_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_112_0 w_113_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_112_0 w_113_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_112_0 w_113_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_112_0 w_113_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_112_0 w_113_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_112_0 w_113_7 ∧
  MxxRuntime.hashSample (hashModel) ([114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 1]) ([]) (w_42_0) w_123_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  generatedRoot.constraints_4 w_123_0 w_124_0 w_126_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.constraints_2 (hashModel : MxxRuntime.HashModel) (w_42_0 : ByteArray) (w_61_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_82_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_83_0 : Int) (w_83_1 : Int) (w_83_2 : Int) (w_83_3 : Int) (w_83_4 : Int) (w_83_5 : Int) (w_83_6 : Int) (w_83_7 : Int) (w_92_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_93_0 : Int) (w_93_1 : Int) (w_93_2 : Int) (w_93_3 : Int) (w_93_4 : Int) (w_93_5 : Int) (w_93_6 : Int) (w_93_7 : Int) (w_102_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_103_0 : Int) (w_103_1 : Int) (w_103_2 : Int) (w_103_3 : Int) (w_103_4 : Int) (w_103_5 : Int) (w_103_6 : Int) (w_103_7 : Int) (w_112_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_113_0 : Int) (w_113_1 : Int) (w_113_2 : Int) (w_113_3 : Int) (w_113_4 : Int) (w_113_5 : Int) (w_113_6 : Int) (w_113_7 : Int) (w_123_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_124_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_126_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_132_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_133_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_135_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_136_0 : Int) (w_136_1 : Int) (w_136_2 : Int) (w_136_3 : Int) (w_136_4 : Int) (w_136_5 : Int) (w_136_6 : Int) (w_136_7 : Int) (w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_146_0 : Int) (w_146_1 : Int) (w_146_2 : Int) (w_146_3 : Int) (w_146_4 : Int) (w_146_5 : Int) (w_146_6 : Int) (w_146_7 : Int) (w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_156_0 : Int) (w_156_1 : Int) (w_156_2 : Int) (w_156_3 : Int) (w_156_4 : Int) (w_156_5 : Int) (w_156_6 : Int) (w_156_7 : Int) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.sliceMatrix w_61_0 0 1 2 3 w_82_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_82_0 w_83_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_82_0 w_83_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_82_0 w_83_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_82_0 w_83_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_82_0 w_83_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_82_0 w_83_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_82_0 w_83_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_82_0 w_83_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 3 ∧
  3 < 4 ∧
  4 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_61_0 0 1 3 4 w_92_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_92_0 w_93_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_92_0 w_93_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_92_0 w_93_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_92_0 w_93_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_92_0 w_93_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_92_0 w_93_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_92_0 w_93_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_92_0 w_93_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 4 ∧
  4 < 5 ∧
  5 ≤ 6 ∧
  generatedRoot.constraints_3 hashModel w_42_0 w_61_0 w_102_0 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_103_6 w_103_7 w_112_0 w_113_0 w_113_1 w_113_2 w_113_3 w_113_4 w_113_5 w_113_6 w_113_7 w_123_0 w_124_0 w_126_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.constraints_1 (hashModel : MxxRuntime.HashModel) (w_42_0 : ByteArray) (w_61_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_62_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_63_0 : Int) (w_63_1 : Int) (w_63_2 : Int) (w_63_3 : Int) (w_63_4 : Int) (w_63_5 : Int) (w_63_6 : Int) (w_63_7 : Int) (w_72_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_73_0 : Int) (w_73_1 : Int) (w_73_2 : Int) (w_73_3 : Int) (w_73_4 : Int) (w_73_5 : Int) (w_73_6 : Int) (w_73_7 : Int) (w_82_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_83_0 : Int) (w_83_1 : Int) (w_83_2 : Int) (w_83_3 : Int) (w_83_4 : Int) (w_83_5 : Int) (w_83_6 : Int) (w_83_7 : Int) (w_92_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_93_0 : Int) (w_93_1 : Int) (w_93_2 : Int) (w_93_3 : Int) (w_93_4 : Int) (w_93_5 : Int) (w_93_6 : Int) (w_93_7 : Int) (w_102_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_103_0 : Int) (w_103_1 : Int) (w_103_2 : Int) (w_103_3 : Int) (w_103_4 : Int) (w_103_5 : Int) (w_103_6 : Int) (w_103_7 : Int) (w_112_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_113_0 : Int) (w_113_1 : Int) (w_113_2 : Int) (w_113_3 : Int) (w_113_4 : Int) (w_113_5 : Int) (w_113_6 : Int) (w_113_7 : Int) (w_123_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_124_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_126_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_132_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_133_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_135_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_136_0 : Int) (w_136_1 : Int) (w_136_2 : Int) (w_136_3 : Int) (w_136_4 : Int) (w_136_5 : Int) (w_136_6 : Int) (w_136_7 : Int) (w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_146_0 : Int) (w_146_1 : Int) (w_146_2 : Int) (w_146_3 : Int) (w_146_4 : Int) (w_146_5 : Int) (w_146_6 : Int) (w_146_7 : Int) (w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_156_0 : Int) (w_156_1 : Int) (w_156_2 : Int) (w_156_3 : Int) (w_156_4 : Int) (w_156_5 : Int) (w_156_6 : Int) (w_156_7 : Int) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.sliceMatrix w_61_0 0 1 0 1 w_62_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_62_0 w_63_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_62_0 w_63_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_62_0 w_63_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_62_0 w_63_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_62_0 w_63_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_62_0 w_63_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_62_0 w_63_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_62_0 w_63_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_61_0 0 1 1 2 w_72_0 ∧
  (8) = 8 ∧
  MxxRuntime.thresholdDecode (2) (8) 0 w_72_0 w_73_0 ∧
  MxxRuntime.thresholdDecode (2) (8) 1 w_72_0 w_73_1 ∧
  MxxRuntime.thresholdDecode (2) (8) 2 w_72_0 w_73_2 ∧
  MxxRuntime.thresholdDecode (2) (8) 3 w_72_0 w_73_3 ∧
  MxxRuntime.thresholdDecode (2) (8) 4 w_72_0 w_73_4 ∧
  MxxRuntime.thresholdDecode (2) (8) 5 w_72_0 w_73_5 ∧
  MxxRuntime.thresholdDecode (2) (8) 6 w_72_0 w_73_6 ∧
  MxxRuntime.thresholdDecode (2) (8) 7 w_72_0 w_73_7 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 2 ∧
  2 < 3 ∧
  3 ≤ 6 ∧
  generatedRoot.constraints_2 hashModel w_42_0 w_61_0 w_82_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_92_0 w_93_0 w_93_1 w_93_2 w_93_3 w_93_4 w_93_5 w_93_6 w_93_7 w_102_0 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_103_6 w_103_7 w_112_0 w_113_0 w_113_1 w_113_2 w_113_3 w_113_4 w_113_5 w_113_6 w_113_7 w_123_0 w_124_0 w_126_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_2_0 : Fin 8 → Int) (w_42_0 : ByteArray) (w_3_0 : Int) (w_7_0 : Int) (w_12_0 : Int) (w_17_0 : Int) (w_22_0 : Int) (w_27_0 : Int) (w_32_0 : Int) (w_37_0 : Int) (w_43_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_44_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_46_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_56_0 : Mxx.Primitives.ExactMatrix 1237937626222020133447349201 8 1 6) (w_57_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_59_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_60_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_61_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_62_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_63_0 : Int) (w_63_1 : Int) (w_63_2 : Int) (w_63_3 : Int) (w_63_4 : Int) (w_63_5 : Int) (w_63_6 : Int) (w_63_7 : Int) (w_72_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_73_0 : Int) (w_73_1 : Int) (w_73_2 : Int) (w_73_3 : Int) (w_73_4 : Int) (w_73_5 : Int) (w_73_6 : Int) (w_73_7 : Int) (w_82_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_83_0 : Int) (w_83_1 : Int) (w_83_2 : Int) (w_83_3 : Int) (w_83_4 : Int) (w_83_5 : Int) (w_83_6 : Int) (w_83_7 : Int) (w_92_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_93_0 : Int) (w_93_1 : Int) (w_93_2 : Int) (w_93_3 : Int) (w_93_4 : Int) (w_93_5 : Int) (w_93_6 : Int) (w_93_7 : Int) (w_102_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_103_0 : Int) (w_103_1 : Int) (w_103_2 : Int) (w_103_3 : Int) (w_103_4 : Int) (w_103_5 : Int) (w_103_6 : Int) (w_103_7 : Int) (w_112_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_113_0 : Int) (w_113_1 : Int) (w_113_2 : Int) (w_113_3 : Int) (w_113_4 : Int) (w_113_5 : Int) (w_113_6 : Int) (w_113_7 : Int) (w_123_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 2 6) (w_124_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_126_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_132_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6) (w_133_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6) (w_135_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_136_0 : Int) (w_136_1 : Int) (w_136_2 : Int) (w_136_3 : Int) (w_136_4 : Int) (w_136_5 : Int) (w_136_6 : Int) (w_136_7 : Int) (w_145_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_146_0 : Int) (w_146_1 : Int) (w_146_2 : Int) (w_146_3 : Int) (w_146_4 : Int) (w_146_5 : Int) (w_146_6 : Int) (w_146_7 : Int) (w_155_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_156_0 : Int) (w_156_1 : Int) (w_156_2 : Int) (w_156_3 : Int) (w_156_4 : Int) (w_156_5 : Int) (w_156_6 : Int) (w_156_7 : Int) (w_165_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_166_0 : Int) (w_166_1 : Int) (w_166_2 : Int) (w_166_3 : Int) (w_166_4 : Int) (w_166_5 : Int) (w_166_6 : Int) (w_166_7 : Int) (w_175_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_176_0 : Int) (w_176_1 : Int) (w_176_2 : Int) (w_176_3 : Int) (w_176_4 : Int) (w_176_5 : Int) (w_176_6 : Int) (w_176_7 : Int) (w_185_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 1) (w_186_0 : Int) (w_186_1 : Int) (w_186_2 : Int) (w_186_3 : Int) (w_186_4 : Int) (w_186_5 : Int) (w_186_6 : Int) (w_186_7 : Int) (w_194_0 : Int) (outputs : Int) : Prop :=
  MxxRuntime.familyGetStatic w_2_0 (0) w_3_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (1) w_7_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (2) w_12_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (3) w_17_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (4) w_22_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (5) w_27_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (6) w_32_0 ∧
  MxxRuntime.familyGetStatic w_2_0 (7) w_37_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 101, 102, 114, 101, 115, 104, 47, 118, 49, 47, 65, 116, 111, 112, 45, 114, 101, 102, 114, 101, 115, 104, 45, 114, 111, 117, 110, 100, 45, 116, 114, 105, 112, 0]) ([]) (w_42_0) w_43_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_43_0 0 1 0 6 w_44_0 ∧
  0 ≤ 1 ∧
  1 < 2 ∧
  2 ≤ 2 ∧
  0 ≤ 0 ∧
  0 < 6 ∧
  6 ≤ 6 ∧
  MxxRuntime.sliceMatrix w_43_0 1 2 0 6 w_46_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 6 w_56_0 ∧
  MxxRuntime.centeredRebaseRuns w_56_0 w_57_0 ∧
  MxxRuntime.modulusReduceRuns w_59_0 w_60_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 6 ∧
  generatedRoot.constraints_1 hashModel w_42_0 w_61_0 w_62_0 w_63_0 w_63_1 w_63_2 w_63_3 w_63_4 w_63_5 w_63_6 w_63_7 w_72_0 w_73_0 w_73_1 w_73_2 w_73_3 w_73_4 w_73_5 w_73_6 w_73_7 w_82_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_92_0 w_93_0 w_93_1 w_93_2 w_93_3 w_93_4 w_93_5 w_93_6 w_93_7 w_102_0 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_103_6 w_103_7 w_112_0 w_113_0 w_113_1 w_113_2 w_113_3 w_113_4 w_113_5 w_113_6 w_113_7 w_123_0 w_124_0 w_126_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × (Fin 8 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Unit) (outputs : Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 := inputs.1
  let w_2_0 : Fin 8 → Int := inputs.2.1
  let w_42_0 : ByteArray := inputs.2.2.1
  let w_49_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := inputs.2.2.2.1
  let w_51_0 : Int := inputs.2.2.2.2.1
  let w_53_0 : Int := inputs.2.2.2.2.2.1
  let w_122_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 := inputs.2.2.2.2.2.2.1
  let w_0_0 : Int := 0
  let w_3_0 := witness.w_3_0
  let w_4_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_3_0
  let w_5_0 : Int := 0
  let w_6_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_4_0 w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_7_0
  let w_9_0 : Int := 1
  let w_10_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_8_0 w_9_0
  let w_11_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_6_0 w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_12_0
  let w_14_0 : Int := 2
  let w_15_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_13_0 w_14_0
  let w_16_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_11_0 w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_17_0
  let w_19_0 : Int := 3
  let w_20_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_18_0 w_19_0
  let w_21_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_16_0 w_20_0
  let w_22_0 := witness.w_22_0
  let w_23_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_22_0
  let w_24_0 : Int := 4
  let w_25_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_23_0 w_24_0
  let w_26_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_21_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_27_0
  let w_29_0 : Int := 5
  let w_30_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_28_0 w_29_0
  let w_31_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_26_0 w_30_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_32_0
  let w_34_0 : Int := 6
  let w_35_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_33_0 w_34_0
  let w_36_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_31_0 w_35_0
  let w_37_0 := witness.w_37_0
  let w_38_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_37_0
  let w_39_0 : Int := 7
  let w_40_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_38_0 w_39_0
  let w_41_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixAdd w_36_0 w_40_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_41_0 w_44_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_45_0 w_46_0
  let w_48_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (MxxRuntime.matrixPolynomial [(1073740529)] : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_50_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_48_0 w_49_0
  let w_52_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.liftInteger w_51_0
  let w_54_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.multiplyMonomial w_52_0 w_53_0
  let w_55_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_50_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_55_0 w_57_0
  let w_59_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixSub w_47_0 w_58_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 := MxxRuntime.matrixSub w_1_0 w_60_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_63_1 := witness.w_63_1
  let w_63_2 := witness.w_63_2
  let w_63_3 := witness.w_63_3
  let w_63_4 := witness.w_63_4
  let w_63_5 := witness.w_63_5
  let w_63_6 := witness.w_63_6
  let w_63_7 := witness.w_63_7
  let w_64_0 : Int := (w_0_0 + w_63_0)
  let w_65_0 : Int := (w_64_0 + w_63_1)
  let w_66_0 : Int := (w_65_0 + w_63_2)
  let w_67_0 : Int := (w_66_0 + w_63_3)
  let w_68_0 : Int := (w_67_0 + w_63_4)
  let w_69_0 : Int := (w_68_0 + w_63_5)
  let w_70_0 : Int := (w_69_0 + w_63_6)
  let w_71_0 : Int := (w_70_0 + w_63_7)
  let w_72_0 := witness.w_72_0
  let w_73_0 := witness.w_73_0
  let w_73_1 := witness.w_73_1
  let w_73_2 := witness.w_73_2
  let w_73_3 := witness.w_73_3
  let w_73_4 := witness.w_73_4
  let w_73_5 := witness.w_73_5
  let w_73_6 := witness.w_73_6
  let w_73_7 := witness.w_73_7
  let w_74_0 : Int := (w_71_0 + w_73_0)
  let w_75_0 : Int := (w_74_0 + w_73_1)
  let w_76_0 : Int := (w_75_0 + w_73_2)
  let w_77_0 : Int := (w_76_0 + w_73_3)
  let w_78_0 : Int := (w_77_0 + w_73_4)
  let w_79_0 : Int := (w_78_0 + w_73_5)
  let w_80_0 : Int := (w_79_0 + w_73_6)
  let w_81_0 : Int := (w_80_0 + w_73_7)
  let w_82_0 := witness.w_82_0
  let w_83_0 := witness.w_83_0
  let w_83_1 := witness.w_83_1
  let w_83_2 := witness.w_83_2
  let w_83_3 := witness.w_83_3
  let w_83_4 := witness.w_83_4
  let w_83_5 := witness.w_83_5
  let w_83_6 := witness.w_83_6
  let w_83_7 := witness.w_83_7
  let w_84_0 : Int := (w_81_0 + w_83_0)
  let w_85_0 : Int := (w_84_0 + w_83_1)
  let w_86_0 : Int := (w_85_0 + w_83_2)
  let w_87_0 : Int := (w_86_0 + w_83_3)
  let w_88_0 : Int := (w_87_0 + w_83_4)
  let w_89_0 : Int := (w_88_0 + w_83_5)
  let w_90_0 : Int := (w_89_0 + w_83_6)
  let w_91_0 : Int := (w_90_0 + w_83_7)
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_93_1 := witness.w_93_1
  let w_93_2 := witness.w_93_2
  let w_93_3 := witness.w_93_3
  let w_93_4 := witness.w_93_4
  let w_93_5 := witness.w_93_5
  let w_93_6 := witness.w_93_6
  let w_93_7 := witness.w_93_7
  let w_94_0 : Int := (w_91_0 + w_93_0)
  let w_95_0 : Int := (w_94_0 + w_93_1)
  let w_96_0 : Int := (w_95_0 + w_93_2)
  let w_97_0 : Int := (w_96_0 + w_93_3)
  let w_98_0 : Int := (w_97_0 + w_93_4)
  let w_99_0 : Int := (w_98_0 + w_93_5)
  let w_100_0 : Int := (w_99_0 + w_93_6)
  let w_101_0 : Int := (w_100_0 + w_93_7)
  let w_102_0 := witness.w_102_0
  let w_103_0 := witness.w_103_0
  let w_103_1 := witness.w_103_1
  let w_103_2 := witness.w_103_2
  let w_103_3 := witness.w_103_3
  let w_103_4 := witness.w_103_4
  let w_103_5 := witness.w_103_5
  let w_103_6 := witness.w_103_6
  let w_103_7 := witness.w_103_7
  let w_104_0 : Int := (w_101_0 + w_103_0)
  let w_105_0 : Int := (w_104_0 + w_103_1)
  let w_106_0 : Int := (w_105_0 + w_103_2)
  let w_107_0 : Int := (w_106_0 + w_103_3)
  let w_108_0 : Int := (w_107_0 + w_103_4)
  let w_109_0 : Int := (w_108_0 + w_103_5)
  let w_110_0 : Int := (w_109_0 + w_103_6)
  let w_111_0 : Int := (w_110_0 + w_103_7)
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_113_1 := witness.w_113_1
  let w_113_2 := witness.w_113_2
  let w_113_3 := witness.w_113_3
  let w_113_4 := witness.w_113_4
  let w_113_5 := witness.w_113_5
  let w_113_6 := witness.w_113_6
  let w_113_7 := witness.w_113_7
  let w_114_0 : Int := (w_111_0 + w_113_0)
  let w_115_0 : Int := (w_114_0 + w_113_1)
  let w_116_0 : Int := (w_115_0 + w_113_2)
  let w_117_0 : Int := (w_116_0 + w_113_3)
  let w_118_0 : Int := (w_117_0 + w_113_4)
  let w_119_0 : Int := (w_118_0 + w_113_5)
  let w_120_0 : Int := (w_119_0 + w_113_6)
  let w_121_0 : Int := (w_120_0 + w_113_7)
  let w_123_0 := witness.w_123_0
  let w_124_0 := witness.w_124_0
  let w_125_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_41_0 w_124_0
  let w_126_0 := witness.w_126_0
  let w_127_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixAdd w_125_0 w_126_0
  let w_128_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := (1 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1)
  let w_129_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_48_0 w_128_0
  let w_130_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 := MxxRuntime.matrixMulScalarLeft w_129_0 w_54_0
  let w_131_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixMulScalarLeft w_130_0 w_57_0
  let w_132_0 : Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 6 := MxxRuntime.matrixSub w_127_0 w_131_0
  let w_133_0 := witness.w_133_0
  let w_134_0 : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 := MxxRuntime.matrixSub w_122_0 w_133_0
  let w_135_0 := witness.w_135_0
  let w_136_0 := witness.w_136_0
  let w_136_1 := witness.w_136_1
  let w_136_2 := witness.w_136_2
  let w_136_3 := witness.w_136_3
  let w_136_4 := witness.w_136_4
  let w_136_5 := witness.w_136_5
  let w_136_6 := witness.w_136_6
  let w_136_7 := witness.w_136_7
  let w_137_0 : Int := (w_121_0 + w_136_0)
  let w_138_0 : Int := (w_137_0 + w_136_1)
  let w_139_0 : Int := (w_138_0 + w_136_2)
  let w_140_0 : Int := (w_139_0 + w_136_3)
  let w_141_0 : Int := (w_140_0 + w_136_4)
  let w_142_0 : Int := (w_141_0 + w_136_5)
  let w_143_0 : Int := (w_142_0 + w_136_6)
  let w_144_0 : Int := (w_143_0 + w_136_7)
  let w_145_0 := witness.w_145_0
  let w_146_0 := witness.w_146_0
  let w_146_1 := witness.w_146_1
  let w_146_2 := witness.w_146_2
  let w_146_3 := witness.w_146_3
  let w_146_4 := witness.w_146_4
  let w_146_5 := witness.w_146_5
  let w_146_6 := witness.w_146_6
  let w_146_7 := witness.w_146_7
  let w_147_0 : Int := (w_144_0 + w_146_0)
  let w_148_0 : Int := (w_147_0 + w_146_1)
  let w_149_0 : Int := (w_148_0 + w_146_2)
  let w_150_0 : Int := (w_149_0 + w_146_3)
  let w_151_0 : Int := (w_150_0 + w_146_4)
  let w_152_0 : Int := (w_151_0 + w_146_5)
  let w_153_0 : Int := (w_152_0 + w_146_6)
  let w_154_0 : Int := (w_153_0 + w_146_7)
  let w_155_0 := witness.w_155_0
  let w_156_0 := witness.w_156_0
  let w_156_1 := witness.w_156_1
  let w_156_2 := witness.w_156_2
  let w_156_3 := witness.w_156_3
  let w_156_4 := witness.w_156_4
  let w_156_5 := witness.w_156_5
  let w_156_6 := witness.w_156_6
  let w_156_7 := witness.w_156_7
  let w_157_0 : Int := (w_154_0 + w_156_0)
  let w_158_0 : Int := (w_157_0 + w_156_1)
  let w_159_0 : Int := (w_158_0 + w_156_2)
  let w_160_0 : Int := (w_159_0 + w_156_3)
  let w_161_0 : Int := (w_160_0 + w_156_4)
  let w_162_0 : Int := (w_161_0 + w_156_5)
  let w_163_0 : Int := (w_162_0 + w_156_6)
  let w_164_0 : Int := (w_163_0 + w_156_7)
  let w_165_0 := witness.w_165_0
  let w_166_0 := witness.w_166_0
  let w_166_1 := witness.w_166_1
  let w_166_2 := witness.w_166_2
  let w_166_3 := witness.w_166_3
  let w_166_4 := witness.w_166_4
  let w_166_5 := witness.w_166_5
  let w_166_6 := witness.w_166_6
  let w_166_7 := witness.w_166_7
  let w_167_0 : Int := (w_164_0 + w_166_0)
  let w_168_0 : Int := (w_167_0 + w_166_1)
  let w_169_0 : Int := (w_168_0 + w_166_2)
  let w_170_0 : Int := (w_169_0 + w_166_3)
  let w_171_0 : Int := (w_170_0 + w_166_4)
  let w_172_0 : Int := (w_171_0 + w_166_5)
  let w_173_0 : Int := (w_172_0 + w_166_6)
  let w_174_0 : Int := (w_173_0 + w_166_7)
  let w_175_0 := witness.w_175_0
  let w_176_0 := witness.w_176_0
  let w_176_1 := witness.w_176_1
  let w_176_2 := witness.w_176_2
  let w_176_3 := witness.w_176_3
  let w_176_4 := witness.w_176_4
  let w_176_5 := witness.w_176_5
  let w_176_6 := witness.w_176_6
  let w_176_7 := witness.w_176_7
  let w_177_0 : Int := (w_174_0 + w_176_0)
  let w_178_0 : Int := (w_177_0 + w_176_1)
  let w_179_0 : Int := (w_178_0 + w_176_2)
  let w_180_0 : Int := (w_179_0 + w_176_3)
  let w_181_0 : Int := (w_180_0 + w_176_4)
  let w_182_0 : Int := (w_181_0 + w_176_5)
  let w_183_0 : Int := (w_182_0 + w_176_6)
  let w_184_0 : Int := (w_183_0 + w_176_7)
  let w_185_0 := witness.w_185_0
  let w_186_0 := witness.w_186_0
  let w_186_1 := witness.w_186_1
  let w_186_2 := witness.w_186_2
  let w_186_3 := witness.w_186_3
  let w_186_4 := witness.w_186_4
  let w_186_5 := witness.w_186_5
  let w_186_6 := witness.w_186_6
  let w_186_7 := witness.w_186_7
  let w_187_0 : Int := (w_184_0 + w_186_0)
  let w_188_0 : Int := (w_187_0 + w_186_1)
  let w_189_0 : Int := (w_188_0 + w_186_2)
  let w_190_0 : Int := (w_189_0 + w_186_3)
  let w_191_0 : Int := (w_190_0 + w_186_4)
  let w_192_0 : Int := (w_191_0 + w_186_5)
  let w_193_0 : Int := (w_192_0 + w_186_6)
  let w_194_0 : Int := (w_193_0 + w_186_7)
  generatedRoot.constraints_0 backend hashModel w_2_0 w_42_0 w_3_0 w_7_0 w_12_0 w_17_0 w_22_0 w_27_0 w_32_0 w_37_0 w_43_0 w_44_0 w_46_0 w_56_0 w_57_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_63_1 w_63_2 w_63_3 w_63_4 w_63_5 w_63_6 w_63_7 w_72_0 w_73_0 w_73_1 w_73_2 w_73_3 w_73_4 w_73_5 w_73_6 w_73_7 w_82_0 w_83_0 w_83_1 w_83_2 w_83_3 w_83_4 w_83_5 w_83_6 w_83_7 w_92_0 w_93_0 w_93_1 w_93_2 w_93_3 w_93_4 w_93_5 w_93_6 w_93_7 w_102_0 w_103_0 w_103_1 w_103_2 w_103_3 w_103_4 w_103_5 w_103_6 w_103_7 w_112_0 w_113_0 w_113_1 w_113_2 w_113_3 w_113_4 w_113_5 w_113_6 w_113_7 w_123_0 w_124_0 w_126_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_136_1 w_136_2 w_136_3 w_136_4 w_136_5 w_136_6 w_136_7 w_145_0 w_146_0 w_146_1 w_146_2 w_146_3 w_146_4 w_146_5 w_146_6 w_146_7 w_155_0 w_156_0 w_156_1 w_156_2 w_156_3 w_156_4 w_156_5 w_156_6 w_156_7 w_165_0 w_166_0 w_166_1 w_166_2 w_166_3 w_166_4 w_166_5 w_166_6 w_166_7 w_175_0 w_176_0 w_176_1 w_176_2 w_176_3 w_176_4 w_176_5 w_176_6 w_176_7 w_185_0 w_186_0 w_186_1 w_186_2 w_186_3 w_186_4 w_186_5 w_186_6 w_186_7 w_194_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × (Fin 8 → Int) × ByteArray × Mxx.Primitives.ExactMatrix 1329223801648636169536407324729467329 8 1 1 × Int × Int × Mxx.Primitives.ExactMatrix 1073741441 8 1 6 × Unit) (outputs : Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end Stage_decode
