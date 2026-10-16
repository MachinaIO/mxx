import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_initial

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 2
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat0
  w_5_0 : mxxMat1
  w_6_0 : mxxMat1
  w_11_0 : mxxMat2
  w_13_0 : mxxMat2
  w_16_0 : mxxMat3
  w_17_0 : mxxMat2
  w_20_0 : mxxMat2
  w_21_0 : mxxMat2
  w_24_0 : Fin 1 → mxxMat2
  w_25_0 : mxxMat2
  w_27_0 : mxxMat2
  w_33_0 : mxxMat2
  w_34_0 : mxxMat2
  w_37_0 : Fin 1 → mxxMat2
  w_39_0 : mxxMat2
  w_41_0 : mxxMat2
  w_44_0 : mxxMat3
  w_45_0 : mxxMat2
  w_48_0 : mxxMat2
  w_49_0 : mxxMat2
  w_52_0 : Fin 1 → mxxMat2
  w_53_0 : mxxMat2
  w_55_0 : mxxMat2
  w_61_0 : mxxMat2
  w_62_0 : mxxMat2
  w_65_0 : Fin 1 → mxxMat2
  w_67_0 : mxxMat2
  w_69_0 : mxxMat2
  w_72_0 : mxxMat3
  w_73_0 : mxxMat2
  w_76_0 : mxxMat2
  w_77_0 : mxxMat2
  w_80_0 : Fin 1 → mxxMat2
  w_81_0 : mxxMat2
  w_83_0 : mxxMat2
  w_89_0 : mxxMat2
  w_90_0 : mxxMat2
  w_93_0 : Fin 1 → mxxMat2
  w_95_0 : mxxMat2
  w_97_0 : mxxMat2
  w_100_0 : mxxMat3
  w_101_0 : mxxMat2
  w_104_0 : mxxMat2
  w_105_0 : mxxMat2
  w_108_0 : Fin 1 → mxxMat2
  w_109_0 : mxxMat2
  w_111_0 : mxxMat2
  w_117_0 : mxxMat2
  w_118_0 : mxxMat2
  w_121_0 : Fin 1 → mxxMat2
  w_123_0 : mxxMat2
  w_125_0 : mxxMat2
  w_128_0 : mxxMat3
  w_129_0 : mxxMat2
  w_132_0 : mxxMat2
  w_133_0 : mxxMat2
  w_136_0 : Fin 1 → mxxMat2
  w_137_0 : mxxMat2
  w_139_0 : mxxMat2
  w_145_0 : mxxMat2
  w_146_0 : mxxMat2
  w_149_0 : Fin 1 → mxxMat2
  w_151_0 : mxxMat2
  w_153_0 : mxxMat2
  w_156_0 : mxxMat3
  w_157_0 : mxxMat2
  w_160_0 : mxxMat2
  w_161_0 : mxxMat2
  w_164_0 : Fin 1 → mxxMat2
  w_165_0 : mxxMat2
  w_167_0 : mxxMat2
  w_173_0 : mxxMat2
  w_174_0 : mxxMat2
  w_177_0 : Fin 1 → mxxMat2
  w_179_0 : mxxMat2
  w_181_0 : mxxMat2
  w_184_0 : mxxMat3
  w_185_0 : mxxMat2
  w_188_0 : mxxMat2
  w_189_0 : mxxMat2
  w_192_0 : Fin 1 → mxxMat2
  w_193_0 : mxxMat2
  w_195_0 : mxxMat2
  w_201_0 : mxxMat2
  w_202_0 : mxxMat2
  w_205_0 : Fin 1 → mxxMat2
  w_207_0 : mxxMat2
  w_209_0 : mxxMat2
  w_212_0 : mxxMat3
  w_213_0 : mxxMat2
  w_216_0 : mxxMat2
  w_217_0 : mxxMat2
  w_220_0 : Fin 8 → mxxMat2
  w_221_0 : mxxMat2
  w_223_0 : mxxMat2
  w_229_0 : mxxMat2
  w_230_0 : mxxMat2
  w_233_0 : Fin 8 → mxxMat2
  w_234_0 : Fin 4 → mxxMat2
  w_235_0 : Fin 4 → mxxMat2
  w_236_0 : mxxMat2
  w_238_0 : mxxMat2
  w_243_0 : Int
  w_248_0 : mxxMat2
  w_251_0 : mxxMat2
  w_252_0 : mxxMat2
  w_255_0 : mxxMat2
  w_257_0 : mxxMat2
  w_262_0 : mxxMat2
  w_265_0 : mxxMat2
  w_266_0 : mxxMat2
  w_269_0 : Fin 2 → mxxMat2
  w_270_0 : mxxMat2
  w_272_0 : mxxMat2
  w_278_0 : mxxMat2
  w_279_0 : mxxMat2
  w_282_0 : mxxMat2
  w_284_0 : mxxMat2
  w_290_0 : mxxMat2
  w_291_0 : mxxMat2
  w_294_0 : Fin 2 → mxxMat2
  w_295_0 : mxxMat2
  w_297_0 : mxxMat2
  w_301_0 : Int
  w_306_0 : mxxMat2
  w_309_0 : mxxMat2
  w_310_0 : mxxMat2
  w_313_0 : mxxMat2
  w_315_0 : mxxMat2
  w_319_0 : Int
  w_324_0 : mxxMat2
  w_327_0 : mxxMat2
  w_328_0 : mxxMat2
  w_331_0 : mxxMat2
  w_333_0 : mxxMat2
  w_337_0 : Int
  w_342_0 : mxxMat2
  w_345_0 : mxxMat2
  w_346_0 : mxxMat2
  w_349_0 : mxxMat2
  w_351_0 : mxxMat2
  w_355_0 : Int
  w_360_0 : mxxMat2
  w_363_0 : mxxMat2
  w_364_0 : mxxMat2
  w_367_0 : Fin 4 → mxxMat2
  w_368_0 : mxxMat2
  w_370_0 : mxxMat2
  w_376_0 : mxxMat2
  w_377_0 : mxxMat2
  w_380_0 : mxxMat2
  w_382_0 : mxxMat2
  w_388_0 : mxxMat2
  w_389_0 : mxxMat2
  w_392_0 : mxxMat2
  w_394_0 : mxxMat2
  w_400_0 : mxxMat2
  w_401_0 : mxxMat2
  w_404_0 : mxxMat2
  w_406_0 : mxxMat2
  w_412_0 : mxxMat2
  w_413_0 : mxxMat2
  w_416_0 : Fin 4 → mxxMat2
  w_417_0 : mxxMat2
  w_419_0 : mxxMat2
  w_425_0 : mxxMat2
  w_428_0 : mxxMat2
  w_429_0 : mxxMat2
  w_432_0 : mxxMat2
  w_434_0 : mxxMat2
  w_441_0 : mxxMat2
  w_442_0 : mxxMat2
  w_445_0 : mxxMat2
  w_447_0 : mxxMat2
  w_456_0 : mxxMat2
  w_459_0 : mxxMat2
  w_460_0 : mxxMat2
  w_463_0 : mxxMat2
  w_465_0 : mxxMat2
  w_471_0 : mxxMat2
  w_472_0 : mxxMat2

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_9_0 : ByteArray) (w_8_0 : mxxMat1) (w_24_0 : Fin 1 → mxxMat2) (w_37_0 : Fin 1 → mxxMat2) (w_52_0 : Fin 1 → mxxMat2) (w_65_0 : Fin 1 → mxxMat2) (w_80_0 : Fin 1 → mxxMat2) (w_93_0 : Fin 1 → mxxMat2) (w_108_0 : Fin 1 → mxxMat2) (w_121_0 : Fin 1 → mxxMat2) (w_136_0 : Fin 1 → mxxMat2) (w_149_0 : Fin 1 → mxxMat2) (w_164_0 : Fin 1 → mxxMat2) (w_177_0 : Fin 1 → mxxMat2) (w_192_0 : Fin 1 → mxxMat2) (w_205_0 : Fin 1 → mxxMat2) (w_220_0 : Fin 8 → mxxMat2) (w_233_0 : Fin 8 → mxxMat2) (w_234_0 : Fin 4 → mxxMat2) (w_235_0 : Fin 4 → mxxMat2) (w_269_0 : Fin 2 → mxxMat2) (w_294_0 : Fin 2 → mxxMat2) (w_367_0 : Fin 4 → mxxMat2) (w_379_0 : mxxMat2) (w_391_0 : mxxMat2) (w_403_0 : mxxMat2) (w_406_0 : mxxMat2) (w_412_0 : mxxMat2) (w_413_0 : mxxMat2) (w_415_0 : mxxMat2) (w_416_0 : Fin 4 → mxxMat2) (w_417_0 : mxxMat2) (w_419_0 : mxxMat2) (w_425_0 : mxxMat2) (w_428_0 : mxxMat2) (w_429_0 : mxxMat2) (w_431_0 : mxxMat2) (w_432_0 : mxxMat2) (w_434_0 : mxxMat2) (w_441_0 : mxxMat2) (w_442_0 : mxxMat2) (w_444_0 : mxxMat2) (w_445_0 : mxxMat2) (w_447_0 : mxxMat2) (w_456_0 : mxxMat2) (w_459_0 : mxxMat2) (w_460_0 : mxxMat2) (w_462_0 : mxxMat2) (w_463_0 : mxxMat2) (w_465_0 : mxxMat2) (w_471_0 : mxxMat2) (w_472_0 : mxxMat2) (w_474_0 : mxxMat2) (outputs : mxxMat1 × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 2 → mxxMat2) × (Fin 2 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_406_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_412_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_413_0 ∧
  MxxRuntime.familyPack (4) [w_379_0, w_391_0, w_403_0, w_415_0] w_416_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_417_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_419_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_425_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_428_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_429_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_432_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_434_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_441_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_442_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_445_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_447_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_456_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_459_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_460_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_463_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_465_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_471_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_472_0 ∧
  outputs = (w_8_0, (w_24_0, (w_37_0, (w_52_0, (w_65_0, (w_80_0, (w_93_0, (w_108_0, (w_121_0, (w_136_0, (w_149_0, (w_164_0, (w_177_0, (w_192_0, (w_205_0, (w_220_0, (w_233_0, (w_234_0, (w_235_0, (w_269_0, (w_294_0, (w_367_0, (w_416_0, (w_431_0, (w_444_0, (w_462_0, (w_474_0, ())))))))))))))))))))))))))))

abbrev generatedRoot.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_9_0 : ByteArray) (w_242_0 : Fin 5 → Int) (w_312_0 : mxxMat2) (w_313_0 : mxxMat2) (w_315_0 : mxxMat2) (w_319_0 : Int) (w_324_0 : mxxMat2) (w_327_0 : mxxMat2) (w_328_0 : mxxMat2) (w_330_0 : mxxMat2) (w_331_0 : mxxMat2) (w_333_0 : mxxMat2) (w_337_0 : Int) (w_342_0 : mxxMat2) (w_345_0 : mxxMat2) (w_346_0 : mxxMat2) (w_348_0 : mxxMat2) (w_349_0 : mxxMat2) (w_351_0 : mxxMat2) (w_355_0 : Int) (w_360_0 : mxxMat2) (w_363_0 : mxxMat2) (w_364_0 : mxxMat2) (w_366_0 : mxxMat2) (w_367_0 : Fin 4 → mxxMat2) (w_368_0 : mxxMat2) (w_370_0 : mxxMat2) (w_376_0 : mxxMat2) (w_377_0 : mxxMat2) (w_380_0 : mxxMat2) (w_382_0 : mxxMat2) (w_388_0 : mxxMat2) (w_389_0 : mxxMat2) (w_392_0 : mxxMat2) (w_394_0 : mxxMat2) (w_400_0 : mxxMat2) (w_401_0 : mxxMat2) (w_404_0 : mxxMat2) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_313_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_315_0 ∧
  MxxRuntime.familyGetStatic w_242_0 (2) w_319_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_324_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_327_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_328_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_331_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_333_0 ∧
  MxxRuntime.familyGetStatic w_242_0 (3) w_337_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_342_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_345_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_346_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_349_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_351_0 ∧
  MxxRuntime.familyGetStatic w_242_0 (4) w_355_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_360_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_363_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_364_0 ∧
  MxxRuntime.familyPack (4) [w_312_0, w_330_0, w_348_0, w_366_0] w_367_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_368_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_370_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_376_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_377_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_380_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_382_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_388_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_389_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_392_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_394_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_400_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_401_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_404_0

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_9_0 : ByteArray) (w_242_0 : Fin 5 → Int) (w_36_0 : mxxMat2) (w_64_0 : mxxMat2) (w_79_0 : mxxMat2) (w_92_0 : mxxMat2) (w_107_0 : mxxMat2) (w_120_0 : mxxMat2) (w_135_0 : mxxMat2) (w_148_0 : mxxMat2) (w_176_0 : mxxMat2) (w_204_0 : mxxMat2) (w_219_0 : mxxMat2) (w_229_0 : mxxMat2) (w_230_0 : mxxMat2) (w_232_0 : mxxMat2) (w_233_0 : Fin 8 → mxxMat2) (w_234_0 : Fin 4 → mxxMat2) (w_235_0 : Fin 4 → mxxMat2) (w_236_0 : mxxMat2) (w_238_0 : mxxMat2) (w_243_0 : Int) (w_248_0 : mxxMat2) (w_251_0 : mxxMat2) (w_252_0 : mxxMat2) (w_254_0 : mxxMat2) (w_255_0 : mxxMat2) (w_257_0 : mxxMat2) (w_262_0 : mxxMat2) (w_265_0 : mxxMat2) (w_266_0 : mxxMat2) (w_268_0 : mxxMat2) (w_269_0 : Fin 2 → mxxMat2) (w_270_0 : mxxMat2) (w_272_0 : mxxMat2) (w_278_0 : mxxMat2) (w_279_0 : mxxMat2) (w_281_0 : mxxMat2) (w_282_0 : mxxMat2) (w_284_0 : mxxMat2) (w_290_0 : mxxMat2) (w_291_0 : mxxMat2) (w_293_0 : mxxMat2) (w_294_0 : Fin 2 → mxxMat2) (w_295_0 : mxxMat2) (w_297_0 : mxxMat2) (w_301_0 : Int) (w_306_0 : mxxMat2) (w_309_0 : mxxMat2) (w_310_0 : mxxMat2) : Prop :=
  MxxRuntime.uniformIntervalSample (0) (1) w_229_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_230_0 ∧
  MxxRuntime.familyPack (8) [w_232_0, w_120_0, w_148_0, w_176_0, w_204_0, w_36_0, w_64_0, w_92_0] w_233_0 ∧
  MxxRuntime.familyPack (4) [w_219_0, w_107_0, w_135_0, w_79_0] w_234_0 ∧
  MxxRuntime.familyPack (4) [w_232_0, w_120_0, w_148_0, w_92_0] w_235_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_236_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_238_0 ∧
  MxxRuntime.familyGetStatic w_242_0 (0) w_243_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_248_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_251_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_252_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_255_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_257_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_262_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_265_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_266_0 ∧
  MxxRuntime.familyPack (2) [w_254_0, w_268_0] w_269_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_270_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_272_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_278_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_279_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_282_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_284_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_290_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_291_0 ∧
  MxxRuntime.familyPack (2) [w_281_0, w_293_0] w_294_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_295_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_297_0 ∧
  MxxRuntime.familyGetStatic w_242_0 (1) w_301_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_306_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_309_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_310_0

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : mxxMat3) (w_9_0 : ByteArray) (w_23_0 : mxxMat2) (w_51_0 : mxxMat2) (w_79_0 : mxxMat2) (w_107_0 : mxxMat2) (w_135_0 : mxxMat2) (w_150_0 : Int) (w_153_0 : mxxMat2) (w_156_0 : mxxMat3) (w_157_0 : mxxMat2) (w_160_0 : mxxMat2) (w_161_0 : mxxMat2) (w_163_0 : mxxMat2) (w_164_0 : Fin 1 → mxxMat2) (w_165_0 : mxxMat2) (w_167_0 : mxxMat2) (w_173_0 : mxxMat2) (w_174_0 : mxxMat2) (w_176_0 : mxxMat2) (w_177_0 : Fin 1 → mxxMat2) (w_178_0 : Int) (w_179_0 : mxxMat2) (w_181_0 : mxxMat2) (w_184_0 : mxxMat3) (w_185_0 : mxxMat2) (w_188_0 : mxxMat2) (w_189_0 : mxxMat2) (w_191_0 : mxxMat2) (w_192_0 : Fin 1 → mxxMat2) (w_193_0 : mxxMat2) (w_195_0 : mxxMat2) (w_201_0 : mxxMat2) (w_202_0 : mxxMat2) (w_204_0 : mxxMat2) (w_205_0 : Fin 1 → mxxMat2) (w_206_0 : Int) (w_207_0 : mxxMat2) (w_209_0 : mxxMat2) (w_212_0 : mxxMat3) (w_213_0 : mxxMat2) (w_216_0 : mxxMat2) (w_217_0 : mxxMat2) (w_219_0 : mxxMat2) (w_220_0 : Fin 8 → mxxMat2) (w_221_0 : mxxMat2) (w_223_0 : mxxMat2) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_150_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_153_0 ∧
  MxxRuntime.ringAutomorphismRuns (7) w_0_0 w_156_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_157_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_160_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_161_0 ∧
  MxxRuntime.familyPack (1) [w_163_0] w_164_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_150_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_165_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_150_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_167_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_173_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_174_0 ∧
  MxxRuntime.familyPack (1) [w_176_0] w_177_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_178_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_179_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_178_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_181_0 ∧
  MxxRuntime.ringAutomorphismRuns (9) w_0_0 w_184_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_185_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_188_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_189_0 ∧
  MxxRuntime.familyPack (1) [w_191_0] w_192_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_178_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_193_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_178_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_195_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_201_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_202_0 ∧
  MxxRuntime.familyPack (1) [w_204_0] w_205_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_206_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_207_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_206_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_209_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_0_0 w_212_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_213_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_216_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_217_0 ∧
  MxxRuntime.familyPack (8) [w_219_0, w_107_0, w_135_0, w_163_0, w_191_0, w_23_0, w_51_0, w_79_0] w_220_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_206_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_221_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_206_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_223_0

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : mxxMat3) (w_9_0 : ByteArray) (w_66_0 : Int) (w_77_0 : mxxMat2) (w_79_0 : mxxMat2) (w_80_0 : Fin 1 → mxxMat2) (w_81_0 : mxxMat2) (w_83_0 : mxxMat2) (w_89_0 : mxxMat2) (w_90_0 : mxxMat2) (w_92_0 : mxxMat2) (w_93_0 : Fin 1 → mxxMat2) (w_94_0 : Int) (w_95_0 : mxxMat2) (w_97_0 : mxxMat2) (w_100_0 : mxxMat3) (w_101_0 : mxxMat2) (w_104_0 : mxxMat2) (w_105_0 : mxxMat2) (w_107_0 : mxxMat2) (w_108_0 : Fin 1 → mxxMat2) (w_109_0 : mxxMat2) (w_111_0 : mxxMat2) (w_117_0 : mxxMat2) (w_118_0 : mxxMat2) (w_120_0 : mxxMat2) (w_121_0 : Fin 1 → mxxMat2) (w_122_0 : Int) (w_123_0 : mxxMat2) (w_125_0 : mxxMat2) (w_128_0 : mxxMat3) (w_129_0 : mxxMat2) (w_132_0 : mxxMat2) (w_133_0 : mxxMat2) (w_135_0 : mxxMat2) (w_136_0 : Fin 1 → mxxMat2) (w_137_0 : mxxMat2) (w_139_0 : mxxMat2) (w_145_0 : mxxMat2) (w_146_0 : mxxMat2) (w_148_0 : mxxMat2) (w_149_0 : Fin 1 → mxxMat2) (w_150_0 : Int) (w_151_0 : mxxMat2) : Prop :=
  MxxRuntime.uniformIntervalSample (0) (1) w_77_0 ∧
  MxxRuntime.familyPack (1) [w_79_0] w_80_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_66_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_81_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_66_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_83_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_89_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_90_0 ∧
  MxxRuntime.familyPack (1) [w_92_0] w_93_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_94_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_95_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_94_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_97_0 ∧
  MxxRuntime.ringAutomorphismRuns (3) w_0_0 w_100_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_101_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_104_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_105_0 ∧
  MxxRuntime.familyPack (1) [w_107_0] w_108_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_94_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_109_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_94_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_111_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_117_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_118_0 ∧
  MxxRuntime.familyPack (1) [w_120_0] w_121_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_122_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_123_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_122_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_125_0 ∧
  MxxRuntime.ringAutomorphismRuns (5) w_0_0 w_128_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_129_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_132_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_133_0 ∧
  MxxRuntime.familyPack (1) [w_135_0] w_136_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_122_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_137_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_122_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_139_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_145_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_146_0 ∧
  MxxRuntime.familyPack (1) [w_148_0] w_149_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_150_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_151_0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (w_0_0 : mxxMat3) (w_9_0 : ByteArray) (w_1_0 : mxxMat3) (w_2_0 : mxxMat0) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : Int) (w_11_0 : mxxMat2) (w_13_0 : mxxMat2) (w_16_0 : mxxMat3) (w_17_0 : mxxMat2) (w_20_0 : mxxMat2) (w_21_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : Fin 1 → mxxMat2) (w_25_0 : mxxMat2) (w_27_0 : mxxMat2) (w_33_0 : mxxMat2) (w_34_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : Fin 1 → mxxMat2) (w_38_0 : Int) (w_39_0 : mxxMat2) (w_41_0 : mxxMat2) (w_44_0 : mxxMat3) (w_45_0 : mxxMat2) (w_48_0 : mxxMat2) (w_49_0 : mxxMat2) (w_51_0 : mxxMat2) (w_52_0 : Fin 1 → mxxMat2) (w_53_0 : mxxMat2) (w_55_0 : mxxMat2) (w_61_0 : mxxMat2) (w_62_0 : mxxMat2) (w_64_0 : mxxMat2) (w_65_0 : Fin 1 → mxxMat2) (w_66_0 : Int) (w_67_0 : mxxMat2) (w_69_0 : mxxMat2) (w_72_0 : mxxMat3) (w_73_0 : mxxMat2) (w_76_0 : mxxMat2) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_5_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_10_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_10_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_13_0 ∧
  MxxRuntime.ringAutomorphismRuns (11) w_0_0 w_16_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_17_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_20_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_21_0 ∧
  MxxRuntime.familyPack (1) [w_23_0] w_24_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_10_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_25_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_10_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_27_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_33_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_34_0 ∧
  MxxRuntime.familyPack (1) [w_36_0] w_37_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_38_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_39_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_38_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_41_0 ∧
  MxxRuntime.ringAutomorphismRuns (13) w_0_0 w_44_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_45_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_48_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_49_0 ∧
  MxxRuntime.familyPack (1) [w_51_0] w_52_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_38_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_53_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_38_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_55_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_61_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_62_0 ∧
  MxxRuntime.familyPack (1) [w_64_0] w_65_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_66_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_67_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_66_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_69_0 ∧
  MxxRuntime.ringAutomorphismRuns (15) w_0_0 w_72_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_73_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_76_0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : mxxMat3 × mxxMat4 × ByteArray × (Fin 5 → Int) × mxxMat3 × Int × Unit) (outputs : mxxMat1 × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 2 → mxxMat2) × (Fin 2 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_3_0 : mxxMat4 := inputs.2.1
  let w_9_0 : ByteArray := inputs.2.2.1
  let w_242_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_423_0 : mxxMat3 := inputs.2.2.2.2.1
  let w_451_0 : Int := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat3 := (1 : mxxMat3)
  let w_2_0 := witness.w_2_0
  let w_4_0 : mxxMat1 := MxxRuntime.matrixMul w_2_0 w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat1 := MxxRuntime.matrixSub w_5_0 w_6_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
  let w_10_0 : Int := 11
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_13_0
  let w_15_0 : mxxMat2 := MxxRuntime.matrixAdd w_12_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_16_0 w_17_0
  let w_19_0 : mxxMat2 := MxxRuntime.matrixSub w_15_0 w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat2 := MxxRuntime.matrixSub w_20_0 w_21_0
  let w_23_0 : mxxMat2 := MxxRuntime.matrixAdd w_19_0 w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_27_0
  let w_29_0 : mxxMat2 := MxxRuntime.matrixAdd w_26_0 w_28_0
  let w_30_0 : mxxMat3 := (1 : mxxMat3)
  let w_31_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_30_0 w_17_0
  let w_32_0 : mxxMat2 := MxxRuntime.matrixSub w_29_0 w_31_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 : mxxMat2 := MxxRuntime.matrixSub w_33_0 w_34_0
  let w_36_0 : mxxMat2 := MxxRuntime.matrixAdd w_32_0 w_35_0
  let w_37_0 := witness.w_37_0
  let w_38_0 : Int := 13
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_41_0
  let w_43_0 : mxxMat2 := MxxRuntime.matrixAdd w_40_0 w_42_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_44_0 w_45_0
  let w_47_0 : mxxMat2 := MxxRuntime.matrixSub w_43_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 : mxxMat2 := MxxRuntime.matrixSub w_48_0 w_49_0
  let w_51_0 : mxxMat2 := MxxRuntime.matrixAdd w_47_0 w_50_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_53_0
  let w_55_0 := witness.w_55_0
  let w_56_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_55_0
  let w_57_0 : mxxMat2 := MxxRuntime.matrixAdd w_54_0 w_56_0
  let w_58_0 : mxxMat3 := (1 : mxxMat3)
  let w_59_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_58_0 w_45_0
  let w_60_0 : mxxMat2 := MxxRuntime.matrixSub w_57_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 : mxxMat2 := MxxRuntime.matrixSub w_61_0 w_62_0
  let w_64_0 : mxxMat2 := MxxRuntime.matrixAdd w_60_0 w_63_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : Int := 15
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_67_0
  let w_69_0 := witness.w_69_0
  let w_70_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_69_0
  let w_71_0 : mxxMat2 := MxxRuntime.matrixAdd w_68_0 w_70_0
  let w_72_0 := witness.w_72_0
  let w_73_0 := witness.w_73_0
  let w_74_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_72_0 w_73_0
  let w_75_0 : mxxMat2 := MxxRuntime.matrixSub w_71_0 w_74_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat2 := MxxRuntime.matrixSub w_76_0 w_77_0
  let w_79_0 : mxxMat2 := MxxRuntime.matrixAdd w_75_0 w_78_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_83_0
  let w_85_0 : mxxMat2 := MxxRuntime.matrixAdd w_82_0 w_84_0
  let w_86_0 : mxxMat3 := (1 : mxxMat3)
  let w_87_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_86_0 w_73_0
  let w_88_0 : mxxMat2 := MxxRuntime.matrixSub w_85_0 w_87_0
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 : mxxMat2 := MxxRuntime.matrixSub w_89_0 w_90_0
  let w_92_0 : mxxMat2 := MxxRuntime.matrixAdd w_88_0 w_91_0
  let w_93_0 := witness.w_93_0
  let w_94_0 : Int := 3
  let w_95_0 := witness.w_95_0
  let w_96_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_95_0
  let w_97_0 := witness.w_97_0
  let w_98_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_97_0
  let w_99_0 : mxxMat2 := MxxRuntime.matrixAdd w_96_0 w_98_0
  let w_100_0 := witness.w_100_0
  let w_101_0 := witness.w_101_0
  let w_102_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_100_0 w_101_0
  let w_103_0 : mxxMat2 := MxxRuntime.matrixSub w_99_0 w_102_0
  let w_104_0 := witness.w_104_0
  let w_105_0 := witness.w_105_0
  let w_106_0 : mxxMat2 := MxxRuntime.matrixSub w_104_0 w_105_0
  let w_107_0 : mxxMat2 := MxxRuntime.matrixAdd w_103_0 w_106_0
  let w_108_0 := witness.w_108_0
  let w_109_0 := witness.w_109_0
  let w_110_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_109_0
  let w_111_0 := witness.w_111_0
  let w_112_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_111_0
  let w_113_0 : mxxMat2 := MxxRuntime.matrixAdd w_110_0 w_112_0
  let w_114_0 : mxxMat3 := (1 : mxxMat3)
  let w_115_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_114_0 w_101_0
  let w_116_0 : mxxMat2 := MxxRuntime.matrixSub w_113_0 w_115_0
  let w_117_0 := witness.w_117_0
  let w_118_0 := witness.w_118_0
  let w_119_0 : mxxMat2 := MxxRuntime.matrixSub w_117_0 w_118_0
  let w_120_0 : mxxMat2 := MxxRuntime.matrixAdd w_116_0 w_119_0
  let w_121_0 := witness.w_121_0
  let w_122_0 : Int := 5
  let w_123_0 := witness.w_123_0
  let w_124_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_123_0
  let w_125_0 := witness.w_125_0
  let w_126_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_125_0
  let w_127_0 : mxxMat2 := MxxRuntime.matrixAdd w_124_0 w_126_0
  let w_128_0 := witness.w_128_0
  let w_129_0 := witness.w_129_0
  let w_130_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_128_0 w_129_0
  let w_131_0 : mxxMat2 := MxxRuntime.matrixSub w_127_0 w_130_0
  let w_132_0 := witness.w_132_0
  let w_133_0 := witness.w_133_0
  let w_134_0 : mxxMat2 := MxxRuntime.matrixSub w_132_0 w_133_0
  let w_135_0 : mxxMat2 := MxxRuntime.matrixAdd w_131_0 w_134_0
  let w_136_0 := witness.w_136_0
  let w_137_0 := witness.w_137_0
  let w_138_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_137_0
  let w_139_0 := witness.w_139_0
  let w_140_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_139_0
  let w_141_0 : mxxMat2 := MxxRuntime.matrixAdd w_138_0 w_140_0
  let w_142_0 : mxxMat3 := (1 : mxxMat3)
  let w_143_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_142_0 w_129_0
  let w_144_0 : mxxMat2 := MxxRuntime.matrixSub w_141_0 w_143_0
  let w_145_0 := witness.w_145_0
  let w_146_0 := witness.w_146_0
  let w_147_0 : mxxMat2 := MxxRuntime.matrixSub w_145_0 w_146_0
  let w_148_0 : mxxMat2 := MxxRuntime.matrixAdd w_144_0 w_147_0
  let w_149_0 := witness.w_149_0
  let w_150_0 : Int := 7
  let w_151_0 := witness.w_151_0
  let w_152_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_151_0
  let w_153_0 := witness.w_153_0
  let w_154_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_153_0
  let w_155_0 : mxxMat2 := MxxRuntime.matrixAdd w_152_0 w_154_0
  let w_156_0 := witness.w_156_0
  let w_157_0 := witness.w_157_0
  let w_158_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_156_0 w_157_0
  let w_159_0 : mxxMat2 := MxxRuntime.matrixSub w_155_0 w_158_0
  let w_160_0 := witness.w_160_0
  let w_161_0 := witness.w_161_0
  let w_162_0 : mxxMat2 := MxxRuntime.matrixSub w_160_0 w_161_0
  let w_163_0 : mxxMat2 := MxxRuntime.matrixAdd w_159_0 w_162_0
  let w_164_0 := witness.w_164_0
  let w_165_0 := witness.w_165_0
  let w_166_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_165_0
  let w_167_0 := witness.w_167_0
  let w_168_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_167_0
  let w_169_0 : mxxMat2 := MxxRuntime.matrixAdd w_166_0 w_168_0
  let w_170_0 : mxxMat3 := (1 : mxxMat3)
  let w_171_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_170_0 w_157_0
  let w_172_0 : mxxMat2 := MxxRuntime.matrixSub w_169_0 w_171_0
  let w_173_0 := witness.w_173_0
  let w_174_0 := witness.w_174_0
  let w_175_0 : mxxMat2 := MxxRuntime.matrixSub w_173_0 w_174_0
  let w_176_0 : mxxMat2 := MxxRuntime.matrixAdd w_172_0 w_175_0
  let w_177_0 := witness.w_177_0
  let w_178_0 : Int := 9
  let w_179_0 := witness.w_179_0
  let w_180_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_179_0
  let w_181_0 := witness.w_181_0
  let w_182_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_181_0
  let w_183_0 : mxxMat2 := MxxRuntime.matrixAdd w_180_0 w_182_0
  let w_184_0 := witness.w_184_0
  let w_185_0 := witness.w_185_0
  let w_186_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_184_0 w_185_0
  let w_187_0 : mxxMat2 := MxxRuntime.matrixSub w_183_0 w_186_0
  let w_188_0 := witness.w_188_0
  let w_189_0 := witness.w_189_0
  let w_190_0 : mxxMat2 := MxxRuntime.matrixSub w_188_0 w_189_0
  let w_191_0 : mxxMat2 := MxxRuntime.matrixAdd w_187_0 w_190_0
  let w_192_0 := witness.w_192_0
  let w_193_0 := witness.w_193_0
  let w_194_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_193_0
  let w_195_0 := witness.w_195_0
  let w_196_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_195_0
  let w_197_0 : mxxMat2 := MxxRuntime.matrixAdd w_194_0 w_196_0
  let w_198_0 : mxxMat3 := (1 : mxxMat3)
  let w_199_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_198_0 w_185_0
  let w_200_0 : mxxMat2 := MxxRuntime.matrixSub w_197_0 w_199_0
  let w_201_0 := witness.w_201_0
  let w_202_0 := witness.w_202_0
  let w_203_0 : mxxMat2 := MxxRuntime.matrixSub w_201_0 w_202_0
  let w_204_0 : mxxMat2 := MxxRuntime.matrixAdd w_200_0 w_203_0
  let w_205_0 := witness.w_205_0
  let w_206_0 : Int := 1
  let w_207_0 := witness.w_207_0
  let w_208_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_207_0
  let w_209_0 := witness.w_209_0
  let w_210_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_209_0
  let w_211_0 : mxxMat2 := MxxRuntime.matrixAdd w_208_0 w_210_0
  let w_212_0 := witness.w_212_0
  let w_213_0 := witness.w_213_0
  let w_214_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_212_0 w_213_0
  let w_215_0 : mxxMat2 := MxxRuntime.matrixSub w_211_0 w_214_0
  let w_216_0 := witness.w_216_0
  let w_217_0 := witness.w_217_0
  let w_218_0 : mxxMat2 := MxxRuntime.matrixSub w_216_0 w_217_0
  let w_219_0 : mxxMat2 := MxxRuntime.matrixAdd w_215_0 w_218_0
  let w_220_0 := witness.w_220_0
  let w_221_0 := witness.w_221_0
  let w_222_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_221_0
  let w_223_0 := witness.w_223_0
  let w_224_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_223_0
  let w_225_0 : mxxMat2 := MxxRuntime.matrixAdd w_222_0 w_224_0
  let w_226_0 : mxxMat3 := (1 : mxxMat3)
  let w_227_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_226_0 w_213_0
  let w_228_0 : mxxMat2 := MxxRuntime.matrixSub w_225_0 w_227_0
  let w_229_0 := witness.w_229_0
  let w_230_0 := witness.w_230_0
  let w_231_0 : mxxMat2 := MxxRuntime.matrixSub w_229_0 w_230_0
  let w_232_0 : mxxMat2 := MxxRuntime.matrixAdd w_228_0 w_231_0
  let w_233_0 := witness.w_233_0
  let w_234_0 := witness.w_234_0
  let w_235_0 := witness.w_235_0
  let w_236_0 := witness.w_236_0
  let w_237_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_236_0
  let w_238_0 := witness.w_238_0
  let w_239_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_238_0
  let w_240_0 : mxxMat2 := MxxRuntime.matrixAdd w_237_0 w_239_0
  let w_241_0 : mxxMat3 := (1 : mxxMat3)
  let w_243_0 := witness.w_243_0
  let w_244_0 : Int := 1
  let w_245_0 : Int := (w_243_0 * w_244_0)
  let w_246_0 : mxxMat3 := MxxRuntime.multiplyMonomial w_241_0 w_245_0
  let w_247_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_246_0 w_0_0
  let w_248_0 := witness.w_248_0
  let w_249_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_247_0 w_248_0
  let w_250_0 : mxxMat2 := MxxRuntime.matrixSub w_240_0 w_249_0
  let w_251_0 := witness.w_251_0
  let w_252_0 := witness.w_252_0
  let w_253_0 : mxxMat2 := MxxRuntime.matrixSub w_251_0 w_252_0
  let w_254_0 : mxxMat2 := MxxRuntime.matrixAdd w_250_0 w_253_0
  let w_255_0 := witness.w_255_0
  let w_256_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_255_0
  let w_257_0 := witness.w_257_0
  let w_258_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_257_0
  let w_259_0 : mxxMat2 := MxxRuntime.matrixAdd w_256_0 w_258_0
  let w_260_0 : mxxMat3 := (1 : mxxMat3)
  let w_261_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_260_0 w_0_0
  let w_262_0 := witness.w_262_0
  let w_263_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_261_0 w_262_0
  let w_264_0 : mxxMat2 := MxxRuntime.matrixSub w_259_0 w_263_0
  let w_265_0 := witness.w_265_0
  let w_266_0 := witness.w_266_0
  let w_267_0 : mxxMat2 := MxxRuntime.matrixSub w_265_0 w_266_0
  let w_268_0 : mxxMat2 := MxxRuntime.matrixAdd w_264_0 w_267_0
  let w_269_0 := witness.w_269_0
  let w_270_0 := witness.w_270_0
  let w_271_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_270_0
  let w_272_0 := witness.w_272_0
  let w_273_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_272_0
  let w_274_0 : mxxMat2 := MxxRuntime.matrixAdd w_271_0 w_273_0
  let w_275_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_246_0 w_1_0
  let w_276_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_275_0 w_248_0
  let w_277_0 : mxxMat2 := MxxRuntime.matrixSub w_274_0 w_276_0
  let w_278_0 := witness.w_278_0
  let w_279_0 := witness.w_279_0
  let w_280_0 : mxxMat2 := MxxRuntime.matrixSub w_278_0 w_279_0
  let w_281_0 : mxxMat2 := MxxRuntime.matrixAdd w_277_0 w_280_0
  let w_282_0 := witness.w_282_0
  let w_283_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_282_0
  let w_284_0 := witness.w_284_0
  let w_285_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_284_0
  let w_286_0 : mxxMat2 := MxxRuntime.matrixAdd w_283_0 w_285_0
  let w_287_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_260_0 w_1_0
  let w_288_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_287_0 w_262_0
  let w_289_0 : mxxMat2 := MxxRuntime.matrixSub w_286_0 w_288_0
  let w_290_0 := witness.w_290_0
  let w_291_0 := witness.w_291_0
  let w_292_0 : mxxMat2 := MxxRuntime.matrixSub w_290_0 w_291_0
  let w_293_0 : mxxMat2 := MxxRuntime.matrixAdd w_289_0 w_292_0
  let w_294_0 := witness.w_294_0
  let w_295_0 := witness.w_295_0
  let w_296_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_295_0
  let w_297_0 := witness.w_297_0
  let w_298_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_297_0
  let w_299_0 : mxxMat2 := MxxRuntime.matrixAdd w_296_0 w_298_0
  let w_300_0 : mxxMat3 := (1 : mxxMat3)
  let w_301_0 := witness.w_301_0
  let w_302_0 : Int := 1
  let w_303_0 : Int := (w_301_0 * w_302_0)
  let w_304_0 : mxxMat3 := MxxRuntime.multiplyMonomial w_300_0 w_303_0
  let w_305_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_304_0 w_0_0
  let w_306_0 := witness.w_306_0
  let w_307_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_305_0 w_306_0
  let w_308_0 : mxxMat2 := MxxRuntime.matrixSub w_299_0 w_307_0
  let w_309_0 := witness.w_309_0
  let w_310_0 := witness.w_310_0
  let w_311_0 : mxxMat2 := MxxRuntime.matrixSub w_309_0 w_310_0
  let w_312_0 : mxxMat2 := MxxRuntime.matrixAdd w_308_0 w_311_0
  let w_313_0 := witness.w_313_0
  let w_314_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_313_0
  let w_315_0 := witness.w_315_0
  let w_316_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_315_0
  let w_317_0 : mxxMat2 := MxxRuntime.matrixAdd w_314_0 w_316_0
  let w_318_0 : mxxMat3 := (1 : mxxMat3)
  let w_319_0 := witness.w_319_0
  let w_320_0 : Int := 1
  let w_321_0 : Int := (w_319_0 * w_320_0)
  let w_322_0 : mxxMat3 := MxxRuntime.multiplyMonomial w_318_0 w_321_0
  let w_323_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_322_0 w_0_0
  let w_324_0 := witness.w_324_0
  let w_325_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_323_0 w_324_0
  let w_326_0 : mxxMat2 := MxxRuntime.matrixSub w_317_0 w_325_0
  let w_327_0 := witness.w_327_0
  let w_328_0 := witness.w_328_0
  let w_329_0 : mxxMat2 := MxxRuntime.matrixSub w_327_0 w_328_0
  let w_330_0 : mxxMat2 := MxxRuntime.matrixAdd w_326_0 w_329_0
  let w_331_0 := witness.w_331_0
  let w_332_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_331_0
  let w_333_0 := witness.w_333_0
  let w_334_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_333_0
  let w_335_0 : mxxMat2 := MxxRuntime.matrixAdd w_332_0 w_334_0
  let w_336_0 : mxxMat3 := (1 : mxxMat3)
  let w_337_0 := witness.w_337_0
  let w_338_0 : Int := 1
  let w_339_0 : Int := (w_337_0 * w_338_0)
  let w_340_0 : mxxMat3 := MxxRuntime.multiplyMonomial w_336_0 w_339_0
  let w_341_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_340_0 w_0_0
  let w_342_0 := witness.w_342_0
  let w_343_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_341_0 w_342_0
  let w_344_0 : mxxMat2 := MxxRuntime.matrixSub w_335_0 w_343_0
  let w_345_0 := witness.w_345_0
  let w_346_0 := witness.w_346_0
  let w_347_0 : mxxMat2 := MxxRuntime.matrixSub w_345_0 w_346_0
  let w_348_0 : mxxMat2 := MxxRuntime.matrixAdd w_344_0 w_347_0
  let w_349_0 := witness.w_349_0
  let w_350_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_349_0
  let w_351_0 := witness.w_351_0
  let w_352_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_351_0
  let w_353_0 : mxxMat2 := MxxRuntime.matrixAdd w_350_0 w_352_0
  let w_354_0 : mxxMat3 := (1 : mxxMat3)
  let w_355_0 := witness.w_355_0
  let w_356_0 : Int := 1
  let w_357_0 : Int := (w_355_0 * w_356_0)
  let w_358_0 : mxxMat3 := MxxRuntime.multiplyMonomial w_354_0 w_357_0
  let w_359_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_358_0 w_0_0
  let w_360_0 := witness.w_360_0
  let w_361_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_359_0 w_360_0
  let w_362_0 : mxxMat2 := MxxRuntime.matrixSub w_353_0 w_361_0
  let w_363_0 := witness.w_363_0
  let w_364_0 := witness.w_364_0
  let w_365_0 : mxxMat2 := MxxRuntime.matrixSub w_363_0 w_364_0
  let w_366_0 : mxxMat2 := MxxRuntime.matrixAdd w_362_0 w_365_0
  let w_367_0 := witness.w_367_0
  let w_368_0 := witness.w_368_0
  let w_369_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_368_0
  let w_370_0 := witness.w_370_0
  let w_371_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_370_0
  let w_372_0 : mxxMat2 := MxxRuntime.matrixAdd w_369_0 w_371_0
  let w_373_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_304_0 w_1_0
  let w_374_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_373_0 w_306_0
  let w_375_0 : mxxMat2 := MxxRuntime.matrixSub w_372_0 w_374_0
  let w_376_0 := witness.w_376_0
  let w_377_0 := witness.w_377_0
  let w_378_0 : mxxMat2 := MxxRuntime.matrixSub w_376_0 w_377_0
  let w_379_0 : mxxMat2 := MxxRuntime.matrixAdd w_375_0 w_378_0
  let w_380_0 := witness.w_380_0
  let w_381_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_380_0
  let w_382_0 := witness.w_382_0
  let w_383_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_382_0
  let w_384_0 : mxxMat2 := MxxRuntime.matrixAdd w_381_0 w_383_0
  let w_385_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_322_0 w_1_0
  let w_386_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_385_0 w_324_0
  let w_387_0 : mxxMat2 := MxxRuntime.matrixSub w_384_0 w_386_0
  let w_388_0 := witness.w_388_0
  let w_389_0 := witness.w_389_0
  let w_390_0 : mxxMat2 := MxxRuntime.matrixSub w_388_0 w_389_0
  let w_391_0 : mxxMat2 := MxxRuntime.matrixAdd w_387_0 w_390_0
  let w_392_0 := witness.w_392_0
  let w_393_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_392_0
  let w_394_0 := witness.w_394_0
  let w_395_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_394_0
  let w_396_0 : mxxMat2 := MxxRuntime.matrixAdd w_393_0 w_395_0
  let w_397_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_340_0 w_1_0
  let w_398_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_397_0 w_342_0
  let w_399_0 : mxxMat2 := MxxRuntime.matrixSub w_396_0 w_398_0
  let w_400_0 := witness.w_400_0
  let w_401_0 := witness.w_401_0
  let w_402_0 : mxxMat2 := MxxRuntime.matrixSub w_400_0 w_401_0
  let w_403_0 : mxxMat2 := MxxRuntime.matrixAdd w_399_0 w_402_0
  let w_404_0 := witness.w_404_0
  let w_405_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_404_0
  let w_406_0 := witness.w_406_0
  let w_407_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_406_0
  let w_408_0 : mxxMat2 := MxxRuntime.matrixAdd w_405_0 w_407_0
  let w_409_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_358_0 w_1_0
  let w_410_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_409_0 w_360_0
  let w_411_0 : mxxMat2 := MxxRuntime.matrixSub w_408_0 w_410_0
  let w_412_0 := witness.w_412_0
  let w_413_0 := witness.w_413_0
  let w_414_0 : mxxMat2 := MxxRuntime.matrixSub w_412_0 w_413_0
  let w_415_0 : mxxMat2 := MxxRuntime.matrixAdd w_411_0 w_414_0
  let w_416_0 := witness.w_416_0
  let w_417_0 := witness.w_417_0
  let w_418_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_417_0
  let w_419_0 := witness.w_419_0
  let w_420_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_419_0
  let w_421_0 : mxxMat2 := MxxRuntime.matrixAdd w_418_0 w_420_0
  let w_422_0 : mxxMat3 := (1 : mxxMat3)
  let w_424_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_422_0 w_423_0
  let w_425_0 := witness.w_425_0
  let w_426_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_424_0 w_425_0
  let w_427_0 : mxxMat2 := MxxRuntime.matrixSub w_421_0 w_426_0
  let w_428_0 := witness.w_428_0
  let w_429_0 := witness.w_429_0
  let w_430_0 : mxxMat2 := MxxRuntime.matrixSub w_428_0 w_429_0
  let w_431_0 : mxxMat2 := MxxRuntime.matrixAdd w_427_0 w_430_0
  let w_432_0 := witness.w_432_0
  let w_433_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_432_0
  let w_434_0 := witness.w_434_0
  let w_435_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_434_0
  let w_436_0 : mxxMat2 := MxxRuntime.matrixAdd w_433_0 w_435_0
  let w_437_0 : mxxMat3 := (1 : mxxMat3)
  let w_438_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_422_0 w_437_0
  let w_439_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_438_0 w_425_0
  let w_440_0 : mxxMat2 := MxxRuntime.matrixSub w_436_0 w_439_0
  let w_441_0 := witness.w_441_0
  let w_442_0 := witness.w_442_0
  let w_443_0 : mxxMat2 := MxxRuntime.matrixSub w_441_0 w_442_0
  let w_444_0 : mxxMat2 := MxxRuntime.matrixAdd w_440_0 w_443_0
  let w_445_0 := witness.w_445_0
  let w_446_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_445_0
  let w_447_0 := witness.w_447_0
  let w_448_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_447_0
  let w_449_0 : mxxMat2 := MxxRuntime.matrixAdd w_446_0 w_448_0
  let w_450_0 : mxxMat3 := (1 : mxxMat3)
  let w_452_0 : Int := 2
  let w_453_0 : Int := (w_451_0 * w_452_0)
  let w_454_0 : mxxMat3 := MxxRuntime.multiplyMonomial w_450_0 w_453_0
  let w_455_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_454_0 w_423_0
  let w_456_0 := witness.w_456_0
  let w_457_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_455_0 w_456_0
  let w_458_0 : mxxMat2 := MxxRuntime.matrixSub w_449_0 w_457_0
  let w_459_0 := witness.w_459_0
  let w_460_0 := witness.w_460_0
  let w_461_0 : mxxMat2 := MxxRuntime.matrixSub w_459_0 w_460_0
  let w_462_0 : mxxMat2 := MxxRuntime.matrixAdd w_458_0 w_461_0
  let w_463_0 := witness.w_463_0
  let w_464_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_0_0 w_463_0
  let w_465_0 := witness.w_465_0
  let w_466_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_465_0
  let w_467_0 : mxxMat2 := MxxRuntime.matrixAdd w_464_0 w_466_0
  let w_468_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_454_0 w_437_0
  let w_469_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_468_0 w_456_0
  let w_470_0 : mxxMat2 := MxxRuntime.matrixSub w_467_0 w_469_0
  let w_471_0 := witness.w_471_0
  let w_472_0 := witness.w_472_0
  let w_473_0 : mxxMat2 := MxxRuntime.matrixSub w_471_0 w_472_0
  let w_474_0 : mxxMat2 := MxxRuntime.matrixAdd w_470_0 w_473_0
  generatedRoot.constraints_0 backend hashModel w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_13_0 w_16_0 w_17_0 w_20_0 w_21_0 w_23_0 w_24_0 w_25_0 w_27_0 w_33_0 w_34_0 w_36_0 w_37_0 w_38_0 w_39_0 w_41_0 w_44_0 w_45_0 w_48_0 w_49_0 w_51_0 w_52_0 w_53_0 w_55_0 w_61_0 w_62_0 w_64_0 w_65_0 w_66_0 w_67_0 w_69_0 w_72_0 w_73_0 w_76_0 ∧
    generatedRoot.constraints_1 backend hashModel w_0_0 w_9_0 w_66_0 w_77_0 w_79_0 w_80_0 w_81_0 w_83_0 w_89_0 w_90_0 w_92_0 w_93_0 w_94_0 w_95_0 w_97_0 w_100_0 w_101_0 w_104_0 w_105_0 w_107_0 w_108_0 w_109_0 w_111_0 w_117_0 w_118_0 w_120_0 w_121_0 w_122_0 w_123_0 w_125_0 w_128_0 w_129_0 w_132_0 w_133_0 w_135_0 w_136_0 w_137_0 w_139_0 w_145_0 w_146_0 w_148_0 w_149_0 w_150_0 w_151_0 ∧
    generatedRoot.constraints_2 backend hashModel w_0_0 w_9_0 w_23_0 w_51_0 w_79_0 w_107_0 w_135_0 w_150_0 w_153_0 w_156_0 w_157_0 w_160_0 w_161_0 w_163_0 w_164_0 w_165_0 w_167_0 w_173_0 w_174_0 w_176_0 w_177_0 w_178_0 w_179_0 w_181_0 w_184_0 w_185_0 w_188_0 w_189_0 w_191_0 w_192_0 w_193_0 w_195_0 w_201_0 w_202_0 w_204_0 w_205_0 w_206_0 w_207_0 w_209_0 w_212_0 w_213_0 w_216_0 w_217_0 w_219_0 w_220_0 w_221_0 w_223_0 ∧
    generatedRoot.constraints_3 backend hashModel w_9_0 w_242_0 w_36_0 w_64_0 w_79_0 w_92_0 w_107_0 w_120_0 w_135_0 w_148_0 w_176_0 w_204_0 w_219_0 w_229_0 w_230_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_238_0 w_243_0 w_248_0 w_251_0 w_252_0 w_254_0 w_255_0 w_257_0 w_262_0 w_265_0 w_266_0 w_268_0 w_269_0 w_270_0 w_272_0 w_278_0 w_279_0 w_281_0 w_282_0 w_284_0 w_290_0 w_291_0 w_293_0 w_294_0 w_295_0 w_297_0 w_301_0 w_306_0 w_309_0 w_310_0 ∧
    generatedRoot.constraints_4 backend hashModel w_9_0 w_242_0 w_312_0 w_313_0 w_315_0 w_319_0 w_324_0 w_327_0 w_328_0 w_330_0 w_331_0 w_333_0 w_337_0 w_342_0 w_345_0 w_346_0 w_348_0 w_349_0 w_351_0 w_355_0 w_360_0 w_363_0 w_364_0 w_366_0 w_367_0 w_368_0 w_370_0 w_376_0 w_377_0 w_380_0 w_382_0 w_388_0 w_389_0 w_392_0 w_394_0 w_400_0 w_401_0 w_404_0 ∧
    generatedRoot.constraints_5 backend hashModel w_9_0 w_8_0 w_24_0 w_37_0 w_52_0 w_65_0 w_80_0 w_93_0 w_108_0 w_121_0 w_136_0 w_149_0 w_164_0 w_177_0 w_192_0 w_205_0 w_220_0 w_233_0 w_234_0 w_235_0 w_269_0 w_294_0 w_367_0 w_379_0 w_391_0 w_403_0 w_406_0 w_412_0 w_413_0 w_415_0 w_416_0 w_417_0 w_419_0 w_425_0 w_428_0 w_429_0 w_431_0 w_432_0 w_434_0 w_441_0 w_442_0 w_444_0 w_445_0 w_447_0 w_456_0 w_459_0 w_460_0 w_462_0 w_463_0 w_465_0 w_471_0 w_472_0 w_474_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : mxxMat3 × mxxMat4 × ByteArray × (Fin 5 → Int) × mxxMat3 × Int × Unit) (outputs : mxxMat1 × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 1 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 8 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 2 → mxxMat2) × (Fin 2 → mxxMat2) × (Fin 4 → mxxMat2) × (Fin 4 → mxxMat2) × mxxMat2 × mxxMat2 × mxxMat2 × mxxMat2 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end Stage_initial
