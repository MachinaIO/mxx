import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_encrypt

structure Params where
  unit : Unit

local notation "mxxNat0" => 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489
local notation "mxxNat1" => 72341562789056444115684979687416806180395197053951206217914228266014512572833794748552875371413453242085860536342419038795864559652387809846630825946845083145910068784615984026523621685186503403424255574409217
local notation "mxxNat2" => 1009041927918404748912066953523025233623263629931203392403660676397045184266783237099217072052453834753

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := mxxNat1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 60
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 32768 30 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 30
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 59

abbrev parallel_generatedRoot_5.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_3_0 : ByteArray) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_27_0 : mxxMat2) (w_28_0 : mxxMat1) (w_29_0 : mxxMat3) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) (w_37_0 : mxxMat3) (w_43_0 : mxxMat3) (w_44_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × Unit) : Prop :=
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (60)) ∧
  (Int.fmod ((Int.ofNat i_0)) (60)) < (1 + (Int.fmod ((Int.ofNat i_0)) (60))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (60))) ≤ 60 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (Int.fmod ((Int.ofNat i_0)) (60)) (1 + (Int.fmod ((Int.ofNat i_0)) (60))) w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 30 w_26_0 w_27_0 ∧
  MxxRuntime.centeredRebaseRuns w_28_0 w_29_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [33]) ((319/100)) (31) w_33_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_3_0) w_35_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_3_0) w_37_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [43]) ((319/100)) (31) w_43_0 ∧
  outputs = (w_34_0, (w_44_0, ()))

abbrev parallel_generatedRoot_5.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_3_0 : ByteArray) (w_15_0 : Fin 45 → mxxMat4) (w_21_0 : Fin 45 → mxxMat4) (w_12_0 : mxxMat5) (w_13_0 : mxxMat4) (w_14_0 : mxxMat4) (w_16_0 : Int) (w_17_0 : mxxMat4) (w_18_0 : mxxMat4) (w_19_0 : mxxMat4) (w_20_0 : mxxMat4) (w_22_0 : mxxMat4) (w_23_0 : mxxMat4) (w_24_0 : mxxMat4) (w_25_0 : mxxMat0) : Prop :=
  (29 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) < (59 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) ∧
  (59 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) ≤ 59 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (29 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) (59 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) w_13_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(65281), (65025), (64769), (64513), (64257), (64001), (63745), (63489), (63233), (62977), (62721), (62465), (62209), (61953), (61697), (61441), (57345), (53249), (49153), (45057), (40961), (36865), (32769), (28673), (24577), (20481), (16385), (12289), (8193), (4097)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_3_0) w_14_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 45 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_16_0 w_17_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])) w_18_0 w_19_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(65281), (65025), (64769), (64513), (64257), (64001), (63745), (63489), (63233), (62977), (62721), (62465), (62209), (61953), (61697), (61441), (57345), (53249), (49153), (45057), (40961), (36865), (32769), (28673), (24577), (20481), (16385), (12289), (8193), (4097)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_3_0) w_20_0 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 45 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_16_0 w_22_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])) w_23_0 w_24_0 ∧
  MxxRuntime.concatColumns w_19_0 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1

abbrev parallel_generatedRoot_5.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat1) (w_3_0 : ByteArray) (w_6_0 : mxxMat1) (w_1_0 : mxxMat1) (w_2_0 : mxxMat3) (w_4_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) : Prop :=
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])) w_0_0 w_1_0 ∧
  MxxRuntime.centeredRebaseRuns w_1_0 w_2_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_3_0) w_4_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1800))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1800))) < 30 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_3_0) w_8_0 ∧
  0 ≤ (29) ∧
  (29) < 59 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  (30) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  (30) ≠ 0 ∧
  0 ≤ (29 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30))))

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × (Fin 45 → mxxMat4) × (Fin 45 → mxxMat4) × Unit) (outputs : mxxMat3 × mxxMat3 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_3_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat1 := inputs.2.2.1
  let w_15_0 : Fin 45 → mxxMat4 := inputs.2.2.2.1
  let w_21_0 : Fin 45 → mxxMat4 := inputs.2.2.2.2.1
  ∃ (w_1_0 : mxxMat1),
    ∃ (w_2_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat3),
        let w_5_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_7_0 : mxxMat3),
          ∃ (w_8_0 : mxxMat3),
            let w_9_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
            let w_10_0 : mxxMat3 := MxxRuntime.matrixAdd w_5_0 w_9_0
            let w_11_0 : mxxMat3 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat3)
            let w_12_0 : mxxMat5 := (MxxRuntime.unitRow (29) : mxxMat5)
            ∃ (w_13_0 : mxxMat4),
              ∃ (w_14_0 : mxxMat4),
                let w_16_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1800))) [(44), (43), (42), (41), (40), (39), (38), (37), (36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
                ∃ (w_17_0 : mxxMat4),
                  let w_18_0 : mxxMat4 := MxxRuntime.matrixSub w_14_0 w_17_0
                  ∃ (w_19_0 : mxxMat4),
                    ∃ (w_20_0 : mxxMat4),
                      ∃ (w_22_0 : mxxMat4),
                        let w_23_0 : mxxMat4 := MxxRuntime.matrixSub w_20_0 w_22_0
                        ∃ (w_24_0 : mxxMat4),
                          ∃ (w_25_0 : mxxMat0),
                            ∃ (w_26_0 : mxxMat1),
                              ∃ (w_27_0 : mxxMat2),
                                let w_28_0 : mxxMat1 := MxxRuntime.matrixMul w_13_0 w_27_0
                                ∃ (w_29_0 : mxxMat3),
                                  let w_30_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_29_0 w_2_0
                                  let w_31_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_11_0 w_30_0
                                  let w_32_0 : mxxMat3 := MxxRuntime.matrixSub w_10_0 w_31_0
                                  ∃ (w_33_0 : mxxMat3),
                                    let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_32_0 w_33_0
                                    ∃ (w_35_0 : mxxMat3),
                                      let w_36_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_2_0 w_35_0
                                      ∃ (w_37_0 : mxxMat3),
                                        let w_38_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_7_0 w_37_0
                                        let w_39_0 : mxxMat3 := MxxRuntime.matrixAdd w_36_0 w_38_0
                                        let w_40_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_29_0 w_7_0
                                        let w_41_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_11_0 w_40_0
                                        let w_42_0 : mxxMat3 := MxxRuntime.matrixSub w_39_0 w_41_0
                                        ∃ (w_43_0 : mxxMat3),
                                          let w_44_0 : mxxMat3 := MxxRuntime.matrixAdd w_42_0 w_43_0
                                          parallel_generatedRoot_5.constraints_0 hashModel i_0 w_0_0 w_3_0 w_6_0 w_1_0 w_2_0 w_4_0 w_7_0 w_8_0 ∧
    parallel_generatedRoot_5.constraints_1 hashModel i_0 w_3_0 w_15_0 w_21_0 w_12_0 w_13_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_25_0 ∧
    parallel_generatedRoot_5.constraints_2 backend hashModel tape path i_0 w_3_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 w_34_0 w_35_0 w_37_0 w_43_0 w_44_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (w_0_0 : mxxMat4) (w_1_0 : mxxMat4) (outputs : mxxMat4 × mxxMat4 × Unit) : Prop :=
  outputs = (w_0_0, (w_1_0, ()))

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (inputs : mxxMat4 × mxxMat4 × Unit) (outputs : mxxMat4 × mxxMat4 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat4 := inputs.1
  let w_1_0 : mxxMat4 := inputs.2.1
  parallel_generatedRoot_6.constraints_0 w_0_0 w_1_0 outputs

abbrev parallel_generatedRoot_37.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_2_0 : ByteArray) (w_16_0 : Fin 2 → mxxMat4) (w_14_0 : Int) (w_15_0 : mxxMat4) (w_17_0 : mxxMat4) (w_18_0 : mxxMat0) (w_19_0 : mxxMat1) (w_20_0 : mxxMat2) (w_21_0 : mxxMat1) (w_22_0 : mxxMat3) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat3) (w_30_0 : mxxMat3) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × Unit) : Prop :=
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_14_0 w_17_0 ∧
  MxxRuntime.concatColumns w_15_0 w_17_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (60)) ∧
  (Int.fmod ((Int.ofNat i_0)) (60)) < (1 + (Int.fmod ((Int.ofNat i_0)) (60))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (60))) ≤ 60 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 (Int.fmod ((Int.ofNat i_0)) (60)) (1 + (Int.fmod ((Int.ofNat i_0)) (60))) w_19_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 4096 30 w_19_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_21_0 w_22_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [26]) ((319/100)) (31) w_26_0 ∧
  (1800) ≠ 0 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (1800))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_28_0 ∧
  (1800) ≠ 0 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (1800))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_30_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [36]) ((319/100)) (31) w_36_0 ∧
  outputs = (w_27_0, (w_37_0, ()))

abbrev parallel_generatedRoot_37.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat1) (w_2_0 : ByteArray) (w_5_0 : mxxMat1) (w_13_0 : Fin 2 → mxxMat4) (w_1_0 : mxxMat3) (w_3_0 : mxxMat3) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_11_0 : mxxMat5) (w_12_0 : mxxMat4) (w_14_0 : Int) (w_15_0 : mxxMat4) : Prop :=
  MxxRuntime.centeredRebaseRuns w_0_0 w_1_0 ∧
  (1800) ≠ 0 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (1800))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_5_0 w_6_0 ∧
  (1800) ≠ 0 ∧
  (60) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (1800))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (60))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_7_0 ∧
  0 ≤ (29) ∧
  (29) < 59 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  (30) ≠ 0 ∧
  (60) ≠ 0 ∧
  (30) ≠ 0 ∧
  (30) ≠ 0 ∧
  0 ≤ (29 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) ∧
  (29 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) < (59 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) ∧
  (59 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) ≤ 59 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (29 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) (59 + (-1 * (Int.fmod ((30 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (60))) (30)))) (30)))) w_12_0 ∧
  (1800) ≠ 0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_14_0 w_15_0 ∧
  0 ≤ w_14_0

def parallel_generatedRoot_37 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × ByteArray × mxxMat1 × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × Unit) (outputs : mxxMat3 × mxxMat3 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat1 := inputs.2.2.1
  let w_13_0 : Fin 2 → mxxMat4 := inputs.2.2.2.1
  let w_16_0 : Fin 2 → mxxMat4 := inputs.2.2.2.2.1
  ∃ (w_1_0 : mxxMat3),
    ∃ (w_3_0 : mxxMat3),
      let w_4_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_1_0 w_3_0
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat3),
          let w_8_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_6_0 w_7_0
          let w_9_0 : mxxMat3 := MxxRuntime.matrixAdd w_4_0 w_8_0
          let w_10_0 : mxxMat3 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat3)
          let w_11_0 : mxxMat5 := (MxxRuntime.unitRow (29) : mxxMat5)
          ∃ (w_12_0 : mxxMat4),
            let w_14_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (1800))
            ∃ (w_15_0 : mxxMat4),
              ∃ (w_17_0 : mxxMat4),
                ∃ (w_18_0 : mxxMat0),
                  ∃ (w_19_0 : mxxMat1),
                    ∃ (w_20_0 : mxxMat2),
                      let w_21_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_20_0
                      ∃ (w_22_0 : mxxMat3),
                        let w_23_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_22_0 w_1_0
                        let w_24_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_10_0 w_23_0
                        let w_25_0 : mxxMat3 := MxxRuntime.matrixSub w_9_0 w_24_0
                        ∃ (w_26_0 : mxxMat3),
                          let w_27_0 : mxxMat3 := MxxRuntime.matrixAdd w_25_0 w_26_0
                          ∃ (w_28_0 : mxxMat3),
                            let w_29_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_1_0 w_28_0
                            ∃ (w_30_0 : mxxMat3),
                              let w_31_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_6_0 w_30_0
                              let w_32_0 : mxxMat3 := MxxRuntime.matrixAdd w_29_0 w_31_0
                              let w_33_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_22_0 w_6_0
                              let w_34_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_10_0 w_33_0
                              let w_35_0 : mxxMat3 := MxxRuntime.matrixSub w_32_0 w_34_0
                              ∃ (w_36_0 : mxxMat3),
                                let w_37_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_36_0
                                parallel_generatedRoot_37.constraints_0 hashModel i_0 w_0_0 w_2_0 w_5_0 w_13_0 w_1_0 w_3_0 w_6_0 w_7_0 w_11_0 w_12_0 w_14_0 w_15_0 ∧
    parallel_generatedRoot_37.constraints_1 backend hashModel tape path i_0 w_2_0 w_16_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_26_0 w_27_0 w_28_0 w_30_0 w_36_0 w_37_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_5_0 : Fin 54000 → mxxMat3
  w_5_1 : Fin 54000 → mxxMat3
  w_7_0 : mxxMat4
  w_9_0 : mxxMat4
  w_12_0 : mxxMat4
  w_13_0 : mxxMat4
  w_23_0 : mxxMat4
  w_25_0 : mxxMat4
  w_27_0 : mxxMat4
  w_33_0 : mxxMat4
  w_37_0 : Fin 3600 → mxxMat3
  w_37_1 : Fin 3600 → mxxMat3

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat1) (w_1_0 : ByteArray) (w_3_0 : Fin 45 → mxxMat4) (w_4_0 : Fin 45 → mxxMat4) (w_35_0 : Fin 2 → mxxMat4) (w_36_0 : Fin 2 → mxxMat4) (w_2_0 : mxxMat1) (w_5_0 : Fin 54000 → mxxMat3) (w_5_1 : Fin 54000 → mxxMat3) (w_6_0 : Fin 45 → mxxMat4) (w_6_1 : Fin 45 → mxxMat4) (w_7_0 : mxxMat4) (w_9_0 : mxxMat4) (w_12_0 : mxxMat4) (w_13_0 : mxxMat4) (w_23_0 : mxxMat4) (w_24_0 : mxxMat4) (w_25_0 : mxxMat4) (w_27_0 : mxxMat4) (w_33_0 : mxxMat4) (w_34_0 : mxxMat4) (w_37_0 : Fin 3600 → mxxMat3) (w_37_1 : Fin 3600 → mxxMat3) (outputs : (Fin 54000 → mxxMat3) × (Fin 54000 → mxxMat3) × (Fin 45 → mxxMat4) × (Fin 45 → mxxMat4) × mxxMat4 × mxxMat4 × (Fin 3600 → mxxMat3) × (Fin 3600 → mxxMat3) × Unit) : Prop :=
  (54000) = 54000 ∧
  (∀ i : Fin 54000, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_4_0, ()))))) ((w_5_0 i), ((w_5_1 i), ()))) ∧
  (45) = 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 30 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_12_0 w_13_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [23]) ((319/100)) (31) w_23_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_25_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_27_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [33]) ((319/100)) (31) w_33_0 ∧
  (3600) = 3600 ∧
  (∀ i : Fin 3600, parallel_generatedRoot_37 backend hashModel tape (path ++ [37, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_35_0, (w_36_0, ()))))) ((w_37_0 i), ((w_37_1 i), ()))) ∧
  outputs = (w_5_0, (w_5_1, (w_6_0, (w_6_1, (w_24_0, (w_34_0, (w_37_0, (w_37_1, ()))))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1 × ByteArray × (Fin 45 → mxxMat4) × (Fin 45 → mxxMat4) × Int × Int × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × Unit) (outputs : (Fin 54000 → mxxMat3) × (Fin 54000 → mxxMat3) × (Fin 45 → mxxMat4) × (Fin 45 → mxxMat4) × mxxMat4 × mxxMat4 × (Fin 3600 → mxxMat3) × (Fin 3600 → mxxMat3) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_3_0 : Fin 45 → mxxMat4 := inputs.2.2.1
  let w_4_0 : Fin 45 → mxxMat4 := inputs.2.2.2.1
  let w_16_0 : Int := inputs.2.2.2.2.1
  let w_18_0 : Int := inputs.2.2.2.2.2.1
  let w_35_0 : Fin 2 → mxxMat4 := inputs.2.2.2.2.2.2.1
  let w_36_0 : Fin 2 → mxxMat4 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : mxxMat1 := (1 : mxxMat1)
  let w_5_0 := witness.w_5_0
  let w_5_1 := witness.w_5_1
  let w_6_0 : Fin 45 → mxxMat4 := w_3_0
  let w_6_1 : Fin 45 → mxxMat4 := w_4_0
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_0_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_2_0 w_9_0
  let w_11_0 : mxxMat4 := MxxRuntime.matrixAdd w_8_0 w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_0_0 w_13_0
  let w_15_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat1)
  let w_17_0 : mxxMat1 := MxxRuntime.liftInteger w_16_0
  let w_19_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_17_0 w_18_0
  let w_20_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_15_0 w_19_0
  let w_21_0 : mxxMat4 := MxxRuntime.matrixMulScalarRight w_14_0 w_20_0
  let w_22_0 : mxxMat4 := MxxRuntime.matrixSub w_11_0 w_21_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat4 := MxxRuntime.matrixAdd w_22_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_0_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_2_0 w_27_0
  let w_29_0 : mxxMat4 := MxxRuntime.matrixAdd w_26_0 w_28_0
  let w_30_0 : mxxMat4 := MxxRuntime.matrixMulScalarLeft w_2_0 w_13_0
  let w_31_0 : mxxMat4 := MxxRuntime.matrixMulScalarRight w_30_0 w_20_0
  let w_32_0 : mxxMat4 := MxxRuntime.matrixSub w_29_0 w_31_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat4 := MxxRuntime.matrixAdd w_32_0 w_33_0
  let w_37_0 := witness.w_37_0
  let w_37_1 := witness.w_37_1
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_3_0 w_4_0 w_35_0 w_36_0 w_2_0 w_5_0 w_5_1 w_6_0 w_6_1 w_7_0 w_9_0 w_12_0 w_13_0 w_23_0 w_24_0 w_25_0 w_27_0 w_33_0 w_34_0 w_37_0 w_37_1 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1 × ByteArray × (Fin 45 → mxxMat4) × (Fin 45 → mxxMat4) × Int × Int × (Fin 2 → mxxMat4) × (Fin 2 → mxxMat4) × Unit) (outputs : (Fin 54000 → mxxMat3) × (Fin 54000 → mxxMat3) × (Fin 45 → mxxMat4) × (Fin 45 → mxxMat4) × mxxMat4 × mxxMat4 × (Fin 3600 → mxxMat3) × (Fin 3600 → mxxMat3) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_encrypt
