import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_prelude

structure Params where
  unit : Unit

local notation "mxxNat0" => 71693316984649896843904253443489203902220097476285449024776635275984659786482860843809654567991896462655489
local notation "mxxNat1" => 72341562789056444115684979687416806180395197053951206217914228266014512572833794748552875371413453242085860536342419038795864559652387809846630825946845083145910068784615984026523621685186503403424255574409217
local notation "mxxNat2" => 35846658492324948421952126721744601951110048738142724512388317637992329893241430421904827283995948231327744

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

/-- The modulus of a matrix type. -/
abbrev modulus_1 : Nat := mxxNat1

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 32768 1 30
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_1 32768 1 295

abbrev parallel_generatedRoot_4.constraints_0 (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat0) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 45 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])) w_0_0 w_8_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [12]) ((319/100)) (31) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 45 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(257), (513), (769), (1025), (1281), (1537), (1793), (2049), (2305), (2561), (2817), (3073), (3329), (3585), (3841), (4097), (8193), (12289), (16385), (20481), (24577), (28673), (32769), (36865), (40961), (45057), (49153), (53249), (57345), (61441), (61697), (61953), (62209), (62465), (62721), (62977), (63233), (63489), (63745), (64001), (64257), (64513), (64769), (65025), (65281)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [20]) ((319/100)) (31) w_20_0 ∧
  outputs = (w_13_0, (w_21_0, ()))

def parallel_generatedRoot_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_9_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      ∃ (w_8_0 : mxxMat0),
        let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
        let w_11_0 : mxxMat1 := MxxRuntime.matrixSub w_7_0 w_10_0
        ∃ (w_12_0 : mxxMat1),
          let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_11_0 w_12_0
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_14_0
            ∃ (w_16_0 : mxxMat1),
              let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_16_0
              let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_15_0 w_17_0
              let w_19_0 : mxxMat1 := MxxRuntime.matrixSub w_18_0 w_9_0
              ∃ (w_20_0 : mxxMat1),
                let w_21_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_20_0
                parallel_generatedRoot_4.constraints_0 hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_5_0 w_8_0 w_12_0 w_13_0 w_14_0 w_16_0 w_20_0 w_21_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_7_0 : ByteArray) (w_1_0 : mxxMat1) (w_8_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_16_0 : mxxMat1) (w_19_0 : mxxMat1) (w_20_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 4096 30 w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_7_0) w_8_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [11]) ((319/100)) (31) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [55, 51, 50, 48, 54, 98, 48, 49, 52, 57, 53, 52, 99, 53, 53, 100, 101, 48, 99, 53, 52, 55, 54, 57, 100, 102, 98, 56, 50, 52, 51, 53, 98, 52, 101, 54, 100, 98, 100, 55, 49, 50, 53, 102, 54, 50, 53, 53, 50, 52, 50, 56, 53, 52, 49, 102, 102, 51, 50, 57, 48, 49, 51, 52], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_7_0) w_16_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [19]) ((319/100)) (31) w_19_0 ∧
  outputs = (w_12_0, (w_20_0, ()))

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × Int × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_4_0 : Int := inputs.2.1
  let w_7_0 : ByteArray := inputs.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.1
  ∃ (w_1_0 : mxxMat1),
    let w_2_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_1_0
    let w_3_0 : mxxMat0 := (1 : mxxMat0)
    let w_5_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
    let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_2_0 w_5_0
    ∃ (w_8_0 : mxxMat1),
      let w_9_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_8_0
      let w_10_0 : mxxMat1 := MxxRuntime.matrixSub w_6_0 w_9_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_10_0 w_11_0
        let w_14_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_13_0 w_1_0
        let w_15_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_14_0 w_5_0
        ∃ (w_16_0 : mxxMat1),
          let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_16_0
          let w_18_0 : mxxMat1 := MxxRuntime.matrixSub w_15_0 w_17_0
          ∃ (w_19_0 : mxxMat1),
            let w_20_0 : mxxMat1 := MxxRuntime.matrixAdd w_18_0 w_19_0
            parallel_generatedRoot_6.constraints_0 backend hashModel tape path i_0 w_7_0 w_1_0 w_8_0 w_11_0 w_12_0 w_16_0 w_19_0 w_20_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : mxxMat1
  w_4_0 : Fin 45 → mxxMat1
  w_4_1 : Fin 45 → mxxMat1
  w_6_0 : Fin 2 → mxxMat1
  w_6_1 : Fin 2 → mxxMat1
  w_7_0 : mxxMat2
  w_10_0 : mxxMat3
  w_12_0 : mxxMat3
  w_14_0 : mxxMat0
  w_16_0 : mxxMat0
  w_18_0 : mxxMat0
  w_22_0 : mxxMat0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_5_0 : Fin 2 → Int) (w_21_0 : Fin 32768 → Int) (w_2_0 : mxxMat0) (w_3_0 : mxxMat1) (w_4_0 : Fin 45 → mxxMat1) (w_4_1 : Fin 45 → mxxMat1) (w_6_0 : Fin 2 → mxxMat1) (w_6_1 : Fin 2 → mxxMat1) (w_7_0 : mxxMat2) (w_10_0 : mxxMat3) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (outputs : (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × mxxMat3 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 4096 30 w_3_0 ∧
  (45) = 45 ∧
  (∀ i : Fin 45, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ()))) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_6 backend hashModel tape (path ++ [6, i.val]) params i (w_0_0, ((w_5_0 i), (w_1_0, (w_2_0, ())))) ((w_6_0 i), ((w_6_1 i), ()))) ∧
  MxxRuntime.centeredRebaseRuns w_0_0 w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_1_0) w_10_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [12]) ((319/100)) (31) w_12_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) (w_1_0) w_14_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [51, 52, 101, 99, 56, 49, 55, 48, 55, 98, 100, 100, 54, 99, 102, 98, 102, 99, 99, 52, 50, 51, 99, 55, 99, 98, 54, 97, 53, 51, 51, 100, 52, 57, 98, 102, 98, 101, 99, 56, 98, 57, 100, 97, 57, 101, 98, 98, 56, 48, 97, 52, 54, 57, 52, 51, 53, 53, 52, 48, 99, 101, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) (w_1_0) w_16_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [18]) ((319/100)) (31) w_18_0 ∧
  MxxRuntime.polynomialFromValues false w_21_0 w_22_0 ∧
  outputs = (w_4_0, (w_4_1, (w_6_0, (w_6_1, (w_13_0, (w_24_0, ()))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × ByteArray × (Fin 2 → Int) × mxxMat3 × (Fin 32768 → Int) × Unit) (outputs : (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × mxxMat3 × mxxMat0 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : Fin 2 → Int := inputs.2.2.1
  let w_8_0 : mxxMat3 := inputs.2.2.2.1
  let w_21_0 : Fin 32768 → Int := inputs.2.2.2.2.1
  let w_2_0 : mxxMat0 := (1 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_6_0 := witness.w_6_0
  let w_6_1 := witness.w_6_1
  let w_7_0 := witness.w_7_0
  let w_9_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat3 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat3 := MxxRuntime.matrixAdd w_11_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_17_0 w_18_0
  let w_20_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat0)
  let w_22_0 := witness.w_22_0
  let w_23_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_20_0 w_22_0
  let w_24_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_23_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_5_0 w_21_0 w_2_0 w_3_0 w_4_0 w_4_1 w_6_0 w_6_1 w_7_0 w_10_0 w_12_0 w_13_0 w_14_0 w_16_0 w_18_0 w_22_0 w_24_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × ByteArray × (Fin 2 → Int) × mxxMat3 × (Fin 32768 → Int) × Unit) (outputs : (Fin 45 → mxxMat1) × (Fin 45 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × mxxMat3 × mxxMat0 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_prelude
