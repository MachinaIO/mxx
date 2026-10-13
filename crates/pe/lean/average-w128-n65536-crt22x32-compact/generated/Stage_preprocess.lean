import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_preprocess

structure Params where
  unit : Unit

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxNat2" => 77469527647209439214259235281224611949048149835750629526602555268330821212381343721480441750241968331304616539815191276988146174287613644747658263498964462003172995682625109900434768370549872016137303635774952449
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 88
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 87
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 88
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 44
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 175
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 1
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 22
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1936
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 43
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix mxxNat0 65536 2 1
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix mxxNat0 65536 88 1
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 105
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 9240
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 209
local notation "mxxMat17" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 44
local notation "mxxMat18" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 11176

abbrev sequential_sequential_generatedRoot_10_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_10_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_10_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_10.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_10_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_10.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_10.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_10.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_10.Witness,
    sequential_generatedRoot_10.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_11_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_11_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_11_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_11.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_11.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_11_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_11.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_11.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_11.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_11.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_11.Witness,
    sequential_generatedRoot_11.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_14_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_14_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_14_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_14.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_14.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_14_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_14.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_14.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_14.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_14.Witness,
    sequential_generatedRoot_14.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_15_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_15_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_15_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_15.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_15.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_15_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_15.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_15.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_15.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_15.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_15.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_15.Witness,
    sequential_generatedRoot_15.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_18_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_18_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_18_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_18.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_18.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_18_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_18.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_18.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_18.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_18.Witness,
    sequential_generatedRoot_18.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_19_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_19_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_19_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_19.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_19.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_19_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_19.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_19.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_19.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_19.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_19.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_19 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_19.Witness,
    sequential_generatedRoot_19.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_22_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_22_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_22_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_22.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_22.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_22_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_22.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_22.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_22.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_22 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_22.Witness,
    sequential_generatedRoot_22.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_23_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_23_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_23_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_23.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_23.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_23_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_23.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_23.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_23.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_23.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_23.Witness,
    sequential_generatedRoot_23.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_44_14.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat0) (w_11_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_5_0 : mxxMat3) (w_9_0 : mxxMat0) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_2_0 w_3_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = (w_9_0, (w_15_0, ()))

def sequential_sequential_generatedRoot_44_14 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_10_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_6_0 : mxxMat4 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_3_0 : mxxMat1),
      let w_4_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
      ∃ (w_5_0 : mxxMat3),
        let w_7_0 : mxxMat0 := MxxRuntime.matrixMul w_5_0 w_6_0
        let w_8_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_3_0 w_7_0
        let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_8_0
        ∃ (w_12_0 : mxxMat1),
          ∃ (w_13_0 : mxxMat1),
            let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_13_0 w_7_0
            let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_14_0
            sequential_sequential_generatedRoot_44_14.constraints_0 i_0 i_1 w_1_0 w_11_0 w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_44.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat4
  w_12_0 : mxxMat4
  w_14_0 : mxxMat0
  w_14_1 : mxxMat0
  w_15_0 : mxxMat10
  w_18_0 : mxxMat10

abbrev sequential_generatedRoot_44.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_13_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat4) (w_12_0 : mxxMat4) (w_14_0 : mxxMat0) (w_14_1 : mxxMat0) (w_15_0 : mxxMat10) (w_16_0 : mxxMat10) (w_18_0 : mxxMat10) (w_19_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (21 + (-1 * (Int.ofNat i_0))) ∧
  (21 + (-1 * (Int.ofNat i_0))) < (43 + (-1 * (Int.ofNat i_0))) ∧
  (43 + (-1 * (Int.ofNat i_0))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.ofNat i_0))) (43 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_11_0 w_12_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_44_14 backend hashModel tape (path ++ [14, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_12_0, (w_13_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_14_0, (w_14_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_14_0 w_15_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_14_1 w_18_0 ∧
  outputs = (w_16_0, (w_19_0, ()))

abbrev sequential_generatedRoot_44.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_44.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_17_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_15_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat10 := MxxRuntime.matrixAdd w_17_0 w_18_0
  sequential_generatedRoot_44.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_13_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_14_1 w_15_0 w_16_0 w_18_0 w_19_0 outputs

def sequential_generatedRoot_44 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_44.Witness,
    sequential_generatedRoot_44.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_45_13.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_45_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_45_13.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_45_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_45_13.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_45.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat8
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat8
  w_13_0 : mxxMat0
  w_13_1 : mxxMat0
  w_14_0 : mxxMat10
  w_17_0 : mxxMat10

abbrev sequential_generatedRoot_45.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_9_0 : mxxMat0) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_10_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_45_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_8_0, (w_12_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_13_0, (w_13_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_13_0 w_14_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_13_1 w_17_0 ∧
  outputs = (w_15_0, (w_18_0, ()))

abbrev sequential_generatedRoot_45.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_9_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_7_0 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_45.constraints_1 backend hashModel tape path params i_0 w_4_0 w_9_0 w_2_0 w_3_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

abbrev sequential_generatedRoot_45.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_45.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_16_0 : mxxMat10 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_13_1 := witness.w_13_1
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat10 := MxxRuntime.matrixAdd w_16_0 w_17_0
  sequential_generatedRoot_45.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_9_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

def sequential_generatedRoot_45 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_45.Witness,
    sequential_generatedRoot_45.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_47_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_parallel_generatedRoot_47_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_parallel_generatedRoot_47_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_47.Witness where
  w_2_0 : mxxMat3
  w_3_0 : mxxMat3
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat3
  w_13_0 : mxxMat3
  w_14_0 : mxxMat0
  w_15_0 : mxxMat3
  w_16_0 : mxxMat3
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_18_1 : mxxMat0
  w_19_0 : mxxMat3
  w_20_0 : mxxMat3
  w_21_0 : mxxMat3
  w_22_0 : mxxMat3

abbrev parallel_generatedRoot_47.constraints_2 (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_20_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 0 44 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 44 88 w_22_0 ∧
  outputs = (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))

abbrev parallel_generatedRoot_47.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 0 44 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 44 88 w_13_0 ∧
  MxxRuntime.concatColumns w_12_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_47_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_9_0, (w_14_0, (w_17_0, ())))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, ())) (w_18_0, (w_18_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_47.constraints_2 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_47.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_5_0 : mxxMat10) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_2_0 : mxxMat3) (w_3_0 : mxxMat3) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.concatColumns w_2_0 w_3_0 w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 0 44 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 44 88 w_8_0 ∧
  MxxRuntime.concatColumns w_7_0 w_8_0 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_47.constraints_1 backend hashModel tape path params i_0 w_1_0 w_10_0 w_0_0 w_4_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_47.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : parallel_generatedRoot_47.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : ByteArray := inputs.1
  let w_5_0 : mxxMat10 := inputs.2.1
  let w_10_0 : mxxMat10 := inputs.2.2.1
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  parallel_generatedRoot_47.constraints_0 backend hashModel tape path params i_0 w_1_0 w_5_0 w_10_0 w_0_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

def parallel_generatedRoot_47 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_47.Witness,
    parallel_generatedRoot_47.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_48_16.constraints_1 (i_0 : Nat) (i_1 : Nat) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_22_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  outputs = (w_14_0, (w_25_0, ()))

abbrev sequential_sequential_generatedRoot_48_16.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_4_0 : Fin 15 → mxxMat3) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat3) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat2) (w_10_0 : mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  sequential_sequential_generatedRoot_48_16.constraints_1 i_0 i_1 w_16_0 w_19_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

def sequential_sequential_generatedRoot_48_16 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × mxxMat4 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_11_0 : mxxMat4 := inputs.2.2.2.2.1
  let w_16_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  ∃ (w_3_0 : mxxMat3),
    ∃ (w_5_0 : mxxMat3),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_8_0 : mxxMat1),
            let w_9_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
            ∃ (w_10_0 : mxxMat3),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_12_0
              let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
              let w_17_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
              ∃ (w_18_0 : mxxMat3),
                ∃ (w_20_0 : mxxMat3),
                  ∃ (w_21_0 : mxxMat0),
                    ∃ (w_22_0 : mxxMat1),
                      ∃ (w_23_0 : mxxMat1),
                        let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_23_0 w_12_0
                        let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_24_0
                        sequential_sequential_generatedRoot_48_16.constraints_0 i_0 i_1 w_1_0 w_4_0 w_16_0 w_19_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_48.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat3
  w_11_0 : mxxMat0
  w_12_0 : mxxMat4
  w_13_0 : mxxMat4
  w_16_0 : mxxMat0
  w_16_1 : mxxMat0
  w_17_0 : mxxMat15
  w_20_0 : mxxMat15

abbrev sequential_generatedRoot_48.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15 → mxxMat3) (w_5_0 : Fin 15 → mxxMat3) (w_6_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_15_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat0) (w_12_0 : mxxMat4) (w_13_0 : mxxMat4) (w_16_0 : mxxMat0) (w_16_1 : mxxMat0) (w_17_0 : mxxMat15) (w_18_0 : mxxMat15) (w_20_0 : mxxMat15) (w_21_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (104 + (-1 * (Int.ofNat i_0))) ∧
  (104 + (-1 * (Int.ofNat i_0))) < (209 + (-1 * (Int.ofNat i_0))) ∧
  (209 + (-1 * (Int.ofNat i_0))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.ofNat i_0))) (209 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_9_0 w_10_0 ∧
  MxxRuntime.concatColumns w_8_0 w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_12_0 w_13_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_48_16 backend hashModel tape (path ++ [16, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_5_0, (w_13_0, (w_14_0, (w_15_0, ()))))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_16_0, (w_16_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_16_0 w_17_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_16_1 w_20_0 ∧
  outputs = (w_18_0, (w_21_0, ()))

abbrev sequential_generatedRoot_48.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_48.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_19_0 : mxxMat15 := inputs.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_6_0 : ByteArray := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_16_0 := witness.w_16_0
  let w_16_1 := witness.w_16_1
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_17_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat15 := MxxRuntime.matrixAdd w_19_0 w_20_0
  sequential_generatedRoot_48.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_6_0 w_14_0 w_15_0 w_1_0 w_2_0 w_3_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_16_0 w_16_1 w_17_0 w_18_0 w_20_0 w_21_0 outputs

def sequential_generatedRoot_48 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_48.Witness,
    sequential_generatedRoot_48.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_49_23.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_49_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_49_23.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_49_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_49_23.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_49.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_13_0 : mxxMat8
  w_16_0 : mxxMat3
  w_18_0 : mxxMat3
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat8
  w_23_0 : mxxMat0
  w_23_1 : mxxMat0
  w_24_0 : mxxMat15
  w_27_0 : mxxMat15

abbrev sequential_generatedRoot_49.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_12_0 w_13_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_15_0 w_16_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_15_0 w_18_0 ∧
  MxxRuntime.concatColumns w_16_0 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_20_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_20_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_49_23 backend hashModel tape (path ++ [23, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_13_0, (w_22_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_23_0, (w_23_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_23_0 w_24_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_23_1 w_27_0 ∧
  outputs = (w_25_0, (w_28_0, ()))

abbrev sequential_generatedRoot_49.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : Fin 15 → mxxMat3) (w_8_0 : Fin 15 → mxxMat3) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_6_0 : Int) (w_7_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_6_0 w_7_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_6_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_11_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_49.constraints_1 backend hashModel tape path params i_0 w_4_0 w_14_0 w_17_0 w_2_0 w_3_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_49.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_49.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_26_0 : mxxMat15 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_8_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_17_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_15_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_23_1 := witness.w_23_1
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_24_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat15 := MxxRuntime.matrixAdd w_26_0 w_27_0
  sequential_generatedRoot_49.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_14_0 w_17_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

def sequential_generatedRoot_49 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_49.Witness,
    sequential_generatedRoot_49.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_55.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_generatedRoot_55 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_generatedRoot_55.constraints_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_64_7.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_64_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_64_7.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_64.Witness where
  w_3_0 : mxxMat3
  w_7_0 : mxxMat1
  w_9_0 : mxxMat3
  w_11_0 : mxxMat17
  w_16_0 : mxxMat3
  w_21_0 : mxxMat3
  w_26_0 : mxxMat3

abbrev sequential_generatedRoot_64.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat3) (w_20_0 : Fin 15 → mxxMat3) (w_25_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat17) (w_13_0 : mxxMat3) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_21_0 : mxxMat3) (w_23_0 : mxxMat3) (w_26_0 : mxxMat3) (w_28_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_64_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_2_0 w_21_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_2_0 w_26_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, (w_28_0, ()))))

abbrev sequential_generatedRoot_64.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_64.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_14_0 : mxxMat3 := inputs.2.1
  let w_19_0 : mxxMat3 := inputs.2.2.1
  let w_24_0 : mxxMat3 := inputs.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_20_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_5_0 : mxxMat1 := (0 : mxxMat1)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat3 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat3 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat3 := MxxRuntime.matrixAdd w_14_0 w_17_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat3 := MxxRuntime.matrixMul w_21_0 w_11_0
  let w_23_0 : mxxMat3 := MxxRuntime.matrixAdd w_19_0 w_22_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMul w_26_0 w_11_0
  let w_28_0 : mxxMat3 := MxxRuntime.matrixAdd w_24_0 w_27_0
  sequential_generatedRoot_64.constraints_0 backend hashModel tape path params i_0 w_1_0 w_6_0 w_15_0 w_20_0 w_25_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 w_26_0 w_28_0 outputs

def sequential_generatedRoot_64 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_64.Witness,
    sequential_generatedRoot_64.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_65_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_65_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_65_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_65_24.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_65_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_65_24.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_65.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_65.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_65_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_65.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_65_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_65.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_65.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat10) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_65.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_65.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_65.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat10 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat10 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_65.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_65 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_65.Witness,
    sequential_generatedRoot_65.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_66_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_66_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_66_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_66_24.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_66_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_66_24.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_66.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_66.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_66_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_66.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_66_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_66.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_66.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat15) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_66.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_66.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_66.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat15 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat15 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_66.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_66 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_66.Witness,
    sequential_generatedRoot_66.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_71_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_71_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_71_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_71.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_71.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_71_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_71.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_71.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_71.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_71 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_71.Witness,
    sequential_generatedRoot_71.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_72_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_72_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_72_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_72.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_72.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_72_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_72.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_72.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_72.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_72.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_72.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_72 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_72.Witness,
    sequential_generatedRoot_72.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_75_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_75_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_75_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_75.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_75.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_75_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_75.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_75.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_75.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_75 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_75.Witness,
    sequential_generatedRoot_75.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_76_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_76_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_76_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_76.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_76.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_76_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_76.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_76.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_76.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_76.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_76.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_76 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_76.Witness,
    sequential_generatedRoot_76.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_79_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_79_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_79_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_79.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_79.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_79_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_79.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_79.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_79.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_79 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_79.Witness,
    sequential_generatedRoot_79.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_80_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_80_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_80_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_80.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_80.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_80_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_80.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_80.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_80.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_80.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_80.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_80 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_80.Witness,
    sequential_generatedRoot_80.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_83_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_83_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_83_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_83.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_83.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_83_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_83.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_83.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_83.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_83 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_83.Witness,
    sequential_generatedRoot_83.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_84_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_84_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_84_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_84.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_84.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_84_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_84.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_84.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_84.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_84.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_84.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_84 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_84.Witness,
    sequential_generatedRoot_84.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_105_14.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat0) (w_11_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_5_0 : mxxMat3) (w_9_0 : mxxMat0) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_2_0 w_3_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = (w_9_0, (w_15_0, ()))

def sequential_sequential_generatedRoot_105_14 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_10_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_6_0 : mxxMat4 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_3_0 : mxxMat1),
      let w_4_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
      ∃ (w_5_0 : mxxMat3),
        let w_7_0 : mxxMat0 := MxxRuntime.matrixMul w_5_0 w_6_0
        let w_8_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_3_0 w_7_0
        let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_8_0
        ∃ (w_12_0 : mxxMat1),
          ∃ (w_13_0 : mxxMat1),
            let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_13_0 w_7_0
            let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_14_0
            sequential_sequential_generatedRoot_105_14.constraints_0 i_0 i_1 w_1_0 w_11_0 w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_105.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat4
  w_12_0 : mxxMat4
  w_14_0 : mxxMat0
  w_14_1 : mxxMat0
  w_15_0 : mxxMat10
  w_18_0 : mxxMat10

abbrev sequential_generatedRoot_105.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_13_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat4) (w_12_0 : mxxMat4) (w_14_0 : mxxMat0) (w_14_1 : mxxMat0) (w_15_0 : mxxMat10) (w_16_0 : mxxMat10) (w_18_0 : mxxMat10) (w_19_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (21 + (-1 * (Int.ofNat i_0))) ∧
  (21 + (-1 * (Int.ofNat i_0))) < (43 + (-1 * (Int.ofNat i_0))) ∧
  (43 + (-1 * (Int.ofNat i_0))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.ofNat i_0))) (43 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_11_0 w_12_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_105_14 backend hashModel tape (path ++ [14, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_12_0, (w_13_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_14_0, (w_14_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_14_0 w_15_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_14_1 w_18_0 ∧
  outputs = (w_16_0, (w_19_0, ()))

abbrev sequential_generatedRoot_105.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_105.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_17_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_15_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat10 := MxxRuntime.matrixAdd w_17_0 w_18_0
  sequential_generatedRoot_105.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_13_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_14_1 w_15_0 w_16_0 w_18_0 w_19_0 outputs

def sequential_generatedRoot_105 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_105.Witness,
    sequential_generatedRoot_105.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_106_13.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_106_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_106_13.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_106_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_106_13.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_106.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat8
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat8
  w_13_0 : mxxMat0
  w_13_1 : mxxMat0
  w_14_0 : mxxMat10
  w_17_0 : mxxMat10

abbrev sequential_generatedRoot_106.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_9_0 : mxxMat0) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_10_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_106_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_8_0, (w_12_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_13_0, (w_13_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_13_0 w_14_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_13_1 w_17_0 ∧
  outputs = (w_15_0, (w_18_0, ()))

abbrev sequential_generatedRoot_106.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_9_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_7_0 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_106.constraints_1 backend hashModel tape path params i_0 w_4_0 w_9_0 w_2_0 w_3_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

abbrev sequential_generatedRoot_106.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_106.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_16_0 : mxxMat10 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_13_1 := witness.w_13_1
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat10 := MxxRuntime.matrixAdd w_16_0 w_17_0
  sequential_generatedRoot_106.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_9_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

def sequential_generatedRoot_106 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_106.Witness,
    sequential_generatedRoot_106.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_108_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_parallel_generatedRoot_108_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_parallel_generatedRoot_108_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_108.Witness where
  w_2_0 : mxxMat3
  w_3_0 : mxxMat3
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat3
  w_13_0 : mxxMat3
  w_14_0 : mxxMat0
  w_15_0 : mxxMat3
  w_16_0 : mxxMat3
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_18_1 : mxxMat0
  w_19_0 : mxxMat3
  w_20_0 : mxxMat3
  w_21_0 : mxxMat3
  w_22_0 : mxxMat3

abbrev parallel_generatedRoot_108.constraints_2 (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_20_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 0 44 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 44 88 w_22_0 ∧
  outputs = (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))

abbrev parallel_generatedRoot_108.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 0 44 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 44 88 w_13_0 ∧
  MxxRuntime.concatColumns w_12_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_108_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_9_0, (w_14_0, (w_17_0, ())))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, ())) (w_18_0, (w_18_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_108.constraints_2 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_108.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_5_0 : mxxMat10) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_2_0 : mxxMat3) (w_3_0 : mxxMat3) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.concatColumns w_2_0 w_3_0 w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 0 44 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 44 88 w_8_0 ∧
  MxxRuntime.concatColumns w_7_0 w_8_0 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_108.constraints_1 backend hashModel tape path params i_0 w_1_0 w_10_0 w_0_0 w_4_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_108.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : parallel_generatedRoot_108.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : ByteArray := inputs.1
  let w_5_0 : mxxMat10 := inputs.2.1
  let w_10_0 : mxxMat10 := inputs.2.2.1
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  parallel_generatedRoot_108.constraints_0 backend hashModel tape path params i_0 w_1_0 w_5_0 w_10_0 w_0_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

def parallel_generatedRoot_108 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_108.Witness,
    parallel_generatedRoot_108.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_109_16.constraints_1 (i_0 : Nat) (i_1 : Nat) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_22_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  outputs = (w_14_0, (w_25_0, ()))

abbrev sequential_sequential_generatedRoot_109_16.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_4_0 : Fin 15 → mxxMat3) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat3) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat2) (w_10_0 : mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  sequential_sequential_generatedRoot_109_16.constraints_1 i_0 i_1 w_16_0 w_19_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

def sequential_sequential_generatedRoot_109_16 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × mxxMat4 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_11_0 : mxxMat4 := inputs.2.2.2.2.1
  let w_16_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  ∃ (w_3_0 : mxxMat3),
    ∃ (w_5_0 : mxxMat3),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_8_0 : mxxMat1),
            let w_9_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
            ∃ (w_10_0 : mxxMat3),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_12_0
              let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
              let w_17_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
              ∃ (w_18_0 : mxxMat3),
                ∃ (w_20_0 : mxxMat3),
                  ∃ (w_21_0 : mxxMat0),
                    ∃ (w_22_0 : mxxMat1),
                      ∃ (w_23_0 : mxxMat1),
                        let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_23_0 w_12_0
                        let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_24_0
                        sequential_sequential_generatedRoot_109_16.constraints_0 i_0 i_1 w_1_0 w_4_0 w_16_0 w_19_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_109.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat3
  w_11_0 : mxxMat0
  w_12_0 : mxxMat4
  w_13_0 : mxxMat4
  w_16_0 : mxxMat0
  w_16_1 : mxxMat0
  w_17_0 : mxxMat15
  w_20_0 : mxxMat15

abbrev sequential_generatedRoot_109.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15 → mxxMat3) (w_5_0 : Fin 15 → mxxMat3) (w_6_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_15_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat0) (w_12_0 : mxxMat4) (w_13_0 : mxxMat4) (w_16_0 : mxxMat0) (w_16_1 : mxxMat0) (w_17_0 : mxxMat15) (w_18_0 : mxxMat15) (w_20_0 : mxxMat15) (w_21_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (104 + (-1 * (Int.ofNat i_0))) ∧
  (104 + (-1 * (Int.ofNat i_0))) < (209 + (-1 * (Int.ofNat i_0))) ∧
  (209 + (-1 * (Int.ofNat i_0))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.ofNat i_0))) (209 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_9_0 w_10_0 ∧
  MxxRuntime.concatColumns w_8_0 w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_12_0 w_13_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_109_16 backend hashModel tape (path ++ [16, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_5_0, (w_13_0, (w_14_0, (w_15_0, ()))))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_16_0, (w_16_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_16_0 w_17_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_16_1 w_20_0 ∧
  outputs = (w_18_0, (w_21_0, ()))

abbrev sequential_generatedRoot_109.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_109.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_19_0 : mxxMat15 := inputs.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_6_0 : ByteArray := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_16_0 := witness.w_16_0
  let w_16_1 := witness.w_16_1
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_17_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat15 := MxxRuntime.matrixAdd w_19_0 w_20_0
  sequential_generatedRoot_109.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_6_0 w_14_0 w_15_0 w_1_0 w_2_0 w_3_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_16_0 w_16_1 w_17_0 w_18_0 w_20_0 w_21_0 outputs

def sequential_generatedRoot_109 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_109.Witness,
    sequential_generatedRoot_109.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_110_23.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_110_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_110_23.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_110_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_110_23.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_110.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_13_0 : mxxMat8
  w_16_0 : mxxMat3
  w_18_0 : mxxMat3
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat8
  w_23_0 : mxxMat0
  w_23_1 : mxxMat0
  w_24_0 : mxxMat15
  w_27_0 : mxxMat15

abbrev sequential_generatedRoot_110.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_12_0 w_13_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_15_0 w_16_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_15_0 w_18_0 ∧
  MxxRuntime.concatColumns w_16_0 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_20_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_20_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_110_23 backend hashModel tape (path ++ [23, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_13_0, (w_22_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_23_0, (w_23_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_23_0 w_24_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_23_1 w_27_0 ∧
  outputs = (w_25_0, (w_28_0, ()))

abbrev sequential_generatedRoot_110.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : Fin 15 → mxxMat3) (w_8_0 : Fin 15 → mxxMat3) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_6_0 : Int) (w_7_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_6_0 w_7_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_6_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_11_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_110.constraints_1 backend hashModel tape path params i_0 w_4_0 w_14_0 w_17_0 w_2_0 w_3_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_110.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_110.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_26_0 : mxxMat15 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_8_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_17_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_15_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_23_1 := witness.w_23_1
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_24_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat15 := MxxRuntime.matrixAdd w_26_0 w_27_0
  sequential_generatedRoot_110.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_14_0 w_17_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

def sequential_generatedRoot_110 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_110.Witness,
    sequential_generatedRoot_110.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_115.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_generatedRoot_115 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_generatedRoot_115.constraints_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_124_7.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_124_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_124_7.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_124.Witness where
  w_3_0 : mxxMat3
  w_7_0 : mxxMat1
  w_9_0 : mxxMat3
  w_11_0 : mxxMat17
  w_16_0 : mxxMat3
  w_21_0 : mxxMat3
  w_26_0 : mxxMat3

abbrev sequential_generatedRoot_124.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat3) (w_20_0 : Fin 15 → mxxMat3) (w_25_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat17) (w_13_0 : mxxMat3) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_21_0 : mxxMat3) (w_23_0 : mxxMat3) (w_26_0 : mxxMat3) (w_28_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_124_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_2_0 w_21_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_2_0 w_26_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, (w_28_0, ()))))

abbrev sequential_generatedRoot_124.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_124.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_14_0 : mxxMat3 := inputs.2.1
  let w_19_0 : mxxMat3 := inputs.2.2.1
  let w_24_0 : mxxMat3 := inputs.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_20_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_5_0 : mxxMat1 := (0 : mxxMat1)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat3 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat3 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat3 := MxxRuntime.matrixAdd w_14_0 w_17_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat3 := MxxRuntime.matrixMul w_21_0 w_11_0
  let w_23_0 : mxxMat3 := MxxRuntime.matrixAdd w_19_0 w_22_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMul w_26_0 w_11_0
  let w_28_0 : mxxMat3 := MxxRuntime.matrixAdd w_24_0 w_27_0
  sequential_generatedRoot_124.constraints_0 backend hashModel tape path params i_0 w_1_0 w_6_0 w_15_0 w_20_0 w_25_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 w_26_0 w_28_0 outputs

def sequential_generatedRoot_124 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_124.Witness,
    sequential_generatedRoot_124.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_125_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_125_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_125_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_125_24.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_125_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_125_24.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_125.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_125.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_125_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_125.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_125_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_125.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_125.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat10) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_125.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_125.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_125.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat10 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat10 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_125.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_125 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_125.Witness,
    sequential_generatedRoot_125.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_126_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_126_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_126_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_126_24.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_126_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_126_24.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_126.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_126.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_126_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_126.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_126_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_126.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_126.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat15) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_126.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_126.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_126.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat15 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat15 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_126.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_126 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_126.Witness,
    sequential_generatedRoot_126.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_131_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_131_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_131_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_131.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_131.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_131_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_131.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_131.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_131.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_131 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_131.Witness,
    sequential_generatedRoot_131.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_132_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_132_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_132_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_132.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_132.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_132_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_132.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_132.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_132.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_132.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_132.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_132 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_132.Witness,
    sequential_generatedRoot_132.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_135_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_135_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_135_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_135.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_135.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_135_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_135.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_135.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_135.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_135 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_135.Witness,
    sequential_generatedRoot_135.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_136_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_136_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_136_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_136.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_136.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_136_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_136.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_136.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_136.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_136.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_136.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_136 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_136.Witness,
    sequential_generatedRoot_136.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_139_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_139_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_139_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_139.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_139.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_139_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_139.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_139.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_139.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_139 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_139.Witness,
    sequential_generatedRoot_139.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_140_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_140_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_140_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_140.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_140.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_140_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_140.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_140.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_140.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_140.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_140.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_140 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_140.Witness,
    sequential_generatedRoot_140.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_143_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_143_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_143_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_143.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_143.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_143_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_143.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_143.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_143.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_143 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_143.Witness,
    sequential_generatedRoot_143.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_144_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_144_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_144_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_144.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_144.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_144_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_144.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_144.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_144.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_144.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_144.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_144 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_144.Witness,
    sequential_generatedRoot_144.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_165_14.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat0) (w_11_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_5_0 : mxxMat3) (w_9_0 : mxxMat0) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_2_0 w_3_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = (w_9_0, (w_15_0, ()))

def sequential_sequential_generatedRoot_165_14 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_10_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_6_0 : mxxMat4 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_3_0 : mxxMat1),
      let w_4_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
      ∃ (w_5_0 : mxxMat3),
        let w_7_0 : mxxMat0 := MxxRuntime.matrixMul w_5_0 w_6_0
        let w_8_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_3_0 w_7_0
        let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_8_0
        ∃ (w_12_0 : mxxMat1),
          ∃ (w_13_0 : mxxMat1),
            let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_13_0 w_7_0
            let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_14_0
            sequential_sequential_generatedRoot_165_14.constraints_0 i_0 i_1 w_1_0 w_11_0 w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_165.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat4
  w_12_0 : mxxMat4
  w_14_0 : mxxMat0
  w_14_1 : mxxMat0
  w_15_0 : mxxMat10
  w_18_0 : mxxMat10

abbrev sequential_generatedRoot_165.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_13_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat4) (w_12_0 : mxxMat4) (w_14_0 : mxxMat0) (w_14_1 : mxxMat0) (w_15_0 : mxxMat10) (w_16_0 : mxxMat10) (w_18_0 : mxxMat10) (w_19_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (21 + (-1 * (Int.ofNat i_0))) ∧
  (21 + (-1 * (Int.ofNat i_0))) < (43 + (-1 * (Int.ofNat i_0))) ∧
  (43 + (-1 * (Int.ofNat i_0))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.ofNat i_0))) (43 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_11_0 w_12_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_165_14 backend hashModel tape (path ++ [14, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_12_0, (w_13_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_14_0, (w_14_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_14_0 w_15_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_14_1 w_18_0 ∧
  outputs = (w_16_0, (w_19_0, ()))

abbrev sequential_generatedRoot_165.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_165.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_17_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_15_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat10 := MxxRuntime.matrixAdd w_17_0 w_18_0
  sequential_generatedRoot_165.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_13_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_14_1 w_15_0 w_16_0 w_18_0 w_19_0 outputs

def sequential_generatedRoot_165 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_165.Witness,
    sequential_generatedRoot_165.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_166_13.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_166_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_166_13.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_166_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_166_13.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_166.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat8
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat8
  w_13_0 : mxxMat0
  w_13_1 : mxxMat0
  w_14_0 : mxxMat10
  w_17_0 : mxxMat10

abbrev sequential_generatedRoot_166.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_9_0 : mxxMat0) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_10_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_166_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_8_0, (w_12_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_13_0, (w_13_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_13_0 w_14_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_13_1 w_17_0 ∧
  outputs = (w_15_0, (w_18_0, ()))

abbrev sequential_generatedRoot_166.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_9_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_7_0 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_166.constraints_1 backend hashModel tape path params i_0 w_4_0 w_9_0 w_2_0 w_3_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

abbrev sequential_generatedRoot_166.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_166.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_16_0 : mxxMat10 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_13_1 := witness.w_13_1
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat10 := MxxRuntime.matrixAdd w_16_0 w_17_0
  sequential_generatedRoot_166.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_9_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

def sequential_generatedRoot_166 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_166.Witness,
    sequential_generatedRoot_166.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_168_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_parallel_generatedRoot_168_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_parallel_generatedRoot_168_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_168.Witness where
  w_2_0 : mxxMat3
  w_3_0 : mxxMat3
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat3
  w_13_0 : mxxMat3
  w_14_0 : mxxMat0
  w_15_0 : mxxMat3
  w_16_0 : mxxMat3
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_18_1 : mxxMat0
  w_19_0 : mxxMat3
  w_20_0 : mxxMat3
  w_21_0 : mxxMat3
  w_22_0 : mxxMat3

abbrev parallel_generatedRoot_168.constraints_2 (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_20_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 0 44 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 44 88 w_22_0 ∧
  outputs = (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))

abbrev parallel_generatedRoot_168.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 0 44 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 44 88 w_13_0 ∧
  MxxRuntime.concatColumns w_12_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_168_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_9_0, (w_14_0, (w_17_0, ())))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, ())) (w_18_0, (w_18_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_168.constraints_2 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_168.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_5_0 : mxxMat10) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_2_0 : mxxMat3) (w_3_0 : mxxMat3) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.concatColumns w_2_0 w_3_0 w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 0 44 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 44 88 w_8_0 ∧
  MxxRuntime.concatColumns w_7_0 w_8_0 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_168.constraints_1 backend hashModel tape path params i_0 w_1_0 w_10_0 w_0_0 w_4_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_168.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : parallel_generatedRoot_168.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : ByteArray := inputs.1
  let w_5_0 : mxxMat10 := inputs.2.1
  let w_10_0 : mxxMat10 := inputs.2.2.1
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  parallel_generatedRoot_168.constraints_0 backend hashModel tape path params i_0 w_1_0 w_5_0 w_10_0 w_0_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

def parallel_generatedRoot_168 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_168.Witness,
    parallel_generatedRoot_168.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_169_16.constraints_1 (i_0 : Nat) (i_1 : Nat) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_22_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  outputs = (w_14_0, (w_25_0, ()))

abbrev sequential_sequential_generatedRoot_169_16.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_4_0 : Fin 15 → mxxMat3) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat3) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat2) (w_10_0 : mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  sequential_sequential_generatedRoot_169_16.constraints_1 i_0 i_1 w_16_0 w_19_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

def sequential_sequential_generatedRoot_169_16 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × mxxMat4 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_11_0 : mxxMat4 := inputs.2.2.2.2.1
  let w_16_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  ∃ (w_3_0 : mxxMat3),
    ∃ (w_5_0 : mxxMat3),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_8_0 : mxxMat1),
            let w_9_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
            ∃ (w_10_0 : mxxMat3),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_12_0
              let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
              let w_17_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
              ∃ (w_18_0 : mxxMat3),
                ∃ (w_20_0 : mxxMat3),
                  ∃ (w_21_0 : mxxMat0),
                    ∃ (w_22_0 : mxxMat1),
                      ∃ (w_23_0 : mxxMat1),
                        let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_23_0 w_12_0
                        let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_24_0
                        sequential_sequential_generatedRoot_169_16.constraints_0 i_0 i_1 w_1_0 w_4_0 w_16_0 w_19_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_169.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat3
  w_11_0 : mxxMat0
  w_12_0 : mxxMat4
  w_13_0 : mxxMat4
  w_16_0 : mxxMat0
  w_16_1 : mxxMat0
  w_17_0 : mxxMat15
  w_20_0 : mxxMat15

abbrev sequential_generatedRoot_169.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15 → mxxMat3) (w_5_0 : Fin 15 → mxxMat3) (w_6_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_15_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat0) (w_12_0 : mxxMat4) (w_13_0 : mxxMat4) (w_16_0 : mxxMat0) (w_16_1 : mxxMat0) (w_17_0 : mxxMat15) (w_18_0 : mxxMat15) (w_20_0 : mxxMat15) (w_21_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (104 + (-1 * (Int.ofNat i_0))) ∧
  (104 + (-1 * (Int.ofNat i_0))) < (209 + (-1 * (Int.ofNat i_0))) ∧
  (209 + (-1 * (Int.ofNat i_0))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.ofNat i_0))) (209 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_9_0 w_10_0 ∧
  MxxRuntime.concatColumns w_8_0 w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_12_0 w_13_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_169_16 backend hashModel tape (path ++ [16, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_5_0, (w_13_0, (w_14_0, (w_15_0, ()))))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_16_0, (w_16_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_16_0 w_17_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_16_1 w_20_0 ∧
  outputs = (w_18_0, (w_21_0, ()))

abbrev sequential_generatedRoot_169.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_169.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_19_0 : mxxMat15 := inputs.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_6_0 : ByteArray := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_16_0 := witness.w_16_0
  let w_16_1 := witness.w_16_1
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_17_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat15 := MxxRuntime.matrixAdd w_19_0 w_20_0
  sequential_generatedRoot_169.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_6_0 w_14_0 w_15_0 w_1_0 w_2_0 w_3_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_16_0 w_16_1 w_17_0 w_18_0 w_20_0 w_21_0 outputs

def sequential_generatedRoot_169 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_169.Witness,
    sequential_generatedRoot_169.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_170_23.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_170_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_170_23.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_170_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_170_23.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_170.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_13_0 : mxxMat8
  w_16_0 : mxxMat3
  w_18_0 : mxxMat3
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat8
  w_23_0 : mxxMat0
  w_23_1 : mxxMat0
  w_24_0 : mxxMat15
  w_27_0 : mxxMat15

abbrev sequential_generatedRoot_170.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_12_0 w_13_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_15_0 w_16_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_15_0 w_18_0 ∧
  MxxRuntime.concatColumns w_16_0 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_20_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_20_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_170_23 backend hashModel tape (path ++ [23, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_13_0, (w_22_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_23_0, (w_23_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_23_0 w_24_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_23_1 w_27_0 ∧
  outputs = (w_25_0, (w_28_0, ()))

abbrev sequential_generatedRoot_170.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : Fin 15 → mxxMat3) (w_8_0 : Fin 15 → mxxMat3) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_6_0 : Int) (w_7_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_6_0 w_7_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_6_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_11_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_170.constraints_1 backend hashModel tape path params i_0 w_4_0 w_14_0 w_17_0 w_2_0 w_3_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_170.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_170.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_26_0 : mxxMat15 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_8_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_17_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_15_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_23_1 := witness.w_23_1
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_24_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat15 := MxxRuntime.matrixAdd w_26_0 w_27_0
  sequential_generatedRoot_170.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_14_0 w_17_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

def sequential_generatedRoot_170 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_170.Witness,
    sequential_generatedRoot_170.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_175.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_generatedRoot_175 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_generatedRoot_175.constraints_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_184_7.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_184_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_184_7.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_184.Witness where
  w_3_0 : mxxMat3
  w_7_0 : mxxMat1
  w_9_0 : mxxMat3
  w_11_0 : mxxMat17
  w_16_0 : mxxMat3
  w_21_0 : mxxMat3
  w_26_0 : mxxMat3

abbrev sequential_generatedRoot_184.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat3) (w_20_0 : Fin 15 → mxxMat3) (w_25_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat17) (w_13_0 : mxxMat3) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_21_0 : mxxMat3) (w_23_0 : mxxMat3) (w_26_0 : mxxMat3) (w_28_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_184_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_2_0 w_21_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_2_0 w_26_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, (w_28_0, ()))))

abbrev sequential_generatedRoot_184.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_184.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_14_0 : mxxMat3 := inputs.2.1
  let w_19_0 : mxxMat3 := inputs.2.2.1
  let w_24_0 : mxxMat3 := inputs.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_20_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_5_0 : mxxMat1 := (0 : mxxMat1)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat3 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat3 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat3 := MxxRuntime.matrixAdd w_14_0 w_17_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat3 := MxxRuntime.matrixMul w_21_0 w_11_0
  let w_23_0 : mxxMat3 := MxxRuntime.matrixAdd w_19_0 w_22_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMul w_26_0 w_11_0
  let w_28_0 : mxxMat3 := MxxRuntime.matrixAdd w_24_0 w_27_0
  sequential_generatedRoot_184.constraints_0 backend hashModel tape path params i_0 w_1_0 w_6_0 w_15_0 w_20_0 w_25_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 w_26_0 w_28_0 outputs

def sequential_generatedRoot_184 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_184.Witness,
    sequential_generatedRoot_184.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_185_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_185_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_185_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_185_24.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_185_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_185_24.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_185.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_185.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_185_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_185.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_185_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_185.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_185.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat10) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_185.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_185.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_185.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat10 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat10 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_185.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_185 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_185.Witness,
    sequential_generatedRoot_185.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_186_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_186_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_186_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_186_24.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_186_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat1 := (1 : mxxMat1)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat1 := (1 : mxxMat1)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat1),
      let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_186_24.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_186.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_186.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_186_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_186.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_186_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_186.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_186.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat15) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_186.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_186.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_186.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat15 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat15 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_186.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_186 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_186.Witness,
    sequential_generatedRoot_186.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_191_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_191_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_191_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_191.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_191.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_191_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_191.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_191.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_191.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_191 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_191.Witness,
    sequential_generatedRoot_191.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_192_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (0), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_192_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_192_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_192.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_192.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_192_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_192.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_192.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_192.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_192.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_192.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_192 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_192.Witness,
    sequential_generatedRoot_192.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_195_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_195_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_195_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_195.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_195.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_195_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_195.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_195.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_195.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_195 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_195.Witness,
    sequential_generatedRoot_195.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_196_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (1), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_196_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_196_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_196.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_196.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_196_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_196.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_196.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_196.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_196.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_196.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_196 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_196.Witness,
    sequential_generatedRoot_196.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_199_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_199_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_199_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_199.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_199.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_199_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_199.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_199.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_199.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_199 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_199.Witness,
    sequential_generatedRoot_199.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_200_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (2), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_200_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_200_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_200.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_200.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_200_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_200.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_200.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_200.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_200.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_200.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_200 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_200.Witness,
    sequential_generatedRoot_200.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_203_12.constraints_0 (i_1 : Nat) (w_1_0 : mxxMat0) (w_10_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat2) (w_4_0 : mxxMat3) (w_8_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_3_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_11_0 ∧
  outputs = (w_8_0, (w_13_0, ()))

def sequential_sequential_generatedRoot_203_12 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_9_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : mxxMat4 := inputs.2.2.2.1
  let w_10_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
    ∃ (w_4_0 : mxxMat3),
      let w_6_0 : mxxMat0 := MxxRuntime.matrixMul w_4_0 w_5_0
      let w_7_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_2_0 w_6_0
      let w_8_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_7_0
      ∃ (w_11_0 : mxxMat1),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_11_0 w_6_0
        let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_9_0 w_12_0
        sequential_sequential_generatedRoot_203_12.constraints_0 i_1 w_1_0 w_10_0 w_2_0 w_3_0 w_4_0 w_8_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_203.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat0
  w_9_0 : mxxMat4
  w_10_0 : mxxMat4
  w_12_0 : mxxMat0
  w_12_1 : mxxMat0
  w_13_0 : mxxMat0
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_203.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_11_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat0) (w_9_0 : mxxMat4) (w_10_0 : mxxMat4) (w_12_0 : mxxMat0) (w_12_1 : mxxMat0) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (-1 * (Int.ofNat i_0)) ∧
  (-1 * (Int.ofNat i_0)) < (1 + (-1 * (Int.ofNat i_0))) ∧
  (1 + (-1 * (Int.ofNat i_0))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.ofNat i_0)) (1 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_7_0 ∧
  MxxRuntime.concatColumns w_6_0 w_7_0 w_8_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_8_0 w_9_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_9_0 w_10_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_203_12 backend hashModel tape (path ++ [12, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_10_0, (w_11_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_12_0, (w_12_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_12_0 w_13_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_12_1 w_16_0 ∧
  outputs = (w_14_0, (w_17_0, ()))

abbrev sequential_generatedRoot_203.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_203.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_16_0
  sequential_generatedRoot_203.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_11_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_12_1 w_13_0 w_14_0 w_16_0 w_17_0 outputs

def sequential_generatedRoot_203 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_203.Witness,
    sequential_generatedRoot_203.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_204_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le (3), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

def sequential_sequential_generatedRoot_204_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_204_11.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_204.Witness where
  w_2_0 : mxxMat1
  w_6_0 : mxxMat1
  w_7_0 : mxxMat8
  w_9_0 : mxxMat1
  w_10_0 : mxxMat8
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat0
  w_15_0 : mxxMat0

abbrev sequential_generatedRoot_204.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_7_0 : mxxMat8) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_204_11 backend hashModel tape (path ++ [11, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_7_0, (w_10_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_11_0, (w_11_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_11_0 w_12_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_11_1 w_15_0 ∧
  outputs = (w_13_0, (w_16_0, ()))

abbrev sequential_generatedRoot_204.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_8_0 : mxxMat0) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat8) (w_9_0 : mxxMat1) (w_10_0 : mxxMat8) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (0) ∧
  (0) < 1 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) < (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 1 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (-1 * (Int.fdiv ((Int.ofNat i_0)) (44))) (1 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_9_0 w_10_0 ∧
  0 ≤ (88) ∧
  sequential_generatedRoot_204.constraints_1 backend hashModel tape path params i_0 w_4_0 w_2_0 w_3_0 w_7_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

abbrev sequential_generatedRoot_204.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_204.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat1 := (MxxRuntime.unitRow (0) : mxxMat1)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_15_0
  sequential_generatedRoot_204.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_11_1 w_12_0 w_13_0 w_15_0 w_16_0 outputs

def sequential_generatedRoot_204 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_204.Witness,
    sequential_generatedRoot_204.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_225_14.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : mxxMat0) (w_11_0 : mxxMat0) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat2) (w_5_0 : mxxMat3) (w_9_0 : mxxMat0) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_2_0 w_3_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = (w_9_0, (w_15_0, ()))

def sequential_sequential_generatedRoot_225_14 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat4 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_10_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_6_0 : mxxMat4 := inputs.2.2.2.1
  let w_11_0 : mxxMat0 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    ∃ (w_3_0 : mxxMat1),
      let w_4_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
      ∃ (w_5_0 : mxxMat3),
        let w_7_0 : mxxMat0 := MxxRuntime.matrixMul w_5_0 w_6_0
        let w_8_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_3_0 w_7_0
        let w_9_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_8_0
        ∃ (w_12_0 : mxxMat1),
          ∃ (w_13_0 : mxxMat1),
            let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_13_0 w_7_0
            let w_15_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_14_0
            sequential_sequential_generatedRoot_225_14.constraints_0 i_0 i_1 w_1_0 w_11_0 w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_225.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat3
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat4
  w_12_0 : mxxMat4
  w_14_0 : mxxMat0
  w_14_1 : mxxMat0
  w_15_0 : mxxMat10
  w_18_0 : mxxMat10

abbrev sequential_generatedRoot_225.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : mxxMat0) (w_5_0 : ByteArray) (w_13_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat3) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat4) (w_12_0 : mxxMat4) (w_14_0 : mxxMat0) (w_14_1 : mxxMat0) (w_15_0 : mxxMat10) (w_16_0 : mxxMat10) (w_18_0 : mxxMat10) (w_19_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (21 + (-1 * (Int.ofNat i_0))) ∧
  (21 + (-1 * (Int.ofNat i_0))) < (43 + (-1 * (Int.ofNat i_0))) ∧
  (43 + (-1 * (Int.ofNat i_0))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.ofNat i_0))) (43 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_5_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_5_0) w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_8_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_11_0 w_12_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_225_14 backend hashModel tape (path ++ [14, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_12_0, (w_13_0, ()))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_14_0, (w_14_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_14_0 w_15_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_14_1 w_18_0 ∧
  outputs = (w_16_0, (w_19_0, ()))

abbrev sequential_generatedRoot_225.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_225.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_17_0 : mxxMat10 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_5_0 : ByteArray := inputs.2.2.2.1
  let w_13_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_15_0 := witness.w_15_0
  let w_16_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_15_0
  let w_18_0 := witness.w_18_0
  let w_19_0 : mxxMat10 := MxxRuntime.matrixAdd w_17_0 w_18_0
  sequential_generatedRoot_225.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_13_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_14_1 w_15_0 w_16_0 w_18_0 w_19_0 outputs

def sequential_generatedRoot_225 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_225.Witness,
    sequential_generatedRoot_225.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_226_13.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_226_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_226_13.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_226_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_226_13.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_226.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat1
  w_7_0 : mxxMat1
  w_8_0 : mxxMat8
  w_10_0 : mxxMat1
  w_11_0 : mxxMat1
  w_12_0 : mxxMat8
  w_13_0 : mxxMat0
  w_13_1 : mxxMat0
  w_14_0 : mxxMat10
  w_17_0 : mxxMat10

abbrev sequential_generatedRoot_226.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_9_0 : mxxMat0) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_10_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_226_13 backend hashModel tape (path ++ [13, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_8_0, (w_12_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_13_0, (w_13_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_13_0 w_14_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_13_1 w_17_0 ∧
  outputs = (w_15_0, (w_18_0, ()))

abbrev sequential_generatedRoot_226.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : mxxMat0) (w_9_0 : mxxMat0) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat0) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_8_0 : mxxMat8) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat8) (w_13_0 : mxxMat0) (w_13_1 : mxxMat0) (w_14_0 : mxxMat10) (w_15_0 : mxxMat10) (w_17_0 : mxxMat10) (w_18_0 : mxxMat10) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  0 ≤ (21) ∧
  (21) < 43 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 43 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (21 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (43 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_6_0 w_7_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_7_0 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_226.constraints_1 backend hashModel tape path params i_0 w_4_0 w_9_0 w_2_0 w_3_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

abbrev sequential_generatedRoot_226.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) (witness : sequential_generatedRoot_226.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_16_0 : mxxMat10 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_13_1 := witness.w_13_1
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_14_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat10 := MxxRuntime.matrixAdd w_16_0 w_17_0
  sequential_generatedRoot_226.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_9_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_13_1 w_14_0 w_15_0 w_17_0 w_18_0 outputs

def sequential_generatedRoot_226 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × mxxMat10 × ByteArray × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat10 × mxxMat10 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_226.Witness,
    sequential_generatedRoot_226.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_228_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_parallel_generatedRoot_228_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_parallel_generatedRoot_228_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_228.Witness where
  w_2_0 : mxxMat3
  w_3_0 : mxxMat3
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat3
  w_13_0 : mxxMat3
  w_14_0 : mxxMat0
  w_15_0 : mxxMat3
  w_16_0 : mxxMat3
  w_17_0 : mxxMat0
  w_18_0 : mxxMat0
  w_18_1 : mxxMat0
  w_19_0 : mxxMat3
  w_20_0 : mxxMat3
  w_21_0 : mxxMat3
  w_22_0 : mxxMat3

abbrev parallel_generatedRoot_228.constraints_2 (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_20_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 0 44 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_1 0 1 44 88 w_22_0 ∧
  outputs = (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))

abbrev parallel_generatedRoot_228.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_4_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_11_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 0 44 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 44 88 w_13_0 ∧
  MxxRuntime.concatColumns w_12_0 w_13_0 w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_parallel_generatedRoot_228_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_9_0, (w_14_0, (w_17_0, ())))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, ())) (w_18_0, (w_18_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_228.constraints_2 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_228.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : ByteArray) (w_5_0 : mxxMat10) (w_10_0 : mxxMat10) (w_0_0 : mxxMat0) (w_2_0 : mxxMat3) (w_3_0 : mxxMat3) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat3) (w_14_0 : mxxMat0) (w_15_0 : mxxMat3) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat3) (w_22_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.concatColumns w_2_0 w_3_0 w_4_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_5_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 0 44 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 44 88 w_8_0 ∧
  MxxRuntime.concatColumns w_7_0 w_8_0 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_228.constraints_1 backend hashModel tape path params i_0 w_1_0 w_10_0 w_0_0 w_4_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

abbrev parallel_generatedRoot_228.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : parallel_generatedRoot_228.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : ByteArray := inputs.1
  let w_5_0 : mxxMat10 := inputs.2.1
  let w_10_0 : mxxMat10 := inputs.2.2.1
  let w_0_0 : mxxMat0 := (0 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  parallel_generatedRoot_228.constraints_0 backend hashModel tape path params i_0 w_1_0 w_5_0 w_10_0 w_0_0 w_2_0 w_3_0 w_4_0 w_6_0 w_7_0 w_8_0 w_9_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_20_0 w_21_0 w_22_0 outputs

def parallel_generatedRoot_228 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × mxxMat10 × mxxMat10 × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_228.Witness,
    parallel_generatedRoot_228.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_229_16.constraints_1 (i_0 : Nat) (i_1 : Nat) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_22_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  outputs = (w_14_0, (w_25_0, ()))

abbrev sequential_sequential_generatedRoot_229_16.constraints_0 (i_0 : Nat) (i_1 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_4_0 : Fin 15 → mxxMat3) (w_16_0 : Fin 15 → mxxMat3) (w_19_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat3) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat2) (w_10_0 : mxxMat3) (w_14_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat3) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_25_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (43 + (-1 * (Int.ofNat i_1))) ∧
  (43 + (-1 * (Int.ofNat i_1))) < (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 (43 + (-1 * (Int.ofNat i_1))) (87 + (-1 * (Int.ofNat i_1))) w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  sequential_sequential_generatedRoot_229_16.constraints_1 i_0 i_1 w_16_0 w_19_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

def sequential_sequential_generatedRoot_229_16 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × mxxMat4 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_15_0 : mxxMat0 := inputs.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_11_0 : mxxMat4 := inputs.2.2.2.2.1
  let w_16_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  ∃ (w_3_0 : mxxMat3),
    ∃ (w_5_0 : mxxMat3),
      ∃ (w_6_0 : mxxMat0),
        ∃ (w_7_0 : mxxMat1),
          ∃ (w_8_0 : mxxMat1),
            let w_9_0 : mxxMat2 := (MxxRuntime.unitRow (43) : mxxMat2)
            ∃ (w_10_0 : mxxMat3),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_10_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_12_0
              let w_14_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_13_0
              let w_17_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
              ∃ (w_18_0 : mxxMat3),
                ∃ (w_20_0 : mxxMat3),
                  ∃ (w_21_0 : mxxMat0),
                    ∃ (w_22_0 : mxxMat1),
                      ∃ (w_23_0 : mxxMat1),
                        let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_23_0 w_12_0
                        let w_25_0 : mxxMat0 := MxxRuntime.matrixAdd w_15_0 w_24_0
                        sequential_sequential_generatedRoot_229_16.constraints_0 i_0 i_1 w_1_0 w_4_0 w_16_0 w_19_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_14_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_25_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_229.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat3
  w_11_0 : mxxMat0
  w_12_0 : mxxMat4
  w_13_0 : mxxMat4
  w_16_0 : mxxMat0
  w_16_1 : mxxMat0
  w_17_0 : mxxMat15
  w_20_0 : mxxMat15

abbrev sequential_generatedRoot_229.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15 → mxxMat3) (w_5_0 : Fin 15 → mxxMat3) (w_6_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_15_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat0) (w_12_0 : mxxMat4) (w_13_0 : mxxMat4) (w_16_0 : mxxMat0) (w_16_1 : mxxMat0) (w_17_0 : mxxMat15) (w_18_0 : mxxMat15) (w_20_0 : mxxMat15) (w_21_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (104 + (-1 * (Int.ofNat i_0))) ∧
  (104 + (-1 * (Int.ofNat i_0))) < (209 + (-1 * (Int.ofNat i_0))) ∧
  (209 + (-1 * (Int.ofNat i_0))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.ofNat i_0))) (209 + (-1 * (Int.ofNat i_0))) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_6_0) w_7_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_7_0 w_8_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_6_0) w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_9_0 w_10_0 ∧
  MxxRuntime.concatColumns w_8_0 w_10_0 w_11_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_11_0 w_12_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_12_0 w_13_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_229_16 backend hashModel tape (path ++ [16, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_5_0, (w_13_0, (w_14_0, (w_15_0, ()))))))) next) (Int.toNat (44)) (w_3_0, (w_3_0, ())) (w_16_0, (w_16_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_16_0 w_17_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_16_1 w_20_0 ∧
  outputs = (w_18_0, (w_21_0, ()))

abbrev sequential_generatedRoot_229.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_229.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_19_0 : mxxMat15 := inputs.2.1
  let w_4_0 : Fin 15 → mxxMat3 := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_6_0 : ByteArray := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_16_0 := witness.w_16_0
  let w_16_1 := witness.w_16_1
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_17_0
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat15 := MxxRuntime.matrixAdd w_19_0 w_20_0
  sequential_generatedRoot_229.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_6_0 w_14_0 w_15_0 w_1_0 w_2_0 w_3_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_16_0 w_16_1 w_17_0 w_18_0 w_20_0 w_21_0 outputs

def sequential_generatedRoot_229 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_229.Witness,
    sequential_generatedRoot_229.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_230_23.constraints_1 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_3_0 : mxxMat3) (w_19_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_27_0 ∧
  MxxRuntime.modulusSwitchRuns w_28_0 w_29_0 ∧
  outputs = (w_19_0, (w_33_0, ()))

abbrev sequential_sequential_generatedRoot_230_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (i_1 : Nat) (w_1_0 : ByteArray) (w_2_0 : mxxMat5) (w_3_0 : mxxMat3) (w_4_0 : mxxMat6) (w_5_0 : mxxMat6) (w_6_0 : mxxMat3) (w_9_0 : mxxMat5) (w_10_0 : mxxMat6) (w_11_0 : mxxMat6) (w_12_0 : mxxMat3) (w_16_0 : mxxMat7) (w_17_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat5) (w_22_0 : mxxMat6) (w_23_0 : mxxMat6) (w_24_0 : mxxMat3) (w_26_0 : mxxMat5) (w_27_0 : mxxMat6) (w_28_0 : mxxMat6) (w_29_0 : mxxMat3) (w_33_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_4_0 ∧
  MxxRuntime.modulusSwitchRuns w_5_0 w_6_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_9_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_10_0 ∧
  MxxRuntime.modulusSwitchRuns w_11_0 w_12_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.ofNat i_1)), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.ofNat i_0)) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_21_0 ∧
  MxxRuntime.centeredRebaseRuns w_3_0 w_22_0 ∧
  MxxRuntime.modulusSwitchRuns w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  sequential_sequential_generatedRoot_230_23.constraints_1 hashModel i_0 i_1 w_1_0 w_3_0 w_19_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

def sequential_sequential_generatedRoot_230_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × ByteArray × mxxMat8 × mxxMat8 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_20_0 : mxxMat0 := inputs.2.1
  let w_1_0 : ByteArray := inputs.2.2.1
  let w_7_0 : mxxMat8 := inputs.2.2.2.1
  let w_13_0 : mxxMat8 := inputs.2.2.2.2.1
  ∃ (w_2_0 : mxxMat5),
    ∃ (w_3_0 : mxxMat3),
      ∃ (w_4_0 : mxxMat6),
        let w_5_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_6_0 : mxxMat3),
          let w_8_0 : mxxMat1 := MxxRuntime.matrixMul w_6_0 w_7_0
          ∃ (w_9_0 : mxxMat5),
            ∃ (w_10_0 : mxxMat6),
              let w_11_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
              ∃ (w_12_0 : mxxMat3),
                let w_14_0 : mxxMat1 := MxxRuntime.matrixMul w_12_0 w_13_0
                let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_8_0 w_14_0
                let w_16_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
                ∃ (w_17_0 : mxxMat0),
                  let w_18_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_15_0 w_17_0
                  let w_19_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  ∃ (w_21_0 : mxxMat5),
                    ∃ (w_22_0 : mxxMat6),
                      let w_23_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_21_0 w_22_0
                      ∃ (w_24_0 : mxxMat3),
                        let w_25_0 : mxxMat1 := MxxRuntime.matrixMul w_24_0 w_7_0
                        ∃ (w_26_0 : mxxMat5),
                          ∃ (w_27_0 : mxxMat6),
                            let w_28_0 : mxxMat6 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
                            ∃ (w_29_0 : mxxMat3),
                              let w_30_0 : mxxMat1 := MxxRuntime.matrixMul w_29_0 w_13_0
                              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_25_0 w_30_0
                              let w_32_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_31_0 w_17_0
                              let w_33_0 : mxxMat0 := MxxRuntime.matrixAdd w_20_0 w_32_0
                              sequential_sequential_generatedRoot_230_23.constraints_0 backend hashModel i_0 i_1 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_29_0 w_33_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_230.Witness where
  w_2_0 : mxxMat14
  w_7_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_12_0 : mxxMat1
  w_13_0 : mxxMat8
  w_16_0 : mxxMat3
  w_18_0 : mxxMat3
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat1
  w_22_0 : mxxMat8
  w_23_0 : mxxMat0
  w_23_1 : mxxMat0
  w_24_0 : mxxMat15
  w_27_0 : mxxMat15

abbrev sequential_generatedRoot_230.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_11_0 w_12_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_12_0 w_13_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_15_0 w_16_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_15_0 w_18_0 ∧
  MxxRuntime.concatColumns w_16_0 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_20_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_20_0 w_21_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_230_23 backend hashModel tape (path ++ [23, i]) params i_0 i (current.1, (current.2.1, (w_4_0, (w_13_0, (w_22_0, ()))))) next) (Int.toNat (88)) (w_3_0, (w_3_0, ())) (w_23_0, (w_23_1, ())) ∧
  MxxRuntime.tensorRuns w_2_0 w_23_0 w_24_0 ∧
  MxxRuntime.tensorRuns w_2_0 w_23_1 w_27_0 ∧
  outputs = (w_25_0, (w_28_0, ()))

abbrev sequential_generatedRoot_230.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : ByteArray) (w_5_0 : Fin 15 → mxxMat3) (w_8_0 : Fin 15 → mxxMat3) (w_14_0 : Fin 15 → mxxMat3) (w_17_0 : Fin 15 → mxxMat3) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat0) (w_6_0 : Int) (w_7_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat8) (w_15_0 : Int) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_22_0 : mxxMat8) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat15) (w_25_0 : mxxMat15) (w_27_0 : mxxMat15) (w_28_0 : mxxMat15) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  0 ≤ (104) ∧
  (104) < 209 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) < (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ∧
  (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) ≤ 209 ∧
  MxxRuntime.sliceMatrix w_1_0 0 1 (104 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) (209 + (-1 * (Int.fdiv ((Int.ofNat i_0)) (44)))) w_2_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_6_0 w_7_0 ∧
  0 ≤ w_6_0 ∧
  w_6_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_8_0 w_6_0 w_9_0 ∧
  MxxRuntime.concatColumns w_7_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_11_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_230.constraints_1 backend hashModel tape path params i_0 w_4_0 w_14_0 w_17_0 w_2_0 w_3_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_230.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) (witness : sequential_generatedRoot_230.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_26_0 : mxxMat15 := inputs.2.1
  let w_4_0 : ByteArray := inputs.2.2.1
  let w_5_0 : Fin 15 → mxxMat3 := inputs.2.2.2.1
  let w_8_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_14_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.1
  let w_17_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat0 := (0 : mxxMat0)
  let w_6_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_15_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_23_1 := witness.w_23_1
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_24_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat15 := MxxRuntime.matrixAdd w_26_0 w_27_0
  sequential_generatedRoot_230.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_8_0 w_14_0 w_17_0 w_1_0 w_2_0 w_3_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_23_1 w_24_0 w_25_0 w_27_0 w_28_0 outputs

def sequential_generatedRoot_230 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × mxxMat15 × ByteArray × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat15 × mxxMat15 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_230.Witness,
    sequential_generatedRoot_230.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_235.constraints_0 (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_generatedRoot_235 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_0)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat1 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat1 := (1 : mxxMat1)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat1),
      let w_14_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_generatedRoot_235.constraints_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

abbrev sequential_sequential_generatedRoot_244_7.constraints_0 (i_0 : Nat) (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_sequential_generatedRoot_244_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat1 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat1 := (1 : mxxMat1)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat1),
      let w_14_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_sequential_generatedRoot_244_7.constraints_0 i_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_244.Witness where
  w_3_0 : mxxMat3
  w_7_0 : mxxMat1
  w_9_0 : mxxMat3
  w_11_0 : mxxMat17
  w_16_0 : mxxMat3
  w_21_0 : mxxMat3
  w_26_0 : mxxMat3

abbrev sequential_generatedRoot_244.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 15 → mxxMat3) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat3) (w_20_0 : Fin 15 → mxxMat3) (w_25_0 : Fin 15 → mxxMat3) (w_2_0 : Int) (w_3_0 : mxxMat3) (w_5_0 : mxxMat1) (w_7_0 : mxxMat1) (w_9_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat17) (w_13_0 : mxxMat3) (w_16_0 : mxxMat3) (w_18_0 : mxxMat3) (w_21_0 : mxxMat3) (w_23_0 : mxxMat3) (w_26_0 : mxxMat3) (w_28_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_244_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_2_0 w_21_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_2_0 w_26_0 ∧
  outputs = (w_13_0, (w_18_0, (w_23_0, (w_28_0, ()))))

abbrev sequential_generatedRoot_244.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_244.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_14_0 : mxxMat3 := inputs.2.1
  let w_19_0 : mxxMat3 := inputs.2.2.1
  let w_24_0 : mxxMat3 := inputs.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.1
  let w_20_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat3 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_5_0 : mxxMat1 := (0 : mxxMat1)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat3 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat3 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat3 := MxxRuntime.matrixAdd w_14_0 w_17_0
  let w_21_0 := witness.w_21_0
  let w_22_0 : mxxMat3 := MxxRuntime.matrixMul w_21_0 w_11_0
  let w_23_0 : mxxMat3 := MxxRuntime.matrixAdd w_19_0 w_22_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMul w_26_0 w_11_0
  let w_28_0 : mxxMat3 := MxxRuntime.matrixAdd w_24_0 w_27_0
  sequential_generatedRoot_244.constraints_0 backend hashModel tape path params i_0 w_1_0 w_6_0 w_15_0 w_20_0 w_25_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 w_21_0 w_23_0 w_26_0 w_28_0 outputs

def sequential_generatedRoot_244 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × (Fin 15 → mxxMat3) × (Fin 512 → Int) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × (Fin 15 → mxxMat3) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_244.Witness,
    sequential_generatedRoot_244.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_245_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_245_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_245_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_245_24.constraints_0 (i_0 : Nat) (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_sequential_generatedRoot_245_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat1 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat1 := (1 : mxxMat1)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat1),
      let w_14_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_sequential_generatedRoot_245_24.constraints_0 i_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_245.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_245.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_245_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_245.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_245_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_245.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_245.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat10) (w_11_0 : mxxMat10) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_245.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_245.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_245.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat10 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat10 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_245.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_245 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat10 × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_245.Witness,
    sequential_generatedRoot_245.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_246_19.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat0) (w_4_0 : mxxMat0) (w_3_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat12) (w_7_0 : mxxMat13) (w_8_0 : mxxMat13) (w_10_0 : mxxMat7) (w_11_0 : mxxMat0) (w_13_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_1) ∧
  (Int.ofNat i_1) < (1 + (Int.ofNat i_1)) ∧
  (1 + (Int.ofNat i_1)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_1) (1 + (Int.ofNat i_1)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_11_0 ∧
  outputs = (w_13_0, (w_18_0, ()))

def sequential_sequential_generatedRoot_246_19 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_1_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_15_0 : mxxMat0 := inputs.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat1),
    ∃ (w_5_0 : mxxMat1),
      ∃ (w_6_0 : mxxMat12),
        ∃ (w_7_0 : mxxMat13),
          ∃ (w_8_0 : mxxMat13),
            let w_9_0 : mxxMat1 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat7 := (MxxRuntime.unitRow (87) : mxxMat7)
            ∃ (w_11_0 : mxxMat0),
              let w_12_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
              sequential_sequential_generatedRoot_246_19.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_246_24.constraints_0 (i_0 : Nat) (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_sequential_generatedRoot_246_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × (Fin 512 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat1 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat1 := (1 : mxxMat1)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat1),
      let w_14_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_sequential_generatedRoot_246_24.constraints_0 i_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_246.Witness where
  w_3_0 : mxxMat3
  w_4_0 : mxxMat3
  w_5_0 : mxxMat0
  w_7_0 : mxxMat0
  w_8_0 : mxxMat3
  w_9_0 : mxxMat3
  w_10_0 : mxxMat0
  w_12_0 : mxxMat0
  w_13_0 : mxxMat3
  w_14_0 : mxxMat3
  w_15_0 : mxxMat0
  w_16_0 : mxxMat3
  w_17_0 : mxxMat3
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_24_0 : mxxMat1
  w_26_0 : mxxMat3
  w_28_0 : mxxMat17
  w_32_0 : mxxMat3
  w_36_0 : mxxMat3
  w_40_0 : mxxMat3

abbrev sequential_generatedRoot_246.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_23_0 : Fin 512 → Int) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_246_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current, (w_23_0, ())) next) (Int.toNat (128)) w_22_0 w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_26_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_27_0 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_40_0 ∧
  outputs = (w_30_0, (w_34_0, (w_38_0, (w_42_0, ()))))

abbrev sequential_generatedRoot_246.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_5_0 : mxxMat0) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 0 44 w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 44 88 w_14_0 ∧
  MxxRuntime.concatColumns w_13_0 w_14_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_17_0 ∧
  MxxRuntime.concatColumns w_16_0 w_17_0 w_18_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_246_19 backend hashModel tape (path ++ [19, i]) params i_0 i (current.1, (current.2.1, (w_5_0, (w_10_0, (w_15_0, (w_18_0, ())))))) next) (Int.toNat (88)) (w_1_0, (w_1_0, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_246.constraints_2 backend hashModel tape path params i_0 w_23_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_246.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_2_0 : ByteArray) (w_6_0 : mxxMat15) (w_11_0 : mxxMat15) (w_23_0 : Fin 512 → Int) (w_1_0 : mxxMat0) (w_3_0 : mxxMat3) (w_4_0 : mxxMat3) (w_5_0 : mxxMat0) (w_7_0 : mxxMat0) (w_8_0 : mxxMat3) (w_9_0 : mxxMat3) (w_10_0 : mxxMat0) (w_12_0 : mxxMat0) (w_13_0 : mxxMat3) (w_14_0 : mxxMat3) (w_15_0 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat3) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat17) (w_30_0 : mxxMat3) (w_32_0 : mxxMat3) (w_34_0 : mxxMat3) (w_36_0 : mxxMat3) (w_38_0 : mxxMat3) (w_40_0 : mxxMat3) (w_42_0 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_4_0 ∧
  MxxRuntime.concatColumns w_3_0 w_4_0 w_5_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_6_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 0 44 w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 44 88 w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_246.constraints_1 backend hashModel tape path params i_0 w_2_0 w_11_0 w_23_0 w_1_0 w_5_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

abbrev sequential_generatedRoot_246.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : sequential_generatedRoot_246.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat3 := inputs.1
  let w_31_0 : mxxMat3 := inputs.2.1
  let w_35_0 : mxxMat3 := inputs.2.2.1
  let w_39_0 : mxxMat3 := inputs.2.2.2.1
  let w_2_0 : ByteArray := inputs.2.2.2.2.1
  let w_6_0 : mxxMat15 := inputs.2.2.2.2.2.1
  let w_11_0 : mxxMat15 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (0 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_22_0 : mxxMat1 := (0 : mxxMat1)
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_25_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat3 := MxxRuntime.matrixMul w_20_0 w_28_0
  let w_30_0 : mxxMat3 := MxxRuntime.matrixAdd w_0_0 w_29_0
  let w_32_0 := witness.w_32_0
  let w_33_0 : mxxMat3 := MxxRuntime.matrixMul w_32_0 w_28_0
  let w_34_0 : mxxMat3 := MxxRuntime.matrixAdd w_31_0 w_33_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat3 := MxxRuntime.matrixMul w_36_0 w_28_0
  let w_38_0 : mxxMat3 := MxxRuntime.matrixAdd w_35_0 w_37_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_28_0
  let w_42_0 : mxxMat3 := MxxRuntime.matrixAdd w_39_0 w_41_0
  sequential_generatedRoot_246.constraints_0 backend hashModel tape path params i_0 w_2_0 w_6_0 w_11_0 w_23_0 w_1_0 w_3_0 w_4_0 w_5_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_19_1 w_20_0 w_22_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 outputs

def sequential_generatedRoot_246 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × ByteArray × mxxMat15 × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_246.Witness,
    sequential_generatedRoot_246.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_1_0 : Int
  w_4_0 : mxxMat3
  w_5_0 : mxxMat3
  w_6_0 : mxxMat0
  w_7_0 : mxxMat3
  w_8_0 : mxxMat3
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_10_1 : mxxMat0
  w_11_0 : mxxMat0
  w_11_1 : mxxMat0
  w_12_0 : mxxMat3
  w_14_0 : mxxMat0
  w_14_1 : mxxMat0
  w_15_0 : mxxMat0
  w_15_1 : mxxMat0
  w_16_0 : mxxMat3
  w_18_0 : mxxMat0
  w_18_1 : mxxMat0
  w_19_0 : mxxMat0
  w_19_1 : mxxMat0
  w_20_0 : mxxMat3
  w_22_0 : mxxMat0
  w_22_1 : mxxMat0
  w_23_0 : mxxMat0
  w_23_1 : mxxMat0
  w_24_0 : mxxMat3
  w_25_0 : mxxMat3
  w_26_0 : mxxMat3
  w_27_0 : mxxMat3
  w_28_0 : mxxMat3
  w_29_0 : mxxMat3
  w_30_0 : mxxMat3
  w_31_0 : mxxMat3
  w_32_0 : mxxMat3
  w_33_0 : mxxMat3
  w_34_0 : mxxMat3
  w_35_0 : mxxMat3
  w_36_0 : mxxMat3
  w_37_0 : mxxMat3
  w_38_0 : mxxMat3
  w_39_0 : mxxMat3
  w_40_0 : mxxMat3
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat10
  w_44_1 : mxxMat10
  w_45_0 : mxxMat10
  w_45_1 : mxxMat10
  w_47_0 : Fin 15 → mxxMat3
  w_47_1 : Fin 15 → mxxMat3
  w_47_2 : Fin 15 → mxxMat3
  w_47_3 : Fin 15 → mxxMat3
  w_48_0 : mxxMat15
  w_48_1 : mxxMat15
  w_49_0 : mxxMat15
  w_49_1 : mxxMat15
  w_50_0 : mxxMat18
  w_51_0 : mxxMat18
  w_55_0 : mxxMat1
  w_57_0 : mxxMat3
  w_59_0 : mxxMat17
  w_64_0 : mxxMat3
  w_64_1 : mxxMat3
  w_64_2 : mxxMat3
  w_64_3 : mxxMat3
  w_65_0 : mxxMat3
  w_65_1 : mxxMat3
  w_65_2 : mxxMat3
  w_65_3 : mxxMat3
  w_66_0 : mxxMat3
  w_66_1 : mxxMat3
  w_66_2 : mxxMat3
  w_66_3 : mxxMat3
  w_67_0 : Int
  w_69_0 : mxxMat0
  w_70_0 : mxxMat0
  w_71_0 : mxxMat0
  w_71_1 : mxxMat0
  w_72_0 : mxxMat0
  w_72_1 : mxxMat0
  w_73_0 : mxxMat3
  w_75_0 : mxxMat0
  w_75_1 : mxxMat0
  w_76_0 : mxxMat0
  w_76_1 : mxxMat0
  w_77_0 : mxxMat3
  w_79_0 : mxxMat0
  w_79_1 : mxxMat0
  w_80_0 : mxxMat0
  w_80_1 : mxxMat0
  w_81_0 : mxxMat3
  w_83_0 : mxxMat0
  w_83_1 : mxxMat0
  w_84_0 : mxxMat0
  w_84_1 : mxxMat0
  w_85_0 : mxxMat3
  w_86_0 : mxxMat3
  w_87_0 : mxxMat3
  w_88_0 : mxxMat3
  w_89_0 : mxxMat3
  w_90_0 : mxxMat3
  w_91_0 : mxxMat3
  w_92_0 : mxxMat3
  w_93_0 : mxxMat3
  w_94_0 : mxxMat3
  w_95_0 : mxxMat3
  w_96_0 : mxxMat3
  w_97_0 : mxxMat3
  w_98_0 : mxxMat3
  w_99_0 : mxxMat3
  w_100_0 : mxxMat3
  w_101_0 : mxxMat3
  w_103_0 : mxxMat0
  w_104_0 : mxxMat0
  w_105_0 : mxxMat10
  w_105_1 : mxxMat10
  w_106_0 : mxxMat10
  w_106_1 : mxxMat10
  w_108_0 : Fin 15 → mxxMat3
  w_108_1 : Fin 15 → mxxMat3
  w_108_2 : Fin 15 → mxxMat3
  w_108_3 : Fin 15 → mxxMat3
  w_109_0 : mxxMat15
  w_109_1 : mxxMat15
  w_110_0 : mxxMat15
  w_110_1 : mxxMat15
  w_111_0 : mxxMat18
  w_112_0 : mxxMat18
  w_115_0 : mxxMat1
  w_117_0 : mxxMat3
  w_119_0 : mxxMat17
  w_124_0 : mxxMat3
  w_124_1 : mxxMat3
  w_124_2 : mxxMat3
  w_124_3 : mxxMat3
  w_125_0 : mxxMat3
  w_125_1 : mxxMat3
  w_125_2 : mxxMat3
  w_125_3 : mxxMat3
  w_126_0 : mxxMat3
  w_126_1 : mxxMat3
  w_126_2 : mxxMat3
  w_126_3 : mxxMat3
  w_127_0 : Int
  w_129_0 : mxxMat0
  w_130_0 : mxxMat0
  w_131_0 : mxxMat0
  w_131_1 : mxxMat0
  w_132_0 : mxxMat0
  w_132_1 : mxxMat0
  w_133_0 : mxxMat3
  w_135_0 : mxxMat0
  w_135_1 : mxxMat0
  w_136_0 : mxxMat0
  w_136_1 : mxxMat0
  w_137_0 : mxxMat3
  w_139_0 : mxxMat0
  w_139_1 : mxxMat0
  w_140_0 : mxxMat0
  w_140_1 : mxxMat0
  w_141_0 : mxxMat3
  w_143_0 : mxxMat0
  w_143_1 : mxxMat0
  w_144_0 : mxxMat0
  w_144_1 : mxxMat0
  w_145_0 : mxxMat3
  w_146_0 : mxxMat3
  w_147_0 : mxxMat3
  w_148_0 : mxxMat3
  w_149_0 : mxxMat3
  w_150_0 : mxxMat3
  w_151_0 : mxxMat3
  w_152_0 : mxxMat3
  w_153_0 : mxxMat3
  w_154_0 : mxxMat3
  w_155_0 : mxxMat3
  w_156_0 : mxxMat3
  w_157_0 : mxxMat3
  w_158_0 : mxxMat3
  w_159_0 : mxxMat3
  w_160_0 : mxxMat3
  w_161_0 : mxxMat3
  w_163_0 : mxxMat0
  w_164_0 : mxxMat0
  w_165_0 : mxxMat10
  w_165_1 : mxxMat10
  w_166_0 : mxxMat10
  w_166_1 : mxxMat10
  w_168_0 : Fin 15 → mxxMat3
  w_168_1 : Fin 15 → mxxMat3
  w_168_2 : Fin 15 → mxxMat3
  w_168_3 : Fin 15 → mxxMat3
  w_169_0 : mxxMat15
  w_169_1 : mxxMat15
  w_170_0 : mxxMat15
  w_170_1 : mxxMat15
  w_171_0 : mxxMat18
  w_172_0 : mxxMat18
  w_175_0 : mxxMat1
  w_177_0 : mxxMat3
  w_179_0 : mxxMat17
  w_184_0 : mxxMat3
  w_184_1 : mxxMat3
  w_184_2 : mxxMat3
  w_184_3 : mxxMat3
  w_185_0 : mxxMat3
  w_185_1 : mxxMat3
  w_185_2 : mxxMat3
  w_185_3 : mxxMat3
  w_186_0 : mxxMat3
  w_186_1 : mxxMat3
  w_186_2 : mxxMat3
  w_186_3 : mxxMat3
  w_187_0 : Int
  w_189_0 : mxxMat0
  w_190_0 : mxxMat0
  w_191_0 : mxxMat0
  w_191_1 : mxxMat0
  w_192_0 : mxxMat0
  w_192_1 : mxxMat0
  w_193_0 : mxxMat3
  w_195_0 : mxxMat0
  w_195_1 : mxxMat0
  w_196_0 : mxxMat0
  w_196_1 : mxxMat0
  w_197_0 : mxxMat3
  w_199_0 : mxxMat0
  w_199_1 : mxxMat0
  w_200_0 : mxxMat0
  w_200_1 : mxxMat0
  w_201_0 : mxxMat3
  w_203_0 : mxxMat0
  w_203_1 : mxxMat0
  w_204_0 : mxxMat0
  w_204_1 : mxxMat0
  w_205_0 : mxxMat3
  w_206_0 : mxxMat3
  w_207_0 : mxxMat3
  w_208_0 : mxxMat3
  w_209_0 : mxxMat3
  w_210_0 : mxxMat3
  w_211_0 : mxxMat3
  w_212_0 : mxxMat3
  w_213_0 : mxxMat3
  w_214_0 : mxxMat3
  w_215_0 : mxxMat3
  w_216_0 : mxxMat3
  w_217_0 : mxxMat3
  w_218_0 : mxxMat3
  w_219_0 : mxxMat3
  w_220_0 : mxxMat3
  w_221_0 : mxxMat3
  w_223_0 : mxxMat0
  w_224_0 : mxxMat0
  w_225_0 : mxxMat10
  w_225_1 : mxxMat10
  w_226_0 : mxxMat10
  w_226_1 : mxxMat10
  w_228_0 : Fin 15 → mxxMat3
  w_228_1 : Fin 15 → mxxMat3
  w_228_2 : Fin 15 → mxxMat3
  w_228_3 : Fin 15 → mxxMat3
  w_229_0 : mxxMat15
  w_229_1 : mxxMat15
  w_230_0 : mxxMat15
  w_230_1 : mxxMat15
  w_231_0 : mxxMat18
  w_232_0 : mxxMat18
  w_235_0 : mxxMat1
  w_237_0 : mxxMat3
  w_239_0 : mxxMat17
  w_244_0 : mxxMat3
  w_244_1 : mxxMat3
  w_244_2 : mxxMat3
  w_244_3 : mxxMat3
  w_245_0 : mxxMat3
  w_245_1 : mxxMat3
  w_245_2 : mxxMat3
  w_245_3 : mxxMat3
  w_246_0 : mxxMat3
  w_246_1 : mxxMat3
  w_246_2 : mxxMat3
  w_246_3 : mxxMat3

abbrev generatedRoot.constraints_21 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_161_0 : mxxMat3) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_206_0 : mxxMat3) (w_211_0 : mxxMat3) (w_216_0 : mxxMat3) (w_221_0 : mxxMat3) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_229 backend hashModel tape (path ++ [229, i]) params i (current.1, (current.2.1, (w_228_0, (w_228_1, (w_3_0, (w_228_2, (w_228_3, ()))))))) next) (Int.toNat (105)) (w_227_0, (w_227_0, ())) (w_229_0, (w_229_1, ())) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_230 backend hashModel tape (path ++ [230, i]) params i (current.1, (current.2.1, (w_3_0, (w_228_0, (w_228_1, (w_228_2, (w_228_3, ()))))))) next) (Int.toNat (4620)) (w_229_0, (w_229_1, ())) (w_230_0, (w_230_1, ())) ∧
  MxxRuntime.concatColumns w_226_0 w_230_0 w_231_0 ∧
  MxxRuntime.concatColumns w_226_1 w_230_1 w_232_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_235 backend hashModel tape (path ++ [235, i]) params i (current, (w_54_0, ())) next) (Int.toNat (128)) w_234_0 w_235_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_237_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_238_0 w_239_0 ∧
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_244 backend hashModel tape (path ++ [244, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_228_0, (w_54_0, (w_228_1, (w_228_2, (w_228_3, ()))))))))) next) (Int.toNat (15)) (w_240_0, (w_241_0, (w_242_0, (w_243_0, ())))) (w_244_0, (w_244_1, (w_244_2, (w_244_3, ())))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_245 backend hashModel tape (path ++ [245, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_226_0, (w_226_1, (w_54_0, ())))))))) next) (Int.toNat (7)) (w_244_0, (w_244_1, (w_244_2, (w_244_3, ())))) (w_245_0, (w_245_1, (w_245_2, (w_245_3, ())))) ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_246 backend hashModel tape (path ++ [246, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_230_0, (w_230_1, (w_54_0, ())))))))) next) (Int.toNat (105)) (w_245_0, (w_245_1, (w_245_2, (w_245_3, ())))) (w_246_0, (w_246_1, (w_246_2, (w_246_3, ())))) ∧
  outputs = (w_25_0, (w_30_0, (w_35_0, (w_40_0, (w_50_0, (w_51_0, (w_66_0, (w_66_1, (w_66_2, (w_66_3, (w_86_0, (w_91_0, (w_96_0, (w_101_0, (w_111_0, (w_112_0, (w_126_0, (w_126_1, (w_126_2, (w_126_3, (w_146_0, (w_151_0, (w_156_0, (w_161_0, (w_171_0, (w_172_0, (w_186_0, (w_186_1, (w_186_2, (w_186_3, (w_206_0, (w_211_0, (w_216_0, (w_221_0, (w_231_0, (w_232_0, (w_246_0, (w_246_1, (w_246_2, (w_246_3, (w_246_0, (w_246_1, (w_246_2, (w_246_3, ()))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_20 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_161_0 : mxxMat3) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_196_1 : mxxMat0) (w_200_1 : mxxMat0) (w_204_1 : mxxMat0) (w_206_0 : mxxMat3) (w_211_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_196_1 0 1 44 88 w_218_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_200_1 0 1 44 88 w_219_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_204_1 0 1 44 88 w_220_0 ∧
  0 ≤ w_187_0 ∧
  w_187_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_187_0 [w_217_0, w_218_0, w_219_0, w_220_0] w_221_0 ∧
  MxxRuntime.concatColumns w_206_0 w_211_0 w_223_0 ∧
  MxxRuntime.concatColumns w_216_0 w_221_0 w_224_0 ∧
  0 ≤ (22) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_225 backend hashModel tape (path ++ [225, i]) params i (current.1, (current.2.1, (w_223_0, (w_3_0, (w_224_0, ()))))) next) (Int.toNat (22)) (w_222_0, (w_222_0, ())) (w_225_0, (w_225_1, ())) ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_226 backend hashModel tape (path ++ [226, i]) params i (current.1, (current.2.1, (w_3_0, (w_223_0, (w_224_0, ()))))) next) (Int.toNat (968)) (w_225_0, (w_225_1, ())) (w_226_0, (w_226_1, ())) ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_228 backend hashModel tape (path ++ [228, i.val]) params i (w_3_0, (w_226_0, (w_226_1, ()))) ((w_228_0 i), ((w_228_1 i), ((w_228_2 i), ((w_228_3 i), ()))))) ∧
  generatedRoot.constraints_21 backend hashModel tape path params w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_146_0 w_151_0 w_156_0 w_161_0 w_171_0 w_172_0 w_186_0 w_186_1 w_186_2 w_186_3 w_206_0 w_211_0 w_216_0 w_221_0 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_19 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_161_0 : mxxMat3) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_192_1 : mxxMat0) (w_196_1 : mxxMat0) (w_200_1 : mxxMat0) (w_204_1 : mxxMat0) (w_206_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_196_1 0 1 0 44 w_213_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_200_1 0 1 0 44 w_214_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_204_1 0 1 0 44 w_215_0 ∧
  0 ≤ w_187_0 ∧
  w_187_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_187_0 [w_212_0, w_213_0, w_214_0, w_215_0] w_216_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_192_1 0 1 44 88 w_217_0 ∧
  0 ≤ 0 ∧
  generatedRoot.constraints_20 backend hashModel tape path params w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_146_0 w_151_0 w_156_0 w_161_0 w_171_0 w_172_0 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_196_1 w_200_1 w_204_1 w_206_0 w_211_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_161_0 : mxxMat3) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_192_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_196_0 0 1 44 88 w_208_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_200_0 0 1 44 88 w_209_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_204_0 0 1 44 88 w_210_0 ∧
  0 ≤ w_187_0 ∧
  w_187_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_187_0 [w_207_0, w_208_0, w_209_0, w_210_0] w_211_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_192_1 0 1 0 44 w_212_0 ∧
  0 ≤ 0 ∧
  generatedRoot.constraints_19 backend hashModel tape path params w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_146_0 w_151_0 w_156_0 w_161_0 w_171_0 w_172_0 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_192_1 w_196_1 w_200_1 w_204_1 w_206_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_17 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_161_0 : mxxMat3) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_200 backend hashModel tape (path ++ [200, i]) params i (current.1, (current.2.1, (w_3_0, (w_189_0, (w_190_0, ()))))) next) (Int.toNat (44)) (w_199_0, (w_199_1, ())) (w_200_0, (w_200_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_200_0 0 1 0 44 w_201_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_203 backend hashModel tape (path ++ [203, i]) params i (current.1, (current.2.1, (w_189_0, (w_3_0, (w_190_0, ()))))) next) (Int.toNat (1)) (w_202_0, (w_202_0, ())) (w_203_0, (w_203_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_204 backend hashModel tape (path ++ [204, i]) params i (current.1, (current.2.1, (w_3_0, (w_189_0, (w_190_0, ()))))) next) (Int.toNat (44)) (w_203_0, (w_203_1, ())) (w_204_0, (w_204_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_204_0 0 1 0 44 w_205_0 ∧
  0 ≤ w_187_0 ∧
  w_187_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_187_0 [w_193_0, w_197_0, w_201_0, w_205_0] w_206_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_192_0 0 1 44 88 w_207_0 ∧
  0 ≤ 0 ∧
  generatedRoot.constraints_18 backend hashModel tape path params w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_146_0 w_151_0 w_156_0 w_161_0 w_171_0 w_172_0 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_192_1 w_196_0 w_196_1 w_200_0 w_200_1 w_204_0 w_204_1 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_161_0 : mxxMat3) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_184 backend hashModel tape (path ++ [184, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_168_0, (w_54_0, (w_168_1, (w_168_2, (w_168_3, ()))))))))) next) (Int.toNat (15)) (w_180_0, (w_181_0, (w_182_0, (w_183_0, ())))) (w_184_0, (w_184_1, (w_184_2, (w_184_3, ())))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_185 backend hashModel tape (path ++ [185, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_166_0, (w_166_1, (w_54_0, ())))))))) next) (Int.toNat (7)) (w_184_0, (w_184_1, (w_184_2, (w_184_3, ())))) (w_185_0, (w_185_1, (w_185_2, (w_185_3, ())))) ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_186 backend hashModel tape (path ++ [186, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_170_0, (w_170_1, (w_54_0, ())))))))) next) (Int.toNat (105)) (w_185_0, (w_185_1, (w_185_2, (w_185_3, ())))) (w_186_0, (w_186_1, (w_186_2, (w_186_3, ())))) ∧
  MxxRuntime.familyGetStatic w_0_0 (3) w_187_0 ∧
  MxxRuntime.concatColumns w_186_0 w_186_1 w_189_0 ∧
  MxxRuntime.concatColumns w_186_2 w_186_3 w_190_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_191 backend hashModel tape (path ++ [191, i]) params i (current.1, (current.2.1, (w_189_0, (w_3_0, (w_190_0, ()))))) next) (Int.toNat (1)) (w_188_0, (w_188_0, ())) (w_191_0, (w_191_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_192 backend hashModel tape (path ++ [192, i]) params i (current.1, (current.2.1, (w_3_0, (w_189_0, (w_190_0, ()))))) next) (Int.toNat (44)) (w_191_0, (w_191_1, ())) (w_192_0, (w_192_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_192_0 0 1 0 44 w_193_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_195 backend hashModel tape (path ++ [195, i]) params i (current.1, (current.2.1, (w_189_0, (w_3_0, (w_190_0, ()))))) next) (Int.toNat (1)) (w_194_0, (w_194_0, ())) (w_195_0, (w_195_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_196 backend hashModel tape (path ++ [196, i]) params i (current.1, (current.2.1, (w_3_0, (w_189_0, (w_190_0, ()))))) next) (Int.toNat (44)) (w_195_0, (w_195_1, ())) (w_196_0, (w_196_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_196_0 0 1 0 44 w_197_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_199 backend hashModel tape (path ++ [199, i]) params i (current.1, (current.2.1, (w_189_0, (w_3_0, (w_190_0, ()))))) next) (Int.toNat (1)) (w_198_0, (w_198_0, ())) (w_199_0, (w_199_1, ())) ∧
  generatedRoot.constraints_17 backend hashModel tape path params w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_146_0 w_151_0 w_156_0 w_161_0 w_171_0 w_172_0 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_189_0 w_190_0 w_192_0 w_192_1 w_193_0 w_196_0 w_196_1 w_197_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_140_1 : mxxMat0) (w_144_1 : mxxMat0) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_140_1 0 1 44 88 w_159_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_144_1 0 1 44 88 w_160_0 ∧
  0 ≤ w_127_0 ∧
  w_127_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_127_0 [w_157_0, w_158_0, w_159_0, w_160_0] w_161_0 ∧
  MxxRuntime.concatColumns w_146_0 w_151_0 w_163_0 ∧
  MxxRuntime.concatColumns w_156_0 w_161_0 w_164_0 ∧
  0 ≤ (22) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_165 backend hashModel tape (path ++ [165, i]) params i (current.1, (current.2.1, (w_163_0, (w_3_0, (w_164_0, ()))))) next) (Int.toNat (22)) (w_162_0, (w_162_0, ())) (w_165_0, (w_165_1, ())) ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_166 backend hashModel tape (path ++ [166, i]) params i (current.1, (current.2.1, (w_3_0, (w_163_0, (w_164_0, ()))))) next) (Int.toNat (968)) (w_165_0, (w_165_1, ())) (w_166_0, (w_166_1, ())) ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_168 backend hashModel tape (path ++ [168, i.val]) params i (w_3_0, (w_166_0, (w_166_1, ()))) ((w_168_0 i), ((w_168_1 i), ((w_168_2 i), ((w_168_3 i), ()))))) ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_169 backend hashModel tape (path ++ [169, i]) params i (current.1, (current.2.1, (w_168_0, (w_168_1, (w_3_0, (w_168_2, (w_168_3, ()))))))) next) (Int.toNat (105)) (w_167_0, (w_167_0, ())) (w_169_0, (w_169_1, ())) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_170 backend hashModel tape (path ++ [170, i]) params i (current.1, (current.2.1, (w_3_0, (w_168_0, (w_168_1, (w_168_2, (w_168_3, ()))))))) next) (Int.toNat (4620)) (w_169_0, (w_169_1, ())) (w_170_0, (w_170_1, ())) ∧
  MxxRuntime.concatColumns w_166_0 w_170_0 w_171_0 ∧
  MxxRuntime.concatColumns w_166_1 w_170_1 w_172_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_175 backend hashModel tape (path ++ [175, i]) params i (current, (w_54_0, ())) next) (Int.toNat (128)) w_174_0 w_175_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_177_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_178_0 w_179_0 ∧
  0 ≤ (15) ∧
  generatedRoot.constraints_16 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_146_0 w_151_0 w_156_0 w_161_0 w_166_0 w_166_1 w_168_0 w_168_1 w_168_2 w_168_3 w_170_0 w_170_1 w_171_0 w_172_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_132_1 : mxxMat0) (w_136_1 : mxxMat0) (w_140_1 : mxxMat0) (w_144_1 : mxxMat0) (w_146_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_140_1 0 1 0 44 w_154_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_144_1 0 1 0 44 w_155_0 ∧
  0 ≤ w_127_0 ∧
  w_127_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_127_0 [w_152_0, w_153_0, w_154_0, w_155_0] w_156_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_132_1 0 1 44 88 w_157_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_136_1 0 1 44 88 w_158_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  generatedRoot.constraints_15 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_140_1 w_144_1 w_146_0 w_151_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_132_1 : mxxMat0) (w_136_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_140_0 0 1 44 88 w_149_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_144_0 0 1 44 88 w_150_0 ∧
  0 ≤ w_127_0 ∧
  w_127_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_127_0 [w_147_0, w_148_0, w_149_0, w_150_0] w_151_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_132_1 0 1 0 44 w_152_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_136_1 0 1 0 44 w_153_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  generatedRoot.constraints_14 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_132_1 w_136_1 w_140_1 w_144_1 w_146_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_144 backend hashModel tape (path ++ [144, i]) params i (current.1, (current.2.1, (w_3_0, (w_129_0, (w_130_0, ()))))) next) (Int.toNat (44)) (w_143_0, (w_143_1, ())) (w_144_0, (w_144_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_144_0 0 1 0 44 w_145_0 ∧
  0 ≤ w_127_0 ∧
  w_127_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_127_0 [w_133_0, w_137_0, w_141_0, w_145_0] w_146_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_132_0 0 1 44 88 w_147_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_136_0 0 1 44 88 w_148_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  generatedRoot.constraints_13 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_132_1 w_136_1 w_140_0 w_140_1 w_144_0 w_144_1 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_101_0 : mxxMat3) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_132 backend hashModel tape (path ++ [132, i]) params i (current.1, (current.2.1, (w_3_0, (w_129_0, (w_130_0, ()))))) next) (Int.toNat (44)) (w_131_0, (w_131_1, ())) (w_132_0, (w_132_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_132_0 0 1 0 44 w_133_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_135 backend hashModel tape (path ++ [135, i]) params i (current.1, (current.2.1, (w_129_0, (w_3_0, (w_130_0, ()))))) next) (Int.toNat (1)) (w_134_0, (w_134_0, ())) (w_135_0, (w_135_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_136 backend hashModel tape (path ++ [136, i]) params i (current.1, (current.2.1, (w_3_0, (w_129_0, (w_130_0, ()))))) next) (Int.toNat (44)) (w_135_0, (w_135_1, ())) (w_136_0, (w_136_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_136_0 0 1 0 44 w_137_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_139 backend hashModel tape (path ++ [139, i]) params i (current.1, (current.2.1, (w_129_0, (w_3_0, (w_130_0, ()))))) next) (Int.toNat (1)) (w_138_0, (w_138_0, ())) (w_139_0, (w_139_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_140 backend hashModel tape (path ++ [140, i]) params i (current.1, (current.2.1, (w_3_0, (w_129_0, (w_130_0, ()))))) next) (Int.toNat (44)) (w_139_0, (w_139_1, ())) (w_140_0, (w_140_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_140_0 0 1 0 44 w_141_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_143 backend hashModel tape (path ++ [143, i]) params i (current.1, (current.2.1, (w_129_0, (w_3_0, (w_130_0, ()))))) next) (Int.toNat (1)) (w_142_0, (w_142_0, ())) (w_143_0, (w_143_1, ())) ∧
  generatedRoot.constraints_12 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_129_0 w_130_0 w_132_0 w_132_1 w_133_0 w_136_0 w_136_1 w_137_0 w_140_0 w_140_1 w_141_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  4 = 4 ∧
  MxxRuntime.select w_67_0 [w_97_0, w_98_0, w_99_0, w_100_0] w_101_0 ∧
  MxxRuntime.concatColumns w_86_0 w_91_0 w_103_0 ∧
  MxxRuntime.concatColumns w_96_0 w_101_0 w_104_0 ∧
  0 ≤ (22) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_105 backend hashModel tape (path ++ [105, i]) params i (current.1, (current.2.1, (w_103_0, (w_3_0, (w_104_0, ()))))) next) (Int.toNat (22)) (w_102_0, (w_102_0, ())) (w_105_0, (w_105_1, ())) ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_106 backend hashModel tape (path ++ [106, i]) params i (current.1, (current.2.1, (w_3_0, (w_103_0, (w_104_0, ()))))) next) (Int.toNat (968)) (w_105_0, (w_105_1, ())) (w_106_0, (w_106_1, ())) ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_108 backend hashModel tape (path ++ [108, i.val]) params i (w_3_0, (w_106_0, (w_106_1, ()))) ((w_108_0 i), ((w_108_1 i), ((w_108_2 i), ((w_108_3 i), ()))))) ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_109 backend hashModel tape (path ++ [109, i]) params i (current.1, (current.2.1, (w_108_0, (w_108_1, (w_3_0, (w_108_2, (w_108_3, ()))))))) next) (Int.toNat (105)) (w_107_0, (w_107_0, ())) (w_109_0, (w_109_1, ())) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_110 backend hashModel tape (path ++ [110, i]) params i (current.1, (current.2.1, (w_3_0, (w_108_0, (w_108_1, (w_108_2, (w_108_3, ()))))))) next) (Int.toNat (4620)) (w_109_0, (w_109_1, ())) (w_110_0, (w_110_1, ())) ∧
  MxxRuntime.concatColumns w_106_0 w_110_0 w_111_0 ∧
  MxxRuntime.concatColumns w_106_1 w_110_1 w_112_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_115 backend hashModel tape (path ++ [115, i]) params i (current, (w_54_0, ())) next) (Int.toNat (128)) w_114_0 w_115_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_117_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_118_0 w_119_0 ∧
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_124 backend hashModel tape (path ++ [124, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_108_0, (w_54_0, (w_108_1, (w_108_2, (w_108_3, ()))))))))) next) (Int.toNat (15)) (w_120_0, (w_121_0, (w_122_0, (w_123_0, ())))) (w_124_0, (w_124_1, (w_124_2, (w_124_3, ())))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_125 backend hashModel tape (path ++ [125, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_106_0, (w_106_1, (w_54_0, ())))))))) next) (Int.toNat (7)) (w_124_0, (w_124_1, (w_124_2, (w_124_3, ())))) (w_125_0, (w_125_1, (w_125_2, (w_125_3, ())))) ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_126 backend hashModel tape (path ++ [126, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_110_0, (w_110_1, (w_54_0, ())))))))) next) (Int.toNat (105)) (w_125_0, (w_125_1, (w_125_2, (w_125_3, ())))) (w_126_0, (w_126_1, (w_126_2, (w_126_3, ())))) ∧
  MxxRuntime.familyGetStatic w_0_0 (2) w_127_0 ∧
  MxxRuntime.concatColumns w_126_0 w_126_1 w_129_0 ∧
  MxxRuntime.concatColumns w_126_2 w_126_3 w_130_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_131 backend hashModel tape (path ++ [131, i]) params i (current.1, (current.2.1, (w_129_0, (w_3_0, (w_130_0, ()))))) next) (Int.toNat (1)) (w_128_0, (w_128_0, ())) (w_131_0, (w_131_1, ())) ∧
  0 ≤ (44) ∧
  generatedRoot.constraints_11 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_86_0 w_91_0 w_96_0 w_101_0 w_111_0 w_112_0 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_9 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_72_1 : mxxMat0) (w_76_1 : mxxMat0) (w_80_1 : mxxMat0) (w_84_1 : mxxMat0) (w_86_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  4 = 4 ∧
  MxxRuntime.select w_67_0 [w_92_0, w_93_0, w_94_0, w_95_0] w_96_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_72_1 0 1 44 88 w_97_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_76_1 0 1 44 88 w_98_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_80_1 0 1 44 88 w_99_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_84_1 0 1 44 88 w_100_0 ∧
  0 ≤ w_67_0 ∧
  w_67_0 < 4 ∧
  generatedRoot.constraints_10 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_86_0 w_91_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_72_1 : mxxMat0) (w_76_1 : mxxMat0) (w_80_1 : mxxMat0) (w_84_1 : mxxMat0) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  4 = 4 ∧
  MxxRuntime.select w_67_0 [w_87_0, w_88_0, w_89_0, w_90_0] w_91_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_72_1 0 1 0 44 w_92_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_76_1 0 1 0 44 w_93_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_80_1 0 1 0 44 w_94_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_84_1 0 1 0 44 w_95_0 ∧
  0 ≤ w_67_0 ∧
  w_67_0 < 4 ∧
  generatedRoot.constraints_9 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_72_1 w_76_1 w_80_1 w_84_1 w_86_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  4 = 4 ∧
  MxxRuntime.select w_67_0 [w_73_0, w_77_0, w_81_0, w_85_0] w_86_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_72_0 0 1 44 88 w_87_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_76_0 0 1 44 88 w_88_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_80_0 0 1 44 88 w_89_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_84_0 0 1 44 88 w_90_0 ∧
  0 ≤ w_67_0 ∧
  w_67_0 < 4 ∧
  generatedRoot.constraints_8 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_72_1 w_76_1 w_80_1 w_84_1 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_76 backend hashModel tape (path ++ [76, i]) params i (current.1, (current.2.1, (w_3_0, (w_69_0, (w_70_0, ()))))) next) (Int.toNat (44)) (w_75_0, (w_75_1, ())) (w_76_0, (w_76_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_76_0 0 1 0 44 w_77_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_79 backend hashModel tape (path ++ [79, i]) params i (current.1, (current.2.1, (w_69_0, (w_3_0, (w_70_0, ()))))) next) (Int.toNat (1)) (w_78_0, (w_78_0, ())) (w_79_0, (w_79_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_80 backend hashModel tape (path ++ [80, i]) params i (current.1, (current.2.1, (w_3_0, (w_69_0, (w_70_0, ()))))) next) (Int.toNat (44)) (w_79_0, (w_79_1, ())) (w_80_0, (w_80_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_80_0 0 1 0 44 w_81_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_83 backend hashModel tape (path ++ [83, i]) params i (current.1, (current.2.1, (w_69_0, (w_3_0, (w_70_0, ()))))) next) (Int.toNat (1)) (w_82_0, (w_82_0, ())) (w_83_0, (w_83_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_84 backend hashModel tape (path ++ [84, i]) params i (current.1, (current.2.1, (w_3_0, (w_69_0, (w_70_0, ()))))) next) (Int.toNat (44)) (w_83_0, (w_83_1, ())) (w_84_0, (w_84_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_84_0 0 1 0 44 w_85_0 ∧
  0 ≤ w_67_0 ∧
  w_67_0 < 4 ∧
  generatedRoot.constraints_7 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_72_0 w_72_1 w_73_0 w_76_0 w_76_1 w_77_0 w_80_0 w_80_1 w_81_0 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_40_0 : mxxMat3) (w_45_0 : mxxMat10) (w_45_1 : mxxMat10) (w_46_0 : mxxMat15) (w_47_0 : Fin 15 → mxxMat3) (w_47_1 : Fin 15 → mxxMat3) (w_47_2 : Fin 15 → mxxMat3) (w_47_3 : Fin 15 → mxxMat3) (w_48_0 : mxxMat15) (w_48_1 : mxxMat15) (w_49_0 : mxxMat15) (w_49_1 : mxxMat15) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_53_0 : mxxMat1) (w_55_0 : mxxMat1) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat17) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_62_0 : mxxMat3) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_64_1 : mxxMat3) (w_64_2 : mxxMat3) (w_64_3 : mxxMat3) (w_65_0 : mxxMat3) (w_65_1 : mxxMat3) (w_65_2 : mxxMat3) (w_65_3 : mxxMat3) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_71_1 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_48 backend hashModel tape (path ++ [48, i]) params i (current.1, (current.2.1, (w_47_0, (w_47_1, (w_3_0, (w_47_2, (w_47_3, ()))))))) next) (Int.toNat (105)) (w_46_0, (w_46_0, ())) (w_48_0, (w_48_1, ())) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15 × mxxMat15 × Unit) => sequential_generatedRoot_49 backend hashModel tape (path ++ [49, i]) params i (current.1, (current.2.1, (w_3_0, (w_47_0, (w_47_1, (w_47_2, (w_47_3, ()))))))) next) (Int.toNat (4620)) (w_48_0, (w_48_1, ())) (w_49_0, (w_49_1, ())) ∧
  MxxRuntime.concatColumns w_45_0 w_49_0 w_50_0 ∧
  MxxRuntime.concatColumns w_45_1 w_49_1 w_51_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_55 backend hashModel tape (path ++ [55, i]) params i (current, (w_54_0, ())) next) (Int.toNat (128)) w_53_0 w_55_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_57_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_58_0 w_59_0 ∧
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_64 backend hashModel tape (path ++ [64, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_47_0, (w_54_0, (w_47_1, (w_47_2, (w_47_3, ()))))))))) next) (Int.toNat (15)) (w_60_0, (w_61_0, (w_62_0, (w_63_0, ())))) (w_64_0, (w_64_1, (w_64_2, (w_64_3, ())))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_65 backend hashModel tape (path ++ [65, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_45_0, (w_45_1, (w_54_0, ())))))))) next) (Int.toNat (7)) (w_64_0, (w_64_1, (w_64_2, (w_64_3, ())))) (w_65_0, (w_65_1, (w_65_2, (w_65_3, ())))) ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) => sequential_generatedRoot_66 backend hashModel tape (path ++ [66, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (w_3_0, (w_49_0, (w_49_1, (w_54_0, ())))))))) next) (Int.toNat (105)) (w_65_0, (w_65_1, (w_65_2, (w_65_3, ())))) (w_66_0, (w_66_1, (w_66_2, (w_66_3, ())))) ∧
  MxxRuntime.familyGetStatic w_0_0 (1) w_67_0 ∧
  MxxRuntime.concatColumns w_66_0 w_66_1 w_69_0 ∧
  MxxRuntime.concatColumns w_66_2 w_66_3 w_70_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_71 backend hashModel tape (path ++ [71, i]) params i (current.1, (current.2.1, (w_69_0, (w_3_0, (w_70_0, ()))))) next) (Int.toNat (1)) (w_68_0, (w_68_0, ())) (w_71_0, (w_71_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_72 backend hashModel tape (path ++ [72, i]) params i (current.1, (current.2.1, (w_3_0, (w_69_0, (w_70_0, ()))))) next) (Int.toNat (44)) (w_71_0, (w_71_1, ())) (w_72_0, (w_72_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_72_0 0 1 0 44 w_73_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_75 backend hashModel tape (path ++ [75, i]) params i (current.1, (current.2.1, (w_69_0, (w_3_0, (w_70_0, ()))))) next) (Int.toNat (1)) (w_74_0, (w_74_0, ())) (w_75_0, (w_75_1, ())) ∧
  0 ≤ (44) ∧
  generatedRoot.constraints_6 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_50_0 w_51_0 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_69_0 w_70_0 w_72_0 w_72_1 w_73_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_1_0 : Int) (w_15_1 : mxxMat0) (w_19_1 : mxxMat0) (w_23_1 : mxxMat0) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_35_0 : mxxMat3) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat10) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat10) (w_44_1 : mxxMat10) (w_45_0 : mxxMat10) (w_45_1 : mxxMat10) (w_46_0 : mxxMat15) (w_47_0 : Fin 15 → mxxMat3) (w_47_1 : Fin 15 → mxxMat3) (w_47_2 : Fin 15 → mxxMat3) (w_47_3 : Fin 15 → mxxMat3) (w_48_0 : mxxMat15) (w_48_1 : mxxMat15) (w_49_0 : mxxMat15) (w_49_1 : mxxMat15) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_53_0 : mxxMat1) (w_55_0 : mxxMat1) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat17) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_62_0 : mxxMat3) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_64_1 : mxxMat3) (w_64_2 : mxxMat3) (w_64_3 : mxxMat3) (w_65_0 : mxxMat3) (w_65_1 : mxxMat3) (w_65_2 : mxxMat3) (w_65_3 : mxxMat3) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_71_1 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_15_1 0 1 44 88 w_37_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 44 88 w_38_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_1 0 1 44 88 w_39_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_1_0 [w_36_0, w_37_0, w_38_0, w_39_0] w_40_0 ∧
  MxxRuntime.concatColumns w_25_0 w_30_0 w_42_0 ∧
  MxxRuntime.concatColumns w_35_0 w_40_0 w_43_0 ∧
  0 ≤ (22) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_44 backend hashModel tape (path ++ [44, i]) params i (current.1, (current.2.1, (w_42_0, (w_3_0, (w_43_0, ()))))) next) (Int.toNat (22)) (w_41_0, (w_41_0, ())) (w_44_0, (w_44_1, ())) ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10 × mxxMat10 × Unit) => sequential_generatedRoot_45 backend hashModel tape (path ++ [45, i]) params i (current.1, (current.2.1, (w_3_0, (w_42_0, (w_43_0, ()))))) next) (Int.toNat (968)) (w_44_0, (w_44_1, ())) (w_45_0, (w_45_1, ())) ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_47 backend hashModel tape (path ++ [47, i.val]) params i (w_3_0, (w_45_0, (w_45_1, ()))) ((w_47_0 i), ((w_47_1 i), ((w_47_2 i), ((w_47_3 i), ()))))) ∧
  0 ≤ (105) ∧
  generatedRoot.constraints_5 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_25_0 w_30_0 w_35_0 w_40_0 w_45_0 w_45_1 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_51_0 w_53_0 w_55_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_1 w_64_2 w_64_3 w_65_0 w_65_1 w_65_2 w_65_3 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_74_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_1_0 : Int) (w_11_1 : mxxMat0) (w_15_1 : mxxMat0) (w_19_1 : mxxMat0) (w_23_1 : mxxMat0) (w_25_0 : mxxMat3) (w_30_0 : mxxMat3) (w_31_0 : mxxMat3) (w_32_0 : mxxMat3) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat10) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat10) (w_44_1 : mxxMat10) (w_45_0 : mxxMat10) (w_45_1 : mxxMat10) (w_46_0 : mxxMat15) (w_47_0 : Fin 15 → mxxMat3) (w_47_1 : Fin 15 → mxxMat3) (w_47_2 : Fin 15 → mxxMat3) (w_47_3 : Fin 15 → mxxMat3) (w_48_0 : mxxMat15) (w_48_1 : mxxMat15) (w_49_0 : mxxMat15) (w_49_1 : mxxMat15) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_53_0 : mxxMat1) (w_55_0 : mxxMat1) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat17) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_62_0 : mxxMat3) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_64_1 : mxxMat3) (w_64_2 : mxxMat3) (w_64_3 : mxxMat3) (w_65_0 : mxxMat3) (w_65_1 : mxxMat3) (w_65_2 : mxxMat3) (w_65_3 : mxxMat3) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_71_1 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_15_1 0 1 0 44 w_32_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_1 0 1 0 44 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_1 0 1 0 44 w_34_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_1_0 [w_31_0, w_32_0, w_33_0, w_34_0] w_35_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_1 0 1 44 88 w_36_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  generatedRoot.constraints_4 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_1_0 w_15_1 w_19_1 w_23_1 w_25_0 w_30_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_51_0 w_53_0 w_55_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_1 w_64_2 w_64_3 w_65_0 w_65_1 w_65_2 w_65_3 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_74_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_1_0 : Int) (w_11_1 : mxxMat0) (w_15_0 : mxxMat0) (w_15_1 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_25_0 : mxxMat3) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat3) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_31_0 : mxxMat3) (w_32_0 : mxxMat3) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat10) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat10) (w_44_1 : mxxMat10) (w_45_0 : mxxMat10) (w_45_1 : mxxMat10) (w_46_0 : mxxMat15) (w_47_0 : Fin 15 → mxxMat3) (w_47_1 : Fin 15 → mxxMat3) (w_47_2 : Fin 15 → mxxMat3) (w_47_3 : Fin 15 → mxxMat3) (w_48_0 : mxxMat15) (w_48_1 : mxxMat15) (w_49_0 : mxxMat15) (w_49_1 : mxxMat15) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_53_0 : mxxMat1) (w_55_0 : mxxMat1) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat17) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_62_0 : mxxMat3) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_64_1 : mxxMat3) (w_64_2 : mxxMat3) (w_64_3 : mxxMat3) (w_65_0 : mxxMat3) (w_65_1 : mxxMat3) (w_65_2 : mxxMat3) (w_65_3 : mxxMat3) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_71_1 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 44 88 w_27_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 44 88 w_28_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_0 0 1 44 88 w_29_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_1_0 [w_26_0, w_27_0, w_28_0, w_29_0] w_30_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_1 0 1 0 44 w_31_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_1_0 w_11_1 w_15_1 w_19_1 w_23_1 w_25_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_51_0 w_53_0 w_55_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_1 w_64_2 w_64_3 w_65_0 w_65_1 w_65_2 w_65_3 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_74_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_1_0 : Int) (w_6_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat3) (w_15_0 : mxxMat0) (w_15_1 : mxxMat0) (w_16_0 : mxxMat3) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_22_1 : mxxMat0) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat3) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_31_0 : mxxMat3) (w_32_0 : mxxMat3) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat10) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat10) (w_44_1 : mxxMat10) (w_45_0 : mxxMat10) (w_45_1 : mxxMat10) (w_46_0 : mxxMat15) (w_47_0 : Fin 15 → mxxMat3) (w_47_1 : Fin 15 → mxxMat3) (w_47_2 : Fin 15 → mxxMat3) (w_47_3 : Fin 15 → mxxMat3) (w_48_0 : mxxMat15) (w_48_1 : mxxMat15) (w_49_0 : mxxMat15) (w_49_1 : mxxMat15) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_53_0 : mxxMat1) (w_55_0 : mxxMat1) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat17) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_62_0 : mxxMat3) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_64_1 : mxxMat3) (w_64_2 : mxxMat3) (w_64_3 : mxxMat3) (w_65_0 : mxxMat3) (w_65_1 : mxxMat3) (w_65_2 : mxxMat3) (w_65_3 : mxxMat3) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_71_1 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_19 backend hashModel tape (path ++ [19, i]) params i (current.1, (current.2.1, (w_3_0, (w_6_0, (w_9_0, ()))))) next) (Int.toNat (44)) (w_18_0, (w_18_1, ())) (w_19_0, (w_19_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 0 44 w_20_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_22 backend hashModel tape (path ++ [22, i]) params i (current.1, (current.2.1, (w_6_0, (w_3_0, (w_9_0, ()))))) next) (Int.toNat (1)) (w_21_0, (w_21_0, ())) (w_22_0, (w_22_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_23 backend hashModel tape (path ++ [23, i]) params i (current.1, (current.2.1, (w_3_0, (w_6_0, (w_9_0, ()))))) next) (Int.toNat (44)) (w_22_0, (w_22_1, ())) (w_23_0, (w_23_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_0 0 1 0 44 w_24_0 ∧
  0 ≤ w_1_0 ∧
  w_1_0 < 4 ∧
  4 = 4 ∧
  MxxRuntime.select w_1_0 [w_12_0, w_16_0, w_20_0, w_24_0] w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 44 88 w_26_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  generatedRoot.constraints_2 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_1_0 w_11_1 w_15_0 w_15_1 w_19_0 w_19_1 w_23_0 w_23_1 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_51_0 w_53_0 w_55_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_1 w_64_2 w_64_3 w_65_0 w_65_1 w_65_2 w_65_3 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_74_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Fin 4 → Int) (w_3_0 : ByteArray) (w_54_0 : Fin 512 → Int) (w_1_0 : Int) (w_2_0 : mxxMat0) (w_4_0 : mxxMat3) (w_5_0 : mxxMat3) (w_6_0 : mxxMat0) (w_7_0 : mxxMat3) (w_8_0 : mxxMat3) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_10_1 : mxxMat0) (w_11_0 : mxxMat0) (w_11_1 : mxxMat0) (w_12_0 : mxxMat3) (w_13_0 : mxxMat0) (w_14_0 : mxxMat0) (w_14_1 : mxxMat0) (w_15_0 : mxxMat0) (w_15_1 : mxxMat0) (w_16_0 : mxxMat3) (w_17_0 : mxxMat0) (w_18_0 : mxxMat0) (w_18_1 : mxxMat0) (w_19_0 : mxxMat0) (w_19_1 : mxxMat0) (w_20_0 : mxxMat3) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_22_1 : mxxMat0) (w_23_0 : mxxMat0) (w_23_1 : mxxMat0) (w_24_0 : mxxMat3) (w_25_0 : mxxMat3) (w_26_0 : mxxMat3) (w_27_0 : mxxMat3) (w_28_0 : mxxMat3) (w_29_0 : mxxMat3) (w_30_0 : mxxMat3) (w_31_0 : mxxMat3) (w_32_0 : mxxMat3) (w_33_0 : mxxMat3) (w_34_0 : mxxMat3) (w_35_0 : mxxMat3) (w_36_0 : mxxMat3) (w_37_0 : mxxMat3) (w_38_0 : mxxMat3) (w_39_0 : mxxMat3) (w_40_0 : mxxMat3) (w_41_0 : mxxMat10) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat10) (w_44_1 : mxxMat10) (w_45_0 : mxxMat10) (w_45_1 : mxxMat10) (w_46_0 : mxxMat15) (w_47_0 : Fin 15 → mxxMat3) (w_47_1 : Fin 15 → mxxMat3) (w_47_2 : Fin 15 → mxxMat3) (w_47_3 : Fin 15 → mxxMat3) (w_48_0 : mxxMat15) (w_48_1 : mxxMat15) (w_49_0 : mxxMat15) (w_49_1 : mxxMat15) (w_50_0 : mxxMat18) (w_51_0 : mxxMat18) (w_53_0 : mxxMat1) (w_55_0 : mxxMat1) (w_57_0 : mxxMat3) (w_58_0 : mxxMat3) (w_59_0 : mxxMat17) (w_60_0 : mxxMat3) (w_61_0 : mxxMat3) (w_62_0 : mxxMat3) (w_63_0 : mxxMat3) (w_64_0 : mxxMat3) (w_64_1 : mxxMat3) (w_64_2 : mxxMat3) (w_64_3 : mxxMat3) (w_65_0 : mxxMat3) (w_65_1 : mxxMat3) (w_65_2 : mxxMat3) (w_65_3 : mxxMat3) (w_66_0 : mxxMat3) (w_66_1 : mxxMat3) (w_66_2 : mxxMat3) (w_66_3 : mxxMat3) (w_67_0 : Int) (w_68_0 : mxxMat0) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_71_1 : mxxMat0) (w_72_0 : mxxMat0) (w_72_1 : mxxMat0) (w_73_0 : mxxMat3) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_75_1 : mxxMat0) (w_76_0 : mxxMat0) (w_76_1 : mxxMat0) (w_77_0 : mxxMat3) (w_78_0 : mxxMat0) (w_79_0 : mxxMat0) (w_79_1 : mxxMat0) (w_80_0 : mxxMat0) (w_80_1 : mxxMat0) (w_81_0 : mxxMat3) (w_82_0 : mxxMat0) (w_83_0 : mxxMat0) (w_83_1 : mxxMat0) (w_84_0 : mxxMat0) (w_84_1 : mxxMat0) (w_85_0 : mxxMat3) (w_86_0 : mxxMat3) (w_87_0 : mxxMat3) (w_88_0 : mxxMat3) (w_89_0 : mxxMat3) (w_90_0 : mxxMat3) (w_91_0 : mxxMat3) (w_92_0 : mxxMat3) (w_93_0 : mxxMat3) (w_94_0 : mxxMat3) (w_95_0 : mxxMat3) (w_96_0 : mxxMat3) (w_97_0 : mxxMat3) (w_98_0 : mxxMat3) (w_99_0 : mxxMat3) (w_100_0 : mxxMat3) (w_101_0 : mxxMat3) (w_102_0 : mxxMat10) (w_103_0 : mxxMat0) (w_104_0 : mxxMat0) (w_105_0 : mxxMat10) (w_105_1 : mxxMat10) (w_106_0 : mxxMat10) (w_106_1 : mxxMat10) (w_107_0 : mxxMat15) (w_108_0 : Fin 15 → mxxMat3) (w_108_1 : Fin 15 → mxxMat3) (w_108_2 : Fin 15 → mxxMat3) (w_108_3 : Fin 15 → mxxMat3) (w_109_0 : mxxMat15) (w_109_1 : mxxMat15) (w_110_0 : mxxMat15) (w_110_1 : mxxMat15) (w_111_0 : mxxMat18) (w_112_0 : mxxMat18) (w_114_0 : mxxMat1) (w_115_0 : mxxMat1) (w_117_0 : mxxMat3) (w_118_0 : mxxMat3) (w_119_0 : mxxMat17) (w_120_0 : mxxMat3) (w_121_0 : mxxMat3) (w_122_0 : mxxMat3) (w_123_0 : mxxMat3) (w_124_0 : mxxMat3) (w_124_1 : mxxMat3) (w_124_2 : mxxMat3) (w_124_3 : mxxMat3) (w_125_0 : mxxMat3) (w_125_1 : mxxMat3) (w_125_2 : mxxMat3) (w_125_3 : mxxMat3) (w_126_0 : mxxMat3) (w_126_1 : mxxMat3) (w_126_2 : mxxMat3) (w_126_3 : mxxMat3) (w_127_0 : Int) (w_128_0 : mxxMat0) (w_129_0 : mxxMat0) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_131_1 : mxxMat0) (w_132_0 : mxxMat0) (w_132_1 : mxxMat0) (w_133_0 : mxxMat3) (w_134_0 : mxxMat0) (w_135_0 : mxxMat0) (w_135_1 : mxxMat0) (w_136_0 : mxxMat0) (w_136_1 : mxxMat0) (w_137_0 : mxxMat3) (w_138_0 : mxxMat0) (w_139_0 : mxxMat0) (w_139_1 : mxxMat0) (w_140_0 : mxxMat0) (w_140_1 : mxxMat0) (w_141_0 : mxxMat3) (w_142_0 : mxxMat0) (w_143_0 : mxxMat0) (w_143_1 : mxxMat0) (w_144_0 : mxxMat0) (w_144_1 : mxxMat0) (w_145_0 : mxxMat3) (w_146_0 : mxxMat3) (w_147_0 : mxxMat3) (w_148_0 : mxxMat3) (w_149_0 : mxxMat3) (w_150_0 : mxxMat3) (w_151_0 : mxxMat3) (w_152_0 : mxxMat3) (w_153_0 : mxxMat3) (w_154_0 : mxxMat3) (w_155_0 : mxxMat3) (w_156_0 : mxxMat3) (w_157_0 : mxxMat3) (w_158_0 : mxxMat3) (w_159_0 : mxxMat3) (w_160_0 : mxxMat3) (w_161_0 : mxxMat3) (w_162_0 : mxxMat10) (w_163_0 : mxxMat0) (w_164_0 : mxxMat0) (w_165_0 : mxxMat10) (w_165_1 : mxxMat10) (w_166_0 : mxxMat10) (w_166_1 : mxxMat10) (w_167_0 : mxxMat15) (w_168_0 : Fin 15 → mxxMat3) (w_168_1 : Fin 15 → mxxMat3) (w_168_2 : Fin 15 → mxxMat3) (w_168_3 : Fin 15 → mxxMat3) (w_169_0 : mxxMat15) (w_169_1 : mxxMat15) (w_170_0 : mxxMat15) (w_170_1 : mxxMat15) (w_171_0 : mxxMat18) (w_172_0 : mxxMat18) (w_174_0 : mxxMat1) (w_175_0 : mxxMat1) (w_177_0 : mxxMat3) (w_178_0 : mxxMat3) (w_179_0 : mxxMat17) (w_180_0 : mxxMat3) (w_181_0 : mxxMat3) (w_182_0 : mxxMat3) (w_183_0 : mxxMat3) (w_184_0 : mxxMat3) (w_184_1 : mxxMat3) (w_184_2 : mxxMat3) (w_184_3 : mxxMat3) (w_185_0 : mxxMat3) (w_185_1 : mxxMat3) (w_185_2 : mxxMat3) (w_185_3 : mxxMat3) (w_186_0 : mxxMat3) (w_186_1 : mxxMat3) (w_186_2 : mxxMat3) (w_186_3 : mxxMat3) (w_187_0 : Int) (w_188_0 : mxxMat0) (w_189_0 : mxxMat0) (w_190_0 : mxxMat0) (w_191_0 : mxxMat0) (w_191_1 : mxxMat0) (w_192_0 : mxxMat0) (w_192_1 : mxxMat0) (w_193_0 : mxxMat3) (w_194_0 : mxxMat0) (w_195_0 : mxxMat0) (w_195_1 : mxxMat0) (w_196_0 : mxxMat0) (w_196_1 : mxxMat0) (w_197_0 : mxxMat3) (w_198_0 : mxxMat0) (w_199_0 : mxxMat0) (w_199_1 : mxxMat0) (w_200_0 : mxxMat0) (w_200_1 : mxxMat0) (w_201_0 : mxxMat3) (w_202_0 : mxxMat0) (w_203_0 : mxxMat0) (w_203_1 : mxxMat0) (w_204_0 : mxxMat0) (w_204_1 : mxxMat0) (w_205_0 : mxxMat3) (w_206_0 : mxxMat3) (w_207_0 : mxxMat3) (w_208_0 : mxxMat3) (w_209_0 : mxxMat3) (w_210_0 : mxxMat3) (w_211_0 : mxxMat3) (w_212_0 : mxxMat3) (w_213_0 : mxxMat3) (w_214_0 : mxxMat3) (w_215_0 : mxxMat3) (w_216_0 : mxxMat3) (w_217_0 : mxxMat3) (w_218_0 : mxxMat3) (w_219_0 : mxxMat3) (w_220_0 : mxxMat3) (w_221_0 : mxxMat3) (w_222_0 : mxxMat10) (w_223_0 : mxxMat0) (w_224_0 : mxxMat0) (w_225_0 : mxxMat10) (w_225_1 : mxxMat10) (w_226_0 : mxxMat10) (w_226_1 : mxxMat10) (w_227_0 : mxxMat15) (w_228_0 : Fin 15 → mxxMat3) (w_228_1 : Fin 15 → mxxMat3) (w_228_2 : Fin 15 → mxxMat3) (w_228_3 : Fin 15 → mxxMat3) (w_229_0 : mxxMat15) (w_229_1 : mxxMat15) (w_230_0 : mxxMat15) (w_230_1 : mxxMat15) (w_231_0 : mxxMat18) (w_232_0 : mxxMat18) (w_234_0 : mxxMat1) (w_235_0 : mxxMat1) (w_237_0 : mxxMat3) (w_238_0 : mxxMat3) (w_239_0 : mxxMat17) (w_240_0 : mxxMat3) (w_241_0 : mxxMat3) (w_242_0 : mxxMat3) (w_243_0 : mxxMat3) (w_244_0 : mxxMat3) (w_244_1 : mxxMat3) (w_244_2 : mxxMat3) (w_244_3 : mxxMat3) (w_245_0 : mxxMat3) (w_245_1 : mxxMat3) (w_245_2 : mxxMat3) (w_245_3 : mxxMat3) (w_246_0 : mxxMat3) (w_246_1 : mxxMat3) (w_246_2 : mxxMat3) (w_246_3 : mxxMat3) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  MxxRuntime.familyGetStatic w_0_0 (0) w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_3_0) w_4_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_3_0) w_5_0 ∧
  MxxRuntime.concatColumns w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_3_0) w_7_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_3_0) w_8_0 ∧
  MxxRuntime.concatColumns w_7_0 w_8_0 w_9_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_10 backend hashModel tape (path ++ [10, i]) params i (current.1, (current.2.1, (w_6_0, (w_3_0, (w_9_0, ()))))) next) (Int.toNat (1)) (w_2_0, (w_2_0, ())) (w_10_0, (w_10_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_11 backend hashModel tape (path ++ [11, i]) params i (current.1, (current.2.1, (w_3_0, (w_6_0, (w_9_0, ()))))) next) (Int.toNat (44)) (w_10_0, (w_10_1, ())) (w_11_0, (w_11_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 0 44 w_12_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_14 backend hashModel tape (path ++ [14, i]) params i (current.1, (current.2.1, (w_6_0, (w_3_0, (w_9_0, ()))))) next) (Int.toNat (1)) (w_13_0, (w_13_0, ())) (w_14_0, (w_14_1, ())) ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_15 backend hashModel tape (path ++ [15, i]) params i (current.1, (current.2.1, (w_3_0, (w_6_0, (w_9_0, ()))))) next) (Int.toNat (44)) (w_14_0, (w_14_1, ())) (w_15_0, (w_15_1, ())) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_15_0 0 1 0 44 w_16_0 ∧
  0 ≤ (1) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_18 backend hashModel tape (path ++ [18, i]) params i (current.1, (current.2.1, (w_6_0, (w_3_0, (w_9_0, ()))))) next) (Int.toNat (1)) (w_17_0, (w_17_0, ())) (w_18_0, (w_18_1, ())) ∧
  0 ≤ (44) ∧
  generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_1_0 w_6_0 w_9_0 w_11_0 w_11_1 w_12_0 w_15_0 w_15_1 w_16_0 w_18_0 w_18_1 w_19_0 w_19_1 w_20_0 w_21_0 w_22_0 w_22_1 w_23_0 w_23_1 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_51_0 w_53_0 w_55_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_1 w_64_2 w_64_3 w_65_0 w_65_1 w_65_2 w_65_3 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_74_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 4 → Int) × ByteArray × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Fin 4 → Int := inputs.1
  let w_3_0 : ByteArray := inputs.2.1
  let w_54_0 : Fin 512 → Int := inputs.2.2.1
  let w_1_0 := witness.w_1_0
  let w_2_0 : mxxMat0 := (0 : mxxMat0)
  let w_4_0 := witness.w_4_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_10_1 := witness.w_10_1
  let w_11_0 := witness.w_11_0
  let w_11_1 := witness.w_11_1
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat0 := (0 : mxxMat0)
  let w_14_0 := witness.w_14_0
  let w_14_1 := witness.w_14_1
  let w_15_0 := witness.w_15_0
  let w_15_1 := witness.w_15_1
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := (0 : mxxMat0)
  let w_18_0 := witness.w_18_0
  let w_18_1 := witness.w_18_1
  let w_19_0 := witness.w_19_0
  let w_19_1 := witness.w_19_1
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat0 := (0 : mxxMat0)
  let w_22_0 := witness.w_22_0
  let w_22_1 := witness.w_22_1
  let w_23_0 := witness.w_23_0
  let w_23_1 := witness.w_23_1
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 : mxxMat10 := (0 : mxxMat10)
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_44_1 := witness.w_44_1
  let w_45_0 := witness.w_45_0
  let w_45_1 := witness.w_45_1
  let w_46_0 : mxxMat15 := (0 : mxxMat15)
  let w_47_0 := witness.w_47_0
  let w_47_1 := witness.w_47_1
  let w_47_2 := witness.w_47_2
  let w_47_3 := witness.w_47_3
  let w_48_0 := witness.w_48_0
  let w_48_1 := witness.w_48_1
  let w_49_0 := witness.w_49_0
  let w_49_1 := witness.w_49_1
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_53_0 : mxxMat1 := (0 : mxxMat1)
  let w_55_0 := witness.w_55_0
  let w_56_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_52_0 w_55_0
  let w_57_0 := witness.w_57_0
  let w_58_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_56_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat3 := MxxRuntime.matrixMul w_25_0 w_59_0
  let w_61_0 : mxxMat3 := MxxRuntime.matrixMul w_30_0 w_59_0
  let w_62_0 : mxxMat3 := MxxRuntime.matrixMul w_35_0 w_59_0
  let w_63_0 : mxxMat3 := MxxRuntime.matrixMul w_40_0 w_59_0
  let w_64_0 := witness.w_64_0
  let w_64_1 := witness.w_64_1
  let w_64_2 := witness.w_64_2
  let w_64_3 := witness.w_64_3
  let w_65_0 := witness.w_65_0
  let w_65_1 := witness.w_65_1
  let w_65_2 := witness.w_65_2
  let w_65_3 := witness.w_65_3
  let w_66_0 := witness.w_66_0
  let w_66_1 := witness.w_66_1
  let w_66_2 := witness.w_66_2
  let w_66_3 := witness.w_66_3
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := (0 : mxxMat0)
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_71_1 := witness.w_71_1
  let w_72_0 := witness.w_72_0
  let w_72_1 := witness.w_72_1
  let w_73_0 := witness.w_73_0
  let w_74_0 : mxxMat0 := (0 : mxxMat0)
  let w_75_0 := witness.w_75_0
  let w_75_1 := witness.w_75_1
  let w_76_0 := witness.w_76_0
  let w_76_1 := witness.w_76_1
  let w_77_0 := witness.w_77_0
  let w_78_0 : mxxMat0 := (0 : mxxMat0)
  let w_79_0 := witness.w_79_0
  let w_79_1 := witness.w_79_1
  let w_80_0 := witness.w_80_0
  let w_80_1 := witness.w_80_1
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat0 := (0 : mxxMat0)
  let w_83_0 := witness.w_83_0
  let w_83_1 := witness.w_83_1
  let w_84_0 := witness.w_84_0
  let w_84_1 := witness.w_84_1
  let w_85_0 := witness.w_85_0
  let w_86_0 := witness.w_86_0
  let w_87_0 := witness.w_87_0
  let w_88_0 := witness.w_88_0
  let w_89_0 := witness.w_89_0
  let w_90_0 := witness.w_90_0
  let w_91_0 := witness.w_91_0
  let w_92_0 := witness.w_92_0
  let w_93_0 := witness.w_93_0
  let w_94_0 := witness.w_94_0
  let w_95_0 := witness.w_95_0
  let w_96_0 := witness.w_96_0
  let w_97_0 := witness.w_97_0
  let w_98_0 := witness.w_98_0
  let w_99_0 := witness.w_99_0
  let w_100_0 := witness.w_100_0
  let w_101_0 := witness.w_101_0
  let w_102_0 : mxxMat10 := (0 : mxxMat10)
  let w_103_0 := witness.w_103_0
  let w_104_0 := witness.w_104_0
  let w_105_0 := witness.w_105_0
  let w_105_1 := witness.w_105_1
  let w_106_0 := witness.w_106_0
  let w_106_1 := witness.w_106_1
  let w_107_0 : mxxMat15 := (0 : mxxMat15)
  let w_108_0 := witness.w_108_0
  let w_108_1 := witness.w_108_1
  let w_108_2 := witness.w_108_2
  let w_108_3 := witness.w_108_3
  let w_109_0 := witness.w_109_0
  let w_109_1 := witness.w_109_1
  let w_110_0 := witness.w_110_0
  let w_110_1 := witness.w_110_1
  let w_111_0 := witness.w_111_0
  let w_112_0 := witness.w_112_0
  let w_113_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_114_0 : mxxMat1 := (0 : mxxMat1)
  let w_115_0 := witness.w_115_0
  let w_116_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_113_0 w_115_0
  let w_117_0 := witness.w_117_0
  let w_118_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_116_0 w_117_0
  let w_119_0 := witness.w_119_0
  let w_120_0 : mxxMat3 := MxxRuntime.matrixMul w_86_0 w_119_0
  let w_121_0 : mxxMat3 := MxxRuntime.matrixMul w_91_0 w_119_0
  let w_122_0 : mxxMat3 := MxxRuntime.matrixMul w_96_0 w_119_0
  let w_123_0 : mxxMat3 := MxxRuntime.matrixMul w_101_0 w_119_0
  let w_124_0 := witness.w_124_0
  let w_124_1 := witness.w_124_1
  let w_124_2 := witness.w_124_2
  let w_124_3 := witness.w_124_3
  let w_125_0 := witness.w_125_0
  let w_125_1 := witness.w_125_1
  let w_125_2 := witness.w_125_2
  let w_125_3 := witness.w_125_3
  let w_126_0 := witness.w_126_0
  let w_126_1 := witness.w_126_1
  let w_126_2 := witness.w_126_2
  let w_126_3 := witness.w_126_3
  let w_127_0 := witness.w_127_0
  let w_128_0 : mxxMat0 := (0 : mxxMat0)
  let w_129_0 := witness.w_129_0
  let w_130_0 := witness.w_130_0
  let w_131_0 := witness.w_131_0
  let w_131_1 := witness.w_131_1
  let w_132_0 := witness.w_132_0
  let w_132_1 := witness.w_132_1
  let w_133_0 := witness.w_133_0
  let w_134_0 : mxxMat0 := (0 : mxxMat0)
  let w_135_0 := witness.w_135_0
  let w_135_1 := witness.w_135_1
  let w_136_0 := witness.w_136_0
  let w_136_1 := witness.w_136_1
  let w_137_0 := witness.w_137_0
  let w_138_0 : mxxMat0 := (0 : mxxMat0)
  let w_139_0 := witness.w_139_0
  let w_139_1 := witness.w_139_1
  let w_140_0 := witness.w_140_0
  let w_140_1 := witness.w_140_1
  let w_141_0 := witness.w_141_0
  let w_142_0 : mxxMat0 := (0 : mxxMat0)
  let w_143_0 := witness.w_143_0
  let w_143_1 := witness.w_143_1
  let w_144_0 := witness.w_144_0
  let w_144_1 := witness.w_144_1
  let w_145_0 := witness.w_145_0
  let w_146_0 := witness.w_146_0
  let w_147_0 := witness.w_147_0
  let w_148_0 := witness.w_148_0
  let w_149_0 := witness.w_149_0
  let w_150_0 := witness.w_150_0
  let w_151_0 := witness.w_151_0
  let w_152_0 := witness.w_152_0
  let w_153_0 := witness.w_153_0
  let w_154_0 := witness.w_154_0
  let w_155_0 := witness.w_155_0
  let w_156_0 := witness.w_156_0
  let w_157_0 := witness.w_157_0
  let w_158_0 := witness.w_158_0
  let w_159_0 := witness.w_159_0
  let w_160_0 := witness.w_160_0
  let w_161_0 := witness.w_161_0
  let w_162_0 : mxxMat10 := (0 : mxxMat10)
  let w_163_0 := witness.w_163_0
  let w_164_0 := witness.w_164_0
  let w_165_0 := witness.w_165_0
  let w_165_1 := witness.w_165_1
  let w_166_0 := witness.w_166_0
  let w_166_1 := witness.w_166_1
  let w_167_0 : mxxMat15 := (0 : mxxMat15)
  let w_168_0 := witness.w_168_0
  let w_168_1 := witness.w_168_1
  let w_168_2 := witness.w_168_2
  let w_168_3 := witness.w_168_3
  let w_169_0 := witness.w_169_0
  let w_169_1 := witness.w_169_1
  let w_170_0 := witness.w_170_0
  let w_170_1 := witness.w_170_1
  let w_171_0 := witness.w_171_0
  let w_172_0 := witness.w_172_0
  let w_173_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_174_0 : mxxMat1 := (0 : mxxMat1)
  let w_175_0 := witness.w_175_0
  let w_176_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_173_0 w_175_0
  let w_177_0 := witness.w_177_0
  let w_178_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_176_0 w_177_0
  let w_179_0 := witness.w_179_0
  let w_180_0 : mxxMat3 := MxxRuntime.matrixMul w_146_0 w_179_0
  let w_181_0 : mxxMat3 := MxxRuntime.matrixMul w_151_0 w_179_0
  let w_182_0 : mxxMat3 := MxxRuntime.matrixMul w_156_0 w_179_0
  let w_183_0 : mxxMat3 := MxxRuntime.matrixMul w_161_0 w_179_0
  let w_184_0 := witness.w_184_0
  let w_184_1 := witness.w_184_1
  let w_184_2 := witness.w_184_2
  let w_184_3 := witness.w_184_3
  let w_185_0 := witness.w_185_0
  let w_185_1 := witness.w_185_1
  let w_185_2 := witness.w_185_2
  let w_185_3 := witness.w_185_3
  let w_186_0 := witness.w_186_0
  let w_186_1 := witness.w_186_1
  let w_186_2 := witness.w_186_2
  let w_186_3 := witness.w_186_3
  let w_187_0 := witness.w_187_0
  let w_188_0 : mxxMat0 := (0 : mxxMat0)
  let w_189_0 := witness.w_189_0
  let w_190_0 := witness.w_190_0
  let w_191_0 := witness.w_191_0
  let w_191_1 := witness.w_191_1
  let w_192_0 := witness.w_192_0
  let w_192_1 := witness.w_192_1
  let w_193_0 := witness.w_193_0
  let w_194_0 : mxxMat0 := (0 : mxxMat0)
  let w_195_0 := witness.w_195_0
  let w_195_1 := witness.w_195_1
  let w_196_0 := witness.w_196_0
  let w_196_1 := witness.w_196_1
  let w_197_0 := witness.w_197_0
  let w_198_0 : mxxMat0 := (0 : mxxMat0)
  let w_199_0 := witness.w_199_0
  let w_199_1 := witness.w_199_1
  let w_200_0 := witness.w_200_0
  let w_200_1 := witness.w_200_1
  let w_201_0 := witness.w_201_0
  let w_202_0 : mxxMat0 := (0 : mxxMat0)
  let w_203_0 := witness.w_203_0
  let w_203_1 := witness.w_203_1
  let w_204_0 := witness.w_204_0
  let w_204_1 := witness.w_204_1
  let w_205_0 := witness.w_205_0
  let w_206_0 := witness.w_206_0
  let w_207_0 := witness.w_207_0
  let w_208_0 := witness.w_208_0
  let w_209_0 := witness.w_209_0
  let w_210_0 := witness.w_210_0
  let w_211_0 := witness.w_211_0
  let w_212_0 := witness.w_212_0
  let w_213_0 := witness.w_213_0
  let w_214_0 := witness.w_214_0
  let w_215_0 := witness.w_215_0
  let w_216_0 := witness.w_216_0
  let w_217_0 := witness.w_217_0
  let w_218_0 := witness.w_218_0
  let w_219_0 := witness.w_219_0
  let w_220_0 := witness.w_220_0
  let w_221_0 := witness.w_221_0
  let w_222_0 : mxxMat10 := (0 : mxxMat10)
  let w_223_0 := witness.w_223_0
  let w_224_0 := witness.w_224_0
  let w_225_0 := witness.w_225_0
  let w_225_1 := witness.w_225_1
  let w_226_0 := witness.w_226_0
  let w_226_1 := witness.w_226_1
  let w_227_0 : mxxMat15 := (0 : mxxMat15)
  let w_228_0 := witness.w_228_0
  let w_228_1 := witness.w_228_1
  let w_228_2 := witness.w_228_2
  let w_228_3 := witness.w_228_3
  let w_229_0 := witness.w_229_0
  let w_229_1 := witness.w_229_1
  let w_230_0 := witness.w_230_0
  let w_230_1 := witness.w_230_1
  let w_231_0 := witness.w_231_0
  let w_232_0 := witness.w_232_0
  let w_233_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_234_0 : mxxMat1 := (0 : mxxMat1)
  let w_235_0 := witness.w_235_0
  let w_236_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_233_0 w_235_0
  let w_237_0 := witness.w_237_0
  let w_238_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_236_0 w_237_0
  let w_239_0 := witness.w_239_0
  let w_240_0 : mxxMat3 := MxxRuntime.matrixMul w_206_0 w_239_0
  let w_241_0 : mxxMat3 := MxxRuntime.matrixMul w_211_0 w_239_0
  let w_242_0 : mxxMat3 := MxxRuntime.matrixMul w_216_0 w_239_0
  let w_243_0 : mxxMat3 := MxxRuntime.matrixMul w_221_0 w_239_0
  let w_244_0 := witness.w_244_0
  let w_244_1 := witness.w_244_1
  let w_244_2 := witness.w_244_2
  let w_244_3 := witness.w_244_3
  let w_245_0 := witness.w_245_0
  let w_245_1 := witness.w_245_1
  let w_245_2 := witness.w_245_2
  let w_245_3 := witness.w_245_3
  let w_246_0 := witness.w_246_0
  let w_246_1 := witness.w_246_1
  let w_246_2 := witness.w_246_2
  let w_246_3 := witness.w_246_3
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_3_0 w_54_0 w_1_0 w_2_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_10_1 w_11_0 w_11_1 w_12_0 w_13_0 w_14_0 w_14_1 w_15_0 w_15_1 w_16_0 w_17_0 w_18_0 w_18_1 w_19_0 w_19_1 w_20_0 w_21_0 w_22_0 w_22_1 w_23_0 w_23_1 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_44_1 w_45_0 w_45_1 w_46_0 w_47_0 w_47_1 w_47_2 w_47_3 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_51_0 w_53_0 w_55_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_64_1 w_64_2 w_64_3 w_65_0 w_65_1 w_65_2 w_65_3 w_66_0 w_66_1 w_66_2 w_66_3 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_74_0 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_78_0 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_82_0 w_83_0 w_83_1 w_84_0 w_84_1 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_105_1 w_106_0 w_106_1 w_107_0 w_108_0 w_108_1 w_108_2 w_108_3 w_109_0 w_109_1 w_110_0 w_110_1 w_111_0 w_112_0 w_114_0 w_115_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_124_1 w_124_2 w_124_3 w_125_0 w_125_1 w_125_2 w_125_3 w_126_0 w_126_1 w_126_2 w_126_3 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_131_1 w_132_0 w_132_1 w_133_0 w_134_0 w_135_0 w_135_1 w_136_0 w_136_1 w_137_0 w_138_0 w_139_0 w_139_1 w_140_0 w_140_1 w_141_0 w_142_0 w_143_0 w_143_1 w_144_0 w_144_1 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_165_1 w_166_0 w_166_1 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_169_0 w_169_1 w_170_0 w_170_1 w_171_0 w_172_0 w_174_0 w_175_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_184_1 w_184_2 w_184_3 w_185_0 w_185_1 w_185_2 w_185_3 w_186_0 w_186_1 w_186_2 w_186_3 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_191_1 w_192_0 w_192_1 w_193_0 w_194_0 w_195_0 w_195_1 w_196_0 w_196_1 w_197_0 w_198_0 w_199_0 w_199_1 w_200_0 w_200_1 w_201_0 w_202_0 w_203_0 w_203_1 w_204_0 w_204_1 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_225_1 w_226_0 w_226_1 w_227_0 w_228_0 w_228_1 w_228_2 w_228_3 w_229_0 w_229_1 w_230_0 w_230_1 w_231_0 w_232_0 w_234_0 w_235_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_244_1 w_244_2 w_244_3 w_245_0 w_245_1 w_245_2 w_245_3 w_246_0 w_246_1 w_246_2 w_246_3 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : (Fin 4 → Int) × ByteArray × (Fin 512 → Int) × Unit) (outputs : mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat18 × mxxMat18 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × mxxMat3 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_preprocess
