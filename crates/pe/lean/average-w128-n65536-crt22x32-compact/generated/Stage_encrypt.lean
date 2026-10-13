import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_encrypt

structure Params where
  unit : Unit

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxNat2" => 67598865718973452911259282735194974463966008168342023559927392187609292957345997883634735633837025843140326117569327049993577673305918340947654355600136203031252562778990577685529163020701990351591297285624954881
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 88
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 2
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 1
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 87

abbrev parallel_generatedRoot_4.constraints_0 (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_8_0 : mxxMat0) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 37 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 37 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 37 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_0_0 w_8_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [12]) ((319/100)) (31) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 37 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 37 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
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

abbrev parallel_generatedRoot_5.constraints_1 (i_0 : Nat) (w_14_0 : Fin 37 → mxxMat1) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_8_0 : Int) (w_11_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ w_8_0 ∧
  w_8_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_8_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = (w_2_0, (w_4_0, (w_11_0, (w_17_0, ()))))

abbrev parallel_generatedRoot_5.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_7_0 : Fin 37 → mxxMat1) (w_14_0 : Fin 37 → mxxMat1) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (w_3_0 : mxxMat1) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_8_0 : Int) (w_9_0 : mxxMat1) (w_10_0 : mxxMat1) (w_11_0 : mxxMat1) (w_12_0 : mxxMat1) (w_13_0 : mxxMat1) (w_15_0 : mxxMat1) (w_16_0 : mxxMat1) (w_17_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_1_0 w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_3_0 w_4_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_8_0 ∧
  w_8_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_8_0 w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_9_0 w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  parallel_generatedRoot_5.constraints_1 i_0 w_14_0 w_2_0 w_4_0 w_8_0 w_11_0 w_15_0 w_16_0 w_17_0 outputs

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_7_0 : Fin 37 → mxxMat1 := inputs.2.1
  let w_14_0 : Fin 37 → mxxMat1 := inputs.2.2.1
  ∃ (w_1_0 : mxxMat1),
    ∃ (w_2_0 : mxxMat1),
      ∃ (w_3_0 : mxxMat1),
        ∃ (w_4_0 : mxxMat1),
          ∃ (w_5_0 : mxxMat1),
            ∃ (w_6_0 : mxxMat1),
              let w_8_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
              ∃ (w_9_0 : mxxMat1),
                ∃ (w_10_0 : mxxMat1),
                  let w_11_0 : mxxMat1 := MxxRuntime.matrixSub w_6_0 w_10_0
                  ∃ (w_12_0 : mxxMat1),
                    ∃ (w_13_0 : mxxMat1),
                      ∃ (w_15_0 : mxxMat1),
                        ∃ (w_16_0 : mxxMat1),
                          let w_17_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_16_0
                          parallel_generatedRoot_5.constraints_0 hashModel i_0 w_0_0 w_7_0 w_14_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (w_2_0 : Int) (outputs : Int) : Prop :=
  (3872) ≠ 0 ∧
  outputs = w_2_0

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (3872))
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_generatedRoot_6.constraints_0 w_2_0 outputs

abbrev parallel_generatedRoot_7.constraints_3 (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_3_0 : ByteArray) (w_36_0 : mxxMat2) (w_39_0 : mxxMat2) (w_45_0 : mxxMat2) (w_46_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (3872))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_3_0) w_39_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [45]) ((319/100)) (31) w_45_0 ∧
  outputs = (w_36_0, (w_46_0, ()))

abbrev parallel_generatedRoot_7.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_3_0 : ByteArray) (w_20_0 : mxxMat0) (w_25_0 : mxxMat3) (w_26_0 : mxxMat0) (w_27_0 : mxxMat4) (w_28_0 : mxxMat0) (w_29_0 : mxxMat5) (w_30_0 : mxxMat0) (w_31_0 : mxxMat2) (w_35_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : mxxMat2) (w_39_0 : mxxMat2) (w_45_0 : mxxMat2) (w_46_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  (Int.fmod ((Int.ofNat i_0)) (88)) < (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_25_0 0 1 (Int.fmod ((Int.ofNat i_0)) (88)) (1 + (Int.fmod ((Int.ofNat i_0)) (88))) w_26_0 ∧
  MxxRuntime.concatColumns w_20_0 w_26_0 w_27_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)) ∧
  (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)) < (1 + (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44))) ∧
  (1 + (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44))) ≤ 2 ∧
  MxxRuntime.sliceMatrix w_27_0 0 1 (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)) (1 + (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44))) w_28_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_28_0 w_29_0 ∧
  MxxRuntime.centeredRebaseRuns w_30_0 w_31_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [35]) ((319/100)) (31) w_35_0 ∧
  (3872) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (3872))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (3872))) < 22 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (3872))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_3_0) w_37_0 ∧
  (3872) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (3872))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (3872))) < 22 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  parallel_generatedRoot_7.constraints_3 hashModel tape path i_0 w_3_0 w_36_0 w_39_0 w_45_0 w_46_0 outputs

abbrev parallel_generatedRoot_7.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_3_0 : ByteArray) (w_14_0 : Fin 22 → mxxMat1) (w_15_0 : Int) (w_17_0 : Fin 22 → mxxMat1) (w_21_0 : Fin 22 → mxxMat1) (w_23_0 : Fin 22 → mxxMat1) (w_12_0 : mxxMat6) (w_13_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat3) (w_20_0 : mxxMat0) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat3) (w_26_0 : mxxMat0) (w_27_0 : mxxMat4) (w_28_0 : mxxMat0) (w_29_0 : mxxMat5) (w_30_0 : mxxMat0) (w_31_0 : mxxMat2) (w_35_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : mxxMat2) (w_39_0 : mxxMat2) (w_45_0 : mxxMat2) (w_46_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  (43 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) < (87 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) ∧
  (87 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (43 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) (87 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) w_13_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 22 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_15_0 w_16_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 22 ∧
  MxxRuntime.familyGetDynamic w_17_0 w_15_0 w_18_0 ∧
  MxxRuntime.concatColumns w_16_0 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (88)) ∧
  (Int.fmod ((Int.ofNat i_0)) (88)) < (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_19_0 0 1 (Int.fmod ((Int.ofNat i_0)) (88)) (1 + (Int.fmod ((Int.ofNat i_0)) (88))) w_20_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 22 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_15_0 w_22_0 ∧
  0 ≤ w_15_0 ∧
  w_15_0 < 22 ∧
  MxxRuntime.familyGetDynamic w_23_0 w_15_0 w_24_0 ∧
  MxxRuntime.concatColumns w_22_0 w_24_0 w_25_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (88)) ∧
  parallel_generatedRoot_7.constraints_2 backend hashModel tape path i_0 w_3_0 w_20_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 w_36_0 w_37_0 w_39_0 w_45_0 w_46_0 outputs

abbrev parallel_generatedRoot_7.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_3_0 : ByteArray) (w_6_0 : mxxMat0) (w_14_0 : Fin 22 → mxxMat1) (w_15_0 : Int) (w_17_0 : Fin 22 → mxxMat1) (w_21_0 : Fin 22 → mxxMat1) (w_23_0 : Fin 22 → mxxMat1) (w_1_0 : mxxMat0) (w_2_0 : mxxMat2) (w_4_0 : mxxMat2) (w_7_0 : mxxMat2) (w_8_0 : mxxMat2) (w_12_0 : mxxMat6) (w_13_0 : mxxMat1) (w_16_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat3) (w_20_0 : mxxMat0) (w_22_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat3) (w_26_0 : mxxMat0) (w_27_0 : mxxMat4) (w_28_0 : mxxMat0) (w_29_0 : mxxMat5) (w_30_0 : mxxMat0) (w_31_0 : mxxMat2) (w_35_0 : mxxMat2) (w_36_0 : mxxMat2) (w_37_0 : mxxMat2) (w_39_0 : mxxMat2) (w_45_0 : mxxMat2) (w_46_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  (3872) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (3872))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (3872))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (3872))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_0_0 w_1_0 ∧
  MxxRuntime.centeredRebaseRuns w_1_0 w_2_0 ∧
  (3872) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (3872))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (3872))) < 22 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (3872))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_3_0) w_4_0 ∧
  MxxRuntime.centeredRebaseRuns w_6_0 w_7_0 ∧
  (3872) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (3872))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (3872))) < 22 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (3872))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_3_0) w_8_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (43 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) ∧
  parallel_generatedRoot_7.constraints_1 backend hashModel tape path i_0 w_3_0 w_14_0 w_15_0 w_17_0 w_21_0 w_23_0 w_12_0 w_13_0 w_16_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 w_36_0 w_37_0 w_39_0 w_45_0 w_46_0 outputs

def parallel_generatedRoot_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 22 → mxxMat1) × Int × (Fin 22 → mxxMat1) × (Fin 22 → mxxMat1) × (Fin 22 → mxxMat1) × Unit) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_3_0 : ByteArray := inputs.2.1
  let w_6_0 : mxxMat0 := inputs.2.2.1
  let w_14_0 : Fin 22 → mxxMat1 := inputs.2.2.2.1
  let w_15_0 : Int := inputs.2.2.2.2.1
  let w_17_0 : Fin 22 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_21_0 : Fin 22 → mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 22 → mxxMat1 := inputs.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : mxxMat0),
    ∃ (w_2_0 : mxxMat2),
      ∃ (w_4_0 : mxxMat2),
        let w_5_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_2_0 w_4_0
        ∃ (w_7_0 : mxxMat2),
          ∃ (w_8_0 : mxxMat2),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_7_0 w_8_0
            let w_10_0 : mxxMat2 := MxxRuntime.matrixAdd w_5_0 w_9_0
            let w_11_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
            let w_12_0 : mxxMat6 := (MxxRuntime.unitRow (43) : mxxMat6)
            ∃ (w_13_0 : mxxMat1),
              ∃ (w_16_0 : mxxMat1),
                ∃ (w_18_0 : mxxMat1),
                  ∃ (w_19_0 : mxxMat3),
                    ∃ (w_20_0 : mxxMat0),
                      ∃ (w_22_0 : mxxMat1),
                        ∃ (w_24_0 : mxxMat1),
                          ∃ (w_25_0 : mxxMat3),
                            ∃ (w_26_0 : mxxMat0),
                              ∃ (w_27_0 : mxxMat4),
                                ∃ (w_28_0 : mxxMat0),
                                  ∃ (w_29_0 : mxxMat5),
                                    let w_30_0 : mxxMat0 := MxxRuntime.matrixMul w_13_0 w_29_0
                                    ∃ (w_31_0 : mxxMat2),
                                      let w_32_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_31_0 w_2_0
                                      let w_33_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_11_0 w_32_0
                                      let w_34_0 : mxxMat2 := MxxRuntime.matrixSub w_10_0 w_33_0
                                      ∃ (w_35_0 : mxxMat2),
                                        let w_36_0 : mxxMat2 := MxxRuntime.matrixAdd w_34_0 w_35_0
                                        ∃ (w_37_0 : mxxMat2),
                                          let w_38_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_2_0 w_37_0
                                          ∃ (w_39_0 : mxxMat2),
                                            let w_40_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_7_0 w_39_0
                                            let w_41_0 : mxxMat2 := MxxRuntime.matrixAdd w_38_0 w_40_0
                                            let w_42_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_31_0 w_7_0
                                            let w_43_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_11_0 w_42_0
                                            let w_44_0 : mxxMat2 := MxxRuntime.matrixSub w_41_0 w_43_0
                                            ∃ (w_45_0 : mxxMat2),
                                              let w_46_0 : mxxMat2 := MxxRuntime.matrixAdd w_44_0 w_45_0
                                              parallel_generatedRoot_7.constraints_0 backend hashModel tape path i_0 w_0_0 w_3_0 w_6_0 w_14_0 w_15_0 w_17_0 w_21_0 w_23_0 w_1_0 w_2_0 w_4_0 w_7_0 w_8_0 w_12_0 w_13_0 w_16_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_35_0 w_36_0 w_37_0 w_39_0 w_45_0 w_46_0 outputs

abbrev parallel_generatedRoot_36.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : mxxMat1) (w_2_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  outputs = (w_1_0, (w_2_0, ()))

def parallel_generatedRoot_36 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray) (outputs : mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs
  ∃ (w_1_0 : mxxMat1),
    ∃ (w_2_0 : mxxMat1),
      parallel_generatedRoot_36.constraints_0 hashModel i_0 w_0_0 w_1_0 w_2_0 outputs

abbrev parallel_generatedRoot_37.constraints_0 (w_2_0 : Int) (outputs : Int) : Prop :=
  (3872) ≠ 0 ∧
  outputs = w_2_0

def parallel_generatedRoot_37 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (3872))
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_generatedRoot_37.constraints_0 w_2_0 outputs

abbrev parallel_generatedRoot_40.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_2_0 : ByteArray) (w_24_0 : mxxMat4) (w_25_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat2) (w_34_0 : mxxMat2) (w_36_0 : mxxMat2) (w_42_0 : mxxMat2) (w_43_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)) < (1 + (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44))) ∧
  (1 + (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44))) ≤ 2 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)) (1 + (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44))) w_25_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_25_0 w_26_0 ∧
  MxxRuntime.centeredRebaseRuns w_27_0 w_28_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [32]) ((319/100)) (31) w_32_0 ∧
  (3872) ≠ 0 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (3872))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_34_0 ∧
  (3872) ≠ 0 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (3872))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_36_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [42]) ((319/100)) (31) w_42_0 ∧
  outputs = (w_33_0, (w_43_0, ()))

abbrev parallel_generatedRoot_40.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_2_0 : ByteArray) (w_14_0 : Int) (w_16_0 : Fin 4 → mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_22_0 : mxxMat3) (w_23_0 : mxxMat0) (w_24_0 : mxxMat4) (w_25_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat2) (w_34_0 : mxxMat2) (w_36_0 : mxxMat2) (w_42_0 : mxxMat2) (w_43_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_16_0 w_14_0 w_17_0 ∧
  MxxRuntime.concatColumns w_15_0 w_17_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (88)) ∧
  (Int.fmod ((Int.ofNat i_0)) (88)) < (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 (Int.fmod ((Int.ofNat i_0)) (88)) (1 + (Int.fmod ((Int.ofNat i_0)) (88))) w_19_0 ∧
  MxxRuntime.concatColumns w_20_0 w_21_0 w_22_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (88)) ∧
  (Int.fmod ((Int.ofNat i_0)) (88)) < (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (88))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_22_0 0 1 (Int.fmod ((Int.ofNat i_0)) (88)) (1 + (Int.fmod ((Int.ofNat i_0)) (88))) w_23_0 ∧
  MxxRuntime.concatColumns w_19_0 w_23_0 w_24_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (Int.fdiv ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)) ∧
  parallel_generatedRoot_40.constraints_2 backend hashModel tape path i_0 w_2_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_32_0 w_33_0 w_34_0 w_36_0 w_42_0 w_43_0 outputs

abbrev parallel_generatedRoot_40.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_2_0 : ByteArray) (w_5_0 : mxxMat0) (w_13_0 : Fin 4 → mxxMat1) (w_14_0 : Int) (w_16_0 : Fin 4 → mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat1) (w_1_0 : mxxMat2) (w_3_0 : mxxMat2) (w_6_0 : mxxMat2) (w_7_0 : mxxMat2) (w_11_0 : mxxMat6) (w_12_0 : mxxMat1) (w_15_0 : mxxMat1) (w_17_0 : mxxMat1) (w_18_0 : mxxMat3) (w_19_0 : mxxMat0) (w_22_0 : mxxMat3) (w_23_0 : mxxMat0) (w_24_0 : mxxMat4) (w_25_0 : mxxMat0) (w_26_0 : mxxMat5) (w_27_0 : mxxMat0) (w_28_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat2) (w_34_0 : mxxMat2) (w_36_0 : mxxMat2) (w_42_0 : mxxMat2) (w_43_0 : mxxMat2) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  MxxRuntime.centeredRebaseRuns w_0_0 w_1_0 ∧
  (3872) ≠ 0 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (3872))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  MxxRuntime.centeredRebaseRuns w_5_0 w_6_0 ∧
  (3872) ≠ 0 ∧
  (88) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (3872))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (88))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_7_0 ∧
  0 ≤ (43) ∧
  (43) < 87 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (43 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) ∧
  (43 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) < (87 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) ∧
  (87 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) ≤ 87 ∧
  MxxRuntime.sliceMatrix w_11_0 0 1 (43 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) (87 + (-1 * (Int.fmod ((44 + (Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (88))) (44)))) (44)))) w_12_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_14_0 w_15_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  parallel_generatedRoot_40.constraints_1 backend hashModel tape path i_0 w_2_0 w_14_0 w_16_0 w_20_0 w_21_0 w_15_0 w_17_0 w_18_0 w_19_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_32_0 w_33_0 w_34_0 w_36_0 w_42_0 w_43_0 outputs

def parallel_generatedRoot_40 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 4 → mxxMat1) × Int × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat2 × mxxMat2 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 4 → mxxMat1 := inputs.2.2.2.1
  let w_14_0 : Int := inputs.2.2.2.2.1
  let w_16_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_21_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : mxxMat2),
    ∃ (w_3_0 : mxxMat2),
      let w_4_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_3_0
      ∃ (w_6_0 : mxxMat2),
        ∃ (w_7_0 : mxxMat2),
          let w_8_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_7_0
          let w_9_0 : mxxMat2 := MxxRuntime.matrixAdd w_4_0 w_8_0
          let w_10_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
          let w_11_0 : mxxMat6 := (MxxRuntime.unitRow (43) : mxxMat6)
          ∃ (w_12_0 : mxxMat1),
            ∃ (w_15_0 : mxxMat1),
              ∃ (w_17_0 : mxxMat1),
                ∃ (w_18_0 : mxxMat3),
                  ∃ (w_19_0 : mxxMat0),
                    ∃ (w_22_0 : mxxMat3),
                      ∃ (w_23_0 : mxxMat0),
                        ∃ (w_24_0 : mxxMat4),
                          ∃ (w_25_0 : mxxMat0),
                            ∃ (w_26_0 : mxxMat5),
                              let w_27_0 : mxxMat0 := MxxRuntime.matrixMul w_12_0 w_26_0
                              ∃ (w_28_0 : mxxMat2),
                                let w_29_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_28_0 w_1_0
                                let w_30_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_10_0 w_29_0
                                let w_31_0 : mxxMat2 := MxxRuntime.matrixSub w_9_0 w_30_0
                                ∃ (w_32_0 : mxxMat2),
                                  let w_33_0 : mxxMat2 := MxxRuntime.matrixAdd w_31_0 w_32_0
                                  ∃ (w_34_0 : mxxMat2),
                                    let w_35_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_1_0 w_34_0
                                    ∃ (w_36_0 : mxxMat2),
                                      let w_37_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_36_0
                                      let w_38_0 : mxxMat2 := MxxRuntime.matrixAdd w_35_0 w_37_0
                                      let w_39_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_28_0 w_6_0
                                      let w_40_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_10_0 w_39_0
                                      let w_41_0 : mxxMat2 := MxxRuntime.matrixSub w_38_0 w_40_0
                                      ∃ (w_42_0 : mxxMat2),
                                        let w_43_0 : mxxMat2 := MxxRuntime.matrixAdd w_41_0 w_42_0
                                        parallel_generatedRoot_40.constraints_0 backend hashModel tape path i_0 w_0_0 w_2_0 w_5_0 w_13_0 w_14_0 w_16_0 w_20_0 w_21_0 w_1_0 w_3_0 w_6_0 w_7_0 w_11_0 w_12_0 w_15_0 w_17_0 w_18_0 w_19_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_32_0 w_33_0 w_34_0 w_36_0 w_42_0 w_43_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : mxxMat1
  w_4_0 : Fin 37 → mxxMat1
  w_4_1 : Fin 37 → mxxMat1
  w_5_0 : Fin 22 → mxxMat1
  w_5_1 : Fin 22 → mxxMat1
  w_5_2 : Fin 22 → mxxMat1
  w_5_3 : Fin 22 → mxxMat1
  w_6_0 : Fin 85184 → Int
  w_7_0 : Fin 85184 → mxxMat2
  w_7_1 : Fin 85184 → mxxMat2
  w_8_0 : mxxMat1
  w_10_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat1
  w_24_0 : mxxMat1
  w_26_0 : mxxMat1
  w_28_0 : mxxMat1
  w_34_0 : mxxMat1
  w_36_0 : Fin 4 → mxxMat1
  w_36_1 : Fin 4 → mxxMat1
  w_37_0 : Fin 15488 → Int
  w_40_0 : Fin 15488 → mxxMat2
  w_40_1 : Fin 15488 → mxxMat2

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_38_0 : Fin 4 → mxxMat1) (w_39_0 : Fin 4 → mxxMat1) (w_2_0 : mxxMat0) (w_3_0 : mxxMat1) (w_4_0 : Fin 37 → mxxMat1) (w_4_1 : Fin 37 → mxxMat1) (w_5_0 : Fin 22 → mxxMat1) (w_5_1 : Fin 22 → mxxMat1) (w_5_2 : Fin 22 → mxxMat1) (w_5_3 : Fin 22 → mxxMat1) (w_6_0 : Fin 85184 → Int) (w_7_0 : Fin 85184 → mxxMat2) (w_7_1 : Fin 85184 → mxxMat2) (w_8_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_24_0 : mxxMat1) (w_25_0 : mxxMat1) (w_26_0 : mxxMat1) (w_28_0 : mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat1) (w_36_0 : Fin 4 → mxxMat1) (w_36_1 : Fin 4 → mxxMat1) (w_37_0 : Fin 15488 → Int) (w_40_0 : Fin 15488 → mxxMat2) (w_40_1 : Fin 15488 → mxxMat2) (outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_3_0 ∧
  (37) = 37 ∧
  (∀ i : Fin 37, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ()))) ∧
  (22) = 22 ∧
  (∀ i : Fin 22, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i (w_1_0, (w_4_0, (w_4_1, ()))) ((w_5_0 i), ((w_5_1 i), ((w_5_2 i), ((w_5_3 i), ()))))) ∧
  (85184) = 85184 ∧
  (∀ i : Fin 85184, parallel_generatedRoot_6 backend hashModel tape (path ++ [6, i.val]) params i () (w_6_0 i)) ∧
  (85184) = 85184 ∧
  (∀ i : Fin 85184, parallel_generatedRoot_7 backend hashModel tape (path ++ [7, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_5_0, ((w_6_0 i), (w_5_1, (w_5_2, (w_5_3, ())))))))) ((w_7_0 i), ((w_7_1 i), ()))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_8_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_13_0 ∧
  MxxRuntime.centeredRebaseRuns w_13_0 w_14_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [24]) ((319/100)) (31) w_24_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_26_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_28_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [34]) ((319/100)) (31) w_34_0 ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_36 backend hashModel tape (path ++ [36, i.val]) params i w_1_0 ((w_36_0 i), ((w_36_1 i), ()))) ∧
  (15488) = 15488 ∧
  (∀ i : Fin 15488, parallel_generatedRoot_37 backend hashModel tape (path ++ [37, i.val]) params i () (w_37_0 i)) ∧
  (15488) = 15488 ∧
  (∀ i : Fin 15488, ∃ w_38_0_gathered : mxxMat1, ∃ w_39_0_gathered : mxxMat1, 0 ≤ (w_37_0 i) ∧
  (w_37_0 i) < 4 ∧
  MxxRuntime.familyGetDynamic w_38_0 (w_37_0 i) w_38_0_gathered ∧
  0 ≤ (w_37_0 i) ∧
  (w_37_0 i) < 4 ∧
  MxxRuntime.familyGetDynamic w_39_0 (w_37_0 i) w_39_0_gathered ∧
  parallel_generatedRoot_40 backend hashModel tape (path ++ [40, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_36_0, ((w_37_0 i), (w_36_1, (w_38_0_gathered, (w_39_0_gathered, ())))))))) ((w_40_0 i), ((w_40_1 i), ()))) ∧
  outputs = (w_7_0, (w_7_1, (w_4_0, (w_4_1, (w_25_0, (w_35_0, (w_40_0, (w_40_1, ()))))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × ByteArray × Int × Int × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × Unit) (outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_17_0 : Int := inputs.2.2.1
  let w_19_0 : Int := inputs.2.2.2.1
  let w_38_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.1
  let w_39_0 : Fin 4 → mxxMat1 := inputs.2.2.2.2.2.1
  let w_2_0 : mxxMat0 := (1 : mxxMat0)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_5_0 := witness.w_5_0
  let w_5_1 := witness.w_5_1
  let w_5_2 := witness.w_5_2
  let w_5_3 := witness.w_5_3
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_7_1 := witness.w_7_1
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_2_0 w_10_0
  let w_12_0 : mxxMat1 := MxxRuntime.matrixAdd w_9_0 w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_14_0
  let w_16_0 : mxxMat0 := (MxxRuntime.matrixPolynomial [(1)] : mxxMat0)
  let w_18_0 : mxxMat0 := MxxRuntime.liftInteger w_17_0
  let w_20_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_18_0 w_19_0
  let w_21_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_16_0 w_20_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_15_0 w_21_0
  let w_23_0 : mxxMat1 := MxxRuntime.matrixSub w_12_0 w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_26_0
  let w_28_0 := witness.w_28_0
  let w_29_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_2_0 w_28_0
  let w_30_0 : mxxMat1 := MxxRuntime.matrixAdd w_27_0 w_29_0
  let w_31_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_2_0 w_14_0
  let w_32_0 : mxxMat1 := MxxRuntime.matrixMulScalarRight w_31_0 w_21_0
  let w_33_0 : mxxMat1 := MxxRuntime.matrixSub w_30_0 w_32_0
  let w_34_0 := witness.w_34_0
  let w_35_0 : mxxMat1 := MxxRuntime.matrixAdd w_33_0 w_34_0
  let w_36_0 := witness.w_36_0
  let w_36_1 := witness.w_36_1
  let w_37_0 := witness.w_37_0
  let w_40_0 := witness.w_40_0
  let w_40_1 := witness.w_40_1
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_38_0 w_39_0 w_2_0 w_3_0 w_4_0 w_4_1 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_7_0 w_7_1 w_8_0 w_10_0 w_13_0 w_14_0 w_24_0 w_25_0 w_26_0 w_28_0 w_34_0 w_35_0 w_36_0 w_36_1 w_37_0 w_40_0 w_40_1 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × ByteArray × Int × Int × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × Unit) (outputs : (Fin 85184 → mxxMat2) × (Fin 85184 → mxxMat2) × (Fin 37 → mxxMat1) × (Fin 37 → mxxMat1) × mxxMat1 × mxxMat1 × (Fin 15488 → mxxMat2) × (Fin 15488 → mxxMat2) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_encrypt
