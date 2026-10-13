import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_prelude

structure Params where
  unit : Unit

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxNat2" => 39039761963948063856004653999987206021567571570771970785059555410812382343247291324210616315082566718137759516127340486041270513026828923337402589479793114710260407273133913650612796659174738653801475847949582336
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 501

abbrev parallel_generatedRoot_4.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_7_0 : ByteArray) (w_1_0 : mxxMat0) (w_8_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat0) (w_16_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_7_0) w_8_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [11]) ((319/100)) (31) w_11_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_7_0) w_16_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [19]) ((319/100)) (31) w_19_0 ∧
  outputs = (w_12_0, (w_20_0, ()))

def parallel_generatedRoot_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × Int × ByteArray × mxxMat1 × Unit) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_4_0 : Int := inputs.2.1
  let w_7_0 : ByteArray := inputs.2.2.1
  let w_13_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_1_0 : mxxMat0),
    let w_2_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_1_0
    let w_3_0 : mxxMat1 := (1 : mxxMat1)
    let w_5_0 : mxxMat1 := MxxRuntime.multiplyMonomial w_3_0 w_4_0
    let w_6_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_2_0 w_5_0
    ∃ (w_8_0 : mxxMat0),
      let w_9_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_8_0
      let w_10_0 : mxxMat0 := MxxRuntime.matrixSub w_6_0 w_9_0
      ∃ (w_11_0 : mxxMat0),
        let w_12_0 : mxxMat0 := MxxRuntime.matrixAdd w_10_0 w_11_0
        let w_14_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_13_0 w_1_0
        let w_15_0 : mxxMat0 := MxxRuntime.matrixMulScalarRight w_14_0 w_5_0
        ∃ (w_16_0 : mxxMat0),
          let w_17_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_0_0 w_16_0
          let w_18_0 : mxxMat0 := MxxRuntime.matrixSub w_15_0 w_17_0
          ∃ (w_19_0 : mxxMat0),
            let w_20_0 : mxxMat0 := MxxRuntime.matrixAdd w_18_0 w_19_0
            parallel_generatedRoot_4.constraints_0 backend hashModel tape path i_0 w_7_0 w_1_0 w_8_0 w_11_0 w_12_0 w_16_0 w_19_0 w_20_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_4_0 : Fin 4 → mxxMat0
  w_4_1 : Fin 4 → mxxMat0
  w_5_0 : mxxMat2
  w_8_0 : mxxMat3
  w_10_0 : mxxMat3
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_16_0 : mxxMat1
  w_20_0 : mxxMat1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat1) (w_1_0 : Fin 4 → Int) (w_2_0 : ByteArray) (w_19_0 : Fin 65536 → Int) (w_3_0 : mxxMat1) (w_4_0 : Fin 4 → mxxMat0) (w_4_1 : Fin 4 → mxxMat0) (w_5_0 : mxxMat2) (w_8_0 : mxxMat3) (w_10_0 : mxxMat3) (w_11_0 : mxxMat3) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : mxxMat1) (outputs : (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat3 × mxxMat1 × Unit) : Prop :=
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i (w_0_0, ((w_1_0 i), (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ()))) ∧
  MxxRuntime.centeredRebaseRuns w_0_0 w_5_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_2_0) w_8_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [10]) ((319/100)) (31) w_10_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (0)]) (w_2_0) w_12_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [97, 53, 51, 49, 99, 48, 48, 55, 55, 49, 55, 51, 99, 54, 53, 51, 56, 52, 49, 48, 98, 55, 57, 97, 99, 54, 97, 97, 53, 52, 48, 49, 50, 101, 98, 54, 48, 48, 53, 55, 53, 57, 98, 55, 102, 100, 50, 48, 97, 55, 52, 56, 97, 55, 52, 54, 53, 51, 98, 55, 54, 52, 50, 48], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) (w_2_0) w_14_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [16]) ((319/100)) (31) w_16_0 ∧
  MxxRuntime.polynomialFromValues false w_19_0 w_20_0 ∧
  outputs = (w_4_0, (w_4_1, (w_11_0, (w_22_0, ()))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1 × (Fin 4 → Int) × ByteArray × mxxMat3 × (Fin 65536 → Int) × Unit) (outputs : (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat3 × mxxMat1 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : Fin 4 → Int := inputs.2.1
  let w_2_0 : ByteArray := inputs.2.2.1
  let w_6_0 : mxxMat3 := inputs.2.2.2.1
  let w_19_0 : Fin 65536 → Int := inputs.2.2.2.2.1
  let w_3_0 : mxxMat1 := (1 : mxxMat1)
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_5_0 := witness.w_5_0
  let w_7_0 : mxxMat3 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat3 := MxxRuntime.matrixAdd w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : mxxMat3 := MxxRuntime.matrixAdd w_9_0 w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : mxxMat1 := MxxRuntime.matrixAdd w_13_0 w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat1 := MxxRuntime.matrixAdd w_15_0 w_16_0
  let w_18_0 : mxxMat1 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat1)
  let w_20_0 := witness.w_20_0
  let w_21_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
  let w_22_0 : mxxMat1 := MxxRuntime.matrixAdd w_17_0 w_21_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_19_0 w_3_0 w_4_0 w_4_1 w_5_0 w_8_0 w_10_0 w_11_0 w_12_0 w_14_0 w_16_0 w_20_0 w_22_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat1 × (Fin 4 → Int) × ByteArray × mxxMat3 × (Fin 65536 → Int) × Unit) (outputs : (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × mxxMat3 × mxxMat1 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_prelude
