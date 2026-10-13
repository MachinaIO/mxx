import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_online_eval

structure Params where
  unit : Unit

local notation "mxxNat0" => 78079523927896127712009307999974412043135143141543941570119110821624764686494582648421232630165133436275519032254680972082541026053657846674805178959586229420520814546267827301225593318349477307602951695899164673
local notation "mxxNat1" => 5278087253403224991333645099610003021785206966119564254171942298247993190952980821552692773158302798199939315804484105262715072792529872674621368544490941390506983498620144494214222200022752952511718466872729239793477806438096725442738627984346925320228738380148171996792888816066148390509478988805065937247347127262069050805334354391385613933775075315634469785233518480876460832181155068060520232362595000272223451214118913
local notation "mxxNat2" => 77469527647209439214259235281224611949048149835750629526602555268330821212381343721480441750241968331304616539815191276988146174287613644747658263498964462003172995682625109900434768370549872016137303635774952449
local notation "mxxMat0" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 44
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 88
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix mxxNat0 65536 2 1
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix mxxNat0 65536 88 1
local notation "mxxMat5" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 175
local notation "mxxMat6" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 1
local notation "mxxMat7" => Mxx.Primitives.ExactMatrix mxxNat1 65536 1 44
local notation "mxxMat8" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 1
local notation "mxxMat9" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 22
local notation "mxxMat10" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 1936
local notation "mxxMat11" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 43
local notation "mxxMat12" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 11176
local notation "mxxMat13" => Mxx.Primitives.ExactMatrix mxxNat0 65536 44 44
local notation "mxxMat14" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 105
local notation "mxxMat15" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 9240
local notation "mxxMat16" => Mxx.Primitives.ExactMatrix mxxNat0 65536 1 209

abbrev parallel_generatedRoot_12.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [114, 104, 115, 45, 116, 111, 112], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_2_0 ∧
  outputs = (w_1_0, (w_2_0, ()))

def parallel_generatedRoot_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray) (outputs : mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs
  ∃ (w_1_0 : mxxMat0),
    ∃ (w_2_0 : mxxMat0),
      parallel_generatedRoot_12.constraints_0 hashModel i_0 w_0_0 w_1_0 w_2_0 outputs

abbrev sequential_generatedRoot_23.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_0))) ∧
  (87 + (-1 * (Int.ofNat i_0))) < (175 + (-1 * (Int.ofNat i_0))) ∧
  (175 + (-1 * (Int.ofNat i_0))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_0))) (175 + (-1 * (Int.ofNat i_0))) w_11_0 ∧
  outputs = w_13_0

def sequential_generatedRoot_23 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_generatedRoot_23.constraints_0 backend i_0 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_33_26.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_12_0 : Fin 15488 → mxxMat6) (w_23_0 : Fin 15488 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_17_0 : Int) (w_18_0 : mxxMat6) (w_19_0 : mxxMat0) (w_20_0 : mxxMat7) (w_21_0 : mxxMat7) (w_22_0 : mxxMat0) (w_24_0 : mxxMat6) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_17_0 w_18_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_19_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_20_0 ∧
  MxxRuntime.modulusSwitchRuns w_21_0 w_22_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_23_0 w_17_0 w_24_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_25_0 ∧
  MxxRuntime.modulusSwitchRuns w_26_0 w_27_0 ∧
  outputs = (w_11_0, (w_22_0, (w_27_0, ())))

def sequential_sequential_generatedRoot_33_26 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_13_0 : Int := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_14_0 : Int := 3872
    let w_15_0 : Int := (w_13_0 * w_14_0)
    let w_16_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88)))
    let w_17_0 : Int := (w_15_0 + w_16_0)
    ∃ (w_18_0 : mxxMat6),
      ∃ (w_19_0 : mxxMat0),
        ∃ (w_20_0 : mxxMat7),
          let w_21_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
          ∃ (w_22_0 : mxxMat0),
            ∃ (w_24_0 : mxxMat6),
              ∃ (w_25_0 : mxxMat7),
                let w_26_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
                ∃ (w_27_0 : mxxMat0),
                  sequential_sequential_generatedRoot_33_26.constraints_0 backend i_1 w_12_0 w_23_0 w_8_0 w_9_0 w_11_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_33.Witness where
  w_2_0 : mxxMat2
  w_10_0 : mxxMat6
  w_11_0 : mxxMat0
  w_12_0 : mxxMat7
  w_14_0 : mxxMat0
  w_16_0 : mxxMat6
  w_17_0 : mxxMat7
  w_19_0 : mxxMat0
  w_21_0 : mxxMat2
  w_22_0 : mxxMat8
  w_24_0 : mxxMat2
  w_25_0 : mxxMat8
  w_26_0 : mxxMat1
  w_26_1 : mxxMat0
  w_26_2 : mxxMat0
  w_27_0 : mxxMat1

abbrev sequential_generatedRoot_33.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_23_0 : mxxMat1) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_14_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_24_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_24_0 w_25_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_33_26 backend hashModel tape (path ++ [26, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_22_0, (w_25_0, (w_4_0, (w_5_0, (w_15_0, ())))))))) next) (Int.toNat (88)) (w_3_0, (w_14_0, (w_19_0, ()))) (w_26_0, (w_26_1, (w_26_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_26_0 w_27_0 ∧
  outputs = w_28_0

abbrev sequential_generatedRoot_33.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_1_0 : mxxMat2) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_9_0 : Int) (w_10_0 : mxxMat6) (w_11_0 : mxxMat0) (w_12_0 : mxxMat7) (w_13_0 : mxxMat7) (w_14_0 : mxxMat0) (w_16_0 : mxxMat6) (w_17_0 : mxxMat7) (w_18_0 : mxxMat7) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_11_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_12_0 ∧
  MxxRuntime.modulusSwitchRuns w_13_0 w_14_0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_9_0 w_16_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_17_0 ∧
  MxxRuntime.modulusSwitchRuns w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  sequential_generatedRoot_33.constraints_1 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_23_0 w_2_0 w_3_0 w_14_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_33.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) (witness : sequential_generatedRoot_33.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_4_0 : Fin 15488 → mxxMat6 := inputs.2.1
  let w_5_0 : Int := inputs.2.2.1
  let w_15_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat2 := (MxxRuntime.unitRow (0) : mxxMat2)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_6_0 : Int := 3872
  let w_7_0 : Int := (w_5_0 * w_6_0)
  let w_8_0 : Int := (88 * (Int.fmod ((Int.ofNat i_0)) (44)))
  let w_9_0 : Int := (w_7_0 + w_8_0)
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_10_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_26_1 := witness.w_26_1
  let w_26_2 := witness.w_26_2
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_27_0
  sequential_generatedRoot_33.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_20_0 w_23_0 w_1_0 w_2_0 w_3_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

def sequential_generatedRoot_33 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  ∃ witness : sequential_generatedRoot_33.Witness,
    sequential_generatedRoot_33.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_39.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_generatedRoot_39 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_generatedRoot_39.constraints_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_64_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_64_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_64_24.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_64.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_17_0 : mxxMat2
  w_18_0 : mxxMat2
  w_19_0 : mxxMat8
  w_21_0 : mxxMat2
  w_22_0 : mxxMat2
  w_23_0 : mxxMat8
  w_24_0 : mxxMat1
  w_24_1 : mxxMat0
  w_24_2 : mxxMat0
  w_25_0 : mxxMat10

abbrev sequential_generatedRoot_64.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_17_0 w_18_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_64_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_19_0, (w_23_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_24_0 w_25_0 ∧
  outputs = w_26_0

abbrev sequential_generatedRoot_64.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
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
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_64.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

abbrev sequential_generatedRoot_64.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) (witness : sequential_generatedRoot_64.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_25_0
  sequential_generatedRoot_64.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

def sequential_generatedRoot_64 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) : Prop :=
  ∃ witness : sequential_generatedRoot_64.Witness,
    sequential_generatedRoot_64.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_65_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_18_0 : mxxMat1) (w_23_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_65_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_14_0 : mxxMat1 := inputs.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.1
  let w_1_0 : mxxMat1 := inputs.2.2.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_14_0 w_17_0
              let w_21_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_8_0
              let w_22_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_11_0
              let w_23_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_22_0
              sequential_parallel_generatedRoot_65_24.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_65_49.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_parallel_generatedRoot_65_49 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_parallel_generatedRoot_65_49.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_65.Witness where
  w_3_0 : mxxMat0
  w_5_0 : mxxMat0
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat0
  w_15_0 : mxxMat0
  w_16_0 : mxxMat1
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat1
  w_24_0 : mxxMat1
  w_24_1 : mxxMat1
  w_24_2 : mxxMat1
  w_25_0 : mxxMat0
  w_28_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat1
  w_37_0 : mxxMat0
  w_38_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_45_0 : mxxMat0
  w_46_0 : mxxMat0
  w_48_0 : mxxMat1
  w_49_0 : mxxMat1
  w_50_0 : mxxMat0
  w_52_0 : mxxMat1
  w_53_0 : mxxMat0
  w_56_0 : mxxMat0
  w_57_0 : mxxMat0
  w_58_0 : mxxMat0
  w_61_0 : mxxMat0
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev parallel_generatedRoot_65.constraints_6 (w_24_2 : mxxMat1) (w_55_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 44 88 w_64_0 ∧
  outputs = (w_55_0, (w_60_0, (w_61_0, (w_62_0, (w_63_0, (w_64_0, ()))))))

abbrev parallel_generatedRoot_65.constraints_5 (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_52_0 : mxxMat1) (w_55_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 44 88 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 0 44 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 0 44 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  parallel_generatedRoot_65.constraints_6 w_24_2 w_55_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_65.constraints_4 (i_0 : Nat) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 0 44 w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_51_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_52_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 44 w_53_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 44 88 w_56_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 44 88 w_57_0 ∧
  parallel_generatedRoot_65.constraints_5 w_24_1 w_24_2 w_52_0 w_55_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_65.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_26_0 : mxxMat1) (w_31_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_37_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_39_0 w_40_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_43_0 w_44_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_39_0 w_45_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_45_0 w_46_0 ∧
  MxxRuntime.concatColumns w_42_0 w_47_0 w_48_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_65_49 backend hashModel tape (path ++ [49, i]) params i_0 i (current, (w_31_0, (w_36_0, (w_48_0, ())))) next) (Int.toNat (88)) w_26_0 w_49_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_65.constraints_4 i_0 w_51_0 w_24_0 w_24_1 w_24_2 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_65.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_22_0 ∧
  MxxRuntime.concatColumns w_21_0 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_65_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_6_0, (w_11_0, (w_16_0, (w_20_0, (w_23_0, ())))))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, (w_0_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 0 44 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_34_0 w_35_0 ∧
  MxxRuntime.concatColumns w_33_0 w_35_0 w_36_0 ∧
  parallel_generatedRoot_65.constraints_3 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_51_0 w_24_0 w_24_1 w_24_2 w_26_0 w_31_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_65.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 0 44 w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 44 88 w_15_0 ∧
  MxxRuntime.concatColumns w_14_0 w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_19_0 ∧
  MxxRuntime.concatColumns w_18_0 w_19_0 w_20_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  parallel_generatedRoot_65.constraints_2 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_16_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_65.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_7_0 : mxxMat12) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat0) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 0 44 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 44 88 w_10_0 ∧
  MxxRuntime.concatColumns w_9_0 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_65.constraints_1 backend hashModel tape path params i_0 w_1_0 w_4_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_65.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_65.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 37 → mxxMat0 := inputs.1
  let w_4_0 : Fin 37 → mxxMat0 := inputs.2.1
  let w_7_0 : mxxMat12 := inputs.2.2.1
  let w_12_0 : mxxMat12 := inputs.2.2.2.1
  let w_17_0 : ByteArray := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_29_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_51_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat1 := (0 : mxxMat1)
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_38_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixSub w_44_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_53_0
  let w_55_0 : mxxMat0 := MxxRuntime.matrixAdd w_25_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixAdd w_57_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_56_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  parallel_generatedRoot_65.constraints_0 backend hashModel tape path params i_0 w_1_0 w_4_0 w_7_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

def parallel_generatedRoot_65 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_65.Witness,
    parallel_generatedRoot_65.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_66_7.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_66_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_66_7.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_66.Witness where
  w_3_0 : mxxMat0
  w_7_0 : mxxMat2
  w_9_0 : mxxMat0
  w_11_0 : mxxMat13
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_66.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_1_0 : Fin 15 → mxxMat0) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat2) (w_7_0 : mxxMat2) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_66_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  outputs = (w_13_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))))

abbrev sequential_generatedRoot_66.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_66.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_20_0 : mxxMat0 := inputs.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_5_0 : mxxMat2 := (0 : mxxMat2)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
  sequential_generatedRoot_66.constraints_0 backend hashModel tape path params i_0 w_19_0 w_20_0 w_21_0 w_22_0 w_1_0 w_6_0 w_15_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 outputs

def sequential_generatedRoot_66 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_66.Witness,
    sequential_generatedRoot_66.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_67_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_67_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_67_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_67_44.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_67_44 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_67_44.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_67_54.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_67_54 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_67_54.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_67.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_22_0 : mxxMat0
  w_24_0 : mxxMat0
  w_25_0 : mxxMat1
  w_27_0 : mxxMat0
  w_28_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat1
  w_44_0 : mxxMat1
  w_45_0 : mxxMat0
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_54_0 : mxxMat2
  w_56_0 : mxxMat0
  w_58_0 : mxxMat13
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev sequential_generatedRoot_67.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_44_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_67_54 backend hashModel tape (path ++ [54, i]) params i_0 i (current, (w_53_0, ())) next) (Int.toNat (128)) w_52_0 w_54_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_57_0 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 44 88 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_64_0 ∧
  outputs = (w_60_0, (w_68_0, (w_69_0, (w_70_0, (w_71_0, (w_72_0, ()))))))

abbrev sequential_generatedRoot_67.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_25_0 : mxxMat1) (w_31_0 : mxxMat1) (w_34_0 : Int) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_26_0) w_38_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_34_0 w_40_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  MxxRuntime.concatColumns w_37_0 w_42_0 w_43_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_67_44 backend hashModel tape (path ++ [44, i]) params i_0 i (current, (w_25_0, (w_31_0, (w_43_0, ())))) next) (Int.toNat (88)) w_20_0 w_44_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 0 44 w_45_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_46_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_67.constraints_4 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_53_0 w_18_0 w_44_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_67.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_23_0 w_24_0 ∧
  MxxRuntime.concatColumns w_22_0 w_24_0 w_25_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_26_0) w_27_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_26_0) w_29_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_26_0) w_32_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_34_0 w_35_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_67.constraints_3 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_5_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_25_0 w_31_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_67.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_67_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  sequential_generatedRoot_67.constraints_2 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_67.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_67.constraints_1 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_67.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_67.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_61_0 : mxxMat0 := inputs.2.1
  let w_69_0 : mxxMat0 := inputs.2.2.1
  let w_70_0 : mxxMat0 := inputs.2.2.2.1
  let w_71_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat0 := MxxRuntime.matrixSub w_33_0 w_36_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_39_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_45_0 w_48_0
  let w_50_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_49_0
  let w_51_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_52_0 : mxxMat2 := (0 : mxxMat2)
  let w_54_0 := witness.w_54_0
  let w_55_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_51_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_55_0 w_56_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_50_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_59_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_63_0 w_64_0
  let w_66_0 : mxxMat0 := MxxRuntime.matrixAdd w_62_0 w_65_0
  let w_67_0 : mxxMat0 := MxxRuntime.matrixMul w_66_0 w_58_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_61_0 w_67_0
  sequential_generatedRoot_67.constraints_0 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

def sequential_generatedRoot_67 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_67.Witness,
    sequential_generatedRoot_67.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_69_34.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_69_34 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_69_34.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_69.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_18_0 : mxxMat0
  w_20_0 : mxxMat0
  w_21_0 : mxxMat1
  w_22_0 : mxxMat2
  w_23_0 : mxxMat2
  w_24_0 : mxxMat8
  w_27_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat1
  w_31_0 : mxxMat2
  w_32_0 : mxxMat2
  w_33_0 : mxxMat8
  w_34_0 : mxxMat1
  w_34_1 : mxxMat0
  w_34_2 : mxxMat0
  w_35_0 : mxxMat15

abbrev sequential_generatedRoot_69.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_24_0 : mxxMat8) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_31_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_32_0 w_33_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_69_34 backend hashModel tape (path ++ [34, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_24_0, (w_33_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_34_0, (w_34_1, (w_34_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_34_0 w_35_0 ∧
  outputs = w_36_0

abbrev sequential_generatedRoot_69.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_22_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_26_0 w_27_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_28_0 w_26_0 w_29_0 ∧
  MxxRuntime.concatColumns w_27_0 w_29_0 w_30_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  sequential_generatedRoot_69.constraints_2 backend hashModel tape path params i_0 w_4_0 w_11_0 w_2_0 w_3_0 w_10_0 w_15_0 w_24_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_69.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : Fin 15 → mxxMat0) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  sequential_generatedRoot_69.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_19_0 w_25_0 w_28_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_69.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) (witness : sequential_generatedRoot_69.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : Fin 15 → mxxMat0 := inputs.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_28_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_18_0 := witness.w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_27_0 := witness.w_27_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_34_1 := witness.w_34_1
  let w_34_2 := witness.w_34_2
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_35_0
  sequential_generatedRoot_69.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_19_0 w_25_0 w_28_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

def sequential_generatedRoot_69 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) : Prop :=
  ∃ witness : sequential_generatedRoot_69.Witness,
    sequential_generatedRoot_69.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_70_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_70_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_70_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_70_47.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_70_47 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_70_47.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_70_57.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_70_57 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 0
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_70_57.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_70.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_23_0 : mxxMat0
  w_24_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat0
  w_28_0 : mxxMat1
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat1
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_41_0 : mxxMat0
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_46_0 : mxxMat1
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_50_0 : mxxMat1
  w_51_0 : mxxMat0
  w_57_0 : mxxMat2
  w_59_0 : mxxMat0
  w_61_0 : mxxMat13
  w_65_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0

abbrev sequential_generatedRoot_70.constraints_5 (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_50_0 : mxxMat1) (w_63_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 44 88 w_67_0 ∧
  outputs = (w_63_0, (w_71_0, (w_72_0, (w_73_0, (w_74_0, (w_75_0, ()))))))

abbrev sequential_generatedRoot_70.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_47_0 : mxxMat1) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 0 44 w_51_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_70_57 backend hashModel tape (path ++ [57, i]) params i_0 i (current, (w_56_0, ())) next) (Int.toNat (128)) w_55_0 w_57_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_60_0 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_65_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_66_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_70.constraints_5 w_72_0 w_73_0 w_74_0 w_75_0 w_50_0 w_63_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_70.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_28_0 : mxxMat1) (w_34_0 : mxxMat1) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_29_0) w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_41_0 w_42_0 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_37_0 w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_43_0 w_44_0 ∧
  MxxRuntime.concatColumns w_40_0 w_45_0 w_46_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_70_47 backend hashModel tape (path ++ [47, i]) params i_0 i (current, (w_28_0, (w_34_0, (w_46_0, ())))) next) (Int.toNat (88)) w_20_0 w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_70.constraints_4 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_49_0 w_56_0 w_18_0 w_47_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_70.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_23_0 w_24_0 ∧
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_22_0 w_26_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_26_0 w_27_0 ∧
  MxxRuntime.concatColumns w_24_0 w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_29_0) w_30_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_29_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_32_0 w_33_0 ∧
  MxxRuntime.concatColumns w_31_0 w_33_0 w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_29_0) w_35_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_70.constraints_3 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_28_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_70.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_70_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  sequential_generatedRoot_70.constraints_2 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_70.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_70.constraints_1 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_70.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_70.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_64_0 : mxxMat0 := inputs.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.1
  let w_73_0 : mxxMat0 := inputs.2.2.2.1
  let w_74_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_75_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : mxxMat15 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_56_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15)])
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat0 := MxxRuntime.matrixSub w_36_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : mxxMat0 := MxxRuntime.matrixSub w_42_0 w_44_0
  let w_46_0 := witness.w_46_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat0 := MxxRuntime.matrixAdd w_48_0 w_51_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_52_0
  let w_54_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_55_0 : mxxMat2 := (0 : mxxMat2)
  let w_57_0 := witness.w_57_0
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_54_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_58_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat0 := MxxRuntime.matrixMul w_53_0 w_61_0
  let w_63_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_62_0
  let w_65_0 := witness.w_65_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat0 := MxxRuntime.matrixAdd w_65_0 w_68_0
  let w_70_0 : mxxMat0 := MxxRuntime.matrixMul w_69_0 w_61_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_64_0 w_70_0
  sequential_generatedRoot_70.constraints_0 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

def sequential_generatedRoot_70 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_70.Witness,
    sequential_generatedRoot_70.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_79.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_0))) ∧
  (87 + (-1 * (Int.ofNat i_0))) < (175 + (-1 * (Int.ofNat i_0))) ∧
  (175 + (-1 * (Int.ofNat i_0))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_0))) (175 + (-1 * (Int.ofNat i_0))) w_11_0 ∧
  outputs = w_13_0

def sequential_generatedRoot_79 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_generatedRoot_79.constraints_0 backend i_0 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_87_26.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_12_0 : Fin 15488 → mxxMat6) (w_23_0 : Fin 15488 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_17_0 : Int) (w_18_0 : mxxMat6) (w_19_0 : mxxMat0) (w_20_0 : mxxMat7) (w_21_0 : mxxMat7) (w_22_0 : mxxMat0) (w_24_0 : mxxMat6) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_17_0 w_18_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_19_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_20_0 ∧
  MxxRuntime.modulusSwitchRuns w_21_0 w_22_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_23_0 w_17_0 w_24_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_25_0 ∧
  MxxRuntime.modulusSwitchRuns w_26_0 w_27_0 ∧
  outputs = (w_11_0, (w_22_0, (w_27_0, ())))

def sequential_sequential_generatedRoot_87_26 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_13_0 : Int := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_14_0 : Int := 3872
    let w_15_0 : Int := (w_13_0 * w_14_0)
    let w_16_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88)))
    let w_17_0 : Int := (w_15_0 + w_16_0)
    ∃ (w_18_0 : mxxMat6),
      ∃ (w_19_0 : mxxMat0),
        ∃ (w_20_0 : mxxMat7),
          let w_21_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
          ∃ (w_22_0 : mxxMat0),
            ∃ (w_24_0 : mxxMat6),
              ∃ (w_25_0 : mxxMat7),
                let w_26_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
                ∃ (w_27_0 : mxxMat0),
                  sequential_sequential_generatedRoot_87_26.constraints_0 backend i_1 w_12_0 w_23_0 w_8_0 w_9_0 w_11_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_87.Witness where
  w_2_0 : mxxMat2
  w_10_0 : mxxMat6
  w_11_0 : mxxMat0
  w_12_0 : mxxMat7
  w_14_0 : mxxMat0
  w_16_0 : mxxMat6
  w_17_0 : mxxMat7
  w_19_0 : mxxMat0
  w_21_0 : mxxMat2
  w_22_0 : mxxMat8
  w_24_0 : mxxMat2
  w_25_0 : mxxMat8
  w_26_0 : mxxMat1
  w_26_1 : mxxMat0
  w_26_2 : mxxMat0
  w_27_0 : mxxMat1

abbrev sequential_generatedRoot_87.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_23_0 : mxxMat1) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_14_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_24_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_24_0 w_25_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_87_26 backend hashModel tape (path ++ [26, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_22_0, (w_25_0, (w_4_0, (w_5_0, (w_15_0, ())))))))) next) (Int.toNat (88)) (w_3_0, (w_14_0, (w_19_0, ()))) (w_26_0, (w_26_1, (w_26_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_26_0 w_27_0 ∧
  outputs = w_28_0

abbrev sequential_generatedRoot_87.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_1_0 : mxxMat2) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_9_0 : Int) (w_10_0 : mxxMat6) (w_11_0 : mxxMat0) (w_12_0 : mxxMat7) (w_13_0 : mxxMat7) (w_14_0 : mxxMat0) (w_16_0 : mxxMat6) (w_17_0 : mxxMat7) (w_18_0 : mxxMat7) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_11_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_12_0 ∧
  MxxRuntime.modulusSwitchRuns w_13_0 w_14_0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_9_0 w_16_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_17_0 ∧
  MxxRuntime.modulusSwitchRuns w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  sequential_generatedRoot_87.constraints_1 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_23_0 w_2_0 w_3_0 w_14_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_87.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) (witness : sequential_generatedRoot_87.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_4_0 : Fin 15488 → mxxMat6 := inputs.2.1
  let w_5_0 : Int := inputs.2.2.1
  let w_15_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat2 := (MxxRuntime.unitRow (0) : mxxMat2)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_6_0 : Int := 3872
  let w_7_0 : Int := (w_5_0 * w_6_0)
  let w_8_0 : Int := (88 * (Int.fmod ((Int.ofNat i_0)) (44)))
  let w_9_0 : Int := (w_7_0 + w_8_0)
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_10_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_26_1 := witness.w_26_1
  let w_26_2 := witness.w_26_2
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_27_0
  sequential_generatedRoot_87.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_20_0 w_23_0 w_1_0 w_2_0 w_3_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

def sequential_generatedRoot_87 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  ∃ witness : sequential_generatedRoot_87.Witness,
    sequential_generatedRoot_87.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_92.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_generatedRoot_92 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_generatedRoot_92.constraints_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_113_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_113_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_113_24.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_113.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_17_0 : mxxMat2
  w_18_0 : mxxMat2
  w_19_0 : mxxMat8
  w_21_0 : mxxMat2
  w_22_0 : mxxMat2
  w_23_0 : mxxMat8
  w_24_0 : mxxMat1
  w_24_1 : mxxMat0
  w_24_2 : mxxMat0
  w_25_0 : mxxMat10

abbrev sequential_generatedRoot_113.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_17_0 w_18_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_113_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_19_0, (w_23_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_24_0 w_25_0 ∧
  outputs = w_26_0

abbrev sequential_generatedRoot_113.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
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
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_113.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

abbrev sequential_generatedRoot_113.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) (witness : sequential_generatedRoot_113.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_25_0
  sequential_generatedRoot_113.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

def sequential_generatedRoot_113 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) : Prop :=
  ∃ witness : sequential_generatedRoot_113.Witness,
    sequential_generatedRoot_113.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_114_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_18_0 : mxxMat1) (w_23_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_114_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_14_0 : mxxMat1 := inputs.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.1
  let w_1_0 : mxxMat1 := inputs.2.2.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_14_0 w_17_0
              let w_21_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_8_0
              let w_22_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_11_0
              let w_23_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_22_0
              sequential_parallel_generatedRoot_114_24.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_114_49.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_parallel_generatedRoot_114_49 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_parallel_generatedRoot_114_49.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_114.Witness where
  w_3_0 : mxxMat0
  w_5_0 : mxxMat0
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat0
  w_15_0 : mxxMat0
  w_16_0 : mxxMat1
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat1
  w_24_0 : mxxMat1
  w_24_1 : mxxMat1
  w_24_2 : mxxMat1
  w_25_0 : mxxMat0
  w_28_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat1
  w_37_0 : mxxMat0
  w_38_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_45_0 : mxxMat0
  w_46_0 : mxxMat0
  w_48_0 : mxxMat1
  w_49_0 : mxxMat1
  w_50_0 : mxxMat0
  w_52_0 : mxxMat1
  w_53_0 : mxxMat0
  w_56_0 : mxxMat0
  w_57_0 : mxxMat0
  w_58_0 : mxxMat0
  w_61_0 : mxxMat0
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev parallel_generatedRoot_114.constraints_6 (w_24_2 : mxxMat1) (w_55_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 44 88 w_64_0 ∧
  outputs = (w_55_0, (w_60_0, (w_61_0, (w_62_0, (w_63_0, (w_64_0, ()))))))

abbrev parallel_generatedRoot_114.constraints_5 (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_52_0 : mxxMat1) (w_55_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 44 88 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 0 44 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 0 44 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  parallel_generatedRoot_114.constraints_6 w_24_2 w_55_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_114.constraints_4 (i_0 : Nat) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 0 44 w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_51_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_52_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 44 w_53_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 44 88 w_56_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 44 88 w_57_0 ∧
  parallel_generatedRoot_114.constraints_5 w_24_1 w_24_2 w_52_0 w_55_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_114.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_26_0 : mxxMat1) (w_31_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_37_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_39_0 w_40_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_43_0 w_44_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_39_0 w_45_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_45_0 w_46_0 ∧
  MxxRuntime.concatColumns w_42_0 w_47_0 w_48_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_114_49 backend hashModel tape (path ++ [49, i]) params i_0 i (current, (w_31_0, (w_36_0, (w_48_0, ())))) next) (Int.toNat (88)) w_26_0 w_49_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_114.constraints_4 i_0 w_51_0 w_24_0 w_24_1 w_24_2 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_114.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_22_0 ∧
  MxxRuntime.concatColumns w_21_0 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_114_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_6_0, (w_11_0, (w_16_0, (w_20_0, (w_23_0, ())))))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, (w_0_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 0 44 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_34_0 w_35_0 ∧
  MxxRuntime.concatColumns w_33_0 w_35_0 w_36_0 ∧
  parallel_generatedRoot_114.constraints_3 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_51_0 w_24_0 w_24_1 w_24_2 w_26_0 w_31_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_114.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 0 44 w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 44 88 w_15_0 ∧
  MxxRuntime.concatColumns w_14_0 w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_19_0 ∧
  MxxRuntime.concatColumns w_18_0 w_19_0 w_20_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  parallel_generatedRoot_114.constraints_2 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_16_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_114.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_7_0 : mxxMat12) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat0) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 0 44 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 44 88 w_10_0 ∧
  MxxRuntime.concatColumns w_9_0 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_114.constraints_1 backend hashModel tape path params i_0 w_1_0 w_4_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_114.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_114.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 37 → mxxMat0 := inputs.1
  let w_4_0 : Fin 37 → mxxMat0 := inputs.2.1
  let w_7_0 : mxxMat12 := inputs.2.2.1
  let w_12_0 : mxxMat12 := inputs.2.2.2.1
  let w_17_0 : ByteArray := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_29_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_51_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat1 := (0 : mxxMat1)
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_38_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixSub w_44_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_53_0
  let w_55_0 : mxxMat0 := MxxRuntime.matrixAdd w_25_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixAdd w_57_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_56_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  parallel_generatedRoot_114.constraints_0 backend hashModel tape path params i_0 w_1_0 w_4_0 w_7_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

def parallel_generatedRoot_114 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_114.Witness,
    parallel_generatedRoot_114.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_115_7.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_115_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_115_7.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_115.Witness where
  w_3_0 : mxxMat0
  w_7_0 : mxxMat2
  w_9_0 : mxxMat0
  w_11_0 : mxxMat13
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_115.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_1_0 : Fin 15 → mxxMat0) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat2) (w_7_0 : mxxMat2) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_115_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  outputs = (w_13_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))))

abbrev sequential_generatedRoot_115.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_115.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_20_0 : mxxMat0 := inputs.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_5_0 : mxxMat2 := (0 : mxxMat2)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
  sequential_generatedRoot_115.constraints_0 backend hashModel tape path params i_0 w_19_0 w_20_0 w_21_0 w_22_0 w_1_0 w_6_0 w_15_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 outputs

def sequential_generatedRoot_115 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_115.Witness,
    sequential_generatedRoot_115.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_116_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_116_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_116_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_116_44.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_116_44 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_116_44.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_116_54.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_116_54 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_116_54.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_116.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_22_0 : mxxMat0
  w_24_0 : mxxMat0
  w_25_0 : mxxMat1
  w_27_0 : mxxMat0
  w_28_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat1
  w_44_0 : mxxMat1
  w_45_0 : mxxMat0
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_54_0 : mxxMat2
  w_56_0 : mxxMat0
  w_58_0 : mxxMat13
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev sequential_generatedRoot_116.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_44_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_116_54 backend hashModel tape (path ++ [54, i]) params i_0 i (current, (w_53_0, ())) next) (Int.toNat (128)) w_52_0 w_54_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_57_0 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 44 88 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_64_0 ∧
  outputs = (w_60_0, (w_68_0, (w_69_0, (w_70_0, (w_71_0, (w_72_0, ()))))))

abbrev sequential_generatedRoot_116.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_25_0 : mxxMat1) (w_31_0 : mxxMat1) (w_34_0 : Int) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_26_0) w_38_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_34_0 w_40_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  MxxRuntime.concatColumns w_37_0 w_42_0 w_43_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_116_44 backend hashModel tape (path ++ [44, i]) params i_0 i (current, (w_25_0, (w_31_0, (w_43_0, ())))) next) (Int.toNat (88)) w_20_0 w_44_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 0 44 w_45_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_46_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_116.constraints_4 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_53_0 w_18_0 w_44_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_116.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_23_0 w_24_0 ∧
  MxxRuntime.concatColumns w_22_0 w_24_0 w_25_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_26_0) w_27_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_26_0) w_29_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_26_0) w_32_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_34_0 w_35_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_116.constraints_3 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_5_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_25_0 w_31_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_116.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_116_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  sequential_generatedRoot_116.constraints_2 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_116.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_116.constraints_1 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_116.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_116.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_61_0 : mxxMat0 := inputs.2.1
  let w_69_0 : mxxMat0 := inputs.2.2.1
  let w_70_0 : mxxMat0 := inputs.2.2.2.1
  let w_71_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat0 := MxxRuntime.matrixSub w_33_0 w_36_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_39_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_45_0 w_48_0
  let w_50_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_49_0
  let w_51_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_52_0 : mxxMat2 := (0 : mxxMat2)
  let w_54_0 := witness.w_54_0
  let w_55_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_51_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_55_0 w_56_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_50_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_59_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_63_0 w_64_0
  let w_66_0 : mxxMat0 := MxxRuntime.matrixAdd w_62_0 w_65_0
  let w_67_0 : mxxMat0 := MxxRuntime.matrixMul w_66_0 w_58_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_61_0 w_67_0
  sequential_generatedRoot_116.constraints_0 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

def sequential_generatedRoot_116 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_116.Witness,
    sequential_generatedRoot_116.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_118_34.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_118_34 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_118_34.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_118.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_18_0 : mxxMat0
  w_20_0 : mxxMat0
  w_21_0 : mxxMat1
  w_22_0 : mxxMat2
  w_23_0 : mxxMat2
  w_24_0 : mxxMat8
  w_27_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat1
  w_31_0 : mxxMat2
  w_32_0 : mxxMat2
  w_33_0 : mxxMat8
  w_34_0 : mxxMat1
  w_34_1 : mxxMat0
  w_34_2 : mxxMat0
  w_35_0 : mxxMat15

abbrev sequential_generatedRoot_118.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_24_0 : mxxMat8) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_31_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_32_0 w_33_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_118_34 backend hashModel tape (path ++ [34, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_24_0, (w_33_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_34_0, (w_34_1, (w_34_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_34_0 w_35_0 ∧
  outputs = w_36_0

abbrev sequential_generatedRoot_118.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_22_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_26_0 w_27_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_28_0 w_26_0 w_29_0 ∧
  MxxRuntime.concatColumns w_27_0 w_29_0 w_30_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  sequential_generatedRoot_118.constraints_2 backend hashModel tape path params i_0 w_4_0 w_11_0 w_2_0 w_3_0 w_10_0 w_15_0 w_24_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_118.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : Fin 15 → mxxMat0) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  sequential_generatedRoot_118.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_19_0 w_25_0 w_28_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_118.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) (witness : sequential_generatedRoot_118.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : Fin 15 → mxxMat0 := inputs.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_28_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_18_0 := witness.w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_27_0 := witness.w_27_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_34_1 := witness.w_34_1
  let w_34_2 := witness.w_34_2
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_35_0
  sequential_generatedRoot_118.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_19_0 w_25_0 w_28_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

def sequential_generatedRoot_118 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) : Prop :=
  ∃ witness : sequential_generatedRoot_118.Witness,
    sequential_generatedRoot_118.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_119_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_119_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_119_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_119_47.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_119_47 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_119_47.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_119_57.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_119_57 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 128
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_119_57.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_119.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_23_0 : mxxMat0
  w_24_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat0
  w_28_0 : mxxMat1
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat1
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_41_0 : mxxMat0
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_46_0 : mxxMat1
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_50_0 : mxxMat1
  w_51_0 : mxxMat0
  w_57_0 : mxxMat2
  w_59_0 : mxxMat0
  w_61_0 : mxxMat13
  w_65_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0

abbrev sequential_generatedRoot_119.constraints_5 (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_50_0 : mxxMat1) (w_63_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 44 88 w_67_0 ∧
  outputs = (w_63_0, (w_71_0, (w_72_0, (w_73_0, (w_74_0, (w_75_0, ()))))))

abbrev sequential_generatedRoot_119.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_47_0 : mxxMat1) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 0 44 w_51_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_119_57 backend hashModel tape (path ++ [57, i]) params i_0 i (current, (w_56_0, ())) next) (Int.toNat (128)) w_55_0 w_57_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_60_0 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_65_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_66_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_119.constraints_5 w_72_0 w_73_0 w_74_0 w_75_0 w_50_0 w_63_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_119.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_28_0 : mxxMat1) (w_34_0 : mxxMat1) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_29_0) w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_41_0 w_42_0 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_37_0 w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_43_0 w_44_0 ∧
  MxxRuntime.concatColumns w_40_0 w_45_0 w_46_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_119_47 backend hashModel tape (path ++ [47, i]) params i_0 i (current, (w_28_0, (w_34_0, (w_46_0, ())))) next) (Int.toNat (88)) w_20_0 w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_119.constraints_4 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_49_0 w_56_0 w_18_0 w_47_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_119.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_23_0 w_24_0 ∧
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_22_0 w_26_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_26_0 w_27_0 ∧
  MxxRuntime.concatColumns w_24_0 w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_29_0) w_30_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_29_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_32_0 w_33_0 ∧
  MxxRuntime.concatColumns w_31_0 w_33_0 w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_29_0) w_35_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_119.constraints_3 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_28_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_119.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_119_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  sequential_generatedRoot_119.constraints_2 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_119.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_119.constraints_1 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_119.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_119.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_64_0 : mxxMat0 := inputs.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.1
  let w_73_0 : mxxMat0 := inputs.2.2.2.1
  let w_74_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_75_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : mxxMat15 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_56_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15)])
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat0 := MxxRuntime.matrixSub w_36_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : mxxMat0 := MxxRuntime.matrixSub w_42_0 w_44_0
  let w_46_0 := witness.w_46_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat0 := MxxRuntime.matrixAdd w_48_0 w_51_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_52_0
  let w_54_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_55_0 : mxxMat2 := (0 : mxxMat2)
  let w_57_0 := witness.w_57_0
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_54_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_58_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat0 := MxxRuntime.matrixMul w_53_0 w_61_0
  let w_63_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_62_0
  let w_65_0 := witness.w_65_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat0 := MxxRuntime.matrixAdd w_65_0 w_68_0
  let w_70_0 : mxxMat0 := MxxRuntime.matrixMul w_69_0 w_61_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_64_0 w_70_0
  sequential_generatedRoot_119.constraints_0 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

def sequential_generatedRoot_119 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_119.Witness,
    sequential_generatedRoot_119.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_128.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_0))) ∧
  (87 + (-1 * (Int.ofNat i_0))) < (175 + (-1 * (Int.ofNat i_0))) ∧
  (175 + (-1 * (Int.ofNat i_0))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_0))) (175 + (-1 * (Int.ofNat i_0))) w_11_0 ∧
  outputs = w_13_0

def sequential_generatedRoot_128 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_generatedRoot_128.constraints_0 backend i_0 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_136_26.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_12_0 : Fin 15488 → mxxMat6) (w_23_0 : Fin 15488 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_17_0 : Int) (w_18_0 : mxxMat6) (w_19_0 : mxxMat0) (w_20_0 : mxxMat7) (w_21_0 : mxxMat7) (w_22_0 : mxxMat0) (w_24_0 : mxxMat6) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_17_0 w_18_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_19_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_20_0 ∧
  MxxRuntime.modulusSwitchRuns w_21_0 w_22_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_23_0 w_17_0 w_24_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_25_0 ∧
  MxxRuntime.modulusSwitchRuns w_26_0 w_27_0 ∧
  outputs = (w_11_0, (w_22_0, (w_27_0, ())))

def sequential_sequential_generatedRoot_136_26 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_13_0 : Int := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_14_0 : Int := 3872
    let w_15_0 : Int := (w_13_0 * w_14_0)
    let w_16_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88)))
    let w_17_0 : Int := (w_15_0 + w_16_0)
    ∃ (w_18_0 : mxxMat6),
      ∃ (w_19_0 : mxxMat0),
        ∃ (w_20_0 : mxxMat7),
          let w_21_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
          ∃ (w_22_0 : mxxMat0),
            ∃ (w_24_0 : mxxMat6),
              ∃ (w_25_0 : mxxMat7),
                let w_26_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
                ∃ (w_27_0 : mxxMat0),
                  sequential_sequential_generatedRoot_136_26.constraints_0 backend i_1 w_12_0 w_23_0 w_8_0 w_9_0 w_11_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_136.Witness where
  w_2_0 : mxxMat2
  w_10_0 : mxxMat6
  w_11_0 : mxxMat0
  w_12_0 : mxxMat7
  w_14_0 : mxxMat0
  w_16_0 : mxxMat6
  w_17_0 : mxxMat7
  w_19_0 : mxxMat0
  w_21_0 : mxxMat2
  w_22_0 : mxxMat8
  w_24_0 : mxxMat2
  w_25_0 : mxxMat8
  w_26_0 : mxxMat1
  w_26_1 : mxxMat0
  w_26_2 : mxxMat0
  w_27_0 : mxxMat1

abbrev sequential_generatedRoot_136.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_23_0 : mxxMat1) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_14_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_24_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_24_0 w_25_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_136_26 backend hashModel tape (path ++ [26, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_22_0, (w_25_0, (w_4_0, (w_5_0, (w_15_0, ())))))))) next) (Int.toNat (88)) (w_3_0, (w_14_0, (w_19_0, ()))) (w_26_0, (w_26_1, (w_26_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_26_0 w_27_0 ∧
  outputs = w_28_0

abbrev sequential_generatedRoot_136.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_1_0 : mxxMat2) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_9_0 : Int) (w_10_0 : mxxMat6) (w_11_0 : mxxMat0) (w_12_0 : mxxMat7) (w_13_0 : mxxMat7) (w_14_0 : mxxMat0) (w_16_0 : mxxMat6) (w_17_0 : mxxMat7) (w_18_0 : mxxMat7) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_11_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_12_0 ∧
  MxxRuntime.modulusSwitchRuns w_13_0 w_14_0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_9_0 w_16_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_17_0 ∧
  MxxRuntime.modulusSwitchRuns w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  sequential_generatedRoot_136.constraints_1 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_23_0 w_2_0 w_3_0 w_14_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_136.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) (witness : sequential_generatedRoot_136.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_4_0 : Fin 15488 → mxxMat6 := inputs.2.1
  let w_5_0 : Int := inputs.2.2.1
  let w_15_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat2 := (MxxRuntime.unitRow (0) : mxxMat2)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_6_0 : Int := 3872
  let w_7_0 : Int := (w_5_0 * w_6_0)
  let w_8_0 : Int := (88 * (Int.fmod ((Int.ofNat i_0)) (44)))
  let w_9_0 : Int := (w_7_0 + w_8_0)
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_10_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_26_1 := witness.w_26_1
  let w_26_2 := witness.w_26_2
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_27_0
  sequential_generatedRoot_136.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_20_0 w_23_0 w_1_0 w_2_0 w_3_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

def sequential_generatedRoot_136 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  ∃ witness : sequential_generatedRoot_136.Witness,
    sequential_generatedRoot_136.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_141.constraints_0 (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_generatedRoot_141 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_0)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_generatedRoot_141.constraints_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

abbrev sequential_sequential_generatedRoot_162_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_162_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_162_24.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_162.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_17_0 : mxxMat2
  w_18_0 : mxxMat2
  w_19_0 : mxxMat8
  w_21_0 : mxxMat2
  w_22_0 : mxxMat2
  w_23_0 : mxxMat8
  w_24_0 : mxxMat1
  w_24_1 : mxxMat0
  w_24_2 : mxxMat0
  w_25_0 : mxxMat10

abbrev sequential_generatedRoot_162.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_17_0 w_18_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_162_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_19_0, (w_23_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_24_0 w_25_0 ∧
  outputs = w_26_0

abbrev sequential_generatedRoot_162.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
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
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_162.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

abbrev sequential_generatedRoot_162.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) (witness : sequential_generatedRoot_162.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_25_0
  sequential_generatedRoot_162.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

def sequential_generatedRoot_162 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) : Prop :=
  ∃ witness : sequential_generatedRoot_162.Witness,
    sequential_generatedRoot_162.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_163_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_18_0 : mxxMat1) (w_23_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_163_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_14_0 : mxxMat1 := inputs.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.1
  let w_1_0 : mxxMat1 := inputs.2.2.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_14_0 w_17_0
              let w_21_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_8_0
              let w_22_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_11_0
              let w_23_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_22_0
              sequential_parallel_generatedRoot_163_24.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_163_49.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_parallel_generatedRoot_163_49 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_parallel_generatedRoot_163_49.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_163.Witness where
  w_3_0 : mxxMat0
  w_5_0 : mxxMat0
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat0
  w_15_0 : mxxMat0
  w_16_0 : mxxMat1
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat1
  w_24_0 : mxxMat1
  w_24_1 : mxxMat1
  w_24_2 : mxxMat1
  w_25_0 : mxxMat0
  w_28_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat1
  w_37_0 : mxxMat0
  w_38_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_45_0 : mxxMat0
  w_46_0 : mxxMat0
  w_48_0 : mxxMat1
  w_49_0 : mxxMat1
  w_50_0 : mxxMat0
  w_52_0 : mxxMat1
  w_53_0 : mxxMat0
  w_56_0 : mxxMat0
  w_57_0 : mxxMat0
  w_58_0 : mxxMat0
  w_61_0 : mxxMat0
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev parallel_generatedRoot_163.constraints_6 (w_24_2 : mxxMat1) (w_55_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 44 88 w_64_0 ∧
  outputs = (w_55_0, (w_60_0, (w_61_0, (w_62_0, (w_63_0, (w_64_0, ()))))))

abbrev parallel_generatedRoot_163.constraints_5 (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_52_0 : mxxMat1) (w_55_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 44 88 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 0 44 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 0 44 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  parallel_generatedRoot_163.constraints_6 w_24_2 w_55_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_163.constraints_4 (i_0 : Nat) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 0 44 w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_51_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_52_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 44 w_53_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 44 88 w_56_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 44 88 w_57_0 ∧
  parallel_generatedRoot_163.constraints_5 w_24_1 w_24_2 w_52_0 w_55_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_163.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_26_0 : mxxMat1) (w_31_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_37_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_39_0 w_40_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_43_0 w_44_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_39_0 w_45_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_45_0 w_46_0 ∧
  MxxRuntime.concatColumns w_42_0 w_47_0 w_48_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_163_49 backend hashModel tape (path ++ [49, i]) params i_0 i (current, (w_31_0, (w_36_0, (w_48_0, ())))) next) (Int.toNat (88)) w_26_0 w_49_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_163.constraints_4 i_0 w_51_0 w_24_0 w_24_1 w_24_2 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_163.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_22_0 ∧
  MxxRuntime.concatColumns w_21_0 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_163_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_6_0, (w_11_0, (w_16_0, (w_20_0, (w_23_0, ())))))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, (w_0_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 0 44 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_34_0 w_35_0 ∧
  MxxRuntime.concatColumns w_33_0 w_35_0 w_36_0 ∧
  parallel_generatedRoot_163.constraints_3 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_51_0 w_24_0 w_24_1 w_24_2 w_26_0 w_31_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_163.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 0 44 w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 44 88 w_15_0 ∧
  MxxRuntime.concatColumns w_14_0 w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_19_0 ∧
  MxxRuntime.concatColumns w_18_0 w_19_0 w_20_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  parallel_generatedRoot_163.constraints_2 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_16_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_163.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_7_0 : mxxMat12) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat0) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 0 44 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 44 88 w_10_0 ∧
  MxxRuntime.concatColumns w_9_0 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_163.constraints_1 backend hashModel tape path params i_0 w_1_0 w_4_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_163.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_163.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 37 → mxxMat0 := inputs.1
  let w_4_0 : Fin 37 → mxxMat0 := inputs.2.1
  let w_7_0 : mxxMat12 := inputs.2.2.1
  let w_12_0 : mxxMat12 := inputs.2.2.2.1
  let w_17_0 : ByteArray := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_29_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_51_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat1 := (0 : mxxMat1)
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_38_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixSub w_44_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_53_0
  let w_55_0 : mxxMat0 := MxxRuntime.matrixAdd w_25_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixAdd w_57_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_56_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  parallel_generatedRoot_163.constraints_0 backend hashModel tape path params i_0 w_1_0 w_4_0 w_7_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

def parallel_generatedRoot_163 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_163.Witness,
    parallel_generatedRoot_163.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_164_7.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_164_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_164_7.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_164.Witness where
  w_3_0 : mxxMat0
  w_7_0 : mxxMat2
  w_9_0 : mxxMat0
  w_11_0 : mxxMat13
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_164.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_1_0 : Fin 15 → mxxMat0) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat2) (w_7_0 : mxxMat2) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_164_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  outputs = (w_13_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))))

abbrev sequential_generatedRoot_164.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_164.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_20_0 : mxxMat0 := inputs.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_5_0 : mxxMat2 := (0 : mxxMat2)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
  sequential_generatedRoot_164.constraints_0 backend hashModel tape path params i_0 w_19_0 w_20_0 w_21_0 w_22_0 w_1_0 w_6_0 w_15_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 outputs

def sequential_generatedRoot_164 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_164.Witness,
    sequential_generatedRoot_164.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_165_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_165_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_165_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_165_44.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_165_44 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_165_44.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_165_54.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_165_54 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_165_54.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_165.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_22_0 : mxxMat0
  w_24_0 : mxxMat0
  w_25_0 : mxxMat1
  w_27_0 : mxxMat0
  w_28_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat1
  w_44_0 : mxxMat1
  w_45_0 : mxxMat0
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_54_0 : mxxMat2
  w_56_0 : mxxMat0
  w_58_0 : mxxMat13
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev sequential_generatedRoot_165.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_44_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_165_54 backend hashModel tape (path ++ [54, i]) params i_0 i (current, (w_53_0, ())) next) (Int.toNat (128)) w_52_0 w_54_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_57_0 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 44 88 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_64_0 ∧
  outputs = (w_60_0, (w_68_0, (w_69_0, (w_70_0, (w_71_0, (w_72_0, ()))))))

abbrev sequential_generatedRoot_165.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_25_0 : mxxMat1) (w_31_0 : mxxMat1) (w_34_0 : Int) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_26_0) w_38_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_34_0 w_40_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  MxxRuntime.concatColumns w_37_0 w_42_0 w_43_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_165_44 backend hashModel tape (path ++ [44, i]) params i_0 i (current, (w_25_0, (w_31_0, (w_43_0, ())))) next) (Int.toNat (88)) w_20_0 w_44_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 0 44 w_45_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_46_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_165.constraints_4 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_53_0 w_18_0 w_44_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_165.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_23_0 w_24_0 ∧
  MxxRuntime.concatColumns w_22_0 w_24_0 w_25_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_26_0) w_27_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_26_0) w_29_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_26_0) w_32_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_34_0 w_35_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_165.constraints_3 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_5_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_25_0 w_31_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_165.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_165_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  sequential_generatedRoot_165.constraints_2 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_165.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_165.constraints_1 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_165.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_165.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_61_0 : mxxMat0 := inputs.2.1
  let w_69_0 : mxxMat0 := inputs.2.2.1
  let w_70_0 : mxxMat0 := inputs.2.2.2.1
  let w_71_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat0 := MxxRuntime.matrixSub w_33_0 w_36_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_39_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_45_0 w_48_0
  let w_50_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_49_0
  let w_51_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_52_0 : mxxMat2 := (0 : mxxMat2)
  let w_54_0 := witness.w_54_0
  let w_55_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_51_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_55_0 w_56_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_50_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_59_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_63_0 w_64_0
  let w_66_0 : mxxMat0 := MxxRuntime.matrixAdd w_62_0 w_65_0
  let w_67_0 : mxxMat0 := MxxRuntime.matrixMul w_66_0 w_58_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_61_0 w_67_0
  sequential_generatedRoot_165.constraints_0 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

def sequential_generatedRoot_165 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_165.Witness,
    sequential_generatedRoot_165.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_167_34.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_167_34 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_167_34.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_167.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_18_0 : mxxMat0
  w_20_0 : mxxMat0
  w_21_0 : mxxMat1
  w_22_0 : mxxMat2
  w_23_0 : mxxMat2
  w_24_0 : mxxMat8
  w_27_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat1
  w_31_0 : mxxMat2
  w_32_0 : mxxMat2
  w_33_0 : mxxMat8
  w_34_0 : mxxMat1
  w_34_1 : mxxMat0
  w_34_2 : mxxMat0
  w_35_0 : mxxMat15

abbrev sequential_generatedRoot_167.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_24_0 : mxxMat8) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_31_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_32_0 w_33_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_167_34 backend hashModel tape (path ++ [34, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_24_0, (w_33_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_34_0, (w_34_1, (w_34_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_34_0 w_35_0 ∧
  outputs = w_36_0

abbrev sequential_generatedRoot_167.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_22_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_26_0 w_27_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_28_0 w_26_0 w_29_0 ∧
  MxxRuntime.concatColumns w_27_0 w_29_0 w_30_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  sequential_generatedRoot_167.constraints_2 backend hashModel tape path params i_0 w_4_0 w_11_0 w_2_0 w_3_0 w_10_0 w_15_0 w_24_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_167.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : Fin 15 → mxxMat0) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  sequential_generatedRoot_167.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_19_0 w_25_0 w_28_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_167.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) (witness : sequential_generatedRoot_167.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : Fin 15 → mxxMat0 := inputs.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_28_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_18_0 := witness.w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_27_0 := witness.w_27_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_34_1 := witness.w_34_1
  let w_34_2 := witness.w_34_2
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_35_0
  sequential_generatedRoot_167.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_19_0 w_25_0 w_28_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

def sequential_generatedRoot_167 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) : Prop :=
  ∃ witness : sequential_generatedRoot_167.Witness,
    sequential_generatedRoot_167.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_168_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_168_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_168_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_168_47.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_168_47 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_168_47.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_168_57.constraints_0 (i_0 : Nat) (w_2_0 : Fin 512 → Int) (w_5_0 : Int) (w_6_0 : Int) (w_15_0 : mxxMat2) (w_16_0 : mxxMat2) (w_18_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_5_0 ∧
  w_5_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_15_0 w_16_0 ∧
  outputs = w_18_0

def sequential_sequential_generatedRoot_168_57 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_2_0 : Fin 512 → Int := inputs.2.1
  let w_1_0 : mxxMat2 := (1 : mxxMat2)
  let w_3_0 : Int := 256
  let w_4_0 : Int := (Int.ofNat i_1)
  let w_5_0 : Int := (w_3_0 + w_4_0)
  ∃ (w_6_0 : Int),
    let w_7_0 : Int := 1
    let w_8_0 : Int := (w_6_0 * w_7_0)
    let w_9_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_1_0 w_8_0
    let w_10_0 : mxxMat2 := (1 : mxxMat2)
    let w_11_0 : Int := 131072
    let w_12_0 : Int := (w_11_0 - w_4_0)
    let w_13_0 : Int := 1
    let w_14_0 : Int := (w_12_0 * w_13_0)
    let w_15_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_10_0 w_14_0
    ∃ (w_16_0 : mxxMat2),
      let w_17_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_9_0 w_16_0
      let w_18_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_17_0
      sequential_sequential_generatedRoot_168_57.constraints_0 i_0 w_2_0 w_5_0 w_6_0 w_15_0 w_16_0 w_18_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_168.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_23_0 : mxxMat0
  w_24_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat0
  w_28_0 : mxxMat1
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat1
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_41_0 : mxxMat0
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_46_0 : mxxMat1
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_50_0 : mxxMat1
  w_51_0 : mxxMat0
  w_57_0 : mxxMat2
  w_59_0 : mxxMat0
  w_61_0 : mxxMat13
  w_65_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0

abbrev sequential_generatedRoot_168.constraints_5 (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_50_0 : mxxMat1) (w_63_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 44 88 w_67_0 ∧
  outputs = (w_63_0, (w_71_0, (w_72_0, (w_73_0, (w_74_0, (w_75_0, ()))))))

abbrev sequential_generatedRoot_168.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_47_0 : mxxMat1) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 0 44 w_51_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_168_57 backend hashModel tape (path ++ [57, i]) params i_0 i (current, (w_56_0, ())) next) (Int.toNat (128)) w_55_0 w_57_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_60_0 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_65_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_66_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_168.constraints_5 w_72_0 w_73_0 w_74_0 w_75_0 w_50_0 w_63_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_168.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_28_0 : mxxMat1) (w_34_0 : mxxMat1) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_29_0) w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_41_0 w_42_0 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_37_0 w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_43_0 w_44_0 ∧
  MxxRuntime.concatColumns w_40_0 w_45_0 w_46_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_168_47 backend hashModel tape (path ++ [47, i]) params i_0 i (current, (w_28_0, (w_34_0, (w_46_0, ())))) next) (Int.toNat (88)) w_20_0 w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_168.constraints_4 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_49_0 w_56_0 w_18_0 w_47_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_168.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_23_0 w_24_0 ∧
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_22_0 w_26_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_26_0 w_27_0 ∧
  MxxRuntime.concatColumns w_24_0 w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_29_0) w_30_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_29_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_32_0 w_33_0 ∧
  MxxRuntime.concatColumns w_31_0 w_33_0 w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_29_0) w_35_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_168.constraints_3 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_28_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_168.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_168_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  sequential_generatedRoot_168.constraints_2 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_168.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_168.constraints_1 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_168.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_168.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_64_0 : mxxMat0 := inputs.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.1
  let w_73_0 : mxxMat0 := inputs.2.2.2.1
  let w_74_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_75_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : mxxMat15 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_56_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15)])
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat0 := MxxRuntime.matrixSub w_36_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : mxxMat0 := MxxRuntime.matrixSub w_42_0 w_44_0
  let w_46_0 := witness.w_46_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat0 := MxxRuntime.matrixAdd w_48_0 w_51_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_52_0
  let w_54_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_55_0 : mxxMat2 := (0 : mxxMat2)
  let w_57_0 := witness.w_57_0
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_54_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_58_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat0 := MxxRuntime.matrixMul w_53_0 w_61_0
  let w_63_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_62_0
  let w_65_0 := witness.w_65_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat0 := MxxRuntime.matrixAdd w_65_0 w_68_0
  let w_70_0 : mxxMat0 := MxxRuntime.matrixMul w_69_0 w_61_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_64_0 w_70_0
  sequential_generatedRoot_168.constraints_0 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

def sequential_generatedRoot_168 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_168.Witness,
    sequential_generatedRoot_168.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_177.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_2_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_3_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (Int.ofNat i_0) ∧
  (Int.ofNat i_0) < (1 + (Int.ofNat i_0)) ∧
  (1 + (Int.ofNat i_0)) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_4_0 0 1 (Int.ofNat i_0) (1 + (Int.ofNat i_0)) w_5_0 ∧
  MxxRuntime.concatRows w_3_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_6_0 w_7_0 ∧
  MxxRuntime.compactCenteredRebaseRuns 32768 w_7_0 w_8_0 ∧
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_0))) ∧
  (87 + (-1 * (Int.ofNat i_0))) < (175 + (-1 * (Int.ofNat i_0))) ∧
  (175 + (-1 * (Int.ofNat i_0))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_10_0 0 1 (87 + (-1 * (Int.ofNat i_0))) (175 + (-1 * (Int.ofNat i_0))) w_11_0 ∧
  outputs = w_13_0

def sequential_generatedRoot_177 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_generatedRoot_177.constraints_0 backend i_0 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_185_26.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_12_0 : Fin 15488 → mxxMat6) (w_23_0 : Fin 15488 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_17_0 : Int) (w_18_0 : mxxMat6) (w_19_0 : mxxMat0) (w_20_0 : mxxMat7) (w_21_0 : mxxMat7) (w_22_0 : mxxMat0) (w_24_0 : mxxMat6) (w_25_0 : mxxMat7) (w_26_0 : mxxMat7) (w_27_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_17_0 w_18_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_19_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_20_0 ∧
  MxxRuntime.modulusSwitchRuns w_21_0 w_22_0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_23_0 w_17_0 w_24_0 ∧
  MxxRuntime.centeredRebaseRuns w_19_0 w_25_0 ∧
  MxxRuntime.modulusSwitchRuns w_26_0 w_27_0 ∧
  outputs = (w_11_0, (w_22_0, (w_27_0, ())))

def sequential_sequential_generatedRoot_185_26 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_13_0 : Int := inputs.2.2.2.2.2.2.1
  let w_23_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_14_0 : Int := 3872
    let w_15_0 : Int := (w_13_0 * w_14_0)
    let w_16_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88)))
    let w_17_0 : Int := (w_15_0 + w_16_0)
    ∃ (w_18_0 : mxxMat6),
      ∃ (w_19_0 : mxxMat0),
        ∃ (w_20_0 : mxxMat7),
          let w_21_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_18_0 w_20_0
          ∃ (w_22_0 : mxxMat0),
            ∃ (w_24_0 : mxxMat6),
              ∃ (w_25_0 : mxxMat7),
                let w_26_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
                ∃ (w_27_0 : mxxMat0),
                  sequential_sequential_generatedRoot_185_26.constraints_0 backend i_1 w_12_0 w_23_0 w_8_0 w_9_0 w_11_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_27_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_185.Witness where
  w_2_0 : mxxMat2
  w_10_0 : mxxMat6
  w_11_0 : mxxMat0
  w_12_0 : mxxMat7
  w_14_0 : mxxMat0
  w_16_0 : mxxMat6
  w_17_0 : mxxMat7
  w_19_0 : mxxMat0
  w_21_0 : mxxMat2
  w_22_0 : mxxMat8
  w_24_0 : mxxMat2
  w_25_0 : mxxMat8
  w_26_0 : mxxMat1
  w_26_1 : mxxMat0
  w_26_2 : mxxMat0
  w_27_0 : mxxMat1

abbrev sequential_generatedRoot_185.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_23_0 : mxxMat1) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_14_0 : mxxMat0) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_21_0 w_22_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_23_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_24_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_24_0 w_25_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_185_26 backend hashModel tape (path ++ [26, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_22_0, (w_25_0, (w_4_0, (w_5_0, (w_15_0, ())))))))) next) (Int.toNat (88)) (w_3_0, (w_14_0, (w_19_0, ()))) (w_26_0, (w_26_1, (w_26_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_26_0 w_27_0 ∧
  outputs = w_28_0

abbrev sequential_generatedRoot_185.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 15488 → mxxMat6) (w_5_0 : Int) (w_15_0 : Fin 15488 → mxxMat6) (w_20_0 : mxxMat1) (w_23_0 : mxxMat1) (w_1_0 : mxxMat2) (w_2_0 : mxxMat2) (w_3_0 : mxxMat1) (w_9_0 : Int) (w_10_0 : mxxMat6) (w_11_0 : mxxMat0) (w_12_0 : mxxMat7) (w_13_0 : mxxMat7) (w_14_0 : mxxMat0) (w_16_0 : mxxMat6) (w_17_0 : mxxMat7) (w_18_0 : mxxMat7) (w_19_0 : mxxMat0) (w_21_0 : mxxMat2) (w_22_0 : mxxMat8) (w_24_0 : mxxMat2) (w_25_0 : mxxMat8) (w_26_0 : mxxMat1) (w_26_1 : mxxMat0) (w_26_2 : mxxMat0) (w_27_0 : mxxMat1) (w_28_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_9_0 w_10_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_11_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_12_0 ∧
  MxxRuntime.modulusSwitchRuns w_13_0 w_14_0 ∧
  0 ≤ w_9_0 ∧
  w_9_0 < 15488 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_9_0 w_16_0 ∧
  MxxRuntime.centeredRebaseRuns w_11_0 w_17_0 ∧
  MxxRuntime.modulusSwitchRuns w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  sequential_generatedRoot_185.constraints_1 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_23_0 w_2_0 w_3_0 w_14_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

abbrev sequential_generatedRoot_185.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) (witness : sequential_generatedRoot_185.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_4_0 : Fin 15488 → mxxMat6 := inputs.2.1
  let w_5_0 : Int := inputs.2.2.1
  let w_15_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_23_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat2 := (MxxRuntime.unitRow (0) : mxxMat2)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_6_0 : Int := 3872
  let w_7_0 : Int := (w_5_0 * w_6_0)
  let w_8_0 : Int := (88 * (Int.fmod ((Int.ofNat i_0)) (44)))
  let w_9_0 : Int := (w_7_0 + w_8_0)
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_10_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_16_0 w_17_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_26_0 := witness.w_26_0
  let w_26_1 := witness.w_26_1
  let w_26_2 := witness.w_26_2
  let w_27_0 := witness.w_27_0
  let w_28_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_27_0
  sequential_generatedRoot_185.constraints_0 backend hashModel tape path params i_0 w_4_0 w_5_0 w_15_0 w_20_0 w_23_0 w_1_0 w_2_0 w_3_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_16_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_24_0 w_25_0 w_26_0 w_26_1 w_26_2 w_27_0 w_28_0 outputs

def sequential_generatedRoot_185 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat1 × (Fin 15488 → mxxMat6) × Int × (Fin 15488 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  ∃ witness : sequential_generatedRoot_185.Witness,
    sequential_generatedRoot_185.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_generatedRoot_190.constraints_0 (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat2) (w_13_0 : mxxMat2) (w_15_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_generatedRoot_190 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_0)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat2 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat2 := (1 : mxxMat2)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat2),
      let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_generatedRoot_190.constraints_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

abbrev sequential_sequential_generatedRoot_211_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_211_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_211_24.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_211.Witness where
  w_2_0 : mxxMat9
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_17_0 : mxxMat2
  w_18_0 : mxxMat2
  w_19_0 : mxxMat8
  w_21_0 : mxxMat2
  w_22_0 : mxxMat2
  w_23_0 : mxxMat8
  w_24_0 : mxxMat1
  w_24_1 : mxxMat0
  w_24_2 : mxxMat0
  w_25_0 : mxxMat10

abbrev sequential_generatedRoot_211.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_17_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_17_0 w_18_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_18_0 w_19_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_20_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_21_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_211_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_19_0, (w_23_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_24_0 w_25_0 ∧
  outputs = w_26_0

abbrev sequential_generatedRoot_211.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_1_0 : mxxMat11) (w_2_0 : mxxMat9) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : mxxMat2) (w_18_0 : mxxMat2) (w_19_0 : mxxMat8) (w_21_0 : mxxMat2) (w_22_0 : mxxMat2) (w_23_0 : mxxMat8) (w_24_0 : mxxMat1) (w_24_1 : mxxMat0) (w_24_2 : mxxMat0) (w_25_0 : mxxMat10) (w_26_0 : mxxMat10) (outputs : mxxMat10) : Prop :=
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
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 22 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  sequential_generatedRoot_211.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

abbrev sequential_generatedRoot_211.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) (witness : sequential_generatedRoot_211.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat10 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : mxxMat1 := inputs.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_1_0 : mxxMat11 := (MxxRuntime.unitRow (21) : mxxMat11)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat10 := MxxRuntime.matrixAdd w_0_0 w_25_0
  sequential_generatedRoot_211.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_20_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 outputs

def sequential_generatedRoot_211 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat10 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat10) : Prop :=
  ∃ witness : sequential_generatedRoot_211.Witness,
    sequential_generatedRoot_211.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_parallel_generatedRoot_212_24.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_18_0 : mxxMat1) (w_23_0 : mxxMat1) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
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
  outputs = (w_13_0, (w_18_0, (w_23_0, ())))

def sequential_parallel_generatedRoot_212_24 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1 × mxxMat1 × mxxMat1 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_14_0 : mxxMat1 := inputs.2.1
  let w_19_0 : mxxMat1 := inputs.2.2.1
  let w_1_0 : mxxMat1 := inputs.2.2.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.2.2.1
  let w_15_0 : mxxMat1 := inputs.2.2.2.2.2.2.1
  let w_20_0 : mxxMat1 := inputs.2.2.2.2.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              let w_16_0 : mxxMat2 := MxxRuntime.matrixMul w_15_0 w_8_0
              let w_17_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_16_0 w_11_0
              let w_18_0 : mxxMat1 := MxxRuntime.matrixAdd w_14_0 w_17_0
              let w_21_0 : mxxMat2 := MxxRuntime.matrixMul w_20_0 w_8_0
              let w_22_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_21_0 w_11_0
              let w_23_0 : mxxMat1 := MxxRuntime.matrixAdd w_19_0 w_22_0
              sequential_parallel_generatedRoot_212_24.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 w_18_0 w_23_0 outputs

abbrev sequential_parallel_generatedRoot_212_49.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_parallel_generatedRoot_212_49 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_parallel_generatedRoot_212_49.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure parallel_generatedRoot_212.Witness where
  w_3_0 : mxxMat0
  w_5_0 : mxxMat0
  w_6_0 : mxxMat1
  w_8_0 : mxxMat1
  w_9_0 : mxxMat0
  w_10_0 : mxxMat0
  w_11_0 : mxxMat1
  w_13_0 : mxxMat1
  w_14_0 : mxxMat0
  w_15_0 : mxxMat0
  w_16_0 : mxxMat1
  w_18_0 : mxxMat0
  w_19_0 : mxxMat0
  w_20_0 : mxxMat1
  w_21_0 : mxxMat0
  w_22_0 : mxxMat0
  w_23_0 : mxxMat1
  w_24_0 : mxxMat1
  w_24_1 : mxxMat1
  w_24_2 : mxxMat1
  w_25_0 : mxxMat0
  w_28_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat1
  w_37_0 : mxxMat0
  w_38_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_45_0 : mxxMat0
  w_46_0 : mxxMat0
  w_48_0 : mxxMat1
  w_49_0 : mxxMat1
  w_50_0 : mxxMat0
  w_52_0 : mxxMat1
  w_53_0 : mxxMat0
  w_56_0 : mxxMat0
  w_57_0 : mxxMat0
  w_58_0 : mxxMat0
  w_61_0 : mxxMat0
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev parallel_generatedRoot_212.constraints_6 (w_24_2 : mxxMat1) (w_55_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 44 88 w_64_0 ∧
  outputs = (w_55_0, (w_60_0, (w_61_0, (w_62_0, (w_63_0, (w_64_0, ()))))))

abbrev parallel_generatedRoot_212.constraints_5 (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_52_0 : mxxMat1) (w_55_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 44 88 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 0 44 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_1 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_2 0 1 0 44 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  parallel_generatedRoot_212.constraints_6 w_24_2 w_55_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_212.constraints_4 (i_0 : Nat) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 0 44 w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_51_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_52_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_52_0 0 1 0 44 w_53_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 44 88 w_56_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 44 88 w_57_0 ∧
  parallel_generatedRoot_212.constraints_5 w_24_1 w_24_2 w_52_0 w_55_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_212.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_51_0 : mxxMat10) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_26_0 : mxxMat1) (w_31_0 : mxxMat1) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_37_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_39_0 w_40_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_43_0 w_44_0 ∧
  0 ≤ w_39_0 ∧
  w_39_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_39_0 w_45_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_45_0 w_46_0 ∧
  MxxRuntime.concatColumns w_42_0 w_47_0 w_48_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_parallel_generatedRoot_212_49 backend hashModel tape (path ++ [49, i]) params i_0 i (current, (w_31_0, (w_36_0, (w_48_0, ())))) next) (Int.toNat (88)) w_26_0 w_49_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  parallel_generatedRoot_212.constraints_4 i_0 w_51_0 w_24_0 w_24_1 w_24_2 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_212.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_16_0 : mxxMat1) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_17_0) w_22_0 ∧
  MxxRuntime.concatColumns w_21_0 w_22_0 w_23_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat1 × mxxMat1 × Unit) => sequential_parallel_generatedRoot_212_24 backend hashModel tape (path ++ [24, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_6_0, (w_11_0, (w_16_0, (w_20_0, (w_23_0, ())))))))) next) (Int.toNat (88)) (w_0_0, (w_0_0, (w_0_0, ()))) (w_24_0, (w_24_1, (w_24_2, ()))) ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 0 44 w_25_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_34_0 w_35_0 ∧
  MxxRuntime.concatColumns w_33_0 w_35_0 w_36_0 ∧
  parallel_generatedRoot_212.constraints_3 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_51_0 w_24_0 w_24_1 w_24_2 w_26_0 w_31_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_212.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_6_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_13_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 0 44 w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 44 88 w_15_0 ∧
  MxxRuntime.concatColumns w_14_0 w_15_0 w_16_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_17_0) w_18_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_17_0) w_19_0 ∧
  MxxRuntime.concatColumns w_18_0 w_19_0 w_20_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_17_0) w_21_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  parallel_generatedRoot_212.constraints_2 backend hashModel tape path params i_0 w_1_0 w_4_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_16_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_212.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_1_0 : Fin 37 → mxxMat0) (w_4_0 : Fin 37 → mxxMat0) (w_7_0 : mxxMat12) (w_12_0 : mxxMat12) (w_17_0 : ByteArray) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_51_0 : mxxMat10) (w_0_0 : mxxMat1) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat0) (w_6_0 : mxxMat1) (w_8_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat0) (w_15_0 : mxxMat0) (w_16_0 : mxxMat1) (w_18_0 : mxxMat0) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_24_1 : mxxMat1) (w_24_2 : mxxMat1) (w_25_0 : mxxMat0) (w_26_0 : mxxMat1) (w_28_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat0) (w_35_0 : mxxMat0) (w_36_0 : mxxMat1) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : Int) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_47_0 : mxxMat0) (w_48_0 : mxxMat1) (w_49_0 : mxxMat1) (w_50_0 : mxxMat0) (w_52_0 : mxxMat1) (w_53_0 : mxxMat0) (w_55_0 : mxxMat0) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_7_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_8_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 0 44 w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 44 88 w_10_0 ∧
  MxxRuntime.concatColumns w_9_0 w_10_0 w_11_0 ∧
  0 ≤ 0 ∧
  parallel_generatedRoot_212.constraints_1 backend hashModel tape path params i_0 w_1_0 w_4_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_6_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

abbrev parallel_generatedRoot_212.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : parallel_generatedRoot_212.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_1_0 : Fin 37 → mxxMat0 := inputs.1
  let w_4_0 : Fin 37 → mxxMat0 := inputs.2.1
  let w_7_0 : mxxMat12 := inputs.2.2.1
  let w_12_0 : mxxMat12 := inputs.2.2.2.1
  let w_17_0 : ByteArray := inputs.2.2.2.2.1
  let w_27_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_29_0 : mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_51_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.1
  let w_0_0 : mxxMat1 := (0 : mxxMat1)
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_3_0 := witness.w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_8_0 := witness.w_8_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_24_1 := witness.w_24_1
  let w_24_2 := witness.w_24_2
  let w_25_0 := witness.w_25_0
  let w_26_0 : mxxMat1 := (0 : mxxMat1)
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 := witness.w_37_0
  let w_38_0 := witness.w_38_0
  let w_39_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_38_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 : mxxMat0 := MxxRuntime.matrixSub w_44_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 : mxxMat0 := MxxRuntime.matrixAdd w_50_0 w_53_0
  let w_55_0 : mxxMat0 := MxxRuntime.matrixAdd w_25_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixAdd w_57_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_56_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  parallel_generatedRoot_212.constraints_0 backend hashModel tape path params i_0 w_1_0 w_4_0 w_7_0 w_12_0 w_17_0 w_27_0 w_29_0 w_51_0 w_0_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_24_1 w_24_2 w_25_0 w_26_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_52_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 outputs

def parallel_generatedRoot_212 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × ByteArray × mxxMat0 × mxxMat0 × mxxMat10 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : parallel_generatedRoot_212.Witness,
    parallel_generatedRoot_212.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_213_7.constraints_0 (i_0 : Nat) (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat2) (w_13_0 : mxxMat2) (w_15_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_sequential_generatedRoot_213_7 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat2 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat2 := (1 : mxxMat2)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat2),
      let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_sequential_generatedRoot_213_7.constraints_0 i_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_213.Witness where
  w_3_0 : mxxMat0
  w_7_0 : mxxMat2
  w_9_0 : mxxMat0
  w_11_0 : mxxMat13
  w_16_0 : mxxMat0

abbrev sequential_generatedRoot_213.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_19_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat0) (w_1_0 : Fin 15 → mxxMat0) (w_6_0 : Fin 512 → Int) (w_15_0 : Fin 15 → mxxMat0) (w_2_0 : Int) (w_3_0 : mxxMat0) (w_5_0 : mxxMat2) (w_7_0 : mxxMat2) (w_9_0 : mxxMat0) (w_10_0 : mxxMat0) (w_11_0 : mxxMat13) (w_13_0 : mxxMat0) (w_16_0 : mxxMat0) (w_18_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_213_7 backend hashModel tape (path ++ [7, i]) params i_0 i (current, (w_6_0, ())) next) (Int.toNat (128)) w_5_0 w_7_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_9_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_10_0 w_11_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_2_0 w_16_0 ∧
  outputs = (w_13_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, ()))))))

abbrev sequential_generatedRoot_213.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_213.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_14_0 : mxxMat0 := inputs.2.1
  let w_19_0 : mxxMat0 := inputs.2.2.1
  let w_20_0 : mxxMat0 := inputs.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_22_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_1_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_6_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.1
  let w_15_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_2_0 : Int := (Int.ofNat i_0)
  let w_3_0 := witness.w_3_0
  let w_4_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_5_0 : mxxMat2 := (0 : mxxMat2)
  let w_7_0 := witness.w_7_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_4_0 w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : mxxMat0 := MxxRuntime.matrixMul w_3_0 w_11_0
  let w_13_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_12_0
  let w_16_0 := witness.w_16_0
  let w_17_0 : mxxMat0 := MxxRuntime.matrixMul w_16_0 w_11_0
  let w_18_0 : mxxMat0 := MxxRuntime.matrixAdd w_14_0 w_17_0
  sequential_generatedRoot_213.constraints_0 backend hashModel tape path params i_0 w_19_0 w_20_0 w_21_0 w_22_0 w_1_0 w_6_0 w_15_0 w_2_0 w_3_0 w_5_0 w_7_0 w_9_0 w_10_0 w_11_0 w_13_0 w_16_0 w_18_0 outputs

def sequential_generatedRoot_213 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 15 → mxxMat0) × (Fin 512 → Int) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_213.Witness,
    sequential_generatedRoot_213.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_214_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_214_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_214_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_214_44.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_214_44 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_214_44.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_214_54.constraints_0 (i_0 : Nat) (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat2) (w_13_0 : mxxMat2) (w_15_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_sequential_generatedRoot_214_54 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat2 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat2 := (1 : mxxMat2)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat2),
      let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_sequential_generatedRoot_214_54.constraints_0 i_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_214.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_22_0 : mxxMat0
  w_24_0 : mxxMat0
  w_25_0 : mxxMat1
  w_27_0 : mxxMat0
  w_28_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat0
  w_31_0 : mxxMat1
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_40_0 : mxxMat0
  w_41_0 : mxxMat0
  w_43_0 : mxxMat1
  w_44_0 : mxxMat1
  w_45_0 : mxxMat0
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_54_0 : mxxMat2
  w_56_0 : mxxMat0
  w_58_0 : mxxMat13
  w_62_0 : mxxMat0
  w_63_0 : mxxMat0
  w_64_0 : mxxMat0

abbrev sequential_generatedRoot_214.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_44_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_214_54 backend hashModel tape (path ++ [54, i]) params i_0 i (current, (w_53_0, ())) next) (Int.toNat (128)) w_52_0 w_54_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_56_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_57_0 w_58_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_62_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 44 88 w_63_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_64_0 ∧
  outputs = (w_60_0, (w_68_0, (w_69_0, (w_70_0, (w_71_0, (w_72_0, ()))))))

abbrev sequential_generatedRoot_214.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_25_0 : mxxMat1) (w_31_0 : mxxMat1) (w_34_0 : Int) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_26_0) w_38_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_34_0 w_40_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_40_0 w_41_0 ∧
  MxxRuntime.concatColumns w_37_0 w_42_0 w_43_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_214_44 backend hashModel tape (path ++ [44, i]) params i_0 i (current, (w_25_0, (w_31_0, (w_43_0, ())))) next) (Int.toNat (88)) w_20_0 w_44_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_44_0 0 1 0 44 w_45_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 1936 ∧
  MxxRuntime.sliceMatrix w_46_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  sequential_generatedRoot_214.constraints_4 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_53_0 w_18_0 w_44_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_214.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_21_0 w_22_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_23_0 w_24_0 ∧
  MxxRuntime.concatColumns w_22_0 w_24_0 w_25_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_26_0) w_27_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_27_0 w_28_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_26_0) w_29_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_29_0 w_30_0 ∧
  MxxRuntime.concatColumns w_28_0 w_30_0 w_31_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(130049), (129025), (128001), (126977), (125953), (124929), (123905), (122881), (121857), (120833), (119809), (118785), (117761), (116737), (115713), (114689), (98305), (81921), (65537), (49153), (32769), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_26_0) w_32_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_32_0 w_33_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_34_0 ∧
  w_34_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_34_0 w_35_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(1025), (2049), (3073), (4097), (5121), (6145), (7169), (8193), (9217), (10241), (11265), (12289), (13313), (14337), (15361), (16385), (32769), (49153), (65537), (81921), (98305), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_214.constraints_3 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_5_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_25_0 w_31_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_214.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_214_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  sequential_generatedRoot_214.constraints_2 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_18_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_214.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_69_0 : mxxMat0) (w_70_0 : mxxMat0) (w_71_0 : mxxMat0) (w_72_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : mxxMat0) (w_23_0 : mxxMat0) (w_26_0 : ByteArray) (w_46_0 : mxxMat10) (w_53_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : mxxMat0) (w_24_0 : mxxMat0) (w_25_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat0) (w_31_0 : mxxMat1) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : Int) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : mxxMat0) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat0) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_52_0 : mxxMat2) (w_54_0 : mxxMat2) (w_56_0 : mxxMat0) (w_57_0 : mxxMat0) (w_58_0 : mxxMat13) (w_60_0 : mxxMat0) (w_62_0 : mxxMat0) (w_63_0 : mxxMat0) (w_64_0 : mxxMat0) (w_68_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((15 + (Int.ofNat i_0))) ∧
  ((15 + (Int.ofNat i_0))) < 22 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1320 + (88 * (Int.ofNat i_0))) ∧
  (1320 + (88 * (Int.ofNat i_0))) < (1408 + (88 * (Int.ofNat i_0))) ∧
  (1408 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1320 + (88 * (Int.ofNat i_0))) (1408 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_214.constraints_1 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

abbrev sequential_generatedRoot_214.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_214.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_61_0 : mxxMat0 := inputs.2.1
  let w_69_0 : mxxMat0 := inputs.2.2.1
  let w_70_0 : mxxMat0 := inputs.2.2.2.1
  let w_71_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_23_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_46_0 : mxxMat10 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (15), (16), (17), (18), (19), (20), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 := witness.w_22_0
  let w_24_0 := witness.w_24_0
  let w_25_0 := witness.w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : Int := (MxxIR.selectInt ((15 + (Int.ofNat i_0))) [(36), (35), (34), (33), (32), (31), (30), (29), (28), (27), (26), (25), (24), (23), (22), (21), (20), (19), (18), (17), (16), (15)])
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : mxxMat0 := MxxRuntime.matrixSub w_33_0 w_36_0
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 := witness.w_40_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixSub w_39_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat0 := MxxRuntime.matrixAdd w_45_0 w_48_0
  let w_50_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_49_0
  let w_51_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_52_0 : mxxMat2 := (0 : mxxMat2)
  let w_54_0 := witness.w_54_0
  let w_55_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_51_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_55_0 w_56_0
  let w_58_0 := witness.w_58_0
  let w_59_0 : mxxMat0 := MxxRuntime.matrixMul w_50_0 w_58_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_59_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 : mxxMat0 := MxxRuntime.matrixAdd w_63_0 w_64_0
  let w_66_0 : mxxMat0 := MxxRuntime.matrixAdd w_62_0 w_65_0
  let w_67_0 : mxxMat0 := MxxRuntime.matrixMul w_66_0 w_58_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_61_0 w_67_0
  sequential_generatedRoot_214.constraints_0 backend hashModel tape path params i_0 w_69_0 w_70_0 w_71_0 w_72_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_23_0 w_26_0 w_46_0 w_53_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_24_0 w_25_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_47_0 w_48_0 w_52_0 w_54_0 w_56_0 w_57_0 w_58_0 w_60_0 w_62_0 w_63_0 w_64_0 w_68_0 outputs

def sequential_generatedRoot_214 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × ByteArray × mxxMat10 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_214.Witness,
    sequential_generatedRoot_214.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_216_34.constraints_0 (backend : MxxRuntime.BackendContext) (i_0 : Nat) (i_1 : Nat) (w_12_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 85184 → mxxMat6) (w_8_0 : mxxMat5) (w_9_0 : mxxMat1) (w_11_0 : mxxMat1) (w_13_0 : Int) (w_14_0 : mxxMat6) (w_15_0 : mxxMat0) (w_16_0 : mxxMat7) (w_17_0 : mxxMat7) (w_18_0 : mxxMat0) (w_20_0 : mxxMat6) (w_21_0 : mxxMat7) (w_22_0 : mxxMat7) (w_23_0 : mxxMat0) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (87) ∧
  (87) < 175 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (87 + (-1 * (Int.ofNat i_1))) ∧
  (87 + (-1 * (Int.ofNat i_1))) < (175 + (-1 * (Int.ofNat i_1))) ∧
  (175 + (-1 * (Int.ofNat i_1))) ≤ 175 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (87 + (-1 * (Int.ofNat i_1))) (175 + (-1 * (Int.ofNat i_1))) w_9_0 ∧
  (44) ≠ 0 ∧
  (88) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_15_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_16_0 ∧
  MxxRuntime.modulusSwitchRuns w_17_0 w_18_0 ∧
  0 ≤ w_13_0 ∧
  w_13_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_13_0 w_20_0 ∧
  MxxRuntime.centeredRebaseRuns w_15_0 w_21_0 ∧
  MxxRuntime.modulusSwitchRuns w_22_0 w_23_0 ∧
  outputs = (w_11_0, (w_18_0, (w_23_0, ())))

def sequential_sequential_generatedRoot_216_34 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat0 × mxxMat0 × mxxMat8 × mxxMat8 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × Unit) (outputs : mxxMat1 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : mxxMat8 := inputs.2.2.2.1
  let w_5_0 : mxxMat8 := inputs.2.2.2.2.1
  let w_12_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.1
  let w_19_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.1
  let w_3_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_2_0
  let w_6_0 : mxxMat2 := MxxRuntime.matrixMul w_4_0 w_5_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixAdd w_3_0 w_6_0
  let w_8_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
  ∃ (w_9_0 : mxxMat1),
    let w_10_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_7_0 w_9_0
    let w_11_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_10_0
    let w_13_0 : Int := (((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (Int.fmod ((1 + (Int.ofNat i_1))) (88))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
    ∃ (w_14_0 : mxxMat6),
      ∃ (w_15_0 : mxxMat0),
        ∃ (w_16_0 : mxxMat7),
          let w_17_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_14_0 w_16_0
          ∃ (w_18_0 : mxxMat0),
            ∃ (w_20_0 : mxxMat6),
              ∃ (w_21_0 : mxxMat7),
                let w_22_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_20_0 w_21_0
                ∃ (w_23_0 : mxxMat0),
                  sequential_sequential_generatedRoot_216_34.constraints_0 backend i_0 i_1 w_12_0 w_19_0 w_8_0 w_9_0 w_11_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_216.Witness where
  w_2_0 : mxxMat14
  w_6_0 : mxxMat6
  w_7_0 : mxxMat0
  w_8_0 : mxxMat7
  w_10_0 : mxxMat0
  w_12_0 : mxxMat6
  w_13_0 : mxxMat7
  w_15_0 : mxxMat0
  w_18_0 : mxxMat0
  w_20_0 : mxxMat0
  w_21_0 : mxxMat1
  w_22_0 : mxxMat2
  w_23_0 : mxxMat2
  w_24_0 : mxxMat8
  w_27_0 : mxxMat0
  w_29_0 : mxxMat0
  w_30_0 : mxxMat1
  w_31_0 : mxxMat2
  w_32_0 : mxxMat2
  w_33_0 : mxxMat8
  w_34_0 : mxxMat1
  w_34_1 : mxxMat0
  w_34_2 : mxxMat0
  w_35_0 : mxxMat15

abbrev sequential_generatedRoot_216.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_24_0 : mxxMat8) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_30_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_31_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_31_0 w_32_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_32_0 w_33_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1 × mxxMat0 × mxxMat0 × Unit) => sequential_sequential_generatedRoot_216_34 backend hashModel tape (path ++ [34, i]) params i_0 i (current.1, (current.2.1, (current.2.2.1, (w_24_0, (w_33_0, (w_4_0, (w_11_0, ()))))))) next) (Int.toNat (88)) (w_3_0, (w_10_0, (w_15_0, ()))) (w_34_0, (w_34_1, (w_34_2, ()))) ∧
  MxxRuntime.tensorRuns w_2_0 w_34_0 w_35_0 ∧
  outputs = w_36_0

abbrev sequential_generatedRoot_216.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_10_0 : mxxMat0) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_19_0 w_17_0 w_20_0 ∧
  MxxRuntime.concatColumns w_18_0 w_20_0 w_21_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  (44) ≠ 0 ∧
  0 ≤ (44 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (44 + (Int.fmod ((Int.ofNat i_0)) (44))) < (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ∧
  (45 + (Int.fmod ((Int.ofNat i_0)) (44))) ≤ 88 ∧
  MxxRuntime.sliceMatrix w_21_0 0 1 (44 + (Int.fmod ((Int.ofNat i_0)) (44))) (45 + (Int.fmod ((Int.ofNat i_0)) (44))) w_22_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_22_0 w_23_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_23_0 w_24_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_26_0 w_27_0 ∧
  0 ≤ w_26_0 ∧
  w_26_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_28_0 w_26_0 w_29_0 ∧
  MxxRuntime.concatColumns w_27_0 w_29_0 w_30_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (44) ≠ 0 ∧
  sequential_generatedRoot_216.constraints_2 backend hashModel tape path params i_0 w_4_0 w_11_0 w_2_0 w_3_0 w_10_0 w_15_0 w_24_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_216.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_4_0 : Fin 85184 → mxxMat6) (w_11_0 : Fin 85184 → mxxMat6) (w_16_0 : Fin 15 → mxxMat0) (w_19_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_28_0 : Fin 15 → mxxMat0) (w_1_0 : mxxMat16) (w_2_0 : mxxMat14) (w_3_0 : mxxMat1) (w_5_0 : Int) (w_6_0 : mxxMat6) (w_7_0 : mxxMat0) (w_8_0 : mxxMat7) (w_9_0 : mxxMat7) (w_10_0 : mxxMat0) (w_12_0 : mxxMat6) (w_13_0 : mxxMat7) (w_14_0 : mxxMat7) (w_15_0 : mxxMat0) (w_17_0 : Int) (w_18_0 : mxxMat0) (w_20_0 : mxxMat0) (w_21_0 : mxxMat1) (w_22_0 : mxxMat2) (w_23_0 : mxxMat2) (w_24_0 : mxxMat8) (w_26_0 : Int) (w_27_0 : mxxMat0) (w_29_0 : mxxMat0) (w_30_0 : mxxMat1) (w_31_0 : mxxMat2) (w_32_0 : mxxMat2) (w_33_0 : mxxMat8) (w_34_0 : mxxMat1) (w_34_1 : mxxMat0) (w_34_2 : mxxMat0) (w_35_0 : mxxMat15) (w_36_0 : mxxMat15) (outputs : mxxMat15) : Prop :=
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
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_5_0 w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_7_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_8_0 ∧
  MxxRuntime.modulusSwitchRuns w_9_0 w_10_0 ∧
  0 ≤ w_5_0 ∧
  w_5_0 < 85184 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_5_0 w_12_0 ∧
  MxxRuntime.centeredRebaseRuns w_7_0 w_13_0 ∧
  MxxRuntime.modulusSwitchRuns w_14_0 w_15_0 ∧
  (44) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (44))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (44))) < 105 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_16_0 w_17_0 w_18_0 ∧
  sequential_generatedRoot_216.constraints_1 backend hashModel tape path params i_0 w_4_0 w_11_0 w_19_0 w_25_0 w_28_0 w_2_0 w_3_0 w_10_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

abbrev sequential_generatedRoot_216.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) (witness : sequential_generatedRoot_216.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat15 := inputs.1
  let w_4_0 : Fin 85184 → mxxMat6 := inputs.2.1
  let w_11_0 : Fin 85184 → mxxMat6 := inputs.2.2.1
  let w_16_0 : Fin 15 → mxxMat0 := inputs.2.2.2.1
  let w_19_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.1
  let w_28_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_1_0 : mxxMat16 := (MxxRuntime.unitRow (104) : mxxMat16)
  let w_2_0 := witness.w_2_0
  let w_3_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : Int := ((88 * (Int.fmod ((Int.ofNat i_0)) (44))) + (3872 * (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])))
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_9_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_6_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 : mxxMat7 := MxxRuntime.matrixMulScalarLeft w_12_0 w_13_0
  let w_15_0 := witness.w_15_0
  let w_17_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_18_0 := witness.w_18_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 : Int := (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (44))) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_27_0 := witness.w_27_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_34_1 := witness.w_34_1
  let w_34_2 := witness.w_34_2
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat15 := MxxRuntime.matrixAdd w_0_0 w_35_0
  sequential_generatedRoot_216.constraints_0 backend hashModel tape path params i_0 w_4_0 w_11_0 w_16_0 w_19_0 w_25_0 w_28_0 w_1_0 w_2_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_12_0 w_13_0 w_14_0 w_15_0 w_17_0 w_18_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_34_1 w_34_2 w_35_0 w_36_0 outputs

def sequential_generatedRoot_216 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat15 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × Unit) (outputs : mxxMat15) : Prop :=
  ∃ witness : sequential_generatedRoot_216.Witness,
    sequential_generatedRoot_216.body backend hashModel tape path params i_0 inputs outputs witness

abbrev sequential_sequential_generatedRoot_217_18.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_217_18 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_217_18.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_217_47.constraints_0 (backend : MxxRuntime.BackendContext) (i_1 : Nat) (w_2_0 : mxxMat1) (w_4_0 : mxxMat1) (w_3_0 : mxxMat2) (w_5_0 : mxxMat2) (w_6_0 : mxxMat3) (w_7_0 : mxxMat4) (w_8_0 : mxxMat4) (w_10_0 : mxxMat5) (w_11_0 : mxxMat1) (w_13_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
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
  outputs = w_13_0

def sequential_sequential_generatedRoot_217_47 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (_ : Nat) (i_1 : Nat) (inputs : mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat1 := inputs.1
  let w_1_0 : mxxMat1 := inputs.2.1
  let w_2_0 : mxxMat1 := inputs.2.2.1
  let w_4_0 : mxxMat1 := inputs.2.2.2.1
  ∃ (w_3_0 : mxxMat2),
    ∃ (w_5_0 : mxxMat2),
      ∃ (w_6_0 : mxxMat3),
        ∃ (w_7_0 : mxxMat4),
          ∃ (w_8_0 : mxxMat4),
            let w_9_0 : mxxMat2 := MxxRuntime.matrixMul w_1_0 w_8_0
            let w_10_0 : mxxMat5 := (MxxRuntime.unitRow (87) : mxxMat5)
            ∃ (w_11_0 : mxxMat1),
              let w_12_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_11_0
              let w_13_0 : mxxMat1 := MxxRuntime.matrixAdd w_0_0 w_12_0
              sequential_sequential_generatedRoot_217_47.constraints_0 backend i_1 w_2_0 w_4_0 w_3_0 w_5_0 w_6_0 w_7_0 w_8_0 w_10_0 w_11_0 w_13_0 outputs

abbrev sequential_sequential_generatedRoot_217_57.constraints_0 (i_0 : Nat) (w_1_0 : Fin 512 → Int) (w_4_0 : Int) (w_5_0 : Int) (w_12_0 : mxxMat2) (w_13_0 : mxxMat2) (w_15_0 : mxxMat2) (outputs : mxxMat2) : Prop :=
  0 ≤ w_4_0 ∧
  w_4_0 < 512 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_4_0 w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17409), (18433), (19457), (20481), (21505), (22529), (23553), (24577), (25601), (26625), (27649), (28673), (29697), (30721), (31745), (33793), (34817), (35841), (36865), (37889), (38913), (39937), (40961), (41985), (43009), (44033), (45057), (46081), (47105), (48129), (50177), (51201), (52225), (53249), (54273), (55297), (56321), (57345), (58369), (59393), (60417), (61441), (62465), (63489), (64513), (66561), (67585), (68609), (69633), (70657), (71681), (72705), (73729), (74753), (75777), (76801), (77825), (78849), (79873), (80897), (82945), (83969), (84993), (86017), (87041), (88065), (89089), (90113), (91137), (92161), (93185), (94209), (95233), (96257), (97281), (99329), (100353), (101377), (102401), (103425), (104449), (105473), (106497), (107521), (108545), (109569), (110593), (111617), (112641), (113665), (115713), (116737), (117761), (118785), (119809), (120833), (121857), (122881), (123905), (124929), (125953), (126977), (128001), (129025), (130049)])) w_12_0 w_13_0 ∧
  outputs = w_15_0

def sequential_sequential_generatedRoot_217_57 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (i_1 : Nat) (inputs : mxxMat2 × (Fin 512 → Int) × Unit) (outputs : mxxMat2) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat2 := inputs.1
  let w_1_0 : Fin 512 → Int := inputs.2.1
  let w_2_0 : Int := 384
  let w_3_0 : Int := (Int.ofNat i_1)
  let w_4_0 : Int := (w_2_0 + w_3_0)
  ∃ (w_5_0 : Int),
    let w_6_0 : mxxMat2 := MxxRuntime.liftInteger w_5_0
    let w_7_0 : mxxMat2 := (1 : mxxMat2)
    let w_8_0 : Int := 131072
    let w_9_0 : Int := (w_8_0 - w_3_0)
    let w_10_0 : Int := 1
    let w_11_0 : Int := (w_9_0 * w_10_0)
    let w_12_0 : mxxMat2 := MxxRuntime.multiplyMonomial w_7_0 w_11_0
    ∃ (w_13_0 : mxxMat2),
      let w_14_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_6_0 w_13_0
      let w_15_0 : mxxMat2 := MxxRuntime.matrixAdd w_0_0 w_14_0
      sequential_sequential_generatedRoot_217_57.constraints_0 i_0 w_1_0 w_4_0 w_5_0 w_12_0 w_13_0 w_15_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure sequential_generatedRoot_217.Witness where
  w_4_0 : mxxMat0
  w_6_0 : mxxMat0
  w_7_0 : mxxMat1
  w_9_0 : mxxMat1
  w_10_0 : mxxMat0
  w_11_0 : mxxMat0
  w_12_0 : mxxMat1
  w_14_0 : mxxMat1
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_18_0 : mxxMat1
  w_19_0 : mxxMat0
  w_23_0 : mxxMat0
  w_24_0 : mxxMat0
  w_26_0 : mxxMat0
  w_27_0 : mxxMat0
  w_28_0 : mxxMat1
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat0
  w_33_0 : mxxMat0
  w_34_0 : mxxMat1
  w_35_0 : mxxMat0
  w_36_0 : mxxMat0
  w_38_0 : mxxMat0
  w_39_0 : mxxMat0
  w_41_0 : mxxMat0
  w_42_0 : mxxMat0
  w_43_0 : mxxMat0
  w_44_0 : mxxMat0
  w_46_0 : mxxMat1
  w_47_0 : mxxMat1
  w_48_0 : mxxMat0
  w_50_0 : mxxMat1
  w_51_0 : mxxMat0
  w_57_0 : mxxMat2
  w_59_0 : mxxMat0
  w_61_0 : mxxMat13
  w_65_0 : mxxMat0
  w_66_0 : mxxMat0
  w_67_0 : mxxMat0

abbrev sequential_generatedRoot_217.constraints_5 (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_50_0 : mxxMat1) (w_63_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 44 88 w_67_0 ∧
  outputs = (w_63_0, (w_71_0, (w_72_0, (w_73_0, (w_74_0, (w_75_0, ()))))))

abbrev sequential_generatedRoot_217.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_47_0 : mxxMat1) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ (88 * (Int.ofNat i_0)) ∧
  (88 * (Int.ofNat i_0)) < (88 + (88 * (Int.ofNat i_0))) ∧
  (88 + (88 * (Int.ofNat i_0))) ≤ 9240 ∧
  MxxRuntime.sliceMatrix w_49_0 0 1 (88 * (Int.ofNat i_0)) (88 + (88 * (Int.ofNat i_0))) w_50_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_50_0 0 1 0 44 w_51_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_sequential_generatedRoot_217_57 backend hashModel tape (path ++ [57, i]) params i_0 i (current, (w_56_0, ())) next) (Int.toNat (128)) w_55_0 w_57_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_59_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_60_0 w_61_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 44 88 w_65_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 44 88 w_66_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_217.constraints_5 w_72_0 w_73_0 w_74_0 w_75_0 w_50_0 w_63_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_217.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_28_0 : mxxMat1) (w_34_0 : mxxMat1) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_37_0 w_38_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_38_0 w_39_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_29_0) w_41_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_41_0 w_42_0 ∧
  0 ≤ w_37_0 ∧
  w_37_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_37_0 w_43_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_43_0 w_44_0 ∧
  MxxRuntime.concatColumns w_40_0 w_45_0 w_46_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_217_47 backend hashModel tape (path ++ [47, i]) params i_0 i (current, (w_28_0, (w_34_0, (w_46_0, ())))) next) (Int.toNat (88)) w_20_0 w_47_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_47_0 0 1 0 44 w_48_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  sequential_generatedRoot_217.constraints_4 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_49_0 w_56_0 w_18_0 w_47_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_217.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_18_0 : mxxMat1) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_21_0 w_22_0 w_23_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_23_0 w_24_0 ∧
  0 ≤ w_22_0 ∧
  w_22_0 < 15 ∧
  MxxRuntime.familyGetDynamic w_25_0 w_22_0 w_26_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_26_0 w_27_0 ∧
  MxxRuntime.concatColumns w_24_0 w_27_0 w_28_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_29_0) w_30_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_30_0 w_31_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_29_0) w_32_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_32_0 w_33_0 ∧
  MxxRuntime.concatColumns w_31_0 w_33_0 w_34_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le ((MxxIR.selectInt ((Int.ofNat i_0)) [(114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385)])), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_29_0) w_35_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (16385), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (32769), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (49153), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (65537), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (81921), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (98305), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689), (114689)])) w_35_0 w_36_0 ∧
  sequential_generatedRoot_217.constraints_3 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_28_0 w_34_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_217.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_7_0 : mxxMat1) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_13_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_14_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 0 44 w_15_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_14_0 0 1 44 88 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_sequential_generatedRoot_217_18 backend hashModel tape (path ++ [18, i]) params i_0 i (current, (w_7_0, (w_12_0, (w_17_0, ())))) next) (Int.toNat (88)) w_1_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 0 44 w_19_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  sequential_generatedRoot_217.constraints_2 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_18_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_217.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (w_72_0 : mxxMat0) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat0) (w_2_0 : Fin 37 → mxxMat0) (w_5_0 : Fin 37 → mxxMat0) (w_8_0 : mxxMat12) (w_13_0 : mxxMat12) (w_21_0 : Fin 15 → mxxMat0) (w_25_0 : Fin 15 → mxxMat0) (w_29_0 : ByteArray) (w_49_0 : mxxMat15) (w_56_0 : Fin 512 → Int) (w_1_0 : mxxMat1) (w_3_0 : Int) (w_4_0 : mxxMat0) (w_6_0 : mxxMat0) (w_7_0 : mxxMat1) (w_9_0 : mxxMat1) (w_10_0 : mxxMat0) (w_11_0 : mxxMat0) (w_12_0 : mxxMat1) (w_14_0 : mxxMat1) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_18_0 : mxxMat1) (w_19_0 : mxxMat0) (w_20_0 : mxxMat1) (w_22_0 : Int) (w_23_0 : mxxMat0) (w_24_0 : mxxMat0) (w_26_0 : mxxMat0) (w_27_0 : mxxMat0) (w_28_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat0) (w_33_0 : mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_36_0 : mxxMat0) (w_37_0 : Int) (w_38_0 : mxxMat0) (w_39_0 : mxxMat0) (w_40_0 : mxxMat0) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat0) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat1) (w_47_0 : mxxMat1) (w_48_0 : mxxMat0) (w_50_0 : mxxMat1) (w_51_0 : mxxMat0) (w_55_0 : mxxMat2) (w_57_0 : mxxMat2) (w_59_0 : mxxMat0) (w_60_0 : mxxMat0) (w_61_0 : mxxMat13) (w_63_0 : mxxMat0) (w_65_0 : mxxMat0) (w_66_0 : mxxMat0) (w_67_0 : mxxMat0) (w_71_0 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 105 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_2_0 w_3_0 w_4_0 ∧
  0 ≤ w_3_0 ∧
  w_3_0 < 37 ∧
  MxxRuntime.familyGetDynamic w_5_0 w_3_0 w_6_0 ∧
  MxxRuntime.concatColumns w_4_0 w_6_0 w_7_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ (1936 + (88 * (Int.ofNat i_0))) ∧
  (1936 + (88 * (Int.ofNat i_0))) < (2024 + (88 * (Int.ofNat i_0))) ∧
  (2024 + (88 * (Int.ofNat i_0))) ≤ 11176 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (1936 + (88 * (Int.ofNat i_0))) (2024 + (88 * (Int.ofNat i_0))) w_9_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 0 44 w_10_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_9_0 0 1 44 88 w_11_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  sequential_generatedRoot_217.constraints_1 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_7_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

abbrev sequential_generatedRoot_217.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : sequential_generatedRoot_217.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_64_0 : mxxMat0 := inputs.2.1
  let w_72_0 : mxxMat0 := inputs.2.2.1
  let w_73_0 : mxxMat0 := inputs.2.2.2.1
  let w_74_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_75_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_2_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.1
  let w_5_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.1
  let w_8_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.1
  let w_13_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_21_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : ByteArray := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : mxxMat15 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_56_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_1_0 : mxxMat1 := (0 : mxxMat1)
  let w_3_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21)])
  let w_4_0 := witness.w_4_0
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_9_0 := witness.w_9_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 : mxxMat1 := (0 : mxxMat1)
  let w_22_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14), (0), (1), (2), (3), (4), (5), (6), (7), (8), (9), (10), (11), (12), (13), (14)])
  let w_23_0 := witness.w_23_0
  let w_24_0 := witness.w_24_0
  let w_26_0 := witness.w_26_0
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 := witness.w_34_0
  let w_35_0 := witness.w_35_0
  let w_36_0 := witness.w_36_0
  let w_37_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (21), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (20), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (19), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (18), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (17), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (16), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15), (15)])
  let w_38_0 := witness.w_38_0
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat0 := MxxRuntime.matrixSub w_36_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 : mxxMat0 := MxxRuntime.matrixSub w_42_0 w_44_0
  let w_46_0 := witness.w_46_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 : mxxMat0 := MxxRuntime.matrixAdd w_48_0 w_51_0
  let w_53_0 : mxxMat0 := MxxRuntime.matrixAdd w_19_0 w_52_0
  let w_54_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_55_0 : mxxMat2 := (0 : mxxMat2)
  let w_57_0 := witness.w_57_0
  let w_58_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_54_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_58_0 w_59_0
  let w_61_0 := witness.w_61_0
  let w_62_0 : mxxMat0 := MxxRuntime.matrixMul w_53_0 w_61_0
  let w_63_0 : mxxMat0 := MxxRuntime.matrixAdd w_0_0 w_62_0
  let w_65_0 := witness.w_65_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 : mxxMat0 := MxxRuntime.matrixAdd w_66_0 w_67_0
  let w_69_0 : mxxMat0 := MxxRuntime.matrixAdd w_65_0 w_68_0
  let w_70_0 : mxxMat0 := MxxRuntime.matrixMul w_69_0 w_61_0
  let w_71_0 : mxxMat0 := MxxRuntime.matrixAdd w_64_0 w_70_0
  sequential_generatedRoot_217.constraints_0 backend hashModel tape path params i_0 w_72_0 w_73_0 w_74_0 w_75_0 w_2_0 w_5_0 w_8_0 w_13_0 w_21_0 w_25_0 w_29_0 w_49_0 w_56_0 w_1_0 w_3_0 w_4_0 w_6_0 w_7_0 w_9_0 w_10_0 w_11_0 w_12_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_22_0 w_23_0 w_24_0 w_26_0 w_27_0 w_28_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_50_0 w_51_0 w_55_0 w_57_0 w_59_0 w_60_0 w_61_0 w_63_0 w_65_0 w_66_0 w_67_0 w_71_0 outputs

def sequential_generatedRoot_217 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 15 → mxxMat0) × (Fin 15 → mxxMat0) × ByteArray × mxxMat15 × (Fin 512 → Int) × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : sequential_generatedRoot_217.Witness,
    sequential_generatedRoot_217.body backend hashModel tape path params i_0 inputs outputs witness

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_10_0 : mxxMat1
  w_12_0 : Fin 4 → mxxMat0
  w_12_1 : Fin 4 → mxxMat0
  w_14_0 : Int
  w_15_0 : mxxMat0
  w_16_0 : mxxMat0
  w_17_0 : mxxMat1
  w_19_0 : mxxMat0
  w_21_0 : mxxMat0
  w_22_0 : mxxMat1
  w_23_0 : mxxMat1
  w_27_0 : mxxMat0
  w_28_0 : mxxMat0
  w_29_0 : mxxMat1
  w_30_0 : mxxMat0
  w_31_0 : mxxMat0
  w_32_0 : mxxMat1
  w_33_0 : mxxMat1
  w_35_0 : mxxMat0
  w_39_0 : mxxMat2
  w_41_0 : mxxMat0
  w_43_0 : mxxMat13
  w_45_0 : mxxMat0
  w_62_0 : mxxMat1
  w_63_0 : mxxMat1
  w_64_0 : mxxMat10
  w_65_0 : Fin 15 → mxxMat0
  w_65_1 : Fin 15 → mxxMat0
  w_65_2 : Fin 15 → mxxMat0
  w_65_3 : Fin 15 → mxxMat0
  w_65_4 : Fin 15 → mxxMat0
  w_65_5 : Fin 15 → mxxMat0
  w_66_0 : mxxMat0
  w_66_1 : mxxMat0
  w_66_2 : mxxMat0
  w_66_3 : mxxMat0
  w_66_4 : mxxMat0
  w_66_5 : mxxMat0
  w_67_0 : mxxMat0
  w_67_1 : mxxMat0
  w_67_2 : mxxMat0
  w_67_3 : mxxMat0
  w_67_4 : mxxMat0
  w_67_5 : mxxMat0
  w_69_0 : mxxMat15
  w_70_0 : mxxMat0
  w_70_1 : mxxMat0
  w_70_2 : mxxMat0
  w_70_3 : mxxMat0
  w_70_4 : mxxMat0
  w_70_5 : mxxMat0
  w_71_0 : mxxMat1
  w_72_0 : Int
  w_73_0 : mxxMat0
  w_74_0 : mxxMat0
  w_75_0 : mxxMat1
  w_76_0 : mxxMat0
  w_77_0 : mxxMat0
  w_78_0 : mxxMat1
  w_79_0 : mxxMat1
  w_83_0 : mxxMat1
  w_86_0 : mxxMat1
  w_87_0 : mxxMat1
  w_89_0 : mxxMat0
  w_92_0 : mxxMat2
  w_94_0 : mxxMat0
  w_96_0 : mxxMat13
  w_98_0 : mxxMat0
  w_111_0 : mxxMat1
  w_112_0 : mxxMat1
  w_113_0 : mxxMat10
  w_114_0 : Fin 15 → mxxMat0
  w_114_1 : Fin 15 → mxxMat0
  w_114_2 : Fin 15 → mxxMat0
  w_114_3 : Fin 15 → mxxMat0
  w_114_4 : Fin 15 → mxxMat0
  w_114_5 : Fin 15 → mxxMat0
  w_115_0 : mxxMat0
  w_115_1 : mxxMat0
  w_115_2 : mxxMat0
  w_115_3 : mxxMat0
  w_115_4 : mxxMat0
  w_115_5 : mxxMat0
  w_116_0 : mxxMat0
  w_116_1 : mxxMat0
  w_116_2 : mxxMat0
  w_116_3 : mxxMat0
  w_116_4 : mxxMat0
  w_116_5 : mxxMat0
  w_118_0 : mxxMat15
  w_119_0 : mxxMat0
  w_119_1 : mxxMat0
  w_119_2 : mxxMat0
  w_119_3 : mxxMat0
  w_119_4 : mxxMat0
  w_119_5 : mxxMat0
  w_120_0 : mxxMat1
  w_121_0 : Int
  w_122_0 : mxxMat0
  w_123_0 : mxxMat0
  w_124_0 : mxxMat1
  w_125_0 : mxxMat0
  w_126_0 : mxxMat0
  w_127_0 : mxxMat1
  w_128_0 : mxxMat1
  w_132_0 : mxxMat1
  w_135_0 : mxxMat1
  w_136_0 : mxxMat1
  w_138_0 : mxxMat0
  w_141_0 : mxxMat2
  w_143_0 : mxxMat0
  w_145_0 : mxxMat13
  w_147_0 : mxxMat0
  w_160_0 : mxxMat1
  w_161_0 : mxxMat1
  w_162_0 : mxxMat10
  w_163_0 : Fin 15 → mxxMat0
  w_163_1 : Fin 15 → mxxMat0
  w_163_2 : Fin 15 → mxxMat0
  w_163_3 : Fin 15 → mxxMat0
  w_163_4 : Fin 15 → mxxMat0
  w_163_5 : Fin 15 → mxxMat0
  w_164_0 : mxxMat0
  w_164_1 : mxxMat0
  w_164_2 : mxxMat0
  w_164_3 : mxxMat0
  w_164_4 : mxxMat0
  w_164_5 : mxxMat0
  w_165_0 : mxxMat0
  w_165_1 : mxxMat0
  w_165_2 : mxxMat0
  w_165_3 : mxxMat0
  w_165_4 : mxxMat0
  w_165_5 : mxxMat0
  w_167_0 : mxxMat15
  w_168_0 : mxxMat0
  w_168_1 : mxxMat0
  w_168_2 : mxxMat0
  w_168_3 : mxxMat0
  w_168_4 : mxxMat0
  w_168_5 : mxxMat0
  w_169_0 : mxxMat1
  w_170_0 : Int
  w_171_0 : mxxMat0
  w_172_0 : mxxMat0
  w_173_0 : mxxMat1
  w_174_0 : mxxMat0
  w_175_0 : mxxMat0
  w_176_0 : mxxMat1
  w_177_0 : mxxMat1
  w_181_0 : mxxMat1
  w_184_0 : mxxMat1
  w_185_0 : mxxMat1
  w_187_0 : mxxMat0
  w_190_0 : mxxMat2
  w_192_0 : mxxMat0
  w_194_0 : mxxMat13
  w_196_0 : mxxMat0
  w_209_0 : mxxMat1
  w_210_0 : mxxMat1
  w_211_0 : mxxMat10
  w_212_0 : Fin 15 → mxxMat0
  w_212_1 : Fin 15 → mxxMat0
  w_212_2 : Fin 15 → mxxMat0
  w_212_3 : Fin 15 → mxxMat0
  w_212_4 : Fin 15 → mxxMat0
  w_212_5 : Fin 15 → mxxMat0
  w_213_0 : mxxMat0
  w_213_1 : mxxMat0
  w_213_2 : mxxMat0
  w_213_3 : mxxMat0
  w_213_4 : mxxMat0
  w_213_5 : mxxMat0
  w_214_0 : mxxMat0
  w_214_1 : mxxMat0
  w_214_2 : mxxMat0
  w_214_3 : mxxMat0
  w_214_4 : mxxMat0
  w_214_5 : mxxMat0
  w_216_0 : mxxMat15
  w_217_0 : mxxMat0
  w_217_1 : mxxMat0
  w_217_2 : mxxMat0
  w_217_3 : mxxMat0
  w_217_4 : mxxMat0
  w_217_5 : mxxMat0

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_11_0 : ByteArray) (w_38_0 : Fin 512 → Int) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_186_0 0 1 0 44 w_187_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_generatedRoot_190 backend hashModel tape (path ++ [190, i]) params i (current, (w_38_0, ())) next) (Int.toNat (128)) w_189_0 w_190_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_192_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_193_0 w_194_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_186_0 0 1 44 88 w_196_0 ∧
  MxxRuntime.concatColumns w_198_0 w_200_0 w_209_0 ∧
  MxxRuntime.concatColumns w_202_0 w_204_0 w_210_0 ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10) => sequential_generatedRoot_211 backend hashModel tape (path ++ [211, i]) params i (current, (w_60_0, (w_61_0, (w_209_0, (w_210_0, ()))))) next) (Int.toNat (968)) w_208_0 w_211_0 ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_212 backend hashModel tape (path ++ [212, i.val]) params i (w_55_0, (w_56_0, (w_206_0, (w_207_0, (w_11_0, (w_187_0, (w_196_0, (w_211_0, ())))))))) ((w_212_0 i), ((w_212_1 i), ((w_212_2 i), ((w_212_3 i), ((w_212_4 i), ((w_212_5 i), ()))))))) ∧
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_213 backend hashModel tape (path ++ [213, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_212_0, (w_38_0, (w_212_1, ()))))))))) next) (Int.toNat (15)) (w_195_0, (w_197_0, (w_199_0, (w_201_0, (w_203_0, (w_205_0, ())))))) (w_213_0, (w_213_1, (w_213_2, (w_213_3, (w_213_4, (w_213_5, ())))))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_214 backend hashModel tape (path ++ [214, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_206_0, (w_207_0, (w_187_0, (w_196_0, (w_11_0, (w_211_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (7)) (w_213_0, (w_213_1, (w_213_2, (w_213_3, (w_213_4, (w_213_5, ())))))) (w_214_0, (w_214_1, (w_214_2, (w_214_3, (w_214_4, (w_214_5, ())))))) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15) => sequential_generatedRoot_216 backend hashModel tape (path ++ [216, i]) params i (current, (w_60_0, (w_61_0, (w_212_2, (w_212_3, (w_212_4, (w_212_5, ()))))))) next) (Int.toNat (4620)) w_215_0 w_216_0 ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_217 backend hashModel tape (path ++ [217, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_206_0, (w_207_0, (w_212_0, (w_212_1, (w_11_0, (w_216_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (105)) (w_214_0, (w_214_1, (w_214_2, (w_214_3, (w_214_4, (w_214_5, ())))))) (w_217_0, (w_217_1, (w_217_2, (w_217_3, (w_217_4, (w_217_5, ())))))) ∧
  outputs = (w_0_0, (w_1_0, (w_2_0, (w_3_0, (w_217_0, (w_217_1, ()))))))

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_11_0 : ByteArray) (w_13_0 : Fin 4 → Int) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 15488 → mxxMat6) (w_26_0 : Fin 15488 → mxxMat6) (w_38_0 : Fin 512 → Int) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_157_0 : mxxMat12) (w_158_0 : mxxMat12) (w_179_0 : mxxMat0) (w_180_0 : mxxMat0) (w_182_0 : mxxMat0) (w_183_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_4_0 : mxxMat1) (w_12_0 : Fin 4 → mxxMat0) (w_12_1 : Fin 4 → mxxMat0) (w_138_0 : mxxMat0) (w_146_0 : mxxMat0) (w_147_0 : mxxMat0) (w_148_0 : mxxMat0) (w_150_0 : mxxMat0) (w_152_0 : mxxMat0) (w_154_0 : mxxMat0) (w_156_0 : mxxMat0) (w_162_0 : mxxMat10) (w_163_0 : Fin 15 → mxxMat0) (w_163_1 : Fin 15 → mxxMat0) (w_163_2 : Fin 15 → mxxMat0) (w_163_3 : Fin 15 → mxxMat0) (w_163_4 : Fin 15 → mxxMat0) (w_163_5 : Fin 15 → mxxMat0) (w_164_0 : mxxMat0) (w_164_1 : mxxMat0) (w_164_2 : mxxMat0) (w_164_3 : mxxMat0) (w_164_4 : mxxMat0) (w_164_5 : mxxMat0) (w_165_0 : mxxMat0) (w_165_1 : mxxMat0) (w_165_2 : mxxMat0) (w_165_3 : mxxMat0) (w_165_4 : mxxMat0) (w_165_5 : mxxMat0) (w_166_0 : mxxMat15) (w_167_0 : mxxMat15) (w_168_0 : mxxMat0) (w_168_1 : mxxMat0) (w_168_2 : mxxMat0) (w_168_3 : mxxMat0) (w_168_4 : mxxMat0) (w_168_5 : mxxMat0) (w_169_0 : mxxMat1) (w_170_0 : Int) (w_171_0 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat1) (w_174_0 : mxxMat0) (w_175_0 : mxxMat0) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_181_0 : mxxMat1) (w_184_0 : mxxMat1) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_164 backend hashModel tape (path ++ [164, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_163_0, (w_38_0, (w_163_1, ()))))))))) next) (Int.toNat (15)) (w_146_0, (w_148_0, (w_150_0, (w_152_0, (w_154_0, (w_156_0, ())))))) (w_164_0, (w_164_1, (w_164_2, (w_164_3, (w_164_4, (w_164_5, ())))))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_165 backend hashModel tape (path ++ [165, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_157_0, (w_158_0, (w_138_0, (w_147_0, (w_11_0, (w_162_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (7)) (w_164_0, (w_164_1, (w_164_2, (w_164_3, (w_164_4, (w_164_5, ())))))) (w_165_0, (w_165_1, (w_165_2, (w_165_3, (w_165_4, (w_165_5, ())))))) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15) => sequential_generatedRoot_167 backend hashModel tape (path ++ [167, i]) params i (current, (w_60_0, (w_61_0, (w_163_2, (w_163_3, (w_163_4, (w_163_5, ()))))))) next) (Int.toNat (4620)) w_166_0 w_167_0 ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_168 backend hashModel tape (path ++ [168, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_157_0, (w_158_0, (w_163_0, (w_163_1, (w_11_0, (w_167_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (105)) (w_165_0, (w_165_1, (w_165_2, (w_165_3, (w_165_4, (w_165_5, ())))))) (w_168_0, (w_168_1, (w_168_2, (w_168_3, (w_168_4, (w_168_5, ())))))) ∧
  MxxRuntime.concatColumns w_168_0 w_168_1 w_169_0 ∧
  MxxRuntime.familyGetStatic w_13_0 (3) w_170_0 ∧
  0 ≤ w_170_0 ∧
  w_170_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_170_0 w_171_0 ∧
  0 ≤ w_170_0 ∧
  w_170_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_1 w_170_0 w_172_0 ∧
  MxxRuntime.concatColumns w_171_0 w_172_0 w_173_0 ∧
  0 ≤ w_170_0 ∧
  w_170_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_170_0 w_174_0 ∧
  0 ≤ w_170_0 ∧
  w_170_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_170_0 w_175_0 ∧
  MxxRuntime.concatColumns w_174_0 w_175_0 w_176_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_177 backend hashModel tape (path ++ [177, i]) params i (current, (w_169_0, (w_173_0, (w_176_0, ())))) next) (Int.toNat (88)) w_4_0 w_177_0 ∧
  MxxRuntime.concatColumns w_179_0 w_180_0 w_181_0 ∧
  MxxRuntime.concatColumns w_182_0 w_183_0 w_184_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_185 backend hashModel tape (path ++ [185, i]) params i (current, (w_25_0, (w_170_0, (w_26_0, (w_181_0, (w_184_0, ())))))) next) (Int.toNat (44)) w_178_0 w_185_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  generatedRoot.constraints_6 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_11_0 w_38_0 w_55_0 w_56_0 w_60_0 w_61_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

abbrev generatedRoot.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_11_0 : ByteArray) (w_13_0 : Fin 4 → Int) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 15488 → mxxMat6) (w_26_0 : Fin 15488 → mxxMat6) (w_38_0 : Fin 512 → Int) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_133_0 : mxxMat0) (w_134_0 : mxxMat0) (w_149_0 : mxxMat0) (w_151_0 : mxxMat0) (w_153_0 : mxxMat0) (w_155_0 : mxxMat0) (w_157_0 : mxxMat12) (w_158_0 : mxxMat12) (w_179_0 : mxxMat0) (w_180_0 : mxxMat0) (w_182_0 : mxxMat0) (w_183_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_12_0 : Fin 4 → mxxMat0) (w_12_1 : Fin 4 → mxxMat0) (w_120_0 : mxxMat1) (w_121_0 : Int) (w_124_0 : mxxMat1) (w_125_0 : mxxMat0) (w_126_0 : mxxMat0) (w_127_0 : mxxMat1) (w_128_0 : mxxMat1) (w_129_0 : mxxMat1) (w_132_0 : mxxMat1) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat1) (w_138_0 : mxxMat0) (w_140_0 : mxxMat2) (w_141_0 : mxxMat2) (w_143_0 : mxxMat0) (w_144_0 : mxxMat0) (w_145_0 : mxxMat13) (w_146_0 : mxxMat0) (w_147_0 : mxxMat0) (w_148_0 : mxxMat0) (w_150_0 : mxxMat0) (w_152_0 : mxxMat0) (w_154_0 : mxxMat0) (w_156_0 : mxxMat0) (w_159_0 : mxxMat10) (w_160_0 : mxxMat1) (w_161_0 : mxxMat1) (w_162_0 : mxxMat10) (w_163_0 : Fin 15 → mxxMat0) (w_163_1 : Fin 15 → mxxMat0) (w_163_2 : Fin 15 → mxxMat0) (w_163_3 : Fin 15 → mxxMat0) (w_163_4 : Fin 15 → mxxMat0) (w_163_5 : Fin 15 → mxxMat0) (w_164_0 : mxxMat0) (w_164_1 : mxxMat0) (w_164_2 : mxxMat0) (w_164_3 : mxxMat0) (w_164_4 : mxxMat0) (w_164_5 : mxxMat0) (w_165_0 : mxxMat0) (w_165_1 : mxxMat0) (w_165_2 : mxxMat0) (w_165_3 : mxxMat0) (w_165_4 : mxxMat0) (w_165_5 : mxxMat0) (w_166_0 : mxxMat15) (w_167_0 : mxxMat15) (w_168_0 : mxxMat0) (w_168_1 : mxxMat0) (w_168_2 : mxxMat0) (w_168_3 : mxxMat0) (w_168_4 : mxxMat0) (w_168_5 : mxxMat0) (w_169_0 : mxxMat1) (w_170_0 : Int) (w_171_0 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat1) (w_174_0 : mxxMat0) (w_175_0 : mxxMat0) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_181_0 : mxxMat1) (w_184_0 : mxxMat1) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_20_0 w_121_0 w_126_0 ∧
  MxxRuntime.concatColumns w_125_0 w_126_0 w_127_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_128 backend hashModel tape (path ++ [128, i]) params i (current, (w_120_0, (w_124_0, (w_127_0, ())))) next) (Int.toNat (88)) w_5_0 w_128_0 ∧
  MxxRuntime.concatColumns w_130_0 w_131_0 w_132_0 ∧
  MxxRuntime.concatColumns w_133_0 w_134_0 w_135_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_136 backend hashModel tape (path ++ [136, i]) params i (current, (w_25_0, (w_121_0, (w_26_0, (w_132_0, (w_135_0, ())))))) next) (Int.toNat (44)) w_129_0 w_136_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_137_0 0 1 0 44 w_138_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_generatedRoot_141 backend hashModel tape (path ++ [141, i]) params i (current, (w_38_0, ())) next) (Int.toNat (128)) w_140_0 w_141_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_143_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_144_0 w_145_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_137_0 0 1 44 88 w_147_0 ∧
  MxxRuntime.concatColumns w_149_0 w_151_0 w_160_0 ∧
  MxxRuntime.concatColumns w_153_0 w_155_0 w_161_0 ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10) => sequential_generatedRoot_162 backend hashModel tape (path ++ [162, i]) params i (current, (w_60_0, (w_61_0, (w_160_0, (w_161_0, ()))))) next) (Int.toNat (968)) w_159_0 w_162_0 ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_163 backend hashModel tape (path ++ [163, i.val]) params i (w_55_0, (w_56_0, (w_157_0, (w_158_0, (w_11_0, (w_138_0, (w_147_0, (w_162_0, ())))))))) ((w_163_0 i), ((w_163_1 i), ((w_163_2 i), ((w_163_3 i), ((w_163_4 i), ((w_163_5 i), ()))))))) ∧
  generatedRoot.constraints_5 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_11_0 w_13_0 w_18_0 w_20_0 w_25_0 w_26_0 w_38_0 w_55_0 w_56_0 w_60_0 w_61_0 w_157_0 w_158_0 w_179_0 w_180_0 w_182_0 w_183_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_4_0 w_12_0 w_12_1 w_138_0 w_146_0 w_147_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_162_0 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_166_0 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_181_0 w_184_0 w_185_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_11_0 : ByteArray) (w_13_0 : Fin 4 → Int) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 15488 → mxxMat6) (w_26_0 : Fin 15488 → mxxMat6) (w_38_0 : Fin 512 → Int) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_100_0 : mxxMat0) (w_102_0 : mxxMat0) (w_104_0 : mxxMat0) (w_106_0 : mxxMat0) (w_108_0 : mxxMat12) (w_109_0 : mxxMat12) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_133_0 : mxxMat0) (w_134_0 : mxxMat0) (w_149_0 : mxxMat0) (w_151_0 : mxxMat0) (w_153_0 : mxxMat0) (w_155_0 : mxxMat0) (w_157_0 : mxxMat12) (w_158_0 : mxxMat12) (w_179_0 : mxxMat0) (w_180_0 : mxxMat0) (w_182_0 : mxxMat0) (w_183_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_12_0 : Fin 4 → mxxMat0) (w_12_1 : Fin 4 → mxxMat0) (w_88_0 : mxxMat1) (w_89_0 : mxxMat0) (w_97_0 : mxxMat0) (w_98_0 : mxxMat0) (w_99_0 : mxxMat0) (w_101_0 : mxxMat0) (w_103_0 : mxxMat0) (w_105_0 : mxxMat0) (w_107_0 : mxxMat0) (w_110_0 : mxxMat10) (w_111_0 : mxxMat1) (w_112_0 : mxxMat1) (w_113_0 : mxxMat10) (w_114_0 : Fin 15 → mxxMat0) (w_114_1 : Fin 15 → mxxMat0) (w_114_2 : Fin 15 → mxxMat0) (w_114_3 : Fin 15 → mxxMat0) (w_114_4 : Fin 15 → mxxMat0) (w_114_5 : Fin 15 → mxxMat0) (w_115_0 : mxxMat0) (w_115_1 : mxxMat0) (w_115_2 : mxxMat0) (w_115_3 : mxxMat0) (w_115_4 : mxxMat0) (w_115_5 : mxxMat0) (w_116_0 : mxxMat0) (w_116_1 : mxxMat0) (w_116_2 : mxxMat0) (w_116_3 : mxxMat0) (w_116_4 : mxxMat0) (w_116_5 : mxxMat0) (w_117_0 : mxxMat15) (w_118_0 : mxxMat15) (w_119_0 : mxxMat0) (w_119_1 : mxxMat0) (w_119_2 : mxxMat0) (w_119_3 : mxxMat0) (w_119_4 : mxxMat0) (w_119_5 : mxxMat0) (w_120_0 : mxxMat1) (w_121_0 : Int) (w_122_0 : mxxMat0) (w_123_0 : mxxMat0) (w_124_0 : mxxMat1) (w_125_0 : mxxMat0) (w_126_0 : mxxMat0) (w_127_0 : mxxMat1) (w_128_0 : mxxMat1) (w_129_0 : mxxMat1) (w_132_0 : mxxMat1) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat1) (w_138_0 : mxxMat0) (w_140_0 : mxxMat2) (w_141_0 : mxxMat2) (w_143_0 : mxxMat0) (w_144_0 : mxxMat0) (w_145_0 : mxxMat13) (w_146_0 : mxxMat0) (w_147_0 : mxxMat0) (w_148_0 : mxxMat0) (w_150_0 : mxxMat0) (w_152_0 : mxxMat0) (w_154_0 : mxxMat0) (w_156_0 : mxxMat0) (w_159_0 : mxxMat10) (w_160_0 : mxxMat1) (w_161_0 : mxxMat1) (w_162_0 : mxxMat10) (w_163_0 : Fin 15 → mxxMat0) (w_163_1 : Fin 15 → mxxMat0) (w_163_2 : Fin 15 → mxxMat0) (w_163_3 : Fin 15 → mxxMat0) (w_163_4 : Fin 15 → mxxMat0) (w_163_5 : Fin 15 → mxxMat0) (w_164_0 : mxxMat0) (w_164_1 : mxxMat0) (w_164_2 : mxxMat0) (w_164_3 : mxxMat0) (w_164_4 : mxxMat0) (w_164_5 : mxxMat0) (w_165_0 : mxxMat0) (w_165_1 : mxxMat0) (w_165_2 : mxxMat0) (w_165_3 : mxxMat0) (w_165_4 : mxxMat0) (w_165_5 : mxxMat0) (w_166_0 : mxxMat15) (w_167_0 : mxxMat15) (w_168_0 : mxxMat0) (w_168_1 : mxxMat0) (w_168_2 : mxxMat0) (w_168_3 : mxxMat0) (w_168_4 : mxxMat0) (w_168_5 : mxxMat0) (w_169_0 : mxxMat1) (w_170_0 : Int) (w_171_0 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat1) (w_174_0 : mxxMat0) (w_175_0 : mxxMat0) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_181_0 : mxxMat1) (w_184_0 : mxxMat1) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_88_0 0 1 44 88 w_98_0 ∧
  MxxRuntime.concatColumns w_100_0 w_102_0 w_111_0 ∧
  MxxRuntime.concatColumns w_104_0 w_106_0 w_112_0 ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10) => sequential_generatedRoot_113 backend hashModel tape (path ++ [113, i]) params i (current, (w_60_0, (w_61_0, (w_111_0, (w_112_0, ()))))) next) (Int.toNat (968)) w_110_0 w_113_0 ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_114 backend hashModel tape (path ++ [114, i.val]) params i (w_55_0, (w_56_0, (w_108_0, (w_109_0, (w_11_0, (w_89_0, (w_98_0, (w_113_0, ())))))))) ((w_114_0 i), ((w_114_1 i), ((w_114_2 i), ((w_114_3 i), ((w_114_4 i), ((w_114_5 i), ()))))))) ∧
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_115 backend hashModel tape (path ++ [115, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_114_0, (w_38_0, (w_114_1, ()))))))))) next) (Int.toNat (15)) (w_97_0, (w_99_0, (w_101_0, (w_103_0, (w_105_0, (w_107_0, ())))))) (w_115_0, (w_115_1, (w_115_2, (w_115_3, (w_115_4, (w_115_5, ())))))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_116 backend hashModel tape (path ++ [116, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_108_0, (w_109_0, (w_89_0, (w_98_0, (w_11_0, (w_113_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (7)) (w_115_0, (w_115_1, (w_115_2, (w_115_3, (w_115_4, (w_115_5, ())))))) (w_116_0, (w_116_1, (w_116_2, (w_116_3, (w_116_4, (w_116_5, ())))))) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15) => sequential_generatedRoot_118 backend hashModel tape (path ++ [118, i]) params i (current, (w_60_0, (w_61_0, (w_114_2, (w_114_3, (w_114_4, (w_114_5, ()))))))) next) (Int.toNat (4620)) w_117_0 w_118_0 ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_119 backend hashModel tape (path ++ [119, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_108_0, (w_109_0, (w_114_0, (w_114_1, (w_11_0, (w_118_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (105)) (w_116_0, (w_116_1, (w_116_2, (w_116_3, (w_116_4, (w_116_5, ())))))) (w_119_0, (w_119_1, (w_119_2, (w_119_3, (w_119_4, (w_119_5, ())))))) ∧
  MxxRuntime.concatColumns w_119_0 w_119_1 w_120_0 ∧
  MxxRuntime.familyGetStatic w_13_0 (2) w_121_0 ∧
  0 ≤ w_121_0 ∧
  w_121_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_121_0 w_122_0 ∧
  0 ≤ w_121_0 ∧
  w_121_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_1 w_121_0 w_123_0 ∧
  MxxRuntime.concatColumns w_122_0 w_123_0 w_124_0 ∧
  0 ≤ w_121_0 ∧
  w_121_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_121_0 w_125_0 ∧
  0 ≤ w_121_0 ∧
  w_121_0 < 4 ∧
  generatedRoot.constraints_4 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_11_0 w_13_0 w_18_0 w_20_0 w_25_0 w_26_0 w_38_0 w_55_0 w_56_0 w_60_0 w_61_0 w_130_0 w_131_0 w_133_0 w_134_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_158_0 w_179_0 w_180_0 w_182_0 w_183_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_4_0 w_5_0 w_12_0 w_12_1 w_120_0 w_121_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_132_0 w_135_0 w_136_0 w_137_0 w_138_0 w_140_0 w_141_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_166_0 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_181_0 w_184_0 w_185_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_11_0 : ByteArray) (w_13_0 : Fin 4 → Int) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 15488 → mxxMat6) (w_26_0 : Fin 15488 → mxxMat6) (w_38_0 : Fin 512 → Int) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_81_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (w_85_0 : mxxMat0) (w_100_0 : mxxMat0) (w_102_0 : mxxMat0) (w_104_0 : mxxMat0) (w_106_0 : mxxMat0) (w_108_0 : mxxMat12) (w_109_0 : mxxMat12) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_133_0 : mxxMat0) (w_134_0 : mxxMat0) (w_149_0 : mxxMat0) (w_151_0 : mxxMat0) (w_153_0 : mxxMat0) (w_155_0 : mxxMat0) (w_157_0 : mxxMat12) (w_158_0 : mxxMat12) (w_179_0 : mxxMat0) (w_180_0 : mxxMat0) (w_182_0 : mxxMat0) (w_183_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_12_0 : Fin 4 → mxxMat0) (w_12_1 : Fin 4 → mxxMat0) (w_71_0 : mxxMat1) (w_72_0 : Int) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat1) (w_76_0 : mxxMat0) (w_77_0 : mxxMat0) (w_78_0 : mxxMat1) (w_79_0 : mxxMat1) (w_80_0 : mxxMat1) (w_83_0 : mxxMat1) (w_86_0 : mxxMat1) (w_87_0 : mxxMat1) (w_88_0 : mxxMat1) (w_89_0 : mxxMat0) (w_91_0 : mxxMat2) (w_92_0 : mxxMat2) (w_94_0 : mxxMat0) (w_95_0 : mxxMat0) (w_96_0 : mxxMat13) (w_97_0 : mxxMat0) (w_98_0 : mxxMat0) (w_99_0 : mxxMat0) (w_101_0 : mxxMat0) (w_103_0 : mxxMat0) (w_105_0 : mxxMat0) (w_107_0 : mxxMat0) (w_110_0 : mxxMat10) (w_111_0 : mxxMat1) (w_112_0 : mxxMat1) (w_113_0 : mxxMat10) (w_114_0 : Fin 15 → mxxMat0) (w_114_1 : Fin 15 → mxxMat0) (w_114_2 : Fin 15 → mxxMat0) (w_114_3 : Fin 15 → mxxMat0) (w_114_4 : Fin 15 → mxxMat0) (w_114_5 : Fin 15 → mxxMat0) (w_115_0 : mxxMat0) (w_115_1 : mxxMat0) (w_115_2 : mxxMat0) (w_115_3 : mxxMat0) (w_115_4 : mxxMat0) (w_115_5 : mxxMat0) (w_116_0 : mxxMat0) (w_116_1 : mxxMat0) (w_116_2 : mxxMat0) (w_116_3 : mxxMat0) (w_116_4 : mxxMat0) (w_116_5 : mxxMat0) (w_117_0 : mxxMat15) (w_118_0 : mxxMat15) (w_119_0 : mxxMat0) (w_119_1 : mxxMat0) (w_119_2 : mxxMat0) (w_119_3 : mxxMat0) (w_119_4 : mxxMat0) (w_119_5 : mxxMat0) (w_120_0 : mxxMat1) (w_121_0 : Int) (w_122_0 : mxxMat0) (w_123_0 : mxxMat0) (w_124_0 : mxxMat1) (w_125_0 : mxxMat0) (w_126_0 : mxxMat0) (w_127_0 : mxxMat1) (w_128_0 : mxxMat1) (w_129_0 : mxxMat1) (w_132_0 : mxxMat1) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat1) (w_138_0 : mxxMat0) (w_140_0 : mxxMat2) (w_141_0 : mxxMat2) (w_143_0 : mxxMat0) (w_144_0 : mxxMat0) (w_145_0 : mxxMat13) (w_146_0 : mxxMat0) (w_147_0 : mxxMat0) (w_148_0 : mxxMat0) (w_150_0 : mxxMat0) (w_152_0 : mxxMat0) (w_154_0 : mxxMat0) (w_156_0 : mxxMat0) (w_159_0 : mxxMat10) (w_160_0 : mxxMat1) (w_161_0 : mxxMat1) (w_162_0 : mxxMat10) (w_163_0 : Fin 15 → mxxMat0) (w_163_1 : Fin 15 → mxxMat0) (w_163_2 : Fin 15 → mxxMat0) (w_163_3 : Fin 15 → mxxMat0) (w_163_4 : Fin 15 → mxxMat0) (w_163_5 : Fin 15 → mxxMat0) (w_164_0 : mxxMat0) (w_164_1 : mxxMat0) (w_164_2 : mxxMat0) (w_164_3 : mxxMat0) (w_164_4 : mxxMat0) (w_164_5 : mxxMat0) (w_165_0 : mxxMat0) (w_165_1 : mxxMat0) (w_165_2 : mxxMat0) (w_165_3 : mxxMat0) (w_165_4 : mxxMat0) (w_165_5 : mxxMat0) (w_166_0 : mxxMat15) (w_167_0 : mxxMat15) (w_168_0 : mxxMat0) (w_168_1 : mxxMat0) (w_168_2 : mxxMat0) (w_168_3 : mxxMat0) (w_168_4 : mxxMat0) (w_168_5 : mxxMat0) (w_169_0 : mxxMat1) (w_170_0 : Int) (w_171_0 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat1) (w_174_0 : mxxMat0) (w_175_0 : mxxMat0) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_181_0 : mxxMat1) (w_184_0 : mxxMat1) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.familyGetDynamic w_12_0 w_72_0 w_73_0 ∧
  0 ≤ w_72_0 ∧
  w_72_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_1 w_72_0 w_74_0 ∧
  MxxRuntime.concatColumns w_73_0 w_74_0 w_75_0 ∧
  0 ≤ w_72_0 ∧
  w_72_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_72_0 w_76_0 ∧
  0 ≤ w_72_0 ∧
  w_72_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_72_0 w_77_0 ∧
  MxxRuntime.concatColumns w_76_0 w_77_0 w_78_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_79 backend hashModel tape (path ++ [79, i]) params i (current, (w_71_0, (w_75_0, (w_78_0, ())))) next) (Int.toNat (88)) w_6_0 w_79_0 ∧
  MxxRuntime.concatColumns w_81_0 w_82_0 w_83_0 ∧
  MxxRuntime.concatColumns w_84_0 w_85_0 w_86_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_87 backend hashModel tape (path ++ [87, i]) params i (current, (w_25_0, (w_72_0, (w_26_0, (w_83_0, (w_86_0, ())))))) next) (Int.toNat (44)) w_80_0 w_87_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_88_0 0 1 0 44 w_89_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_generatedRoot_92 backend hashModel tape (path ++ [92, i]) params i (current, (w_38_0, ())) next) (Int.toNat (128)) w_91_0 w_92_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_94_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_95_0 w_96_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_11_0 w_13_0 w_18_0 w_20_0 w_25_0 w_26_0 w_38_0 w_55_0 w_56_0 w_60_0 w_61_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_109_0 w_130_0 w_131_0 w_133_0 w_134_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_158_0 w_179_0 w_180_0 w_182_0 w_183_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_4_0 w_5_0 w_12_0 w_12_1 w_88_0 w_89_0 w_97_0 w_98_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_117_0 w_118_0 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_132_0 w_135_0 w_136_0 w_137_0 w_138_0 w_140_0 w_141_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_166_0 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_181_0 w_184_0 w_185_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_11_0 : ByteArray) (w_13_0 : Fin 4 → Int) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 15488 → mxxMat6) (w_26_0 : Fin 15488 → mxxMat6) (w_38_0 : Fin 512 → Int) (w_47_0 : mxxMat0) (w_49_0 : mxxMat0) (w_51_0 : mxxMat0) (w_53_0 : mxxMat0) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_57_0 : mxxMat12) (w_58_0 : mxxMat12) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_81_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (w_85_0 : mxxMat0) (w_100_0 : mxxMat0) (w_102_0 : mxxMat0) (w_104_0 : mxxMat0) (w_106_0 : mxxMat0) (w_108_0 : mxxMat12) (w_109_0 : mxxMat12) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_133_0 : mxxMat0) (w_134_0 : mxxMat0) (w_149_0 : mxxMat0) (w_151_0 : mxxMat0) (w_153_0 : mxxMat0) (w_155_0 : mxxMat0) (w_157_0 : mxxMat12) (w_158_0 : mxxMat12) (w_179_0 : mxxMat0) (w_180_0 : mxxMat0) (w_182_0 : mxxMat0) (w_183_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_12_0 : Fin 4 → mxxMat0) (w_12_1 : Fin 4 → mxxMat0) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_37_0 : mxxMat2) (w_39_0 : mxxMat2) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat13) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_48_0 : mxxMat0) (w_50_0 : mxxMat0) (w_52_0 : mxxMat0) (w_54_0 : mxxMat0) (w_59_0 : mxxMat10) (w_62_0 : mxxMat1) (w_63_0 : mxxMat1) (w_64_0 : mxxMat10) (w_65_0 : Fin 15 → mxxMat0) (w_65_1 : Fin 15 → mxxMat0) (w_65_2 : Fin 15 → mxxMat0) (w_65_3 : Fin 15 → mxxMat0) (w_65_4 : Fin 15 → mxxMat0) (w_65_5 : Fin 15 → mxxMat0) (w_66_0 : mxxMat0) (w_66_1 : mxxMat0) (w_66_2 : mxxMat0) (w_66_3 : mxxMat0) (w_66_4 : mxxMat0) (w_66_5 : mxxMat0) (w_67_0 : mxxMat0) (w_67_1 : mxxMat0) (w_67_2 : mxxMat0) (w_67_3 : mxxMat0) (w_67_4 : mxxMat0) (w_67_5 : mxxMat0) (w_68_0 : mxxMat15) (w_69_0 : mxxMat15) (w_70_0 : mxxMat0) (w_70_1 : mxxMat0) (w_70_2 : mxxMat0) (w_70_3 : mxxMat0) (w_70_4 : mxxMat0) (w_70_5 : mxxMat0) (w_71_0 : mxxMat1) (w_72_0 : Int) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat1) (w_76_0 : mxxMat0) (w_77_0 : mxxMat0) (w_78_0 : mxxMat1) (w_79_0 : mxxMat1) (w_80_0 : mxxMat1) (w_83_0 : mxxMat1) (w_86_0 : mxxMat1) (w_87_0 : mxxMat1) (w_88_0 : mxxMat1) (w_89_0 : mxxMat0) (w_91_0 : mxxMat2) (w_92_0 : mxxMat2) (w_94_0 : mxxMat0) (w_95_0 : mxxMat0) (w_96_0 : mxxMat13) (w_97_0 : mxxMat0) (w_98_0 : mxxMat0) (w_99_0 : mxxMat0) (w_101_0 : mxxMat0) (w_103_0 : mxxMat0) (w_105_0 : mxxMat0) (w_107_0 : mxxMat0) (w_110_0 : mxxMat10) (w_111_0 : mxxMat1) (w_112_0 : mxxMat1) (w_113_0 : mxxMat10) (w_114_0 : Fin 15 → mxxMat0) (w_114_1 : Fin 15 → mxxMat0) (w_114_2 : Fin 15 → mxxMat0) (w_114_3 : Fin 15 → mxxMat0) (w_114_4 : Fin 15 → mxxMat0) (w_114_5 : Fin 15 → mxxMat0) (w_115_0 : mxxMat0) (w_115_1 : mxxMat0) (w_115_2 : mxxMat0) (w_115_3 : mxxMat0) (w_115_4 : mxxMat0) (w_115_5 : mxxMat0) (w_116_0 : mxxMat0) (w_116_1 : mxxMat0) (w_116_2 : mxxMat0) (w_116_3 : mxxMat0) (w_116_4 : mxxMat0) (w_116_5 : mxxMat0) (w_117_0 : mxxMat15) (w_118_0 : mxxMat15) (w_119_0 : mxxMat0) (w_119_1 : mxxMat0) (w_119_2 : mxxMat0) (w_119_3 : mxxMat0) (w_119_4 : mxxMat0) (w_119_5 : mxxMat0) (w_120_0 : mxxMat1) (w_121_0 : Int) (w_122_0 : mxxMat0) (w_123_0 : mxxMat0) (w_124_0 : mxxMat1) (w_125_0 : mxxMat0) (w_126_0 : mxxMat0) (w_127_0 : mxxMat1) (w_128_0 : mxxMat1) (w_129_0 : mxxMat1) (w_132_0 : mxxMat1) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat1) (w_138_0 : mxxMat0) (w_140_0 : mxxMat2) (w_141_0 : mxxMat2) (w_143_0 : mxxMat0) (w_144_0 : mxxMat0) (w_145_0 : mxxMat13) (w_146_0 : mxxMat0) (w_147_0 : mxxMat0) (w_148_0 : mxxMat0) (w_150_0 : mxxMat0) (w_152_0 : mxxMat0) (w_154_0 : mxxMat0) (w_156_0 : mxxMat0) (w_159_0 : mxxMat10) (w_160_0 : mxxMat1) (w_161_0 : mxxMat1) (w_162_0 : mxxMat10) (w_163_0 : Fin 15 → mxxMat0) (w_163_1 : Fin 15 → mxxMat0) (w_163_2 : Fin 15 → mxxMat0) (w_163_3 : Fin 15 → mxxMat0) (w_163_4 : Fin 15 → mxxMat0) (w_163_5 : Fin 15 → mxxMat0) (w_164_0 : mxxMat0) (w_164_1 : mxxMat0) (w_164_2 : mxxMat0) (w_164_3 : mxxMat0) (w_164_4 : mxxMat0) (w_164_5 : mxxMat0) (w_165_0 : mxxMat0) (w_165_1 : mxxMat0) (w_165_2 : mxxMat0) (w_165_3 : mxxMat0) (w_165_4 : mxxMat0) (w_165_5 : mxxMat0) (w_166_0 : mxxMat15) (w_167_0 : mxxMat15) (w_168_0 : mxxMat0) (w_168_1 : mxxMat0) (w_168_2 : mxxMat0) (w_168_3 : mxxMat0) (w_168_4 : mxxMat0) (w_168_5 : mxxMat0) (w_169_0 : mxxMat1) (w_170_0 : Int) (w_171_0 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat1) (w_174_0 : mxxMat0) (w_175_0 : mxxMat0) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_181_0 : mxxMat1) (w_184_0 : mxxMat1) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  0 < 44 ∧
  44 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 0 44 w_35_0 ∧
  0 ≤ (128) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat2) => sequential_generatedRoot_39 backend hashModel tape (path ++ [39, i]) params i (current, (w_38_0, ())) next) (Int.toNat (128)) w_37_0 w_39_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 65536 44 w_41_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 65536 44 w_42_0 w_43_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 44 ∧
  44 < 88 ∧
  88 ≤ 88 ∧
  MxxRuntime.sliceMatrix w_34_0 0 1 44 88 w_45_0 ∧
  MxxRuntime.concatColumns w_47_0 w_49_0 w_62_0 ∧
  MxxRuntime.concatColumns w_51_0 w_53_0 w_63_0 ∧
  0 ≤ (968) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat10) => sequential_generatedRoot_64 backend hashModel tape (path ++ [64, i]) params i (current, (w_60_0, (w_61_0, (w_62_0, (w_63_0, ()))))) next) (Int.toNat (968)) w_59_0 w_64_0 ∧
  (15) = 15 ∧
  (∀ i : Fin 15, parallel_generatedRoot_65 backend hashModel tape (path ++ [65, i.val]) params i (w_55_0, (w_56_0, (w_57_0, (w_58_0, (w_11_0, (w_35_0, (w_45_0, (w_64_0, ())))))))) ((w_65_0 i), ((w_65_1 i), ((w_65_2 i), ((w_65_3 i), ((w_65_4 i), ((w_65_5 i), ()))))))) ∧
  0 ≤ (15) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_66 backend hashModel tape (path ++ [66, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_65_0, (w_38_0, (w_65_1, ()))))))))) next) (Int.toNat (15)) (w_44_0, (w_46_0, (w_48_0, (w_50_0, (w_52_0, (w_54_0, ())))))) (w_66_0, (w_66_1, (w_66_2, (w_66_3, (w_66_4, (w_66_5, ())))))) ∧
  0 ≤ (7) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_67 backend hashModel tape (path ++ [67, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_57_0, (w_58_0, (w_35_0, (w_45_0, (w_11_0, (w_64_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (7)) (w_66_0, (w_66_1, (w_66_2, (w_66_3, (w_66_4, (w_66_5, ())))))) (w_67_0, (w_67_1, (w_67_2, (w_67_3, (w_67_4, (w_67_5, ())))))) ∧
  0 ≤ (4620) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat15) => sequential_generatedRoot_69 backend hashModel tape (path ++ [69, i]) params i (current, (w_60_0, (w_61_0, (w_65_2, (w_65_3, (w_65_4, (w_65_5, ()))))))) next) (Int.toNat (4620)) w_68_0 w_69_0 ∧
  0 ≤ (105) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) => sequential_generatedRoot_70 backend hashModel tape (path ++ [70, i]) params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_55_0, (w_56_0, (w_57_0, (w_58_0, (w_65_0, (w_65_1, (w_11_0, (w_69_0, (w_38_0, ()))))))))))))))) next) (Int.toNat (105)) (w_67_0, (w_67_1, (w_67_2, (w_67_3, (w_67_4, (w_67_5, ())))))) (w_70_0, (w_70_1, (w_70_2, (w_70_3, (w_70_4, (w_70_5, ())))))) ∧
  MxxRuntime.concatColumns w_70_0 w_70_1 w_71_0 ∧
  MxxRuntime.familyGetStatic w_13_0 (1) w_72_0 ∧
  0 ≤ w_72_0 ∧
  w_72_0 < 4 ∧
  generatedRoot.constraints_2 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_11_0 w_13_0 w_18_0 w_20_0 w_25_0 w_26_0 w_38_0 w_55_0 w_56_0 w_60_0 w_61_0 w_81_0 w_82_0 w_84_0 w_85_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_109_0 w_130_0 w_131_0 w_133_0 w_134_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_158_0 w_179_0 w_180_0 w_182_0 w_183_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_4_0 w_5_0 w_6_0 w_12_0 w_12_1 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_83_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_117_0 w_118_0 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_132_0 w_135_0 w_136_0 w_137_0 w_138_0 w_140_0 w_141_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_166_0 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_181_0 w_184_0 w_185_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_1_0 : mxxMat0) (w_2_0 : mxxMat0) (w_3_0 : mxxMat0) (w_8_0 : mxxMat0) (w_9_0 : mxxMat0) (w_11_0 : ByteArray) (w_13_0 : Fin 4 → Int) (w_18_0 : Fin 4 → mxxMat0) (w_20_0 : Fin 4 → mxxMat0) (w_25_0 : Fin 15488 → mxxMat6) (w_26_0 : Fin 15488 → mxxMat6) (w_38_0 : Fin 512 → Int) (w_47_0 : mxxMat0) (w_49_0 : mxxMat0) (w_51_0 : mxxMat0) (w_53_0 : mxxMat0) (w_55_0 : Fin 37 → mxxMat0) (w_56_0 : Fin 37 → mxxMat0) (w_57_0 : mxxMat12) (w_58_0 : mxxMat12) (w_60_0 : Fin 85184 → mxxMat6) (w_61_0 : Fin 85184 → mxxMat6) (w_81_0 : mxxMat0) (w_82_0 : mxxMat0) (w_84_0 : mxxMat0) (w_85_0 : mxxMat0) (w_100_0 : mxxMat0) (w_102_0 : mxxMat0) (w_104_0 : mxxMat0) (w_106_0 : mxxMat0) (w_108_0 : mxxMat12) (w_109_0 : mxxMat12) (w_130_0 : mxxMat0) (w_131_0 : mxxMat0) (w_133_0 : mxxMat0) (w_134_0 : mxxMat0) (w_149_0 : mxxMat0) (w_151_0 : mxxMat0) (w_153_0 : mxxMat0) (w_155_0 : mxxMat0) (w_157_0 : mxxMat12) (w_158_0 : mxxMat12) (w_179_0 : mxxMat0) (w_180_0 : mxxMat0) (w_182_0 : mxxMat0) (w_183_0 : mxxMat0) (w_198_0 : mxxMat0) (w_200_0 : mxxMat0) (w_202_0 : mxxMat0) (w_204_0 : mxxMat0) (w_206_0 : mxxMat12) (w_207_0 : mxxMat12) (w_4_0 : mxxMat1) (w_5_0 : mxxMat1) (w_6_0 : mxxMat1) (w_7_0 : mxxMat1) (w_10_0 : mxxMat1) (w_12_0 : Fin 4 → mxxMat0) (w_12_1 : Fin 4 → mxxMat0) (w_14_0 : Int) (w_15_0 : mxxMat0) (w_16_0 : mxxMat0) (w_17_0 : mxxMat1) (w_19_0 : mxxMat0) (w_21_0 : mxxMat0) (w_22_0 : mxxMat1) (w_23_0 : mxxMat1) (w_24_0 : mxxMat1) (w_27_0 : mxxMat0) (w_28_0 : mxxMat0) (w_29_0 : mxxMat1) (w_30_0 : mxxMat0) (w_31_0 : mxxMat0) (w_32_0 : mxxMat1) (w_33_0 : mxxMat1) (w_34_0 : mxxMat1) (w_35_0 : mxxMat0) (w_37_0 : mxxMat2) (w_39_0 : mxxMat2) (w_41_0 : mxxMat0) (w_42_0 : mxxMat0) (w_43_0 : mxxMat13) (w_44_0 : mxxMat0) (w_45_0 : mxxMat0) (w_46_0 : mxxMat0) (w_48_0 : mxxMat0) (w_50_0 : mxxMat0) (w_52_0 : mxxMat0) (w_54_0 : mxxMat0) (w_59_0 : mxxMat10) (w_62_0 : mxxMat1) (w_63_0 : mxxMat1) (w_64_0 : mxxMat10) (w_65_0 : Fin 15 → mxxMat0) (w_65_1 : Fin 15 → mxxMat0) (w_65_2 : Fin 15 → mxxMat0) (w_65_3 : Fin 15 → mxxMat0) (w_65_4 : Fin 15 → mxxMat0) (w_65_5 : Fin 15 → mxxMat0) (w_66_0 : mxxMat0) (w_66_1 : mxxMat0) (w_66_2 : mxxMat0) (w_66_3 : mxxMat0) (w_66_4 : mxxMat0) (w_66_5 : mxxMat0) (w_67_0 : mxxMat0) (w_67_1 : mxxMat0) (w_67_2 : mxxMat0) (w_67_3 : mxxMat0) (w_67_4 : mxxMat0) (w_67_5 : mxxMat0) (w_68_0 : mxxMat15) (w_69_0 : mxxMat15) (w_70_0 : mxxMat0) (w_70_1 : mxxMat0) (w_70_2 : mxxMat0) (w_70_3 : mxxMat0) (w_70_4 : mxxMat0) (w_70_5 : mxxMat0) (w_71_0 : mxxMat1) (w_72_0 : Int) (w_73_0 : mxxMat0) (w_74_0 : mxxMat0) (w_75_0 : mxxMat1) (w_76_0 : mxxMat0) (w_77_0 : mxxMat0) (w_78_0 : mxxMat1) (w_79_0 : mxxMat1) (w_80_0 : mxxMat1) (w_83_0 : mxxMat1) (w_86_0 : mxxMat1) (w_87_0 : mxxMat1) (w_88_0 : mxxMat1) (w_89_0 : mxxMat0) (w_91_0 : mxxMat2) (w_92_0 : mxxMat2) (w_94_0 : mxxMat0) (w_95_0 : mxxMat0) (w_96_0 : mxxMat13) (w_97_0 : mxxMat0) (w_98_0 : mxxMat0) (w_99_0 : mxxMat0) (w_101_0 : mxxMat0) (w_103_0 : mxxMat0) (w_105_0 : mxxMat0) (w_107_0 : mxxMat0) (w_110_0 : mxxMat10) (w_111_0 : mxxMat1) (w_112_0 : mxxMat1) (w_113_0 : mxxMat10) (w_114_0 : Fin 15 → mxxMat0) (w_114_1 : Fin 15 → mxxMat0) (w_114_2 : Fin 15 → mxxMat0) (w_114_3 : Fin 15 → mxxMat0) (w_114_4 : Fin 15 → mxxMat0) (w_114_5 : Fin 15 → mxxMat0) (w_115_0 : mxxMat0) (w_115_1 : mxxMat0) (w_115_2 : mxxMat0) (w_115_3 : mxxMat0) (w_115_4 : mxxMat0) (w_115_5 : mxxMat0) (w_116_0 : mxxMat0) (w_116_1 : mxxMat0) (w_116_2 : mxxMat0) (w_116_3 : mxxMat0) (w_116_4 : mxxMat0) (w_116_5 : mxxMat0) (w_117_0 : mxxMat15) (w_118_0 : mxxMat15) (w_119_0 : mxxMat0) (w_119_1 : mxxMat0) (w_119_2 : mxxMat0) (w_119_3 : mxxMat0) (w_119_4 : mxxMat0) (w_119_5 : mxxMat0) (w_120_0 : mxxMat1) (w_121_0 : Int) (w_122_0 : mxxMat0) (w_123_0 : mxxMat0) (w_124_0 : mxxMat1) (w_125_0 : mxxMat0) (w_126_0 : mxxMat0) (w_127_0 : mxxMat1) (w_128_0 : mxxMat1) (w_129_0 : mxxMat1) (w_132_0 : mxxMat1) (w_135_0 : mxxMat1) (w_136_0 : mxxMat1) (w_137_0 : mxxMat1) (w_138_0 : mxxMat0) (w_140_0 : mxxMat2) (w_141_0 : mxxMat2) (w_143_0 : mxxMat0) (w_144_0 : mxxMat0) (w_145_0 : mxxMat13) (w_146_0 : mxxMat0) (w_147_0 : mxxMat0) (w_148_0 : mxxMat0) (w_150_0 : mxxMat0) (w_152_0 : mxxMat0) (w_154_0 : mxxMat0) (w_156_0 : mxxMat0) (w_159_0 : mxxMat10) (w_160_0 : mxxMat1) (w_161_0 : mxxMat1) (w_162_0 : mxxMat10) (w_163_0 : Fin 15 → mxxMat0) (w_163_1 : Fin 15 → mxxMat0) (w_163_2 : Fin 15 → mxxMat0) (w_163_3 : Fin 15 → mxxMat0) (w_163_4 : Fin 15 → mxxMat0) (w_163_5 : Fin 15 → mxxMat0) (w_164_0 : mxxMat0) (w_164_1 : mxxMat0) (w_164_2 : mxxMat0) (w_164_3 : mxxMat0) (w_164_4 : mxxMat0) (w_164_5 : mxxMat0) (w_165_0 : mxxMat0) (w_165_1 : mxxMat0) (w_165_2 : mxxMat0) (w_165_3 : mxxMat0) (w_165_4 : mxxMat0) (w_165_5 : mxxMat0) (w_166_0 : mxxMat15) (w_167_0 : mxxMat15) (w_168_0 : mxxMat0) (w_168_1 : mxxMat0) (w_168_2 : mxxMat0) (w_168_3 : mxxMat0) (w_168_4 : mxxMat0) (w_168_5 : mxxMat0) (w_169_0 : mxxMat1) (w_170_0 : Int) (w_171_0 : mxxMat0) (w_172_0 : mxxMat0) (w_173_0 : mxxMat1) (w_174_0 : mxxMat0) (w_175_0 : mxxMat0) (w_176_0 : mxxMat1) (w_177_0 : mxxMat1) (w_178_0 : mxxMat1) (w_181_0 : mxxMat1) (w_184_0 : mxxMat1) (w_185_0 : mxxMat1) (w_186_0 : mxxMat1) (w_187_0 : mxxMat0) (w_189_0 : mxxMat2) (w_190_0 : mxxMat2) (w_192_0 : mxxMat0) (w_193_0 : mxxMat0) (w_194_0 : mxxMat13) (w_195_0 : mxxMat0) (w_196_0 : mxxMat0) (w_197_0 : mxxMat0) (w_199_0 : mxxMat0) (w_201_0 : mxxMat0) (w_203_0 : mxxMat0) (w_205_0 : mxxMat0) (w_208_0 : mxxMat10) (w_209_0 : mxxMat1) (w_210_0 : mxxMat1) (w_211_0 : mxxMat10) (w_212_0 : Fin 15 → mxxMat0) (w_212_1 : Fin 15 → mxxMat0) (w_212_2 : Fin 15 → mxxMat0) (w_212_3 : Fin 15 → mxxMat0) (w_212_4 : Fin 15 → mxxMat0) (w_212_5 : Fin 15 → mxxMat0) (w_213_0 : mxxMat0) (w_213_1 : mxxMat0) (w_213_2 : mxxMat0) (w_213_3 : mxxMat0) (w_213_4 : mxxMat0) (w_213_5 : mxxMat0) (w_214_0 : mxxMat0) (w_214_1 : mxxMat0) (w_214_2 : mxxMat0) (w_214_3 : mxxMat0) (w_214_4 : mxxMat0) (w_214_5 : mxxMat0) (w_215_0 : mxxMat15) (w_216_0 : mxxMat15) (w_217_0 : mxxMat0) (w_217_1 : mxxMat0) (w_217_2 : mxxMat0) (w_217_3 : mxxMat0) (w_217_4 : mxxMat0) (w_217_5 : mxxMat0) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_12 backend hashModel tape (path ++ [12, i.val]) params i w_11_0 ((w_12_0 i), ((w_12_1 i), ()))) ∧
  MxxRuntime.familyGetStatic w_13_0 (0) w_14_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_0 w_14_0 w_15_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_12_1 w_14_0 w_16_0 ∧
  MxxRuntime.concatColumns w_15_0 w_16_0 w_17_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_14_0 w_19_0 ∧
  0 ≤ w_14_0 ∧
  w_14_0 < 4 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_14_0 w_21_0 ∧
  MxxRuntime.concatColumns w_19_0 w_21_0 w_22_0 ∧
  0 ≤ (88) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_23 backend hashModel tape (path ++ [23, i]) params i (current, (w_10_0, (w_17_0, (w_22_0, ())))) next) (Int.toNat (88)) w_7_0 w_23_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_11_0) w_27_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_11_0) w_28_0 ∧
  MxxRuntime.concatColumns w_27_0 w_28_0 w_29_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_11_0) w_30_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [98, 99, 49, 51, 97, 99, 48, 49, 54, 57, 53, 102, 102, 55, 49, 101, 57, 48, 102, 54, 97, 52, 50, 56, 51, 100, 50, 52, 51, 51, 102, 48, 51, 56, 53, 51, 51, 56, 53, 50, 54, 56, 50, 50, 53, 48, 100, 50, 49, 49, 100, 54, 56, 53, 52, 100, 98, 48, 53, 102, 101, 51, 55, 49], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_11_0) w_31_0 ∧
  MxxRuntime.concatColumns w_30_0 w_31_0 w_32_0 ∧
  0 ≤ (44) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : mxxMat1) => sequential_generatedRoot_33 backend hashModel tape (path ++ [33, i]) params i (current, (w_25_0, (w_14_0, (w_26_0, (w_29_0, (w_32_0, ())))))) next) (Int.toNat (44)) w_24_0 w_33_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  0 ≤ 0 ∧
  generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_11_0 w_13_0 w_18_0 w_20_0 w_25_0 w_26_0 w_38_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_81_0 w_82_0 w_84_0 w_85_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_109_0 w_130_0 w_131_0 w_133_0 w_134_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_158_0 w_179_0 w_180_0 w_182_0 w_183_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_4_0 w_5_0 w_6_0 w_12_0 w_12_1 w_34_0 w_35_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_59_0 w_62_0 w_63_0 w_64_0 w_65_0 w_65_1 w_65_2 w_65_3 w_65_4 w_65_5 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_67_0 w_67_1 w_67_2 w_67_3 w_67_4 w_67_5 w_68_0 w_69_0 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_83_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_117_0 w_118_0 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_132_0 w_135_0 w_136_0 w_137_0 w_138_0 w_140_0 w_141_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_166_0 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_181_0 w_184_0 w_185_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × (Fin 4 → Int) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 15488 → mxxMat6) × (Fin 15488 → mxxMat6) × (Fin 512 → Int) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat12 × mxxMat12 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : mxxMat0 := inputs.2.1
  let w_2_0 : mxxMat0 := inputs.2.2.1
  let w_3_0 : mxxMat0 := inputs.2.2.2.1
  let w_8_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_9_0 : mxxMat0 := inputs.2.2.2.2.2.1
  let w_11_0 : ByteArray := inputs.2.2.2.2.2.2.1
  let w_13_0 : Fin 4 → Int := inputs.2.2.2.2.2.2.2.1
  let w_18_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.1
  let w_20_0 : Fin 4 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_25_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_26_0 : Fin 15488 → mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_38_0 : Fin 512 → Int := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_47_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_49_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_51_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_53_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_55_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_56_0 : Fin 37 → mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_57_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_58_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_60_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_61_0 : Fin 85184 → mxxMat6 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_81_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_82_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_84_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_85_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_100_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_102_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_104_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_106_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_108_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_109_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_130_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_131_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_133_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_134_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_149_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_151_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_153_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_155_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_157_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_158_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_179_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_180_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_182_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_183_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_198_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_200_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_202_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_204_0 : mxxMat0 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_206_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_207_0 : mxxMat12 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_4_0 : mxxMat1 := (0 : mxxMat1)
  let w_5_0 : mxxMat1 := (0 : mxxMat1)
  let w_6_0 : mxxMat1 := (0 : mxxMat1)
  let w_7_0 : mxxMat1 := (0 : mxxMat1)
  let w_10_0 := witness.w_10_0
  let w_12_0 := witness.w_12_0
  let w_12_1 := witness.w_12_1
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_19_0 := witness.w_19_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : mxxMat1 := (0 : mxxMat1)
  let w_27_0 := witness.w_27_0
  let w_28_0 := witness.w_28_0
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat1 := MxxRuntime.matrixAdd w_23_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_37_0 : mxxMat2 := (0 : mxxMat2)
  let w_39_0 := witness.w_39_0
  let w_40_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_36_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_40_0 w_41_0
  let w_43_0 := witness.w_43_0
  let w_44_0 : mxxMat0 := MxxRuntime.matrixMul w_35_0 w_43_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat0 := MxxRuntime.matrixMul w_45_0 w_43_0
  let w_48_0 : mxxMat0 := MxxRuntime.matrixMul w_47_0 w_43_0
  let w_50_0 : mxxMat0 := MxxRuntime.matrixMul w_49_0 w_43_0
  let w_52_0 : mxxMat0 := MxxRuntime.matrixMul w_51_0 w_43_0
  let w_54_0 : mxxMat0 := MxxRuntime.matrixMul w_53_0 w_43_0
  let w_59_0 : mxxMat10 := (0 : mxxMat10)
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_65_1 := witness.w_65_1
  let w_65_2 := witness.w_65_2
  let w_65_3 := witness.w_65_3
  let w_65_4 := witness.w_65_4
  let w_65_5 := witness.w_65_5
  let w_66_0 := witness.w_66_0
  let w_66_1 := witness.w_66_1
  let w_66_2 := witness.w_66_2
  let w_66_3 := witness.w_66_3
  let w_66_4 := witness.w_66_4
  let w_66_5 := witness.w_66_5
  let w_67_0 := witness.w_67_0
  let w_67_1 := witness.w_67_1
  let w_67_2 := witness.w_67_2
  let w_67_3 := witness.w_67_3
  let w_67_4 := witness.w_67_4
  let w_67_5 := witness.w_67_5
  let w_68_0 : mxxMat15 := (0 : mxxMat15)
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_70_1 := witness.w_70_1
  let w_70_2 := witness.w_70_2
  let w_70_3 := witness.w_70_3
  let w_70_4 := witness.w_70_4
  let w_70_5 := witness.w_70_5
  let w_71_0 := witness.w_71_0
  let w_72_0 := witness.w_72_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 := witness.w_78_0
  let w_79_0 := witness.w_79_0
  let w_80_0 : mxxMat1 := (0 : mxxMat1)
  let w_83_0 := witness.w_83_0
  let w_86_0 := witness.w_86_0
  let w_87_0 := witness.w_87_0
  let w_88_0 : mxxMat1 := MxxRuntime.matrixAdd w_79_0 w_87_0
  let w_89_0 := witness.w_89_0
  let w_90_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_91_0 : mxxMat2 := (0 : mxxMat2)
  let w_92_0 := witness.w_92_0
  let w_93_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_90_0 w_92_0
  let w_94_0 := witness.w_94_0
  let w_95_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_93_0 w_94_0
  let w_96_0 := witness.w_96_0
  let w_97_0 : mxxMat0 := MxxRuntime.matrixMul w_89_0 w_96_0
  let w_98_0 := witness.w_98_0
  let w_99_0 : mxxMat0 := MxxRuntime.matrixMul w_98_0 w_96_0
  let w_101_0 : mxxMat0 := MxxRuntime.matrixMul w_100_0 w_96_0
  let w_103_0 : mxxMat0 := MxxRuntime.matrixMul w_102_0 w_96_0
  let w_105_0 : mxxMat0 := MxxRuntime.matrixMul w_104_0 w_96_0
  let w_107_0 : mxxMat0 := MxxRuntime.matrixMul w_106_0 w_96_0
  let w_110_0 : mxxMat10 := (0 : mxxMat10)
  let w_111_0 := witness.w_111_0
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_114_0 := witness.w_114_0
  let w_114_1 := witness.w_114_1
  let w_114_2 := witness.w_114_2
  let w_114_3 := witness.w_114_3
  let w_114_4 := witness.w_114_4
  let w_114_5 := witness.w_114_5
  let w_115_0 := witness.w_115_0
  let w_115_1 := witness.w_115_1
  let w_115_2 := witness.w_115_2
  let w_115_3 := witness.w_115_3
  let w_115_4 := witness.w_115_4
  let w_115_5 := witness.w_115_5
  let w_116_0 := witness.w_116_0
  let w_116_1 := witness.w_116_1
  let w_116_2 := witness.w_116_2
  let w_116_3 := witness.w_116_3
  let w_116_4 := witness.w_116_4
  let w_116_5 := witness.w_116_5
  let w_117_0 : mxxMat15 := (0 : mxxMat15)
  let w_118_0 := witness.w_118_0
  let w_119_0 := witness.w_119_0
  let w_119_1 := witness.w_119_1
  let w_119_2 := witness.w_119_2
  let w_119_3 := witness.w_119_3
  let w_119_4 := witness.w_119_4
  let w_119_5 := witness.w_119_5
  let w_120_0 := witness.w_120_0
  let w_121_0 := witness.w_121_0
  let w_122_0 := witness.w_122_0
  let w_123_0 := witness.w_123_0
  let w_124_0 := witness.w_124_0
  let w_125_0 := witness.w_125_0
  let w_126_0 := witness.w_126_0
  let w_127_0 := witness.w_127_0
  let w_128_0 := witness.w_128_0
  let w_129_0 : mxxMat1 := (0 : mxxMat1)
  let w_132_0 := witness.w_132_0
  let w_135_0 := witness.w_135_0
  let w_136_0 := witness.w_136_0
  let w_137_0 : mxxMat1 := MxxRuntime.matrixAdd w_128_0 w_136_0
  let w_138_0 := witness.w_138_0
  let w_139_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_140_0 : mxxMat2 := (0 : mxxMat2)
  let w_141_0 := witness.w_141_0
  let w_142_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_139_0 w_141_0
  let w_143_0 := witness.w_143_0
  let w_144_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_142_0 w_143_0
  let w_145_0 := witness.w_145_0
  let w_146_0 : mxxMat0 := MxxRuntime.matrixMul w_138_0 w_145_0
  let w_147_0 := witness.w_147_0
  let w_148_0 : mxxMat0 := MxxRuntime.matrixMul w_147_0 w_145_0
  let w_150_0 : mxxMat0 := MxxRuntime.matrixMul w_149_0 w_145_0
  let w_152_0 : mxxMat0 := MxxRuntime.matrixMul w_151_0 w_145_0
  let w_154_0 : mxxMat0 := MxxRuntime.matrixMul w_153_0 w_145_0
  let w_156_0 : mxxMat0 := MxxRuntime.matrixMul w_155_0 w_145_0
  let w_159_0 : mxxMat10 := (0 : mxxMat10)
  let w_160_0 := witness.w_160_0
  let w_161_0 := witness.w_161_0
  let w_162_0 := witness.w_162_0
  let w_163_0 := witness.w_163_0
  let w_163_1 := witness.w_163_1
  let w_163_2 := witness.w_163_2
  let w_163_3 := witness.w_163_3
  let w_163_4 := witness.w_163_4
  let w_163_5 := witness.w_163_5
  let w_164_0 := witness.w_164_0
  let w_164_1 := witness.w_164_1
  let w_164_2 := witness.w_164_2
  let w_164_3 := witness.w_164_3
  let w_164_4 := witness.w_164_4
  let w_164_5 := witness.w_164_5
  let w_165_0 := witness.w_165_0
  let w_165_1 := witness.w_165_1
  let w_165_2 := witness.w_165_2
  let w_165_3 := witness.w_165_3
  let w_165_4 := witness.w_165_4
  let w_165_5 := witness.w_165_5
  let w_166_0 : mxxMat15 := (0 : mxxMat15)
  let w_167_0 := witness.w_167_0
  let w_168_0 := witness.w_168_0
  let w_168_1 := witness.w_168_1
  let w_168_2 := witness.w_168_2
  let w_168_3 := witness.w_168_3
  let w_168_4 := witness.w_168_4
  let w_168_5 := witness.w_168_5
  let w_169_0 := witness.w_169_0
  let w_170_0 := witness.w_170_0
  let w_171_0 := witness.w_171_0
  let w_172_0 := witness.w_172_0
  let w_173_0 := witness.w_173_0
  let w_174_0 := witness.w_174_0
  let w_175_0 := witness.w_175_0
  let w_176_0 := witness.w_176_0
  let w_177_0 := witness.w_177_0
  let w_178_0 : mxxMat1 := (0 : mxxMat1)
  let w_181_0 := witness.w_181_0
  let w_184_0 := witness.w_184_0
  let w_185_0 := witness.w_185_0
  let w_186_0 : mxxMat1 := MxxRuntime.matrixAdd w_177_0 w_185_0
  let w_187_0 := witness.w_187_0
  let w_188_0 : mxxMat2 := (MxxRuntime.matrixPolynomial [(mxxNat2)] : mxxMat2)
  let w_189_0 : mxxMat2 := (0 : mxxMat2)
  let w_190_0 := witness.w_190_0
  let w_191_0 : mxxMat2 := MxxRuntime.matrixMulScalarLeft w_188_0 w_190_0
  let w_192_0 := witness.w_192_0
  let w_193_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_191_0 w_192_0
  let w_194_0 := witness.w_194_0
  let w_195_0 : mxxMat0 := MxxRuntime.matrixMul w_187_0 w_194_0
  let w_196_0 := witness.w_196_0
  let w_197_0 : mxxMat0 := MxxRuntime.matrixMul w_196_0 w_194_0
  let w_199_0 : mxxMat0 := MxxRuntime.matrixMul w_198_0 w_194_0
  let w_201_0 : mxxMat0 := MxxRuntime.matrixMul w_200_0 w_194_0
  let w_203_0 : mxxMat0 := MxxRuntime.matrixMul w_202_0 w_194_0
  let w_205_0 : mxxMat0 := MxxRuntime.matrixMul w_204_0 w_194_0
  let w_208_0 : mxxMat10 := (0 : mxxMat10)
  let w_209_0 := witness.w_209_0
  let w_210_0 := witness.w_210_0
  let w_211_0 := witness.w_211_0
  let w_212_0 := witness.w_212_0
  let w_212_1 := witness.w_212_1
  let w_212_2 := witness.w_212_2
  let w_212_3 := witness.w_212_3
  let w_212_4 := witness.w_212_4
  let w_212_5 := witness.w_212_5
  let w_213_0 := witness.w_213_0
  let w_213_1 := witness.w_213_1
  let w_213_2 := witness.w_213_2
  let w_213_3 := witness.w_213_3
  let w_213_4 := witness.w_213_4
  let w_213_5 := witness.w_213_5
  let w_214_0 := witness.w_214_0
  let w_214_1 := witness.w_214_1
  let w_214_2 := witness.w_214_2
  let w_214_3 := witness.w_214_3
  let w_214_4 := witness.w_214_4
  let w_214_5 := witness.w_214_5
  let w_215_0 : mxxMat15 := (0 : mxxMat15)
  let w_216_0 := witness.w_216_0
  let w_217_0 := witness.w_217_0
  let w_217_1 := witness.w_217_1
  let w_217_2 := witness.w_217_2
  let w_217_3 := witness.w_217_3
  let w_217_4 := witness.w_217_4
  let w_217_5 := witness.w_217_5
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_8_0 w_9_0 w_11_0 w_13_0 w_18_0 w_20_0 w_25_0 w_26_0 w_38_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_56_0 w_57_0 w_58_0 w_60_0 w_61_0 w_81_0 w_82_0 w_84_0 w_85_0 w_100_0 w_102_0 w_104_0 w_106_0 w_108_0 w_109_0 w_130_0 w_131_0 w_133_0 w_134_0 w_149_0 w_151_0 w_153_0 w_155_0 w_157_0 w_158_0 w_179_0 w_180_0 w_182_0 w_183_0 w_198_0 w_200_0 w_202_0 w_204_0 w_206_0 w_207_0 w_4_0 w_5_0 w_6_0 w_7_0 w_10_0 w_12_0 w_12_1 w_14_0 w_15_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_37_0 w_39_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_59_0 w_62_0 w_63_0 w_64_0 w_65_0 w_65_1 w_65_2 w_65_3 w_65_4 w_65_5 w_66_0 w_66_1 w_66_2 w_66_3 w_66_4 w_66_5 w_67_0 w_67_1 w_67_2 w_67_3 w_67_4 w_67_5 w_68_0 w_69_0 w_70_0 w_70_1 w_70_2 w_70_3 w_70_4 w_70_5 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_83_0 w_86_0 w_87_0 w_88_0 w_89_0 w_91_0 w_92_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_101_0 w_103_0 w_105_0 w_107_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_114_1 w_114_2 w_114_3 w_114_4 w_114_5 w_115_0 w_115_1 w_115_2 w_115_3 w_115_4 w_115_5 w_116_0 w_116_1 w_116_2 w_116_3 w_116_4 w_116_5 w_117_0 w_118_0 w_119_0 w_119_1 w_119_2 w_119_3 w_119_4 w_119_5 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_132_0 w_135_0 w_136_0 w_137_0 w_138_0 w_140_0 w_141_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_150_0 w_152_0 w_154_0 w_156_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_163_1 w_163_2 w_163_3 w_163_4 w_163_5 w_164_0 w_164_1 w_164_2 w_164_3 w_164_4 w_164_5 w_165_0 w_165_1 w_165_2 w_165_3 w_165_4 w_165_5 w_166_0 w_167_0 w_168_0 w_168_1 w_168_2 w_168_3 w_168_4 w_168_5 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_181_0 w_184_0 w_185_0 w_186_0 w_187_0 w_189_0 w_190_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_199_0 w_201_0 w_203_0 w_205_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_212_1 w_212_2 w_212_3 w_212_4 w_212_5 w_213_0 w_213_1 w_213_2 w_213_3 w_213_4 w_213_5 w_214_0 w_214_1 w_214_2 w_214_3 w_214_4 w_214_5 w_215_0 w_216_0 w_217_0 w_217_1 w_217_2 w_217_3 w_217_4 w_217_5 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × ByteArray × (Fin 4 → Int) × (Fin 4 → mxxMat0) × (Fin 4 → mxxMat0) × (Fin 15488 → mxxMat6) × (Fin 15488 → mxxMat6) × (Fin 512 → Int) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × (Fin 37 → mxxMat0) × (Fin 37 → mxxMat0) × mxxMat12 × mxxMat12 × (Fin 85184 → mxxMat6) × (Fin 85184 → mxxMat6) × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat12 × mxxMat12 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat12 × mxxMat12 × Unit) (outputs : mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × mxxMat0 × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_online_eval
