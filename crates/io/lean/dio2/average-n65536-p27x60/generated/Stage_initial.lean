import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_initial

structure Params where
  unit : Unit

local notation "mxxNat0" => 46622222787362522255596712427438640158977719430215665481763441408391999111509151608999934968273383011642417378633647840562514917807804816922347656847676979519702813451900357865140701070628284347817487928891263004147023796456197114198655078174334397712252374369987792302056981468300367512375728844419128305889690907960192125350868940719263879317724644989970506210974573641462431438389817409111571168135085524040305727541538565027353777260752527223173138698768973043853522644067239267926017

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 270
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 544
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 65536 1 2
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 65536 2 544

abbrev parallel_generatedRoot_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(103257)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(103257)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_10.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(103257)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_11.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_12.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(107063)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(107063)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_12.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(107063)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_13.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_14.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(10725)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(10725)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_14.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_15.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(10725)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_15.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_16.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(109395)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(109395)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_16.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_17 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(109395)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_17.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_18.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(10965)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(10965)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_18.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_19.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_19 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(10965)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_19.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_20.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1105)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_20 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1105)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_20.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_21.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_21 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1105)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_21.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_22.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(110877)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_22 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(110877)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_22.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(110877)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_23.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_24.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11473)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_24 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11473)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_24.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_25.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_25 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11473)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_25.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_26.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(114939)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_26 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(114939)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_26.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_27.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(114939)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_27.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_28.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(115181)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_28 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(115181)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_28.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_29.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_29 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(115181)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_29.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_30.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1161)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_30 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1161)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_30.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_31.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_31 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1161)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_31.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_32.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(116465)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_32 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(116465)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_32.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_33.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_33 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(116465)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_33.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_34.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(118955)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_34 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(118955)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_34.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_35.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_35 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(118955)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_35.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_36.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(123069)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_36 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(123069)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_36.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_37.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_37 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(123069)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_37.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_38.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(125207)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_38 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(125207)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_38.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_39.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_39 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(125207)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_39.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_40.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(125775)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_40 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(125775)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_40.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_41.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_41 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(125775)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_41.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_42.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(127033)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_42 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(127033)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_42.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_43.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_43 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(127033)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_43.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_44.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(12771)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_44 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(12771)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_44.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_45.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_45 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(12771)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_45.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_46.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(129)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_46 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(129)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_46.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_47.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_47 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(129)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_47.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_48.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(129117)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_48 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(129117)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_48.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_49.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_49 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(129117)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_49.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_50.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(129899)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_50 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(129899)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_50.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_51.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_51 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(129899)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_51.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_52.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(130681)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_52 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(130681)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_52.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_53.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_53 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(130681)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_53.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_54.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(130727)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_54 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(130727)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_54.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_55.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_55 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(130727)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_55.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_56.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(130957)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_56 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(130957)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_56.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_57.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_57 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(130957)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_57.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_58.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(131003)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_58 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(131003)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_58.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_59.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_59 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(131003)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_59.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_60.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(131049)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_60 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(131049)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_60.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_61.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_61 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(131049)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_61.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_62.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(135)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_62 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(135)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_62.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_63.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_63 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(135)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_63.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_64.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(14561)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_64 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(14561)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_64.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_65.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_65 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(14561)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_65.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_66.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1485)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_66 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1485)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_66.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_67.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_67 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1485)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_67.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_68.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_68 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_68.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_69.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_69 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_69.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_70.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15147)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_70 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15147)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_70.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_71.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_71 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(15147)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_71.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_72.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(153)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_72 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(153)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_72.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_73.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_73 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(153)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_73.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_74.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16385)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_74 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(16385)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_74.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_75.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_75 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(16385)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_75.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_76.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(165)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_76 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(165)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_76.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_77.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_77 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(165)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_77.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_78.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(16575)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_78 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(16575)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_78.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_79.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_79 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(16575)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_79.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_80.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1683)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_80 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1683)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_80.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_81.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_81 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1683)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_81.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_82.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_82 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(17)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_82.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_83.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_83 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(17)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_83.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_84.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(17415)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_84 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(17415)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_84.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_85.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_85 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(17415)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_85.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_86.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1755)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_86 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1755)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_86.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_87.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_87 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1755)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_87.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_88.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(18103)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_88 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(18103)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_88.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_89.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_89 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(18103)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_89.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_90.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(19305)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_90 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(19305)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_90.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_91.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_91 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(19305)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_91.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_92.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1935)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_92 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1935)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_92.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_93.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_93 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1935)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_93.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_94.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(195)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_94 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(195)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_94.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_95.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_95 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(195)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_95.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_96.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(19737)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_96 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(19737)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_96.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_97.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_97 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(19737)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_97.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_98.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(21285)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_98 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(21285)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_98.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_99.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_99 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(21285)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_99.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_100.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(2145)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_100 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2145)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_100.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_101.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_101 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2145)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_101.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_102.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(2193)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_102 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2193)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_102.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_103.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_103 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2193)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_103.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_104.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(22687)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_104 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(22687)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_104.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_105.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_105 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(22687)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_105.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_106.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(2295)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_106 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2295)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_106.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_107.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_107 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2295)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_107.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_108.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(25155)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_108 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(25155)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_108.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_109.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_109 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(25155)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_109.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_110.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(25245)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_110 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(25245)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_110.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_111.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_111 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(25245)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_111.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_112.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(255)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_112 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(255)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_112.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_113.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_113 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(255)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_113.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_114.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(27)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_114 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(27)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_114.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_115.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_115 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(27)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_115.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_116.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(27431)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_116 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(27431)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_116.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_117.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_117 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(27431)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_117.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_118.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(2805)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_118 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2805)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_118.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_119.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_119 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2805)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_119.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_120.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(2925)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_120 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2925)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_120.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_121.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_121 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(2925)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_121.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_122.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(297)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_122 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(297)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_122.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_123.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_123 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(297)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_123.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_124.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(29835)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_124 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(29835)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_124.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_125.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_125 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(29835)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_125.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_126.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_126 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_126.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_127.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_127 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_127.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_128.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(32175)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_128 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(32175)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_128.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_129.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_129 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(32175)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_129.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_130.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(325)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_130 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(325)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_130.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_131.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_131 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(325)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_131.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_132.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(32769)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_132 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(32769)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_132.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_133.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_133 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(32769)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_133.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_134.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(32895)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_134 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(32895)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_134.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_135.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_135 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(32895)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_135.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_136.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(33)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_136 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(33)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_136.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_137.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_137 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(33)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_137.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_138.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3315)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_138 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3315)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_138.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_139.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_139 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3315)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_139.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_140.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(33911)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_140 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(33911)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_140.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_141.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_141 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(33911)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_141.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_142.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(34419)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_142 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(34419)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_142.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_143.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_143 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(34419)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_143.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_144.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3483)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_144 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3483)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_144.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_145.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_145 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(3483)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_145.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_146.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(36465)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_146 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36465)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_146.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_147.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_147 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36465)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_147.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_148.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(36959)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_148 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36959)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_148.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_149.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_149 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(36959)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_149.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_150.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(38313)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_150 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(38313)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_150.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_151.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_151 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(38313)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_151.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_152.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(387)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_152 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(387)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_152.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_153.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_153 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(387)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_153.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_154.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(41023)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_154 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(41023)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_154.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_155.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_155 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(41023)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_155.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_156.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(41925)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_156 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(41925)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_156.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_157.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_157 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(41925)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_157.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_158.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(4257)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_158 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(4257)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_158.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_159.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_159 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(4257)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_159.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_160.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(43039)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_160 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(43039)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_160.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_161.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_161 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(43039)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_161.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_162.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(43683)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_162 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(43683)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_162.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_163.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_163 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(43683)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_163.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_164.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(4455)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_164 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(4455)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_164.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_165.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_165 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(4455)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_165.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_166.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(45)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_166 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(45)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_166.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_167.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_167 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(45)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_167.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_168.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(459)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_168 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(459)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_168.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_169.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_169 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(459)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_169.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_170.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(47627)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_170 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(47627)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_170.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_171.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_171 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(47627)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_171.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_172.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(49153)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_172 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(49153)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_172.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_173.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_173 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(49153)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_173.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_174.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(495)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_174 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(495)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_174.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_175.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_175 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(495)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_175.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_176.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(49725)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_176 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(49725)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_176.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_177.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_177 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(49725)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_177.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_178.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_178 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_178.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_179.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_179 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_179.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_180.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(50407)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_180 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(50407)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_180.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_181.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_181 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(50407)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_181.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_182.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5049)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_182 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5049)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_182.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_183.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_183 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5049)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_183.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_184.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(51)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_184 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(51)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_184.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_185.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_185 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(51)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_185.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_186.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(51253)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_186 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(51253)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_186.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_187.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_187 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(51253)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_187.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_188.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5525)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_188 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5525)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_188.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_189.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_189 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5525)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_189.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_190.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(561)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_190 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(561)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_190.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_191.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_191 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(561)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_191.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_192.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(57365)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_192 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(57365)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_192.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_193.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_193 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(57365)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_193.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_194.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(57915)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_194 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(57915)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_194.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_195.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_195 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(57915)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_195.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_196.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(58037)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_196 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(58037)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_196.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_197.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_197 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(58037)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_197.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_198.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5805)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_198 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5805)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_198.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_199.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_199 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(5805)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_199.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_200.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(585)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_200 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(585)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_200.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_201.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_201 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(585)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_201.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_202.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(59211)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_202 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(59211)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_202.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_203.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_203 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(59211)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_203.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_204.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(60493)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_204 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(60493)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_204.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_205.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_205 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(60493)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_205.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_206.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(63855)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_206 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(63855)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_206.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_207.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_207 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(63855)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_207.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_208.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(6435)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_208 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(6435)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_208.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_209.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_209 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(6435)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_209.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_210.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(645)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_210 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(645)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_210.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_211.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_211 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(645)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_211.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_212.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(65)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_212 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(65)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_212.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_213.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_213 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(65)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_213.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_214.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(6579)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_214 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(6579)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_214.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_215.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_215 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(6579)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_215.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_216.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(66041)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_216 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(66041)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_216.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_217.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_217 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(66041)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_217.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_218.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(67051)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_218 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(67051)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_218.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_219.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_219 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(67051)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_219.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_220.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(68061)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_220 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(68061)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_220.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_221.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_221 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(68061)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_221.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_222.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(70487)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_222 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(70487)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_222.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_223.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_223 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(70487)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_223.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_224.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(72369)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_224 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(72369)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_224.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_225.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_225 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(72369)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_225.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_226.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(72805)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_226 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(72805)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_226.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_227.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_227 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(72805)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_227.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_228.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(73111)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_228 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(73111)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_228.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_229.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_229 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(73111)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_229.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_230.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(75465)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_230 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(75465)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_230.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_231.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_231 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(75465)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_231.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_232.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(75735)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_232 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(75735)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_232.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_233.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_233 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(75735)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_233.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_234.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(765)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_234 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(765)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_234.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_235.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_235 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(765)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_235.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_236.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(83399)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_236 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(83399)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_236.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_237.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_237 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(83399)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_237.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_238.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(8385)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_238 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(8385)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_238.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_239.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_239 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(8385)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_239.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_240.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(8415)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_240 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(8415)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_240.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_241.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_241 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(8415)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_241.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_242.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(85)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_242 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(85)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_242.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_243.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_243 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(85)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_243.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_244.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(86035)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_244 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(86035)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_244.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_245.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_245 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(86035)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_245.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_246.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(87251)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_246 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(87251)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_246.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_247.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_247 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(87251)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_247.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_248.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(87343)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_248 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(87343)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_248.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_249.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_249 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(87343)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_249.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_250.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(8775)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_250 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(8775)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_250.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_251.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_251 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(8775)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_251.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_252.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(891)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_252 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(891)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_252.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_253.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_253 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(891)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_253.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_254.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_254 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_254.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_255.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_255 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_255.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_256.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(95323)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_256 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(95323)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_256.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_257.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_257 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(95323)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_257.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_258.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(96525)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_258 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(96525)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_258.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_259.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_259 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(96525)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_259.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_260.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(975)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_260 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(975)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_260.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_261.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_261 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(975)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_261.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_262.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(98685)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_262 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(98685)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_262.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_263.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_263 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(98685)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_263.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_264.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(99)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_264 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(99)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_264.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_265.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_265 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(99)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_265.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_266.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9945)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_266 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9945)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_266.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_267.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_267 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(9945)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_267.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_268.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(99701)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_268 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(99701)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_268.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_269.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_269 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(99701)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_269.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_270.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (16385), (32769), (49153)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_270 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (16385), (32769), (49153)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_270.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_271.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_271 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (16385), (32769), (49153)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_271.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_272.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 32 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_272 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_272.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_273.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 32 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_273 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (85), (99), (135), (153), (165), (255), (297), (459), (495), (561), (765), (891), (1485), (1683), (2295), (2805), (4455), (5049), (8415), (15147), (25245), (75735)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_273.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_274.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 128 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_274 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      ∃ (w_9_0 : mxxMat0),
        ∃ (w_10_0 : mxxMat1),
          let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
          let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
          ∃ (w_13_0 : mxxMat1),
            ∃ (w_14_0 : mxxMat1),
              let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
              let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
              parallel_generatedRoot_274.constraints_0 backend hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_275.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 128 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_10_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [13]) (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [14]) (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_275 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (9), (15), (17), (27), (33), (45), (51), (65), (85), (99), (129), (135), (153), (165), (195), (255), (297), (325), (387), (459), (495), (561), (585), (645), (765), (891), (975), (1105), (1161), (1485), (1683), (1755), (1935), (2145), (2193), (2295), (2805), (2925), (3315), (3483), (4257), (4455), (5049), (5525), (5805), (6435), (6579), (8385), (8415), (8775), (9945), (10725), (10965), (11473), (12771), (14561), (15147), (16575), (17415), (18103), (19305), (19737), (21285), (22687), (25155), (25245), (27431), (29835), (32175), (32895), (33911), (34419), (36465), (36959), (38313), (41023), (41925), (43039), (43683), (47627), (49725), (50407), (51253), (57365), (57915), (58037), (59211), (60493), (63855), (66041), (67051), (68061), (70487), (72369), (72805), (73111), (75465), (75735), (83399), (86035), (87251), (87343), (95323), (96525), (98685), (99701), (103257), (107063), (109395), (110877), (114939), (115181), (116465), (118955), (123069), (125207), (125775), (127033), (129117), (129899), (130681), (130727), (130957), (131003), (131049)])
  ∃ (w_3_0 : mxxMat1),
    let w_4_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_3_0
    ∃ (w_6_0 : mxxMat1),
      let w_7_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
      let w_8_0 : mxxMat1 := MxxRuntime.matrixAdd w_4_0 w_7_0
      let w_9_0 : mxxMat0 := (1 : mxxMat0)
      ∃ (w_10_0 : mxxMat1),
        let w_11_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_9_0 w_10_0
        let w_12_0 : mxxMat1 := MxxRuntime.matrixSub w_8_0 w_11_0
        ∃ (w_13_0 : mxxMat1),
          ∃ (w_14_0 : mxxMat1),
            let w_15_0 : mxxMat1 := MxxRuntime.matrixSub w_13_0 w_14_0
            let w_16_0 : mxxMat1 := MxxRuntime.matrixAdd w_12_0 w_15_0
            parallel_generatedRoot_275.constraints_0 backend hashModel tape path i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_277.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_13_0 : Fin 8389632 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_12_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_24_0 : Int) (w_27_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_33_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  w_14_0 ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_17_0 w_18_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_20_0, w_23_0] w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_27_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [30]) (0) (1) w_30_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [31]) (0) (1) w_31_0 ∧
  outputs = w_33_0

def parallel_generatedRoot_277 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 8389632 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 8389632 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_9_0 : Int := (Int.ofNat i_0)
      let w_10_0 : Int := 1024
      let w_11_0 : Bool := decide (w_9_0 = w_10_0)
      let w_12_0 : Int := if w_11_0 then 1 else 0
      let w_14_0 : Int := 1024
      let w_15_0 : Int := (w_9_0 % w_14_0)
      let w_16_0 : Int := 8193
      let w_17_0 : Int := (w_15_0 * w_16_0)
      ∃ (w_18_0 : Int),
        let w_19_0 : Int := 512
        let w_20_0 : Int := (w_18_0 * w_19_0)
        let w_21_0 : Int := 0
        let w_22_0 : Int := 0
        let w_23_0 : Int := (w_21_0 + w_22_0)
        ∃ (w_24_0 : Int),
          let w_25_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_8_0 w_24_0
          let w_26_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_25_0 w_0_0
          ∃ (w_27_0 : mxxMat1),
            let w_28_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
            let w_29_0 : mxxMat1 := MxxRuntime.matrixSub w_7_0 w_28_0
            ∃ (w_30_0 : mxxMat1),
              ∃ (w_31_0 : mxxMat1),
                let w_32_0 : mxxMat1 := MxxRuntime.matrixSub w_30_0 w_31_0
                let w_33_0 : mxxMat1 := MxxRuntime.matrixAdd w_29_0 w_32_0
                parallel_generatedRoot_277.constraints_0 backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs

abbrev parallel_generatedRoot_278.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_13_0 : Fin 8389632 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_12_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_24_0 : Int) (w_27_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_33_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_5_0 ∧
  w_14_0 ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_17_0 w_18_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_20_0, w_23_0] w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_27_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [30]) (0) (1) w_30_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [31]) (0) (1) w_31_0 ∧
  outputs = w_33_0

def parallel_generatedRoot_278 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 8389632 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 8389632 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_9_0 : Int := (Int.ofNat i_0)
      let w_10_0 : Int := 1024
      let w_11_0 : Bool := decide (w_9_0 = w_10_0)
      let w_12_0 : Int := if w_11_0 then 1 else 0
      let w_14_0 : Int := 1024
      let w_15_0 : Int := (w_9_0 % w_14_0)
      let w_16_0 : Int := 8193
      let w_17_0 : Int := (w_15_0 * w_16_0)
      ∃ (w_18_0 : Int),
        let w_19_0 : Int := 512
        let w_20_0 : Int := (w_18_0 * w_19_0)
        let w_21_0 : Int := 0
        let w_22_0 : Int := 0
        let w_23_0 : Int := (w_21_0 + w_22_0)
        ∃ (w_24_0 : Int),
          let w_25_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_8_0 w_24_0
          let w_26_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_25_0 w_4_0
          ∃ (w_27_0 : mxxMat1),
            let w_28_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_26_0 w_27_0
            let w_29_0 : mxxMat1 := MxxRuntime.matrixSub w_7_0 w_28_0
            ∃ (w_30_0 : mxxMat1),
              ∃ (w_31_0 : mxxMat1),
                let w_32_0 : mxxMat1 := MxxRuntime.matrixSub w_30_0 w_31_0
                let w_33_0 : mxxMat1 := MxxRuntime.matrixAdd w_29_0 w_32_0
                parallel_generatedRoot_278.constraints_0 backend hashModel tape path i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs

abbrev parallel_generatedRoot_279.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_9_0 : Fin 8389632 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_11_0 : Int) (w_15_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_25_0 : mxxMat1) (w_28_0 : mxxMat1) (w_29_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  w_11_0 ≠ 0 ∧
  w_15_0 ≠ 0 ∧
  0 ≤ w_19_0 ∧
  w_19_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_19_0 w_20_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_25_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [28]) (0) (1) w_28_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [29]) (0) (1) w_29_0 ∧
  outputs = w_31_0

def parallel_generatedRoot_279 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 8389632 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_9_0 : Fin 8389632 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_10_0 : Int := (Int.ofNat i_0)
      let w_11_0 : Int := 8192
      let w_12_0 : Int := (w_10_0 / w_11_0)
      let w_13_0 : Int := 8193
      let w_14_0 : Int := (w_12_0 * w_13_0)
      let w_15_0 : Int := 8192
      let w_16_0 : Int := (w_10_0 % w_15_0)
      let w_17_0 : Int := (w_14_0 + w_16_0)
      let w_18_0 : Int := 1
      let w_19_0 : Int := (w_17_0 + w_18_0)
      ∃ (w_20_0 : Int),
        let w_21_0 : Int := 512
        let w_22_0 : Int := (w_20_0 * w_21_0)
        let w_23_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_8_0 w_22_0
        let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_23_0 w_0_0
        ∃ (w_25_0 : mxxMat1),
          let w_26_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
          let w_27_0 : mxxMat1 := MxxRuntime.matrixSub w_7_0 w_26_0
          ∃ (w_28_0 : mxxMat1),
            ∃ (w_29_0 : mxxMat1),
              let w_30_0 : mxxMat1 := MxxRuntime.matrixSub w_28_0 w_29_0
              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_27_0 w_30_0
              parallel_generatedRoot_279.constraints_0 backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs

abbrev parallel_generatedRoot_280.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_9_0 : Fin 8389632 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_11_0 : Int) (w_15_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_25_0 : mxxMat1) (w_28_0 : mxxMat1) (w_29_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_5_0 ∧
  w_11_0 ≠ 0 ∧
  w_15_0 ≠ 0 ∧
  0 ≤ w_19_0 ∧
  w_19_0 < 8389632 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_19_0 w_20_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_25_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [28]) (0) (1) w_28_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [29]) (0) (1) w_29_0 ∧
  outputs = w_31_0

def parallel_generatedRoot_280 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 8389632 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_9_0 : Fin 8389632 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_10_0 : Int := (Int.ofNat i_0)
      let w_11_0 : Int := 8192
      let w_12_0 : Int := (w_10_0 / w_11_0)
      let w_13_0 : Int := 8193
      let w_14_0 : Int := (w_12_0 * w_13_0)
      let w_15_0 : Int := 8192
      let w_16_0 : Int := (w_10_0 % w_15_0)
      let w_17_0 : Int := (w_14_0 + w_16_0)
      let w_18_0 : Int := 1
      let w_19_0 : Int := (w_17_0 + w_18_0)
      ∃ (w_20_0 : Int),
        let w_21_0 : Int := 512
        let w_22_0 : Int := (w_20_0 * w_21_0)
        let w_23_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_8_0 w_22_0
        let w_24_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_23_0 w_4_0
        ∃ (w_25_0 : mxxMat1),
          let w_26_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_24_0 w_25_0
          let w_27_0 : mxxMat1 := MxxRuntime.matrixSub w_7_0 w_26_0
          ∃ (w_28_0 : mxxMat1),
            ∃ (w_29_0 : mxxMat1),
              let w_30_0 : mxxMat1 := MxxRuntime.matrixSub w_28_0 w_29_0
              let w_31_0 : mxxMat1 := MxxRuntime.matrixAdd w_27_0 w_30_0
              parallel_generatedRoot_280.constraints_0 backend hashModel tape path i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs

abbrev generatedRoot.OutputType : Type :=
  (mxxMat2 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × Unit) × ((Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 32 → mxxMat1) × (Fin 32 → mxxMat1) × (Fin 128 → mxxMat1) × (Fin 128 → mxxMat1) × (Fin 1025 → mxxMat1) × (Fin 1025 → mxxMat1) × (Fin 8388608 → mxxMat1) × (Fin 8388608 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit)

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : mxxMat3
  w_5_0 : mxxMat2
  w_6_0 : mxxMat2
  w_10_0 : Fin 1 → mxxMat1
  w_11_0 : Fin 1 → mxxMat1
  w_12_0 : Fin 1 → mxxMat1
  w_13_0 : Fin 1 → mxxMat1
  w_14_0 : Fin 1 → mxxMat1
  w_15_0 : Fin 1 → mxxMat1
  w_16_0 : Fin 1 → mxxMat1
  w_17_0 : Fin 1 → mxxMat1
  w_18_0 : Fin 1 → mxxMat1
  w_19_0 : Fin 1 → mxxMat1
  w_20_0 : Fin 1 → mxxMat1
  w_21_0 : Fin 1 → mxxMat1
  w_22_0 : Fin 1 → mxxMat1
  w_23_0 : Fin 1 → mxxMat1
  w_24_0 : Fin 1 → mxxMat1
  w_25_0 : Fin 1 → mxxMat1
  w_26_0 : Fin 1 → mxxMat1
  w_27_0 : Fin 1 → mxxMat1
  w_28_0 : Fin 1 → mxxMat1
  w_29_0 : Fin 1 → mxxMat1
  w_30_0 : Fin 1 → mxxMat1
  w_31_0 : Fin 1 → mxxMat1
  w_32_0 : Fin 1 → mxxMat1
  w_33_0 : Fin 1 → mxxMat1
  w_34_0 : Fin 1 → mxxMat1
  w_35_0 : Fin 1 → mxxMat1
  w_36_0 : Fin 1 → mxxMat1
  w_37_0 : Fin 1 → mxxMat1
  w_38_0 : Fin 1 → mxxMat1
  w_39_0 : Fin 1 → mxxMat1
  w_40_0 : Fin 1 → mxxMat1
  w_41_0 : Fin 1 → mxxMat1
  w_42_0 : Fin 1 → mxxMat1
  w_43_0 : Fin 1 → mxxMat1
  w_44_0 : Fin 1 → mxxMat1
  w_45_0 : Fin 1 → mxxMat1
  w_46_0 : Fin 1 → mxxMat1
  w_47_0 : Fin 1 → mxxMat1
  w_48_0 : Fin 1 → mxxMat1
  w_49_0 : Fin 1 → mxxMat1
  w_50_0 : Fin 1 → mxxMat1
  w_51_0 : Fin 1 → mxxMat1
  w_52_0 : Fin 1 → mxxMat1
  w_53_0 : Fin 1 → mxxMat1
  w_54_0 : Fin 1 → mxxMat1
  w_55_0 : Fin 1 → mxxMat1
  w_56_0 : Fin 1 → mxxMat1
  w_57_0 : Fin 1 → mxxMat1
  w_58_0 : Fin 1 → mxxMat1
  w_59_0 : Fin 1 → mxxMat1
  w_60_0 : Fin 1 → mxxMat1
  w_61_0 : Fin 1 → mxxMat1
  w_62_0 : Fin 1 → mxxMat1
  w_63_0 : Fin 1 → mxxMat1
  w_64_0 : Fin 1 → mxxMat1
  w_65_0 : Fin 1 → mxxMat1
  w_66_0 : Fin 1 → mxxMat1
  w_67_0 : Fin 1 → mxxMat1
  w_68_0 : Fin 1 → mxxMat1
  w_69_0 : Fin 1 → mxxMat1
  w_70_0 : Fin 1 → mxxMat1
  w_71_0 : Fin 1 → mxxMat1
  w_72_0 : Fin 1 → mxxMat1
  w_73_0 : Fin 1 → mxxMat1
  w_74_0 : Fin 1 → mxxMat1
  w_75_0 : Fin 1 → mxxMat1
  w_76_0 : Fin 1 → mxxMat1
  w_77_0 : Fin 1 → mxxMat1
  w_78_0 : Fin 1 → mxxMat1
  w_79_0 : Fin 1 → mxxMat1
  w_80_0 : Fin 1 → mxxMat1
  w_81_0 : Fin 1 → mxxMat1
  w_82_0 : Fin 1 → mxxMat1
  w_83_0 : Fin 1 → mxxMat1
  w_84_0 : Fin 1 → mxxMat1
  w_85_0 : Fin 1 → mxxMat1
  w_86_0 : Fin 1 → mxxMat1
  w_87_0 : Fin 1 → mxxMat1
  w_88_0 : Fin 1 → mxxMat1
  w_89_0 : Fin 1 → mxxMat1
  w_90_0 : Fin 1 → mxxMat1
  w_91_0 : Fin 1 → mxxMat1
  w_92_0 : Fin 1 → mxxMat1
  w_93_0 : Fin 1 → mxxMat1
  w_94_0 : Fin 1 → mxxMat1
  w_95_0 : Fin 1 → mxxMat1
  w_96_0 : Fin 1 → mxxMat1
  w_97_0 : Fin 1 → mxxMat1
  w_98_0 : Fin 1 → mxxMat1
  w_99_0 : Fin 1 → mxxMat1
  w_100_0 : Fin 1 → mxxMat1
  w_101_0 : Fin 1 → mxxMat1
  w_102_0 : Fin 1 → mxxMat1
  w_103_0 : Fin 1 → mxxMat1
  w_104_0 : Fin 1 → mxxMat1
  w_105_0 : Fin 1 → mxxMat1
  w_106_0 : Fin 1 → mxxMat1
  w_107_0 : Fin 1 → mxxMat1
  w_108_0 : Fin 1 → mxxMat1
  w_109_0 : Fin 1 → mxxMat1
  w_110_0 : Fin 1 → mxxMat1
  w_111_0 : Fin 1 → mxxMat1
  w_112_0 : Fin 1 → mxxMat1
  w_113_0 : Fin 1 → mxxMat1
  w_114_0 : Fin 1 → mxxMat1
  w_115_0 : Fin 1 → mxxMat1
  w_116_0 : Fin 1 → mxxMat1
  w_117_0 : Fin 1 → mxxMat1
  w_118_0 : Fin 1 → mxxMat1
  w_119_0 : Fin 1 → mxxMat1
  w_120_0 : Fin 1 → mxxMat1
  w_121_0 : Fin 1 → mxxMat1
  w_122_0 : Fin 1 → mxxMat1
  w_123_0 : Fin 1 → mxxMat1
  w_124_0 : Fin 1 → mxxMat1
  w_125_0 : Fin 1 → mxxMat1
  w_126_0 : Fin 1 → mxxMat1
  w_127_0 : Fin 1 → mxxMat1
  w_128_0 : Fin 1 → mxxMat1
  w_129_0 : Fin 1 → mxxMat1
  w_130_0 : Fin 1 → mxxMat1
  w_131_0 : Fin 1 → mxxMat1
  w_132_0 : Fin 1 → mxxMat1
  w_133_0 : Fin 1 → mxxMat1
  w_134_0 : Fin 1 → mxxMat1
  w_135_0 : Fin 1 → mxxMat1
  w_136_0 : Fin 1 → mxxMat1
  w_137_0 : Fin 1 → mxxMat1
  w_138_0 : Fin 1 → mxxMat1
  w_139_0 : Fin 1 → mxxMat1
  w_140_0 : Fin 1 → mxxMat1
  w_141_0 : Fin 1 → mxxMat1
  w_142_0 : Fin 1 → mxxMat1
  w_143_0 : Fin 1 → mxxMat1
  w_144_0 : Fin 1 → mxxMat1
  w_145_0 : Fin 1 → mxxMat1
  w_146_0 : Fin 1 → mxxMat1
  w_147_0 : Fin 1 → mxxMat1
  w_148_0 : Fin 1 → mxxMat1
  w_149_0 : Fin 1 → mxxMat1
  w_150_0 : Fin 1 → mxxMat1
  w_151_0 : Fin 1 → mxxMat1
  w_152_0 : Fin 1 → mxxMat1
  w_153_0 : Fin 1 → mxxMat1
  w_154_0 : Fin 1 → mxxMat1
  w_155_0 : Fin 1 → mxxMat1
  w_156_0 : Fin 1 → mxxMat1
  w_157_0 : Fin 1 → mxxMat1
  w_158_0 : Fin 1 → mxxMat1
  w_159_0 : Fin 1 → mxxMat1
  w_160_0 : Fin 1 → mxxMat1
  w_161_0 : Fin 1 → mxxMat1
  w_162_0 : Fin 1 → mxxMat1
  w_163_0 : Fin 1 → mxxMat1
  w_164_0 : Fin 1 → mxxMat1
  w_165_0 : Fin 1 → mxxMat1
  w_166_0 : Fin 1 → mxxMat1
  w_167_0 : Fin 1 → mxxMat1
  w_168_0 : Fin 1 → mxxMat1
  w_169_0 : Fin 1 → mxxMat1
  w_170_0 : Fin 1 → mxxMat1
  w_171_0 : Fin 1 → mxxMat1
  w_172_0 : Fin 1 → mxxMat1
  w_173_0 : Fin 1 → mxxMat1
  w_174_0 : Fin 1 → mxxMat1
  w_175_0 : Fin 1 → mxxMat1
  w_176_0 : Fin 1 → mxxMat1
  w_177_0 : Fin 1 → mxxMat1
  w_178_0 : Fin 1 → mxxMat1
  w_179_0 : Fin 1 → mxxMat1
  w_180_0 : Fin 1 → mxxMat1
  w_181_0 : Fin 1 → mxxMat1
  w_182_0 : Fin 1 → mxxMat1
  w_183_0 : Fin 1 → mxxMat1
  w_184_0 : Fin 1 → mxxMat1
  w_185_0 : Fin 1 → mxxMat1
  w_186_0 : Fin 1 → mxxMat1
  w_187_0 : Fin 1 → mxxMat1
  w_188_0 : Fin 1 → mxxMat1
  w_189_0 : Fin 1 → mxxMat1
  w_190_0 : Fin 1 → mxxMat1
  w_191_0 : Fin 1 → mxxMat1
  w_192_0 : Fin 1 → mxxMat1
  w_193_0 : Fin 1 → mxxMat1
  w_194_0 : Fin 1 → mxxMat1
  w_195_0 : Fin 1 → mxxMat1
  w_196_0 : Fin 1 → mxxMat1
  w_197_0 : Fin 1 → mxxMat1
  w_198_0 : Fin 1 → mxxMat1
  w_199_0 : Fin 1 → mxxMat1
  w_200_0 : Fin 1 → mxxMat1
  w_201_0 : Fin 1 → mxxMat1
  w_202_0 : Fin 1 → mxxMat1
  w_203_0 : Fin 1 → mxxMat1
  w_204_0 : Fin 1 → mxxMat1
  w_205_0 : Fin 1 → mxxMat1
  w_206_0 : Fin 1 → mxxMat1
  w_207_0 : Fin 1 → mxxMat1
  w_208_0 : Fin 1 → mxxMat1
  w_209_0 : Fin 1 → mxxMat1
  w_210_0 : Fin 1 → mxxMat1
  w_211_0 : Fin 1 → mxxMat1
  w_212_0 : Fin 1 → mxxMat1
  w_213_0 : Fin 1 → mxxMat1
  w_214_0 : Fin 1 → mxxMat1
  w_215_0 : Fin 1 → mxxMat1
  w_216_0 : Fin 1 → mxxMat1
  w_217_0 : Fin 1 → mxxMat1
  w_218_0 : Fin 1 → mxxMat1
  w_219_0 : Fin 1 → mxxMat1
  w_220_0 : Fin 1 → mxxMat1
  w_221_0 : Fin 1 → mxxMat1
  w_222_0 : Fin 1 → mxxMat1
  w_223_0 : Fin 1 → mxxMat1
  w_224_0 : Fin 1 → mxxMat1
  w_225_0 : Fin 1 → mxxMat1
  w_226_0 : Fin 1 → mxxMat1
  w_227_0 : Fin 1 → mxxMat1
  w_228_0 : Fin 1 → mxxMat1
  w_229_0 : Fin 1 → mxxMat1
  w_230_0 : Fin 1 → mxxMat1
  w_231_0 : Fin 1 → mxxMat1
  w_232_0 : Fin 1 → mxxMat1
  w_233_0 : Fin 1 → mxxMat1
  w_234_0 : Fin 1 → mxxMat1
  w_235_0 : Fin 1 → mxxMat1
  w_236_0 : Fin 1 → mxxMat1
  w_237_0 : Fin 1 → mxxMat1
  w_238_0 : Fin 1 → mxxMat1
  w_239_0 : Fin 1 → mxxMat1
  w_240_0 : Fin 1 → mxxMat1
  w_241_0 : Fin 1 → mxxMat1
  w_242_0 : Fin 1 → mxxMat1
  w_243_0 : Fin 1 → mxxMat1
  w_244_0 : Fin 1 → mxxMat1
  w_245_0 : Fin 1 → mxxMat1
  w_246_0 : Fin 1 → mxxMat1
  w_247_0 : Fin 1 → mxxMat1
  w_248_0 : Fin 1 → mxxMat1
  w_249_0 : Fin 1 → mxxMat1
  w_250_0 : Fin 1 → mxxMat1
  w_251_0 : Fin 1 → mxxMat1
  w_252_0 : Fin 1 → mxxMat1
  w_253_0 : Fin 1 → mxxMat1
  w_254_0 : Fin 1 → mxxMat1
  w_255_0 : Fin 1 → mxxMat1
  w_256_0 : Fin 1 → mxxMat1
  w_257_0 : Fin 1 → mxxMat1
  w_258_0 : Fin 1 → mxxMat1
  w_259_0 : Fin 1 → mxxMat1
  w_260_0 : Fin 1 → mxxMat1
  w_261_0 : Fin 1 → mxxMat1
  w_262_0 : Fin 1 → mxxMat1
  w_263_0 : Fin 1 → mxxMat1
  w_264_0 : Fin 1 → mxxMat1
  w_265_0 : Fin 1 → mxxMat1
  w_266_0 : Fin 1 → mxxMat1
  w_267_0 : Fin 1 → mxxMat1
  w_268_0 : Fin 1 → mxxMat1
  w_269_0 : Fin 1 → mxxMat1
  w_270_0 : Fin 4 → mxxMat1
  w_271_0 : Fin 4 → mxxMat1
  w_272_0 : Fin 32 → mxxMat1
  w_273_0 : Fin 32 → mxxMat1
  w_274_0 : Fin 128 → mxxMat1
  w_275_0 : Fin 128 → mxxMat1
  w_277_0 : Fin 1025 → mxxMat1
  w_278_0 : Fin 1025 → mxxMat1
  w_279_0 : Fin 8388608 → mxxMat1
  w_280_0 : Fin 8388608 → mxxMat1
  w_281_0 : mxxMat1
  w_283_0 : mxxMat1
  w_289_0 : mxxMat1
  w_292_0 : mxxMat1
  w_293_0 : mxxMat1
  w_296_0 : mxxMat1
  w_298_0 : mxxMat1
  w_304_0 : mxxMat1
  w_307_0 : mxxMat1
  w_308_0 : mxxMat1
  w_311_0 : mxxMat1
  w_313_0 : mxxMat1
  w_322_0 : mxxMat1
  w_325_0 : mxxMat1
  w_326_0 : mxxMat1
  w_329_0 : mxxMat1
  w_331_0 : mxxMat1
  w_339_0 : mxxMat1
  w_342_0 : mxxMat1
  w_343_0 : mxxMat1

abbrev generatedRoot.constraints_17 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (w_9_0 : ByteArray) (w_8_0 : mxxMat2) (w_10_0 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_16_0 : Fin 1 → mxxMat1) (w_17_0 : Fin 1 → mxxMat1) (w_18_0 : Fin 1 → mxxMat1) (w_19_0 : Fin 1 → mxxMat1) (w_20_0 : Fin 1 → mxxMat1) (w_21_0 : Fin 1 → mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_23_0 : Fin 1 → mxxMat1) (w_24_0 : Fin 1 → mxxMat1) (w_25_0 : Fin 1 → mxxMat1) (w_26_0 : Fin 1 → mxxMat1) (w_27_0 : Fin 1 → mxxMat1) (w_28_0 : Fin 1 → mxxMat1) (w_29_0 : Fin 1 → mxxMat1) (w_30_0 : Fin 1 → mxxMat1) (w_31_0 : Fin 1 → mxxMat1) (w_32_0 : Fin 1 → mxxMat1) (w_33_0 : Fin 1 → mxxMat1) (w_34_0 : Fin 1 → mxxMat1) (w_35_0 : Fin 1 → mxxMat1) (w_36_0 : Fin 1 → mxxMat1) (w_37_0 : Fin 1 → mxxMat1) (w_38_0 : Fin 1 → mxxMat1) (w_39_0 : Fin 1 → mxxMat1) (w_40_0 : Fin 1 → mxxMat1) (w_41_0 : Fin 1 → mxxMat1) (w_42_0 : Fin 1 → mxxMat1) (w_43_0 : Fin 1 → mxxMat1) (w_44_0 : Fin 1 → mxxMat1) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : Fin 1 → mxxMat1) (w_47_0 : Fin 1 → mxxMat1) (w_48_0 : Fin 1 → mxxMat1) (w_49_0 : Fin 1 → mxxMat1) (w_50_0 : Fin 1 → mxxMat1) (w_51_0 : Fin 1 → mxxMat1) (w_52_0 : Fin 1 → mxxMat1) (w_53_0 : Fin 1 → mxxMat1) (w_54_0 : Fin 1 → mxxMat1) (w_55_0 : Fin 1 → mxxMat1) (w_56_0 : Fin 1 → mxxMat1) (w_57_0 : Fin 1 → mxxMat1) (w_58_0 : Fin 1 → mxxMat1) (w_59_0 : Fin 1 → mxxMat1) (w_60_0 : Fin 1 → mxxMat1) (w_61_0 : Fin 1 → mxxMat1) (w_62_0 : Fin 1 → mxxMat1) (w_63_0 : Fin 1 → mxxMat1) (w_64_0 : Fin 1 → mxxMat1) (w_65_0 : Fin 1 → mxxMat1) (w_66_0 : Fin 1 → mxxMat1) (w_67_0 : Fin 1 → mxxMat1) (w_68_0 : Fin 1 → mxxMat1) (w_69_0 : Fin 1 → mxxMat1) (w_70_0 : Fin 1 → mxxMat1) (w_71_0 : Fin 1 → mxxMat1) (w_72_0 : Fin 1 → mxxMat1) (w_73_0 : Fin 1 → mxxMat1) (w_74_0 : Fin 1 → mxxMat1) (w_75_0 : Fin 1 → mxxMat1) (w_76_0 : Fin 1 → mxxMat1) (w_77_0 : Fin 1 → mxxMat1) (w_78_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_80_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_82_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_84_0 : Fin 1 → mxxMat1) (w_85_0 : Fin 1 → mxxMat1) (w_86_0 : Fin 1 → mxxMat1) (w_87_0 : Fin 1 → mxxMat1) (w_88_0 : Fin 1 → mxxMat1) (w_89_0 : Fin 1 → mxxMat1) (w_90_0 : Fin 1 → mxxMat1) (w_91_0 : Fin 1 → mxxMat1) (w_92_0 : Fin 1 → mxxMat1) (w_93_0 : Fin 1 → mxxMat1) (w_94_0 : Fin 1 → mxxMat1) (w_95_0 : Fin 1 → mxxMat1) (w_96_0 : Fin 1 → mxxMat1) (w_97_0 : Fin 1 → mxxMat1) (w_98_0 : Fin 1 → mxxMat1) (w_99_0 : Fin 1 → mxxMat1) (w_100_0 : Fin 1 → mxxMat1) (w_101_0 : Fin 1 → mxxMat1) (w_102_0 : Fin 1 → mxxMat1) (w_103_0 : Fin 1 → mxxMat1) (w_104_0 : Fin 1 → mxxMat1) (w_105_0 : Fin 1 → mxxMat1) (w_106_0 : Fin 1 → mxxMat1) (w_107_0 : Fin 1 → mxxMat1) (w_108_0 : Fin 1 → mxxMat1) (w_109_0 : Fin 1 → mxxMat1) (w_110_0 : Fin 1 → mxxMat1) (w_111_0 : Fin 1 → mxxMat1) (w_112_0 : Fin 1 → mxxMat1) (w_113_0 : Fin 1 → mxxMat1) (w_114_0 : Fin 1 → mxxMat1) (w_115_0 : Fin 1 → mxxMat1) (w_116_0 : Fin 1 → mxxMat1) (w_117_0 : Fin 1 → mxxMat1) (w_118_0 : Fin 1 → mxxMat1) (w_119_0 : Fin 1 → mxxMat1) (w_120_0 : Fin 1 → mxxMat1) (w_121_0 : Fin 1 → mxxMat1) (w_122_0 : Fin 1 → mxxMat1) (w_123_0 : Fin 1 → mxxMat1) (w_124_0 : Fin 1 → mxxMat1) (w_125_0 : Fin 1 → mxxMat1) (w_126_0 : Fin 1 → mxxMat1) (w_127_0 : Fin 1 → mxxMat1) (w_128_0 : Fin 1 → mxxMat1) (w_129_0 : Fin 1 → mxxMat1) (w_130_0 : Fin 1 → mxxMat1) (w_131_0 : Fin 1 → mxxMat1) (w_132_0 : Fin 1 → mxxMat1) (w_133_0 : Fin 1 → mxxMat1) (w_134_0 : Fin 1 → mxxMat1) (w_135_0 : Fin 1 → mxxMat1) (w_136_0 : Fin 1 → mxxMat1) (w_137_0 : Fin 1 → mxxMat1) (w_138_0 : Fin 1 → mxxMat1) (w_139_0 : Fin 1 → mxxMat1) (w_140_0 : Fin 1 → mxxMat1) (w_141_0 : Fin 1 → mxxMat1) (w_142_0 : Fin 1 → mxxMat1) (w_143_0 : Fin 1 → mxxMat1) (w_144_0 : Fin 1 → mxxMat1) (w_145_0 : Fin 1 → mxxMat1) (w_146_0 : Fin 1 → mxxMat1) (w_147_0 : Fin 1 → mxxMat1) (w_148_0 : Fin 1 → mxxMat1) (w_149_0 : Fin 1 → mxxMat1) (w_150_0 : Fin 1 → mxxMat1) (w_151_0 : Fin 1 → mxxMat1) (w_152_0 : Fin 1 → mxxMat1) (w_153_0 : Fin 1 → mxxMat1) (w_154_0 : Fin 1 → mxxMat1) (w_155_0 : Fin 1 → mxxMat1) (w_156_0 : Fin 1 → mxxMat1) (w_157_0 : Fin 1 → mxxMat1) (w_158_0 : Fin 1 → mxxMat1) (w_159_0 : Fin 1 → mxxMat1) (w_160_0 : Fin 1 → mxxMat1) (w_161_0 : Fin 1 → mxxMat1) (w_162_0 : Fin 1 → mxxMat1) (w_163_0 : Fin 1 → mxxMat1) (w_164_0 : Fin 1 → mxxMat1) (w_165_0 : Fin 1 → mxxMat1) (w_166_0 : Fin 1 → mxxMat1) (w_167_0 : Fin 1 → mxxMat1) (w_168_0 : Fin 1 → mxxMat1) (w_169_0 : Fin 1 → mxxMat1) (w_170_0 : Fin 1 → mxxMat1) (w_171_0 : Fin 1 → mxxMat1) (w_172_0 : Fin 1 → mxxMat1) (w_173_0 : Fin 1 → mxxMat1) (w_174_0 : Fin 1 → mxxMat1) (w_175_0 : Fin 1 → mxxMat1) (w_176_0 : Fin 1 → mxxMat1) (w_177_0 : Fin 1 → mxxMat1) (w_178_0 : Fin 1 → mxxMat1) (w_179_0 : Fin 1 → mxxMat1) (w_180_0 : Fin 1 → mxxMat1) (w_181_0 : Fin 1 → mxxMat1) (w_182_0 : Fin 1 → mxxMat1) (w_183_0 : Fin 1 → mxxMat1) (w_184_0 : Fin 1 → mxxMat1) (w_185_0 : Fin 1 → mxxMat1) (w_186_0 : Fin 1 → mxxMat1) (w_187_0 : Fin 1 → mxxMat1) (w_188_0 : Fin 1 → mxxMat1) (w_189_0 : Fin 1 → mxxMat1) (w_190_0 : Fin 1 → mxxMat1) (w_191_0 : Fin 1 → mxxMat1) (w_192_0 : Fin 1 → mxxMat1) (w_193_0 : Fin 1 → mxxMat1) (w_194_0 : Fin 1 → mxxMat1) (w_195_0 : Fin 1 → mxxMat1) (w_196_0 : Fin 1 → mxxMat1) (w_197_0 : Fin 1 → mxxMat1) (w_198_0 : Fin 1 → mxxMat1) (w_199_0 : Fin 1 → mxxMat1) (w_200_0 : Fin 1 → mxxMat1) (w_201_0 : Fin 1 → mxxMat1) (w_202_0 : Fin 1 → mxxMat1) (w_203_0 : Fin 1 → mxxMat1) (w_204_0 : Fin 1 → mxxMat1) (w_205_0 : Fin 1 → mxxMat1) (w_206_0 : Fin 1 → mxxMat1) (w_207_0 : Fin 1 → mxxMat1) (w_208_0 : Fin 1 → mxxMat1) (w_209_0 : Fin 1 → mxxMat1) (w_210_0 : Fin 1 → mxxMat1) (w_211_0 : Fin 1 → mxxMat1) (w_212_0 : Fin 1 → mxxMat1) (w_213_0 : Fin 1 → mxxMat1) (w_214_0 : Fin 1 → mxxMat1) (w_215_0 : Fin 1 → mxxMat1) (w_216_0 : Fin 1 → mxxMat1) (w_217_0 : Fin 1 → mxxMat1) (w_218_0 : Fin 1 → mxxMat1) (w_219_0 : Fin 1 → mxxMat1) (w_220_0 : Fin 1 → mxxMat1) (w_221_0 : Fin 1 → mxxMat1) (w_222_0 : Fin 1 → mxxMat1) (w_223_0 : Fin 1 → mxxMat1) (w_224_0 : Fin 1 → mxxMat1) (w_225_0 : Fin 1 → mxxMat1) (w_226_0 : Fin 1 → mxxMat1) (w_227_0 : Fin 1 → mxxMat1) (w_228_0 : Fin 1 → mxxMat1) (w_229_0 : Fin 1 → mxxMat1) (w_230_0 : Fin 1 → mxxMat1) (w_231_0 : Fin 1 → mxxMat1) (w_232_0 : Fin 1 → mxxMat1) (w_233_0 : Fin 1 → mxxMat1) (w_234_0 : Fin 1 → mxxMat1) (w_235_0 : Fin 1 → mxxMat1) (w_236_0 : Fin 1 → mxxMat1) (w_237_0 : Fin 1 → mxxMat1) (w_238_0 : Fin 1 → mxxMat1) (w_239_0 : Fin 1 → mxxMat1) (w_240_0 : Fin 1 → mxxMat1) (w_241_0 : Fin 1 → mxxMat1) (w_242_0 : Fin 1 → mxxMat1) (w_243_0 : Fin 1 → mxxMat1) (w_244_0 : Fin 1 → mxxMat1) (w_245_0 : Fin 1 → mxxMat1) (w_246_0 : Fin 1 → mxxMat1) (w_247_0 : Fin 1 → mxxMat1) (w_248_0 : Fin 1 → mxxMat1) (w_249_0 : Fin 1 → mxxMat1) (w_250_0 : Fin 1 → mxxMat1) (w_251_0 : Fin 1 → mxxMat1) (w_252_0 : Fin 1 → mxxMat1) (w_253_0 : Fin 1 → mxxMat1) (w_254_0 : Fin 1 → mxxMat1) (w_255_0 : Fin 1 → mxxMat1) (w_256_0 : Fin 1 → mxxMat1) (w_257_0 : Fin 1 → mxxMat1) (w_258_0 : Fin 1 → mxxMat1) (w_259_0 : Fin 1 → mxxMat1) (w_260_0 : Fin 1 → mxxMat1) (w_261_0 : Fin 1 → mxxMat1) (w_262_0 : Fin 1 → mxxMat1) (w_263_0 : Fin 1 → mxxMat1) (w_264_0 : Fin 1 → mxxMat1) (w_265_0 : Fin 1 → mxxMat1) (w_266_0 : Fin 1 → mxxMat1) (w_267_0 : Fin 1 → mxxMat1) (w_268_0 : Fin 1 → mxxMat1) (w_269_0 : Fin 1 → mxxMat1) (w_270_0 : Fin 4 → mxxMat1) (w_271_0 : Fin 4 → mxxMat1) (w_272_0 : Fin 32 → mxxMat1) (w_273_0 : Fin 32 → mxxMat1) (w_274_0 : Fin 128 → mxxMat1) (w_275_0 : Fin 128 → mxxMat1) (w_277_0 : Fin 1025 → mxxMat1) (w_278_0 : Fin 1025 → mxxMat1) (w_279_0 : Fin 8388608 → mxxMat1) (w_280_0 : Fin 8388608 → mxxMat1) (w_283_0 : mxxMat1) (w_289_0 : mxxMat1) (w_292_0 : mxxMat1) (w_293_0 : mxxMat1) (w_295_0 : mxxMat1) (w_296_0 : mxxMat1) (w_298_0 : mxxMat1) (w_304_0 : mxxMat1) (w_307_0 : mxxMat1) (w_308_0 : mxxMat1) (w_310_0 : mxxMat1) (w_311_0 : mxxMat1) (w_313_0 : mxxMat1) (w_322_0 : mxxMat1) (w_325_0 : mxxMat1) (w_326_0 : mxxMat1) (w_328_0 : mxxMat1) (w_329_0 : mxxMat1) (w_331_0 : mxxMat1) (w_339_0 : mxxMat1) (w_342_0 : mxxMat1) (w_343_0 : mxxMat1) (w_345_0 : mxxMat1) (outputs : (generatedRoot.OutputType)) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_283_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_289_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [292]) (0) (1) w_292_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [293]) (0) (1) w_293_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_296_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_298_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_304_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [307]) (0) (1) w_307_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [308]) (0) (1) w_308_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_311_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_313_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_322_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [325]) (0) (1) w_325_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [326]) (0) (1) w_326_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_329_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_331_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 64 270 w_339_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [342]) (0) (1) w_342_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [343]) (0) (1) w_343_0 ∧
  outputs = ((w_8_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, (w_23_0, (w_24_0, (w_25_0, (w_26_0, (w_27_0, (w_28_0, (w_29_0, (w_30_0, (w_31_0, (w_32_0, (w_33_0, (w_34_0, (w_35_0, (w_36_0, (w_37_0, (w_38_0, (w_39_0, (w_40_0, (w_41_0, (w_42_0, (w_43_0, (w_44_0, (w_45_0, (w_46_0, (w_47_0, (w_48_0, (w_49_0, (w_50_0, (w_51_0, (w_52_0, (w_53_0, (w_54_0, (w_55_0, (w_56_0, (w_57_0, (w_58_0, (w_59_0, (w_60_0, (w_61_0, (w_62_0, (w_63_0, (w_64_0, (w_65_0, (w_66_0, (w_67_0, (w_68_0, (w_69_0, (w_70_0, (w_71_0, (w_72_0, (w_73_0, (w_74_0, (w_75_0, (w_76_0, (w_77_0, (w_78_0, (w_79_0, (w_80_0, (w_81_0, (w_82_0, (w_83_0, (w_84_0, (w_85_0, (w_86_0, (w_87_0, (w_88_0, (w_89_0, (w_90_0, (w_91_0, (w_92_0, (w_93_0, (w_94_0, (w_95_0, (w_96_0, (w_97_0, (w_98_0, (w_99_0, (w_100_0, (w_101_0, (w_102_0, (w_103_0, (w_104_0, (w_105_0, (w_106_0, (w_107_0, (w_108_0, (w_109_0, (w_110_0, (w_111_0, (w_112_0, (w_113_0, (w_114_0, (w_115_0, (w_116_0, (w_117_0, (w_118_0, (w_119_0, (w_120_0, (w_121_0, (w_122_0, (w_123_0, (w_124_0, (w_125_0, (w_126_0, (w_127_0, (w_128_0, (w_129_0, (w_130_0, (w_131_0, (w_132_0, (w_133_0, (w_134_0, (w_135_0, (w_136_0, (w_137_0, (w_138_0, (w_139_0, (w_140_0, (w_141_0, (w_142_0, (w_143_0, (w_144_0, (w_145_0, ()))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))), (w_146_0, (w_147_0, (w_148_0, (w_149_0, (w_150_0, (w_151_0, (w_152_0, (w_153_0, (w_154_0, (w_155_0, (w_156_0, (w_157_0, (w_158_0, (w_159_0, (w_160_0, (w_161_0, (w_162_0, (w_163_0, (w_164_0, (w_165_0, (w_166_0, (w_167_0, (w_168_0, (w_169_0, (w_170_0, (w_171_0, (w_172_0, (w_173_0, (w_174_0, (w_175_0, (w_176_0, (w_177_0, (w_178_0, (w_179_0, (w_180_0, (w_181_0, (w_182_0, (w_183_0, (w_184_0, (w_185_0, (w_186_0, (w_187_0, (w_188_0, (w_189_0, (w_190_0, (w_191_0, (w_192_0, (w_193_0, (w_194_0, (w_195_0, (w_196_0, (w_197_0, (w_198_0, (w_199_0, (w_200_0, (w_201_0, (w_202_0, (w_203_0, (w_204_0, (w_205_0, (w_206_0, (w_207_0, (w_208_0, (w_209_0, (w_210_0, (w_211_0, (w_212_0, (w_213_0, (w_214_0, (w_215_0, (w_216_0, (w_217_0, (w_218_0, (w_219_0, (w_220_0, (w_221_0, (w_222_0, (w_223_0, (w_224_0, (w_225_0, (w_226_0, (w_227_0, (w_228_0, (w_229_0, (w_230_0, (w_231_0, (w_232_0, (w_233_0, (w_234_0, (w_235_0, (w_236_0, (w_237_0, (w_238_0, (w_239_0, (w_240_0, (w_241_0, (w_242_0, (w_243_0, (w_244_0, (w_245_0, (w_246_0, (w_247_0, (w_248_0, (w_249_0, (w_250_0, (w_251_0, (w_252_0, (w_253_0, (w_254_0, (w_255_0, (w_256_0, (w_257_0, (w_258_0, (w_259_0, (w_260_0, (w_261_0, (w_262_0, (w_263_0, (w_264_0, (w_265_0, (w_266_0, (w_267_0, (w_268_0, (w_269_0, (w_270_0, (w_271_0, (w_272_0, (w_273_0, (w_274_0, (w_275_0, (w_277_0, (w_278_0, (w_279_0, (w_280_0, (w_295_0, (w_310_0, (w_328_0, (w_345_0, ())))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))

abbrev generatedRoot.constraints_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_276_0 : Fin 8389632 → Int) (w_1_0 : mxxMat0) (w_264_0 : Fin 1 → mxxMat1) (w_265_0 : Fin 1 → mxxMat1) (w_266_0 : Fin 1 → mxxMat1) (w_267_0 : Fin 1 → mxxMat1) (w_268_0 : Fin 1 → mxxMat1) (w_269_0 : Fin 1 → mxxMat1) (w_270_0 : Fin 4 → mxxMat1) (w_271_0 : Fin 4 → mxxMat1) (w_272_0 : Fin 32 → mxxMat1) (w_273_0 : Fin 32 → mxxMat1) (w_274_0 : Fin 128 → mxxMat1) (w_275_0 : Fin 128 → mxxMat1) (w_277_0 : Fin 1025 → mxxMat1) (w_278_0 : Fin 1025 → mxxMat1) (w_279_0 : Fin 8388608 → mxxMat1) (w_280_0 : Fin 8388608 → mxxMat1) (w_281_0 : mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_264 backend hashModel tape (path ++ [264, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_264_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_265 backend hashModel tape (path ++ [265, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_265_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_266 backend hashModel tape (path ++ [266, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_266_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_267 backend hashModel tape (path ++ [267, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_267_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_268 backend hashModel tape (path ++ [268, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_268_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_269 backend hashModel tape (path ++ [269, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_269_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_270 backend hashModel tape (path ++ [270, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_270_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_271 backend hashModel tape (path ++ [271, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_271_0 i)) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_generatedRoot_272 backend hashModel tape (path ++ [272, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_272_0 i)) ∧
  (32) = 32 ∧
  (∀ i : Fin 32, parallel_generatedRoot_273 backend hashModel tape (path ++ [273, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_273_0 i)) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_generatedRoot_274 backend hashModel tape (path ++ [274, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_274_0 i)) ∧
  (128) = 128 ∧
  (∀ i : Fin 128, parallel_generatedRoot_275 backend hashModel tape (path ++ [275, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_275_0 i)) ∧
  (1025) = 1025 ∧
  (∀ i : Fin 1025, parallel_generatedRoot_277 backend hashModel tape (path ++ [277, i.val]) params i (w_0_0, (w_9_0, (w_1_0, (w_276_0, ())))) (w_277_0 i)) ∧
  (1025) = 1025 ∧
  (∀ i : Fin 1025, parallel_generatedRoot_278 backend hashModel tape (path ++ [278, i.val]) params i (w_0_0, (w_9_0, (w_1_0, (w_276_0, ())))) (w_278_0 i)) ∧
  (8388608) = 8388608 ∧
  (∀ i : Fin 8388608, parallel_generatedRoot_279 backend hashModel tape (path ++ [279, i.val]) params i (w_0_0, (w_9_0, (w_1_0, (w_276_0, ())))) (w_279_0 i)) ∧
  (8388608) = 8388608 ∧
  (∀ i : Fin 8388608, parallel_generatedRoot_280 backend hashModel tape (path ++ [280, i.val]) params i (w_0_0, (w_9_0, (w_1_0, (w_276_0, ())))) (w_280_0 i)) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_281_0

abbrev generatedRoot.constraints_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_248_0 : Fin 1 → mxxMat1) (w_249_0 : Fin 1 → mxxMat1) (w_250_0 : Fin 1 → mxxMat1) (w_251_0 : Fin 1 → mxxMat1) (w_252_0 : Fin 1 → mxxMat1) (w_253_0 : Fin 1 → mxxMat1) (w_254_0 : Fin 1 → mxxMat1) (w_255_0 : Fin 1 → mxxMat1) (w_256_0 : Fin 1 → mxxMat1) (w_257_0 : Fin 1 → mxxMat1) (w_258_0 : Fin 1 → mxxMat1) (w_259_0 : Fin 1 → mxxMat1) (w_260_0 : Fin 1 → mxxMat1) (w_261_0 : Fin 1 → mxxMat1) (w_262_0 : Fin 1 → mxxMat1) (w_263_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_248 backend hashModel tape (path ++ [248, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_248_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_249 backend hashModel tape (path ++ [249, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_249_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_250 backend hashModel tape (path ++ [250, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_250_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_251 backend hashModel tape (path ++ [251, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_251_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_252 backend hashModel tape (path ++ [252, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_252_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_253 backend hashModel tape (path ++ [253, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_253_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_254 backend hashModel tape (path ++ [254, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_254_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_255 backend hashModel tape (path ++ [255, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_255_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_256 backend hashModel tape (path ++ [256, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_256_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_257 backend hashModel tape (path ++ [257, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_257_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_258 backend hashModel tape (path ++ [258, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_258_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_259 backend hashModel tape (path ++ [259, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_259_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_260 backend hashModel tape (path ++ [260, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_260_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_261 backend hashModel tape (path ++ [261, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_261_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_262 backend hashModel tape (path ++ [262, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_262_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_263 backend hashModel tape (path ++ [263, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_263_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_232_0 : Fin 1 → mxxMat1) (w_233_0 : Fin 1 → mxxMat1) (w_234_0 : Fin 1 → mxxMat1) (w_235_0 : Fin 1 → mxxMat1) (w_236_0 : Fin 1 → mxxMat1) (w_237_0 : Fin 1 → mxxMat1) (w_238_0 : Fin 1 → mxxMat1) (w_239_0 : Fin 1 → mxxMat1) (w_240_0 : Fin 1 → mxxMat1) (w_241_0 : Fin 1 → mxxMat1) (w_242_0 : Fin 1 → mxxMat1) (w_243_0 : Fin 1 → mxxMat1) (w_244_0 : Fin 1 → mxxMat1) (w_245_0 : Fin 1 → mxxMat1) (w_246_0 : Fin 1 → mxxMat1) (w_247_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_232 backend hashModel tape (path ++ [232, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_232_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_233 backend hashModel tape (path ++ [233, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_233_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_234 backend hashModel tape (path ++ [234, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_234_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_235 backend hashModel tape (path ++ [235, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_235_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_236 backend hashModel tape (path ++ [236, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_236_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_237 backend hashModel tape (path ++ [237, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_237_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_238 backend hashModel tape (path ++ [238, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_238_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_239 backend hashModel tape (path ++ [239, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_239_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_240 backend hashModel tape (path ++ [240, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_240_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_241 backend hashModel tape (path ++ [241, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_241_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_242 backend hashModel tape (path ++ [242, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_242_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_243 backend hashModel tape (path ++ [243, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_243_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_244 backend hashModel tape (path ++ [244, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_244_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_245 backend hashModel tape (path ++ [245, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_245_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_246 backend hashModel tape (path ++ [246, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_246_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_247 backend hashModel tape (path ++ [247, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_247_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_216_0 : Fin 1 → mxxMat1) (w_217_0 : Fin 1 → mxxMat1) (w_218_0 : Fin 1 → mxxMat1) (w_219_0 : Fin 1 → mxxMat1) (w_220_0 : Fin 1 → mxxMat1) (w_221_0 : Fin 1 → mxxMat1) (w_222_0 : Fin 1 → mxxMat1) (w_223_0 : Fin 1 → mxxMat1) (w_224_0 : Fin 1 → mxxMat1) (w_225_0 : Fin 1 → mxxMat1) (w_226_0 : Fin 1 → mxxMat1) (w_227_0 : Fin 1 → mxxMat1) (w_228_0 : Fin 1 → mxxMat1) (w_229_0 : Fin 1 → mxxMat1) (w_230_0 : Fin 1 → mxxMat1) (w_231_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_216 backend hashModel tape (path ++ [216, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_216_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_217 backend hashModel tape (path ++ [217, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_217_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_218 backend hashModel tape (path ++ [218, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_218_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_219 backend hashModel tape (path ++ [219, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_219_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_220 backend hashModel tape (path ++ [220, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_220_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_221 backend hashModel tape (path ++ [221, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_221_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_222 backend hashModel tape (path ++ [222, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_222_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_223 backend hashModel tape (path ++ [223, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_223_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_224 backend hashModel tape (path ++ [224, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_224_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_225 backend hashModel tape (path ++ [225, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_225_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_226 backend hashModel tape (path ++ [226, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_226_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_227 backend hashModel tape (path ++ [227, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_227_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_228 backend hashModel tape (path ++ [228, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_228_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_229 backend hashModel tape (path ++ [229, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_229_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_230 backend hashModel tape (path ++ [230, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_230_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_231 backend hashModel tape (path ++ [231, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_231_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_200_0 : Fin 1 → mxxMat1) (w_201_0 : Fin 1 → mxxMat1) (w_202_0 : Fin 1 → mxxMat1) (w_203_0 : Fin 1 → mxxMat1) (w_204_0 : Fin 1 → mxxMat1) (w_205_0 : Fin 1 → mxxMat1) (w_206_0 : Fin 1 → mxxMat1) (w_207_0 : Fin 1 → mxxMat1) (w_208_0 : Fin 1 → mxxMat1) (w_209_0 : Fin 1 → mxxMat1) (w_210_0 : Fin 1 → mxxMat1) (w_211_0 : Fin 1 → mxxMat1) (w_212_0 : Fin 1 → mxxMat1) (w_213_0 : Fin 1 → mxxMat1) (w_214_0 : Fin 1 → mxxMat1) (w_215_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_200 backend hashModel tape (path ++ [200, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_200_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_201 backend hashModel tape (path ++ [201, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_201_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_202 backend hashModel tape (path ++ [202, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_202_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_203 backend hashModel tape (path ++ [203, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_203_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_204 backend hashModel tape (path ++ [204, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_204_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_205 backend hashModel tape (path ++ [205, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_205_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_206 backend hashModel tape (path ++ [206, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_206_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_207 backend hashModel tape (path ++ [207, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_207_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_208 backend hashModel tape (path ++ [208, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_208_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_209 backend hashModel tape (path ++ [209, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_209_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_210 backend hashModel tape (path ++ [210, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_210_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_211 backend hashModel tape (path ++ [211, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_211_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_212 backend hashModel tape (path ++ [212, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_212_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_213 backend hashModel tape (path ++ [213, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_213_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_214 backend hashModel tape (path ++ [214, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_214_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_215 backend hashModel tape (path ++ [215, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_215_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_184_0 : Fin 1 → mxxMat1) (w_185_0 : Fin 1 → mxxMat1) (w_186_0 : Fin 1 → mxxMat1) (w_187_0 : Fin 1 → mxxMat1) (w_188_0 : Fin 1 → mxxMat1) (w_189_0 : Fin 1 → mxxMat1) (w_190_0 : Fin 1 → mxxMat1) (w_191_0 : Fin 1 → mxxMat1) (w_192_0 : Fin 1 → mxxMat1) (w_193_0 : Fin 1 → mxxMat1) (w_194_0 : Fin 1 → mxxMat1) (w_195_0 : Fin 1 → mxxMat1) (w_196_0 : Fin 1 → mxxMat1) (w_197_0 : Fin 1 → mxxMat1) (w_198_0 : Fin 1 → mxxMat1) (w_199_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_184 backend hashModel tape (path ++ [184, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_184_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_185 backend hashModel tape (path ++ [185, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_185_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_186 backend hashModel tape (path ++ [186, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_186_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_187 backend hashModel tape (path ++ [187, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_187_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_188 backend hashModel tape (path ++ [188, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_188_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_189 backend hashModel tape (path ++ [189, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_189_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_190 backend hashModel tape (path ++ [190, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_190_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_191 backend hashModel tape (path ++ [191, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_191_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_192 backend hashModel tape (path ++ [192, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_192_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_193 backend hashModel tape (path ++ [193, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_193_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_194 backend hashModel tape (path ++ [194, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_194_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_195 backend hashModel tape (path ++ [195, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_195_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_196 backend hashModel tape (path ++ [196, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_196_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_197 backend hashModel tape (path ++ [197, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_197_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_198 backend hashModel tape (path ++ [198, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_198_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_199 backend hashModel tape (path ++ [199, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_199_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_168_0 : Fin 1 → mxxMat1) (w_169_0 : Fin 1 → mxxMat1) (w_170_0 : Fin 1 → mxxMat1) (w_171_0 : Fin 1 → mxxMat1) (w_172_0 : Fin 1 → mxxMat1) (w_173_0 : Fin 1 → mxxMat1) (w_174_0 : Fin 1 → mxxMat1) (w_175_0 : Fin 1 → mxxMat1) (w_176_0 : Fin 1 → mxxMat1) (w_177_0 : Fin 1 → mxxMat1) (w_178_0 : Fin 1 → mxxMat1) (w_179_0 : Fin 1 → mxxMat1) (w_180_0 : Fin 1 → mxxMat1) (w_181_0 : Fin 1 → mxxMat1) (w_182_0 : Fin 1 → mxxMat1) (w_183_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_168 backend hashModel tape (path ++ [168, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_168_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_169 backend hashModel tape (path ++ [169, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_169_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_170 backend hashModel tape (path ++ [170, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_170_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_171 backend hashModel tape (path ++ [171, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_171_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_172 backend hashModel tape (path ++ [172, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_172_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_173 backend hashModel tape (path ++ [173, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_173_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_174 backend hashModel tape (path ++ [174, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_174_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_175 backend hashModel tape (path ++ [175, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_175_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_176 backend hashModel tape (path ++ [176, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_176_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_177 backend hashModel tape (path ++ [177, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_177_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_178 backend hashModel tape (path ++ [178, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_178_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_179 backend hashModel tape (path ++ [179, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_179_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_180 backend hashModel tape (path ++ [180, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_180_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_181 backend hashModel tape (path ++ [181, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_181_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_182 backend hashModel tape (path ++ [182, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_182_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_183 backend hashModel tape (path ++ [183, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_183_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_9 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_152_0 : Fin 1 → mxxMat1) (w_153_0 : Fin 1 → mxxMat1) (w_154_0 : Fin 1 → mxxMat1) (w_155_0 : Fin 1 → mxxMat1) (w_156_0 : Fin 1 → mxxMat1) (w_157_0 : Fin 1 → mxxMat1) (w_158_0 : Fin 1 → mxxMat1) (w_159_0 : Fin 1 → mxxMat1) (w_160_0 : Fin 1 → mxxMat1) (w_161_0 : Fin 1 → mxxMat1) (w_162_0 : Fin 1 → mxxMat1) (w_163_0 : Fin 1 → mxxMat1) (w_164_0 : Fin 1 → mxxMat1) (w_165_0 : Fin 1 → mxxMat1) (w_166_0 : Fin 1 → mxxMat1) (w_167_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_152 backend hashModel tape (path ++ [152, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_152_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_153 backend hashModel tape (path ++ [153, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_153_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_154 backend hashModel tape (path ++ [154, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_154_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_155 backend hashModel tape (path ++ [155, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_155_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_156 backend hashModel tape (path ++ [156, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_156_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_157 backend hashModel tape (path ++ [157, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_157_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_158 backend hashModel tape (path ++ [158, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_158_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_159 backend hashModel tape (path ++ [159, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_159_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_160 backend hashModel tape (path ++ [160, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_160_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_161 backend hashModel tape (path ++ [161, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_161_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_162 backend hashModel tape (path ++ [162, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_162_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_163 backend hashModel tape (path ++ [163, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_163_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_164 backend hashModel tape (path ++ [164, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_164_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_165 backend hashModel tape (path ++ [165, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_165_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_166 backend hashModel tape (path ++ [166, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_166_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_167 backend hashModel tape (path ++ [167, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_167_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_136_0 : Fin 1 → mxxMat1) (w_137_0 : Fin 1 → mxxMat1) (w_138_0 : Fin 1 → mxxMat1) (w_139_0 : Fin 1 → mxxMat1) (w_140_0 : Fin 1 → mxxMat1) (w_141_0 : Fin 1 → mxxMat1) (w_142_0 : Fin 1 → mxxMat1) (w_143_0 : Fin 1 → mxxMat1) (w_144_0 : Fin 1 → mxxMat1) (w_145_0 : Fin 1 → mxxMat1) (w_146_0 : Fin 1 → mxxMat1) (w_147_0 : Fin 1 → mxxMat1) (w_148_0 : Fin 1 → mxxMat1) (w_149_0 : Fin 1 → mxxMat1) (w_150_0 : Fin 1 → mxxMat1) (w_151_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_136 backend hashModel tape (path ++ [136, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_136_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_137 backend hashModel tape (path ++ [137, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_137_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_138 backend hashModel tape (path ++ [138, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_138_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_139 backend hashModel tape (path ++ [139, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_139_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_140 backend hashModel tape (path ++ [140, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_140_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_141 backend hashModel tape (path ++ [141, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_141_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_142 backend hashModel tape (path ++ [142, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_142_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_143 backend hashModel tape (path ++ [143, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_143_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_144 backend hashModel tape (path ++ [144, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_144_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_145 backend hashModel tape (path ++ [145, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_145_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_146 backend hashModel tape (path ++ [146, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_146_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_147 backend hashModel tape (path ++ [147, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_147_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_148 backend hashModel tape (path ++ [148, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_148_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_149 backend hashModel tape (path ++ [149, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_149_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_150 backend hashModel tape (path ++ [150, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_150_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_151 backend hashModel tape (path ++ [151, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_151_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_7 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_120_0 : Fin 1 → mxxMat1) (w_121_0 : Fin 1 → mxxMat1) (w_122_0 : Fin 1 → mxxMat1) (w_123_0 : Fin 1 → mxxMat1) (w_124_0 : Fin 1 → mxxMat1) (w_125_0 : Fin 1 → mxxMat1) (w_126_0 : Fin 1 → mxxMat1) (w_127_0 : Fin 1 → mxxMat1) (w_128_0 : Fin 1 → mxxMat1) (w_129_0 : Fin 1 → mxxMat1) (w_130_0 : Fin 1 → mxxMat1) (w_131_0 : Fin 1 → mxxMat1) (w_132_0 : Fin 1 → mxxMat1) (w_133_0 : Fin 1 → mxxMat1) (w_134_0 : Fin 1 → mxxMat1) (w_135_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_120 backend hashModel tape (path ++ [120, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_120_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_121 backend hashModel tape (path ++ [121, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_121_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_122 backend hashModel tape (path ++ [122, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_122_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_123 backend hashModel tape (path ++ [123, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_123_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_124 backend hashModel tape (path ++ [124, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_124_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_125 backend hashModel tape (path ++ [125, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_125_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_126 backend hashModel tape (path ++ [126, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_126_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_127 backend hashModel tape (path ++ [127, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_127_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_128 backend hashModel tape (path ++ [128, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_128_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_129 backend hashModel tape (path ++ [129, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_129_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_130 backend hashModel tape (path ++ [130, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_130_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_131 backend hashModel tape (path ++ [131, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_131_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_132 backend hashModel tape (path ++ [132, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_132_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_133 backend hashModel tape (path ++ [133, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_133_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_134 backend hashModel tape (path ++ [134, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_134_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_135 backend hashModel tape (path ++ [135, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_135_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_6 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_104_0 : Fin 1 → mxxMat1) (w_105_0 : Fin 1 → mxxMat1) (w_106_0 : Fin 1 → mxxMat1) (w_107_0 : Fin 1 → mxxMat1) (w_108_0 : Fin 1 → mxxMat1) (w_109_0 : Fin 1 → mxxMat1) (w_110_0 : Fin 1 → mxxMat1) (w_111_0 : Fin 1 → mxxMat1) (w_112_0 : Fin 1 → mxxMat1) (w_113_0 : Fin 1 → mxxMat1) (w_114_0 : Fin 1 → mxxMat1) (w_115_0 : Fin 1 → mxxMat1) (w_116_0 : Fin 1 → mxxMat1) (w_117_0 : Fin 1 → mxxMat1) (w_118_0 : Fin 1 → mxxMat1) (w_119_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_104 backend hashModel tape (path ++ [104, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_104_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_105 backend hashModel tape (path ++ [105, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_105_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_106 backend hashModel tape (path ++ [106, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_106_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_107 backend hashModel tape (path ++ [107, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_107_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_108 backend hashModel tape (path ++ [108, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_108_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_109 backend hashModel tape (path ++ [109, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_109_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_110 backend hashModel tape (path ++ [110, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_110_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_111 backend hashModel tape (path ++ [111, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_111_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_112 backend hashModel tape (path ++ [112, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_112_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_113 backend hashModel tape (path ++ [113, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_113_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_114 backend hashModel tape (path ++ [114, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_114_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_115 backend hashModel tape (path ++ [115, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_115_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_116 backend hashModel tape (path ++ [116, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_116_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_117 backend hashModel tape (path ++ [117, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_117_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_118 backend hashModel tape (path ++ [118, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_118_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_119 backend hashModel tape (path ++ [119, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_119_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_88_0 : Fin 1 → mxxMat1) (w_89_0 : Fin 1 → mxxMat1) (w_90_0 : Fin 1 → mxxMat1) (w_91_0 : Fin 1 → mxxMat1) (w_92_0 : Fin 1 → mxxMat1) (w_93_0 : Fin 1 → mxxMat1) (w_94_0 : Fin 1 → mxxMat1) (w_95_0 : Fin 1 → mxxMat1) (w_96_0 : Fin 1 → mxxMat1) (w_97_0 : Fin 1 → mxxMat1) (w_98_0 : Fin 1 → mxxMat1) (w_99_0 : Fin 1 → mxxMat1) (w_100_0 : Fin 1 → mxxMat1) (w_101_0 : Fin 1 → mxxMat1) (w_102_0 : Fin 1 → mxxMat1) (w_103_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_88 backend hashModel tape (path ++ [88, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_88_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_89 backend hashModel tape (path ++ [89, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_89_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_90 backend hashModel tape (path ++ [90, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_90_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_91 backend hashModel tape (path ++ [91, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_91_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_92 backend hashModel tape (path ++ [92, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_92_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_93 backend hashModel tape (path ++ [93, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_93_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_94 backend hashModel tape (path ++ [94, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_94_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_95 backend hashModel tape (path ++ [95, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_95_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_96 backend hashModel tape (path ++ [96, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_96_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_97 backend hashModel tape (path ++ [97, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_97_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_98 backend hashModel tape (path ++ [98, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_98_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_99 backend hashModel tape (path ++ [99, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_99_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_100 backend hashModel tape (path ++ [100, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_100_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_101 backend hashModel tape (path ++ [101, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_101_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_102 backend hashModel tape (path ++ [102, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_102_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_103 backend hashModel tape (path ++ [103, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_103_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_72_0 : Fin 1 → mxxMat1) (w_73_0 : Fin 1 → mxxMat1) (w_74_0 : Fin 1 → mxxMat1) (w_75_0 : Fin 1 → mxxMat1) (w_76_0 : Fin 1 → mxxMat1) (w_77_0 : Fin 1 → mxxMat1) (w_78_0 : Fin 1 → mxxMat1) (w_79_0 : Fin 1 → mxxMat1) (w_80_0 : Fin 1 → mxxMat1) (w_81_0 : Fin 1 → mxxMat1) (w_82_0 : Fin 1 → mxxMat1) (w_83_0 : Fin 1 → mxxMat1) (w_84_0 : Fin 1 → mxxMat1) (w_85_0 : Fin 1 → mxxMat1) (w_86_0 : Fin 1 → mxxMat1) (w_87_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_72 backend hashModel tape (path ++ [72, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_72_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_73 backend hashModel tape (path ++ [73, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_73_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_74 backend hashModel tape (path ++ [74, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_74_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_75 backend hashModel tape (path ++ [75, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_75_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_76 backend hashModel tape (path ++ [76, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_76_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_77 backend hashModel tape (path ++ [77, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_77_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_78 backend hashModel tape (path ++ [78, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_78_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_79 backend hashModel tape (path ++ [79, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_79_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_80 backend hashModel tape (path ++ [80, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_80_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_81 backend hashModel tape (path ++ [81, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_81_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_82 backend hashModel tape (path ++ [82, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_82_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_83 backend hashModel tape (path ++ [83, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_83_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_84 backend hashModel tape (path ++ [84, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_84_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_85 backend hashModel tape (path ++ [85, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_85_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_86 backend hashModel tape (path ++ [86, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_86_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_87 backend hashModel tape (path ++ [87, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_87_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_3 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_56_0 : Fin 1 → mxxMat1) (w_57_0 : Fin 1 → mxxMat1) (w_58_0 : Fin 1 → mxxMat1) (w_59_0 : Fin 1 → mxxMat1) (w_60_0 : Fin 1 → mxxMat1) (w_61_0 : Fin 1 → mxxMat1) (w_62_0 : Fin 1 → mxxMat1) (w_63_0 : Fin 1 → mxxMat1) (w_64_0 : Fin 1 → mxxMat1) (w_65_0 : Fin 1 → mxxMat1) (w_66_0 : Fin 1 → mxxMat1) (w_67_0 : Fin 1 → mxxMat1) (w_68_0 : Fin 1 → mxxMat1) (w_69_0 : Fin 1 → mxxMat1) (w_70_0 : Fin 1 → mxxMat1) (w_71_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_56 backend hashModel tape (path ++ [56, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_56_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_57 backend hashModel tape (path ++ [57, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_57_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_58 backend hashModel tape (path ++ [58, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_58_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_59 backend hashModel tape (path ++ [59, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_59_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_60 backend hashModel tape (path ++ [60, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_60_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_61 backend hashModel tape (path ++ [61, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_61_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_62 backend hashModel tape (path ++ [62, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_62_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_63 backend hashModel tape (path ++ [63, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_63_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_64 backend hashModel tape (path ++ [64, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_64_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_65 backend hashModel tape (path ++ [65, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_65_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_66 backend hashModel tape (path ++ [66, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_66_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_67 backend hashModel tape (path ++ [67, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_67_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_68 backend hashModel tape (path ++ [68, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_68_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_69 backend hashModel tape (path ++ [69, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_69_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_70 backend hashModel tape (path ++ [70, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_70_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_71 backend hashModel tape (path ++ [71, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_71_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_40_0 : Fin 1 → mxxMat1) (w_41_0 : Fin 1 → mxxMat1) (w_42_0 : Fin 1 → mxxMat1) (w_43_0 : Fin 1 → mxxMat1) (w_44_0 : Fin 1 → mxxMat1) (w_45_0 : Fin 1 → mxxMat1) (w_46_0 : Fin 1 → mxxMat1) (w_47_0 : Fin 1 → mxxMat1) (w_48_0 : Fin 1 → mxxMat1) (w_49_0 : Fin 1 → mxxMat1) (w_50_0 : Fin 1 → mxxMat1) (w_51_0 : Fin 1 → mxxMat1) (w_52_0 : Fin 1 → mxxMat1) (w_53_0 : Fin 1 → mxxMat1) (w_54_0 : Fin 1 → mxxMat1) (w_55_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_40 backend hashModel tape (path ++ [40, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_40_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_41 backend hashModel tape (path ++ [41, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_41_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_42 backend hashModel tape (path ++ [42, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_42_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_43 backend hashModel tape (path ++ [43, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_43_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_44 backend hashModel tape (path ++ [44, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_44_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_45 backend hashModel tape (path ++ [45, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_45_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_46 backend hashModel tape (path ++ [46, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_46_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_47 backend hashModel tape (path ++ [47, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_47_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_48 backend hashModel tape (path ++ [48, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_48_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_49 backend hashModel tape (path ++ [49, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_49_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_50 backend hashModel tape (path ++ [50, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_50_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_51 backend hashModel tape (path ++ [51, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_51_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_52 backend hashModel tape (path ++ [52, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_52_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_53 backend hashModel tape (path ++ [53, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_53_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_54 backend hashModel tape (path ++ [54, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_54_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_55 backend hashModel tape (path ++ [55, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_55_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_24_0 : Fin 1 → mxxMat1) (w_25_0 : Fin 1 → mxxMat1) (w_26_0 : Fin 1 → mxxMat1) (w_27_0 : Fin 1 → mxxMat1) (w_28_0 : Fin 1 → mxxMat1) (w_29_0 : Fin 1 → mxxMat1) (w_30_0 : Fin 1 → mxxMat1) (w_31_0 : Fin 1 → mxxMat1) (w_32_0 : Fin 1 → mxxMat1) (w_33_0 : Fin 1 → mxxMat1) (w_34_0 : Fin 1 → mxxMat1) (w_35_0 : Fin 1 → mxxMat1) (w_36_0 : Fin 1 → mxxMat1) (w_37_0 : Fin 1 → mxxMat1) (w_38_0 : Fin 1 → mxxMat1) (w_39_0 : Fin 1 → mxxMat1) : Prop :=
  (∀ i : Fin 1, parallel_generatedRoot_24 backend hashModel tape (path ++ [24, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_24_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_25 backend hashModel tape (path ++ [25, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_25_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_26 backend hashModel tape (path ++ [26, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_26_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_27 backend hashModel tape (path ++ [27, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_27_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_28 backend hashModel tape (path ++ [28, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_28_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_29 backend hashModel tape (path ++ [29, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_29_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_30 backend hashModel tape (path ++ [30, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_30_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_31 backend hashModel tape (path ++ [31, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_31_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_32 backend hashModel tape (path ++ [32, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_32_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_33 backend hashModel tape (path ++ [33, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_33_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_34 backend hashModel tape (path ++ [34, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_34_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_35 backend hashModel tape (path ++ [35, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_35_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_36 backend hashModel tape (path ++ [36, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_36_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_37 backend hashModel tape (path ++ [37, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_37_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_38 backend hashModel tape (path ++ [38, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_38_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_39 backend hashModel tape (path ++ [39, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_39_0 i)) ∧
  (1) = 1

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_2_0 : mxxMat3) (w_5_0 : mxxMat2) (w_6_0 : mxxMat2) (w_10_0 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_16_0 : Fin 1 → mxxMat1) (w_17_0 : Fin 1 → mxxMat1) (w_18_0 : Fin 1 → mxxMat1) (w_19_0 : Fin 1 → mxxMat1) (w_20_0 : Fin 1 → mxxMat1) (w_21_0 : Fin 1 → mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_23_0 : Fin 1 → mxxMat1) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [5]) (0) (1) w_5_0 ∧
  MxxRuntime.uniformIntervalSampleAt tape (path ++ [6]) (0) (1) w_6_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_10 backend hashModel tape (path ++ [10, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_10_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel tape (path ++ [11, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_11_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_12 backend hashModel tape (path ++ [12, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_12_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_13 backend hashModel tape (path ++ [13, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_13_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_14 backend hashModel tape (path ++ [14, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_14_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_15 backend hashModel tape (path ++ [15, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_15_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_16 backend hashModel tape (path ++ [16, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_16_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_17 backend hashModel tape (path ++ [17, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_17_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_18 backend hashModel tape (path ++ [18, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_18_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_19 backend hashModel tape (path ++ [19, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_19_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_20 backend hashModel tape (path ++ [20, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_20_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_21 backend hashModel tape (path ++ [21, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_21_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_22 backend hashModel tape (path ++ [22, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_22_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_23 backend hashModel tape (path ++ [23, i.val]) params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_23_0 i)) ∧
  (1) = 1

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat4 × ByteArray × (Fin 8389632 → Int) × mxxMat0 × Int × Unit) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_3_0 : mxxMat4 := inputs.2.1
  let w_9_0 : ByteArray := inputs.2.2.1
  let w_276_0 : Fin 8389632 → Int := inputs.2.2.2.1
  let w_287_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_317_0 : Int := inputs.2.2.2.2.2.1
  let w_1_0 : mxxMat0 := (1 : mxxMat0)
  let w_2_0 := witness.w_2_0
  let w_4_0 : mxxMat2 := MxxRuntime.matrixMul w_2_0 w_3_0
  let w_5_0 := witness.w_5_0
  let w_6_0 := witness.w_6_0
  let w_7_0 : mxxMat2 := MxxRuntime.matrixSub w_5_0 w_6_0
  let w_8_0 : mxxMat2 := MxxRuntime.matrixAdd w_4_0 w_7_0
  let w_10_0 := witness.w_10_0
  let w_11_0 := witness.w_11_0
  let w_12_0 := witness.w_12_0
  let w_13_0 := witness.w_13_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 := witness.w_16_0
  let w_17_0 := witness.w_17_0
  let w_18_0 := witness.w_18_0
  let w_19_0 := witness.w_19_0
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_22_0 := witness.w_22_0
  let w_23_0 := witness.w_23_0
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
  let w_41_0 := witness.w_41_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 := witness.w_46_0
  let w_47_0 := witness.w_47_0
  let w_48_0 := witness.w_48_0
  let w_49_0 := witness.w_49_0
  let w_50_0 := witness.w_50_0
  let w_51_0 := witness.w_51_0
  let w_52_0 := witness.w_52_0
  let w_53_0 := witness.w_53_0
  let w_54_0 := witness.w_54_0
  let w_55_0 := witness.w_55_0
  let w_56_0 := witness.w_56_0
  let w_57_0 := witness.w_57_0
  let w_58_0 := witness.w_58_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 := witness.w_61_0
  let w_62_0 := witness.w_62_0
  let w_63_0 := witness.w_63_0
  let w_64_0 := witness.w_64_0
  let w_65_0 := witness.w_65_0
  let w_66_0 := witness.w_66_0
  let w_67_0 := witness.w_67_0
  let w_68_0 := witness.w_68_0
  let w_69_0 := witness.w_69_0
  let w_70_0 := witness.w_70_0
  let w_71_0 := witness.w_71_0
  let w_72_0 := witness.w_72_0
  let w_73_0 := witness.w_73_0
  let w_74_0 := witness.w_74_0
  let w_75_0 := witness.w_75_0
  let w_76_0 := witness.w_76_0
  let w_77_0 := witness.w_77_0
  let w_78_0 := witness.w_78_0
  let w_79_0 := witness.w_79_0
  let w_80_0 := witness.w_80_0
  let w_81_0 := witness.w_81_0
  let w_82_0 := witness.w_82_0
  let w_83_0 := witness.w_83_0
  let w_84_0 := witness.w_84_0
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
  let w_102_0 := witness.w_102_0
  let w_103_0 := witness.w_103_0
  let w_104_0 := witness.w_104_0
  let w_105_0 := witness.w_105_0
  let w_106_0 := witness.w_106_0
  let w_107_0 := witness.w_107_0
  let w_108_0 := witness.w_108_0
  let w_109_0 := witness.w_109_0
  let w_110_0 := witness.w_110_0
  let w_111_0 := witness.w_111_0
  let w_112_0 := witness.w_112_0
  let w_113_0 := witness.w_113_0
  let w_114_0 := witness.w_114_0
  let w_115_0 := witness.w_115_0
  let w_116_0 := witness.w_116_0
  let w_117_0 := witness.w_117_0
  let w_118_0 := witness.w_118_0
  let w_119_0 := witness.w_119_0
  let w_120_0 := witness.w_120_0
  let w_121_0 := witness.w_121_0
  let w_122_0 := witness.w_122_0
  let w_123_0 := witness.w_123_0
  let w_124_0 := witness.w_124_0
  let w_125_0 := witness.w_125_0
  let w_126_0 := witness.w_126_0
  let w_127_0 := witness.w_127_0
  let w_128_0 := witness.w_128_0
  let w_129_0 := witness.w_129_0
  let w_130_0 := witness.w_130_0
  let w_131_0 := witness.w_131_0
  let w_132_0 := witness.w_132_0
  let w_133_0 := witness.w_133_0
  let w_134_0 := witness.w_134_0
  let w_135_0 := witness.w_135_0
  let w_136_0 := witness.w_136_0
  let w_137_0 := witness.w_137_0
  let w_138_0 := witness.w_138_0
  let w_139_0 := witness.w_139_0
  let w_140_0 := witness.w_140_0
  let w_141_0 := witness.w_141_0
  let w_142_0 := witness.w_142_0
  let w_143_0 := witness.w_143_0
  let w_144_0 := witness.w_144_0
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
  let w_162_0 := witness.w_162_0
  let w_163_0 := witness.w_163_0
  let w_164_0 := witness.w_164_0
  let w_165_0 := witness.w_165_0
  let w_166_0 := witness.w_166_0
  let w_167_0 := witness.w_167_0
  let w_168_0 := witness.w_168_0
  let w_169_0 := witness.w_169_0
  let w_170_0 := witness.w_170_0
  let w_171_0 := witness.w_171_0
  let w_172_0 := witness.w_172_0
  let w_173_0 := witness.w_173_0
  let w_174_0 := witness.w_174_0
  let w_175_0 := witness.w_175_0
  let w_176_0 := witness.w_176_0
  let w_177_0 := witness.w_177_0
  let w_178_0 := witness.w_178_0
  let w_179_0 := witness.w_179_0
  let w_180_0 := witness.w_180_0
  let w_181_0 := witness.w_181_0
  let w_182_0 := witness.w_182_0
  let w_183_0 := witness.w_183_0
  let w_184_0 := witness.w_184_0
  let w_185_0 := witness.w_185_0
  let w_186_0 := witness.w_186_0
  let w_187_0 := witness.w_187_0
  let w_188_0 := witness.w_188_0
  let w_189_0 := witness.w_189_0
  let w_190_0 := witness.w_190_0
  let w_191_0 := witness.w_191_0
  let w_192_0 := witness.w_192_0
  let w_193_0 := witness.w_193_0
  let w_194_0 := witness.w_194_0
  let w_195_0 := witness.w_195_0
  let w_196_0 := witness.w_196_0
  let w_197_0 := witness.w_197_0
  let w_198_0 := witness.w_198_0
  let w_199_0 := witness.w_199_0
  let w_200_0 := witness.w_200_0
  let w_201_0 := witness.w_201_0
  let w_202_0 := witness.w_202_0
  let w_203_0 := witness.w_203_0
  let w_204_0 := witness.w_204_0
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
  let w_222_0 := witness.w_222_0
  let w_223_0 := witness.w_223_0
  let w_224_0 := witness.w_224_0
  let w_225_0 := witness.w_225_0
  let w_226_0 := witness.w_226_0
  let w_227_0 := witness.w_227_0
  let w_228_0 := witness.w_228_0
  let w_229_0 := witness.w_229_0
  let w_230_0 := witness.w_230_0
  let w_231_0 := witness.w_231_0
  let w_232_0 := witness.w_232_0
  let w_233_0 := witness.w_233_0
  let w_234_0 := witness.w_234_0
  let w_235_0 := witness.w_235_0
  let w_236_0 := witness.w_236_0
  let w_237_0 := witness.w_237_0
  let w_238_0 := witness.w_238_0
  let w_239_0 := witness.w_239_0
  let w_240_0 := witness.w_240_0
  let w_241_0 := witness.w_241_0
  let w_242_0 := witness.w_242_0
  let w_243_0 := witness.w_243_0
  let w_244_0 := witness.w_244_0
  let w_245_0 := witness.w_245_0
  let w_246_0 := witness.w_246_0
  let w_247_0 := witness.w_247_0
  let w_248_0 := witness.w_248_0
  let w_249_0 := witness.w_249_0
  let w_250_0 := witness.w_250_0
  let w_251_0 := witness.w_251_0
  let w_252_0 := witness.w_252_0
  let w_253_0 := witness.w_253_0
  let w_254_0 := witness.w_254_0
  let w_255_0 := witness.w_255_0
  let w_256_0 := witness.w_256_0
  let w_257_0 := witness.w_257_0
  let w_258_0 := witness.w_258_0
  let w_259_0 := witness.w_259_0
  let w_260_0 := witness.w_260_0
  let w_261_0 := witness.w_261_0
  let w_262_0 := witness.w_262_0
  let w_263_0 := witness.w_263_0
  let w_264_0 := witness.w_264_0
  let w_265_0 := witness.w_265_0
  let w_266_0 := witness.w_266_0
  let w_267_0 := witness.w_267_0
  let w_268_0 := witness.w_268_0
  let w_269_0 := witness.w_269_0
  let w_270_0 := witness.w_270_0
  let w_271_0 := witness.w_271_0
  let w_272_0 := witness.w_272_0
  let w_273_0 := witness.w_273_0
  let w_274_0 := witness.w_274_0
  let w_275_0 := witness.w_275_0
  let w_277_0 := witness.w_277_0
  let w_278_0 := witness.w_278_0
  let w_279_0 := witness.w_279_0
  let w_280_0 := witness.w_280_0
  let w_281_0 := witness.w_281_0
  let w_282_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_281_0
  let w_283_0 := witness.w_283_0
  let w_284_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_283_0
  let w_285_0 : mxxMat1 := MxxRuntime.matrixAdd w_282_0 w_284_0
  let w_286_0 : mxxMat0 := (1 : mxxMat0)
  let w_288_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_286_0 w_287_0
  let w_289_0 := witness.w_289_0
  let w_290_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_288_0 w_289_0
  let w_291_0 : mxxMat1 := MxxRuntime.matrixSub w_285_0 w_290_0
  let w_292_0 := witness.w_292_0
  let w_293_0 := witness.w_293_0
  let w_294_0 : mxxMat1 := MxxRuntime.matrixSub w_292_0 w_293_0
  let w_295_0 : mxxMat1 := MxxRuntime.matrixAdd w_291_0 w_294_0
  let w_296_0 := witness.w_296_0
  let w_297_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_296_0
  let w_298_0 := witness.w_298_0
  let w_299_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_298_0
  let w_300_0 : mxxMat1 := MxxRuntime.matrixAdd w_297_0 w_299_0
  let w_301_0 : mxxMat0 := (1 : mxxMat0)
  let w_302_0 : mxxMat0 := (1 : mxxMat0)
  let w_303_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_301_0 w_302_0
  let w_304_0 := witness.w_304_0
  let w_305_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_303_0 w_304_0
  let w_306_0 : mxxMat1 := MxxRuntime.matrixSub w_300_0 w_305_0
  let w_307_0 := witness.w_307_0
  let w_308_0 := witness.w_308_0
  let w_309_0 : mxxMat1 := MxxRuntime.matrixSub w_307_0 w_308_0
  let w_310_0 : mxxMat1 := MxxRuntime.matrixAdd w_306_0 w_309_0
  let w_311_0 := witness.w_311_0
  let w_312_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_311_0
  let w_313_0 := witness.w_313_0
  let w_314_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_313_0
  let w_315_0 : mxxMat1 := MxxRuntime.matrixAdd w_312_0 w_314_0
  let w_316_0 : mxxMat0 := (1 : mxxMat0)
  let w_318_0 : Int := 2
  let w_319_0 : Int := (w_317_0 * w_318_0)
  let w_320_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_316_0 w_319_0
  let w_321_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_320_0 w_287_0
  let w_322_0 := witness.w_322_0
  let w_323_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_321_0 w_322_0
  let w_324_0 : mxxMat1 := MxxRuntime.matrixSub w_315_0 w_323_0
  let w_325_0 := witness.w_325_0
  let w_326_0 := witness.w_326_0
  let w_327_0 : mxxMat1 := MxxRuntime.matrixSub w_325_0 w_326_0
  let w_328_0 : mxxMat1 := MxxRuntime.matrixAdd w_324_0 w_327_0
  let w_329_0 := witness.w_329_0
  let w_330_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_329_0
  let w_331_0 := witness.w_331_0
  let w_332_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_331_0
  let w_333_0 : mxxMat1 := MxxRuntime.matrixAdd w_330_0 w_332_0
  let w_334_0 : mxxMat0 := (1 : mxxMat0)
  let w_335_0 : Int := 2
  let w_336_0 : Int := (w_317_0 * w_335_0)
  let w_337_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_334_0 w_336_0
  let w_338_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_337_0 w_302_0
  let w_339_0 := witness.w_339_0
  let w_340_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_338_0 w_339_0
  let w_341_0 : mxxMat1 := MxxRuntime.matrixSub w_333_0 w_340_0
  let w_342_0 := witness.w_342_0
  let w_343_0 := witness.w_343_0
  let w_344_0 : mxxMat1 := MxxRuntime.matrixSub w_342_0 w_343_0
  let w_345_0 : mxxMat1 := MxxRuntime.matrixAdd w_341_0 w_344_0
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 ∧
    generatedRoot.constraints_1 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 ∧
    generatedRoot.constraints_2 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 ∧
    generatedRoot.constraints_3 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 ∧
    generatedRoot.constraints_4 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 ∧
    generatedRoot.constraints_5 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 ∧
    generatedRoot.constraints_6 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 ∧
    generatedRoot.constraints_7 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 ∧
    generatedRoot.constraints_8 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 ∧
    generatedRoot.constraints_9 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 ∧
    generatedRoot.constraints_10 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 ∧
    generatedRoot.constraints_11 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 ∧
    generatedRoot.constraints_12 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 ∧
    generatedRoot.constraints_13 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 ∧
    generatedRoot.constraints_14 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 ∧
    generatedRoot.constraints_15 backend hashModel tape path params w_0_0 w_9_0 w_1_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 ∧
    generatedRoot.constraints_16 backend hashModel tape path params w_0_0 w_9_0 w_276_0 w_1_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_281_0 ∧
    generatedRoot.constraints_17 backend hashModel tape path w_9_0 w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 w_275_0 w_277_0 w_278_0 w_279_0 w_280_0 w_283_0 w_289_0 w_292_0 w_293_0 w_295_0 w_296_0 w_298_0 w_304_0 w_307_0 w_308_0 w_310_0 w_311_0 w_313_0 w_322_0 w_325_0 w_326_0 w_328_0 w_329_0 w_331_0 w_339_0 w_342_0 w_343_0 w_345_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : mxxMat0 × mxxMat4 × ByteArray × (Fin 8389632 → Int) × mxxMat0 × Int × Unit) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_initial
