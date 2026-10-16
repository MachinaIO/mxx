import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace W_Stage_initial

structure Params where
  unit : Unit

local notation "mxxNat0" => 1809251394327365744250316798798350396453362684746139641520435240552807944993

/-- The modulus of a matrix type. -/
abbrev modulus_0 : Nat := mxxNat0

local notation "mxxMat0" => Mxx.Primitives.ExactMatrix modulus_0 8 1 1
local notation "mxxMat1" => Mxx.Primitives.ExactMatrix modulus_0 8 1 25
local notation "mxxMat2" => Mxx.Primitives.ExactMatrix modulus_0 8 1 54
local notation "mxxMat3" => Mxx.Primitives.ExactMatrix modulus_0 8 1 2
local notation "mxxMat4" => Mxx.Primitives.ExactMatrix modulus_0 8 2 54

abbrev parallel_generatedRoot_10.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(11)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_10 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11)])
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
              parallel_generatedRoot_10.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_11.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_11 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(11)])
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
            parallel_generatedRoot_11.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_12.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(13)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_12 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(13)])
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
              parallel_generatedRoot_12.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_13.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_13 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(13)])
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
            parallel_generatedRoot_13.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_14.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(15)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_14 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
              parallel_generatedRoot_14.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_15.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_15 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
            parallel_generatedRoot_15.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_16.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(3)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_16 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
              parallel_generatedRoot_16.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_17.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_17 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
            parallel_generatedRoot_17.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_18.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(5)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_18 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
              parallel_generatedRoot_18.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_19.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_19 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
            parallel_generatedRoot_19.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_20.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(7)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_20 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(7)])
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
              parallel_generatedRoot_20.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_21.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_21 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(7)])
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
            parallel_generatedRoot_21.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_22.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(9)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_22 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
              parallel_generatedRoot_22.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_23.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
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
            parallel_generatedRoot_23.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_24.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_24 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])
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
              parallel_generatedRoot_24.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_25.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 8 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_25 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (7), (9), (11), (13), (15)])
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
            parallel_generatedRoot_25.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_26.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : mxxMat0) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_9_0 : mxxMat0) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])) w_0_0 w_9_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_26 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])
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
              parallel_generatedRoot_26.constraints_0 backend hashModel i_0 w_0_0 w_1_0 w_2_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_27.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_2_0 : Int) (w_3_0 : mxxMat1) (w_6_0 : mxxMat1) (w_10_0 : mxxMat1) (w_13_0 : mxxMat1) (w_14_0 : mxxMat1) (w_16_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 4 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_3_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 108, 117, 116, 45, 97]) ([.integer w_2_0, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_6_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_10_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_13_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_14_0 ∧
  outputs = w_16_0

def parallel_generatedRoot_27 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : mxxMat0 := inputs.2.2.1
  let w_2_0 : Int := (MxxIR.selectInt ((Int.ofNat i_0)) [(1), (3), (5), (15)])
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
            parallel_generatedRoot_27.constraints_0 backend hashModel i_0 w_1_0 w_2_0 w_3_0 w_6_0 w_10_0 w_13_0 w_14_0 w_16_0 outputs

abbrev parallel_generatedRoot_29.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_13_0 : Fin 5 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_12_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_24_0 : Int) (w_27_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_33_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  w_14_0 ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_17_0 w_18_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_20_0, w_23_0] w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_27_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_30_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_31_0 ∧
  outputs = w_33_0

def parallel_generatedRoot_29 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 5 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 5 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_9_0 : Int := (Int.ofNat i_0)
      let w_10_0 : Int := 1
      let w_11_0 : Bool := decide (w_9_0 = w_10_0)
      let w_12_0 : Int := if w_11_0 then 1 else 0
      let w_14_0 : Int := 1
      let w_15_0 : Int := (w_9_0 % w_14_0)
      let w_16_0 : Int := 5
      let w_17_0 : Int := (w_15_0 * w_16_0)
      ∃ (w_18_0 : Int),
        let w_19_0 : Int := 1
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
                parallel_generatedRoot_29.constraints_0 backend hashModel i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs

abbrev parallel_generatedRoot_30.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_13_0 : Fin 5 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_12_0 : Int) (w_14_0 : Int) (w_17_0 : Int) (w_18_0 : Int) (w_20_0 : Int) (w_23_0 : Int) (w_24_0 : Int) (w_27_0 : mxxMat1) (w_30_0 : mxxMat1) (w_31_0 : mxxMat1) (w_33_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108, 45, 48]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_5_0 ∧
  w_14_0 ≠ 0 ∧
  0 ≤ w_17_0 ∧
  w_17_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_13_0 w_17_0 w_18_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 2 ∧
  2 = 2 ∧
  MxxRuntime.select w_12_0 [w_20_0, w_23_0] w_24_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_27_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_30_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_31_0 ∧
  outputs = w_33_0

def parallel_generatedRoot_30 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 5 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_13_0 : Fin 5 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_9_0 : Int := (Int.ofNat i_0)
      let w_10_0 : Int := 1
      let w_11_0 : Bool := decide (w_9_0 = w_10_0)
      let w_12_0 : Int := if w_11_0 then 1 else 0
      let w_14_0 : Int := 1
      let w_15_0 : Int := (w_9_0 % w_14_0)
      let w_16_0 : Int := 5
      let w_17_0 : Int := (w_15_0 * w_16_0)
      ∃ (w_18_0 : Int),
        let w_19_0 : Int := 1
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
                parallel_generatedRoot_30.constraints_0 backend hashModel i_0 w_1_0 w_13_0 w_2_0 w_5_0 w_12_0 w_14_0 w_17_0 w_18_0 w_20_0 w_23_0 w_24_0 w_27_0 w_30_0 w_31_0 w_33_0 outputs

abbrev parallel_generatedRoot_31.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_9_0 : Fin 5 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_11_0 : Int) (w_15_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_25_0 : mxxMat1) (w_28_0 : mxxMat1) (w_29_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  w_11_0 ≠ 0 ∧
  w_15_0 ≠ 0 ∧
  0 ≤ w_19_0 ∧
  w_19_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_19_0 w_20_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_25_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_28_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_29_0 ∧
  outputs = w_31_0

def parallel_generatedRoot_31 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 5 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_9_0 : Fin 5 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_10_0 : Int := (Int.ofNat i_0)
      let w_11_0 : Int := 4
      let w_12_0 : Int := (w_10_0 / w_11_0)
      let w_13_0 : Int := 5
      let w_14_0 : Int := (w_12_0 * w_13_0)
      let w_15_0 : Int := 4
      let w_16_0 : Int := (w_10_0 % w_15_0)
      let w_17_0 : Int := (w_14_0 + w_16_0)
      let w_18_0 : Int := 1
      let w_19_0 : Int := (w_17_0 + w_18_0)
      ∃ (w_20_0 : Int),
        let w_21_0 : Int := 1
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
              parallel_generatedRoot_31.constraints_0 backend hashModel i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs

abbrev parallel_generatedRoot_32.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : ByteArray) (w_9_0 : Fin 5 → Int) (w_2_0 : mxxMat1) (w_5_0 : mxxMat1) (w_11_0 : Int) (w_15_0 : Int) (w_19_0 : Int) (w_20_0 : Int) (w_25_0 : mxxMat1) (w_28_0 : mxxMat1) (w_29_0 : mxxMat1) (w_31_0 : mxxMat1) (outputs : mxxMat1) : Prop :=
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 108, 97, 98, 101, 108]) ([.bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i_0)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_5_0 ∧
  w_11_0 ≠ 0 ∧
  w_15_0 ≠ 0 ∧
  0 ≤ w_19_0 ∧
  w_19_0 < 5 ∧
  MxxRuntime.familyGetDynamic w_9_0 w_19_0 w_20_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_25_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_28_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_29_0 ∧
  outputs = w_31_0

def parallel_generatedRoot_32 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (i_0 : Nat) (inputs : mxxMat0 × ByteArray × mxxMat0 × (Fin 5 → Int) × Unit) (outputs : mxxMat1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : mxxMat0 := inputs.2.2.1
  let w_9_0 : Fin 5 → Int := inputs.2.2.2.1
  ∃ (w_2_0 : mxxMat1),
    let w_3_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : mxxMat1),
      let w_6_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : mxxMat1 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : mxxMat0 := (1 : mxxMat0)
      let w_10_0 : Int := (Int.ofNat i_0)
      let w_11_0 : Int := 4
      let w_12_0 : Int := (w_10_0 / w_11_0)
      let w_13_0 : Int := 5
      let w_14_0 : Int := (w_12_0 * w_13_0)
      let w_15_0 : Int := 4
      let w_16_0 : Int := (w_10_0 % w_15_0)
      let w_17_0 : Int := (w_14_0 + w_16_0)
      let w_18_0 : Int := 1
      let w_19_0 : Int := (w_17_0 + w_18_0)
      ∃ (w_20_0 : Int),
        let w_21_0 : Int := 1
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
              parallel_generatedRoot_32.constraints_0 backend hashModel i_0 w_1_0 w_9_0 w_2_0 w_5_0 w_11_0 w_15_0 w_19_0 w_20_0 w_25_0 w_28_0 w_29_0 w_31_0 outputs

abbrev generatedRoot.OutputType : Type :=
  mxxMat2 × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 1 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 8 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 2 → mxxMat1) × (Fin 4 → mxxMat1) × (Fin 4 → mxxMat1) × mxxMat1 × mxxMat1 × mxxMat1 × mxxMat1 × Unit

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
  w_24_0 : Fin 8 → mxxMat1
  w_25_0 : Fin 8 → mxxMat1
  w_26_0 : Fin 4 → mxxMat1
  w_27_0 : Fin 4 → mxxMat1
  w_29_0 : Fin 2 → mxxMat1
  w_30_0 : Fin 2 → mxxMat1
  w_31_0 : Fin 4 → mxxMat1
  w_32_0 : Fin 4 → mxxMat1
  w_33_0 : mxxMat1
  w_35_0 : mxxMat1
  w_41_0 : mxxMat1
  w_44_0 : mxxMat1
  w_45_0 : mxxMat1
  w_48_0 : mxxMat1
  w_50_0 : mxxMat1
  w_56_0 : mxxMat1
  w_59_0 : mxxMat1
  w_60_0 : mxxMat1
  w_63_0 : mxxMat1
  w_65_0 : mxxMat1
  w_74_0 : mxxMat1
  w_77_0 : mxxMat1
  w_78_0 : mxxMat1
  w_81_0 : mxxMat1
  w_83_0 : mxxMat1
  w_91_0 : mxxMat1
  w_94_0 : mxxMat1
  w_95_0 : mxxMat1

abbrev generatedRoot.constraints_2 (backend : MxxRuntime.BackendContext) (w_8_0 : mxxMat2) (w_10_0 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_16_0 : Fin 1 → mxxMat1) (w_17_0 : Fin 1 → mxxMat1) (w_18_0 : Fin 1 → mxxMat1) (w_19_0 : Fin 1 → mxxMat1) (w_20_0 : Fin 1 → mxxMat1) (w_21_0 : Fin 1 → mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_23_0 : Fin 1 → mxxMat1) (w_24_0 : Fin 8 → mxxMat1) (w_25_0 : Fin 8 → mxxMat1) (w_26_0 : Fin 4 → mxxMat1) (w_27_0 : Fin 4 → mxxMat1) (w_29_0 : Fin 2 → mxxMat1) (w_30_0 : Fin 2 → mxxMat1) (w_31_0 : Fin 4 → mxxMat1) (w_32_0 : Fin 4 → mxxMat1) (w_47_0 : mxxMat1) (w_62_0 : mxxMat1) (w_80_0 : mxxMat1) (w_91_0 : mxxMat1) (w_94_0 : mxxMat1) (w_95_0 : mxxMat1) (w_97_0 : mxxMat1) (outputs : (generatedRoot.OutputType)) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_91_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_94_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_95_0 ∧
  outputs = (w_8_0, (w_10_0, (w_11_0, (w_12_0, (w_13_0, (w_14_0, (w_15_0, (w_16_0, (w_17_0, (w_18_0, (w_19_0, (w_20_0, (w_21_0, (w_22_0, (w_23_0, (w_24_0, (w_25_0, (w_26_0, (w_27_0, (w_29_0, (w_30_0, (w_31_0, (w_32_0, (w_47_0, (w_62_0, (w_80_0, (w_97_0, ())))))))))))))))))))))))))))

abbrev generatedRoot.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_28_0 : Fin 5 → Int) (w_1_0 : mxxMat0) (w_24_0 : Fin 8 → mxxMat1) (w_25_0 : Fin 8 → mxxMat1) (w_26_0 : Fin 4 → mxxMat1) (w_27_0 : Fin 4 → mxxMat1) (w_29_0 : Fin 2 → mxxMat1) (w_30_0 : Fin 2 → mxxMat1) (w_31_0 : Fin 4 → mxxMat1) (w_32_0 : Fin 4 → mxxMat1) (w_33_0 : mxxMat1) (w_35_0 : mxxMat1) (w_41_0 : mxxMat1) (w_44_0 : mxxMat1) (w_45_0 : mxxMat1) (w_48_0 : mxxMat1) (w_50_0 : mxxMat1) (w_56_0 : mxxMat1) (w_59_0 : mxxMat1) (w_60_0 : mxxMat1) (w_63_0 : mxxMat1) (w_65_0 : mxxMat1) (w_74_0 : mxxMat1) (w_77_0 : mxxMat1) (w_78_0 : mxxMat1) (w_81_0 : mxxMat1) (w_83_0 : mxxMat1) : Prop :=
  (∀ i : Fin 8, parallel_generatedRoot_24 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_24_0 i)) ∧
  (8) = 8 ∧
  (∀ i : Fin 8, parallel_generatedRoot_25 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_25_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_26 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_26_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_27 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_27_0 i)) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_29 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, (w_28_0, ())))) (w_29_0 i)) ∧
  (2) = 2 ∧
  (∀ i : Fin 2, parallel_generatedRoot_30 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, (w_28_0, ())))) (w_30_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_31 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, (w_28_0, ())))) (w_31_0 i)) ∧
  (4) = 4 ∧
  (∀ i : Fin 4, parallel_generatedRoot_32 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, (w_28_0, ())))) (w_32_0 i)) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_33_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_35_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_41_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_44_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_45_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_48_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 111, 110, 101, 45, 112, 114, 105, 118, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_50_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_56_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_59_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_60_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_9_0) w_63_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_9_0) w_65_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 4096 25 w_74_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_77_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_78_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_9_0) w_81_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100, 105, 111, 50, 47, 115, 116, 97, 116, 101]) ([.bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_9_0) w_83_0

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (w_0_0 : mxxMat0) (w_9_0 : ByteArray) (w_1_0 : mxxMat0) (w_2_0 : mxxMat3) (w_5_0 : mxxMat2) (w_6_0 : mxxMat2) (w_10_0 : Fin 1 → mxxMat1) (w_11_0 : Fin 1 → mxxMat1) (w_12_0 : Fin 1 → mxxMat1) (w_13_0 : Fin 1 → mxxMat1) (w_14_0 : Fin 1 → mxxMat1) (w_15_0 : Fin 1 → mxxMat1) (w_16_0 : Fin 1 → mxxMat1) (w_17_0 : Fin 1 → mxxMat1) (w_18_0 : Fin 1 → mxxMat1) (w_19_0 : Fin 1 → mxxMat1) (w_20_0 : Fin 1 → mxxMat1) (w_21_0 : Fin 1 → mxxMat1) (w_22_0 : Fin 1 → mxxMat1) (w_23_0 : Fin 1 → mxxMat1) : Prop :=
  MxxRuntime.concatColumns w_0_0 w_1_0 w_2_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_5_0 ∧
  MxxRuntime.uniformIntervalSample (0) (1) w_6_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_10 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_10_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_11 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_11_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_12 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_12_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_13 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_13_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_14 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_14_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_15 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_15_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_16 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_16_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_17 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_17_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_18 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_18_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_19 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_19_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_20 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_20_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_21 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_21_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_22 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_22_0 i)) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_23 backend hashModel params i (w_0_0, (w_9_0, (w_1_0, ()))) (w_23_0 i)) ∧
  (8) = 8

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : mxxMat0 × mxxMat4 × ByteArray × (Fin 5 → Int) × mxxMat0 × Int × Unit) (outputs : (generatedRoot.OutputType)) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : mxxMat0 := inputs.1
  let w_3_0 : mxxMat4 := inputs.2.1
  let w_9_0 : ByteArray := inputs.2.2.1
  let w_28_0 : Fin 5 → Int := inputs.2.2.2.1
  let w_39_0 : mxxMat0 := inputs.2.2.2.2.1
  let w_69_0 : Int := inputs.2.2.2.2.2.1
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
  let w_29_0 := witness.w_29_0
  let w_30_0 := witness.w_30_0
  let w_31_0 := witness.w_31_0
  let w_32_0 := witness.w_32_0
  let w_33_0 := witness.w_33_0
  let w_34_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_35_0
  let w_37_0 : mxxMat1 := MxxRuntime.matrixAdd w_34_0 w_36_0
  let w_38_0 : mxxMat0 := (1 : mxxMat0)
  let w_40_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_38_0 w_39_0
  let w_41_0 := witness.w_41_0
  let w_42_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_40_0 w_41_0
  let w_43_0 : mxxMat1 := MxxRuntime.matrixSub w_37_0 w_42_0
  let w_44_0 := witness.w_44_0
  let w_45_0 := witness.w_45_0
  let w_46_0 : mxxMat1 := MxxRuntime.matrixSub w_44_0 w_45_0
  let w_47_0 : mxxMat1 := MxxRuntime.matrixAdd w_43_0 w_46_0
  let w_48_0 := witness.w_48_0
  let w_49_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_48_0
  let w_50_0 := witness.w_50_0
  let w_51_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_50_0
  let w_52_0 : mxxMat1 := MxxRuntime.matrixAdd w_49_0 w_51_0
  let w_53_0 : mxxMat0 := (1 : mxxMat0)
  let w_54_0 : mxxMat0 := (1 : mxxMat0)
  let w_55_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_53_0 w_54_0
  let w_56_0 := witness.w_56_0
  let w_57_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_55_0 w_56_0
  let w_58_0 : mxxMat1 := MxxRuntime.matrixSub w_52_0 w_57_0
  let w_59_0 := witness.w_59_0
  let w_60_0 := witness.w_60_0
  let w_61_0 : mxxMat1 := MxxRuntime.matrixSub w_59_0 w_60_0
  let w_62_0 : mxxMat1 := MxxRuntime.matrixAdd w_58_0 w_61_0
  let w_63_0 := witness.w_63_0
  let w_64_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_63_0
  let w_65_0 := witness.w_65_0
  let w_66_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_65_0
  let w_67_0 : mxxMat1 := MxxRuntime.matrixAdd w_64_0 w_66_0
  let w_68_0 : mxxMat0 := (1 : mxxMat0)
  let w_70_0 : Int := 2
  let w_71_0 : Int := (w_69_0 * w_70_0)
  let w_72_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_68_0 w_71_0
  let w_73_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_72_0 w_39_0
  let w_74_0 := witness.w_74_0
  let w_75_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_73_0 w_74_0
  let w_76_0 : mxxMat1 := MxxRuntime.matrixSub w_67_0 w_75_0
  let w_77_0 := witness.w_77_0
  let w_78_0 := witness.w_78_0
  let w_79_0 : mxxMat1 := MxxRuntime.matrixSub w_77_0 w_78_0
  let w_80_0 : mxxMat1 := MxxRuntime.matrixAdd w_76_0 w_79_0
  let w_81_0 := witness.w_81_0
  let w_82_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_0_0 w_81_0
  let w_83_0 := witness.w_83_0
  let w_84_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_1_0 w_83_0
  let w_85_0 : mxxMat1 := MxxRuntime.matrixAdd w_82_0 w_84_0
  let w_86_0 : mxxMat0 := (1 : mxxMat0)
  let w_87_0 : Int := 2
  let w_88_0 : Int := (w_69_0 * w_87_0)
  let w_89_0 : mxxMat0 := MxxRuntime.multiplyMonomial w_86_0 w_88_0
  let w_90_0 : mxxMat0 := MxxRuntime.matrixMulScalarLeft w_89_0 w_54_0
  let w_91_0 := witness.w_91_0
  let w_92_0 : mxxMat1 := MxxRuntime.matrixMulScalarLeft w_90_0 w_91_0
  let w_93_0 : mxxMat1 := MxxRuntime.matrixSub w_85_0 w_92_0
  let w_94_0 := witness.w_94_0
  let w_95_0 := witness.w_95_0
  let w_96_0 : mxxMat1 := MxxRuntime.matrixSub w_94_0 w_95_0
  let w_97_0 : mxxMat1 := MxxRuntime.matrixAdd w_93_0 w_96_0
  generatedRoot.constraints_0 backend hashModel params w_0_0 w_9_0 w_1_0 w_2_0 w_5_0 w_6_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 ∧
    generatedRoot.constraints_1 backend hashModel params w_0_0 w_9_0 w_28_0 w_1_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_35_0 w_41_0 w_44_0 w_45_0 w_48_0 w_50_0 w_56_0 w_59_0 w_60_0 w_63_0 w_65_0 w_74_0 w_77_0 w_78_0 w_81_0 w_83_0 ∧
    generatedRoot.constraints_2 backend w_8_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_30_0 w_31_0 w_32_0 w_47_0 w_62_0 w_80_0 w_91_0 w_94_0 w_95_0 w_97_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (params : Params) (inputs : mxxMat0 × mxxMat4 × ByteArray × (Fin 5 → Int) × mxxMat0 × Int × Unit) (outputs : (generatedRoot.OutputType)) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel params inputs outputs witness


end W_Stage_initial
