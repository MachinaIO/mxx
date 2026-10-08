import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_encrypt

structure Params where
  unit : Unit

abbrev parallel_generatedRoot_4.constraints_0 (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_1_0 : ByteArray) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_0_0 w_8_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [12]) ((319/100)) (20) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_14_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_16_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [20]) ((319/100)) (20) w_20_0 ∧
  outputs = (w_13_0, (w_21_0, ()))

def parallel_generatedRoot_4 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.2.2.1
  let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.1
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
    let w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      let w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_3_0 w_6_0
      ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
        let w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_8_0 w_9_0
        let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_7_0 w_10_0
        ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
          let w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_11_0 w_12_0
          ∃ (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
            let w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_14_0
            ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
              let w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_4_0 w_16_0
              let w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_15_0 w_17_0
              let w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_18_0 w_9_0
              ∃ (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                let w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_19_0 w_20_0
                parallel_generatedRoot_4.constraints_0 hashModel tape path i_0 w_0_0 w_1_0 w_2_0 w_5_0 w_8_0 w_12_0 w_13_0 w_14_0 w_16_0 w_20_0 w_21_0 outputs

abbrev parallel_generatedRoot_5.constraints_1 (i_0 : Nat) (w_14_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_8_0 : Int) (w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  0 ≤ w_8_0 ∧
  w_8_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_8_0 w_15_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_15_0 w_16_0 ∧
  outputs = (w_2_0, (w_4_0, (w_11_0, (w_17_0, ()))))

abbrev parallel_generatedRoot_5.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_0_0 : ByteArray) (w_7_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_14_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_8_0 : Int) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_0_0) w_1_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_1_0 w_2_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_0_0) w_3_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_3_0 w_4_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_0_0) w_5_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_5_0 w_6_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  0 ≤ w_8_0 ∧
  w_8_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_7_0 w_8_0 w_9_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_9_0 w_10_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_0_0) w_12_0 ∧
  0 ≤ ((Int.ofNat i_0)) ∧
  ((Int.ofNat i_0)) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.ofNat i_0)) [(8193)]) w_12_0 w_13_0 ∧
  parallel_generatedRoot_5.constraints_1 i_0 w_14_0 w_2_0 w_4_0 w_8_0 w_11_0 w_15_0 w_16_0 w_17_0 outputs

def parallel_generatedRoot_5 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : ByteArray × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : ByteArray := inputs.1
  let w_7_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.1
  let w_14_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
    ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
        ∃ (w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
          ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
            ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
              let w_8_0 : Int := MxxIR.selectInt ((Int.ofNat i_0)) [(0)]
              ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                  let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_6_0 w_10_0
                  ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                    ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                      ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                        ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                          let w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_13_0 w_16_0
                          parallel_generatedRoot_5.constraints_0 hashModel i_0 w_0_0 w_7_0 w_14_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 outputs

abbrev parallel_generatedRoot_6.constraints_0 (w_2_0 : Int) (outputs : Int) : Prop :=
  (400) ≠ 0 ∧
  outputs = w_2_0

def parallel_generatedRoot_6 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (400))
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_generatedRoot_6.constraints_0 w_2_0 outputs

abbrev parallel_generatedRoot_8.constraints_3 (tape : MxxRuntime.SampleTape) (path : List Nat) (w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  MxxRuntime.gaussianSampleAt tape (path ++ [41]) ((319/100)) (20) w_41_0 ∧
  outputs = (w_33_0, (w_42_0, ()))

abbrev parallel_generatedRoot_8.constraints_2 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_2_0 : ByteArray) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2) (w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_22_0 0 1 (Int.fmod ((Int.ofNat i_0)) (20)) (1 + (Int.fmod ((Int.ofNat i_0)) (20))) w_23_0 ∧
  MxxRuntime.concatColumns w_17_0 w_23_0 w_24_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  0 ≤ (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)) ∧
  (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)) < (1 + (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10))) ∧
  (1 + (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10))) ≤ 2 ∧
  MxxRuntime.sliceMatrix w_24_0 0 1 (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)) (1 + (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10))) w_25_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 32768 10 w_25_0 w_26_0 ∧
  MxxRuntime.tensorRuns w_27_0 w_28_0 w_29_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [32]) ((319/100)) (20) w_32_0 ∧
  (400) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (400))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (400))) < 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (400))) [(8193)]), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_2_0) w_34_0 ∧
  (400) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (400))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (400))) < 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (400))) [(8193)]), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_2_0) w_36_0 ∧
  parallel_generatedRoot_8.constraints_3 tape path w_33_0 w_41_0 w_42_0 outputs

abbrev parallel_generatedRoot_8.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_2_0 : ByteArray) (w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Int) (w_14_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_18_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_20_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2) (w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  MxxRuntime.sliceMatrix w_9_0 0 1 (9 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) (19 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) w_10_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_11_0 w_12_0 w_13_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_14_0 w_12_0 w_15_0 ∧
  MxxRuntime.concatColumns w_13_0 w_15_0 w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (20)) ∧
  (Int.fmod ((Int.ofNat i_0)) (20)) < (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ≤ 20 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (Int.fmod ((Int.ofNat i_0)) (20)) (1 + (Int.fmod ((Int.ofNat i_0)) (20))) w_17_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_18_0 w_12_0 w_19_0 ∧
  0 ≤ w_12_0 ∧
  w_12_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_20_0 w_12_0 w_21_0 ∧
  MxxRuntime.concatColumns w_19_0 w_21_0 w_22_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (20)) ∧
  (Int.fmod ((Int.ofNat i_0)) (20)) < (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ≤ 20 ∧
  parallel_generatedRoot_8.constraints_2 backend hashModel tape path i_0 w_2_0 w_28_0 w_17_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_32_0 w_33_0 w_34_0 w_36_0 w_41_0 w_42_0 outputs

abbrev parallel_generatedRoot_8.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_2_0 : ByteArray) (w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Int) (w_14_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_18_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_20_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2) (w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_41_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  (400) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (400))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (400))) < 1 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (400))) [(8193)]) w_0_0 w_1_0 ∧
  (400) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (400))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (400))) < 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (400))) [(8193)]), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_2_0) w_3_0 ∧
  (400) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (400))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (400))) < 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [97, 117, 116, 111, 109, 111, 114, 112, 104, 105, 115, 109], .u64Le (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (400))) [(8193)]), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_2_0) w_6_0 ∧
  0 ≤ (9) ∧
  (9) < 19 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  0 ≤ (9 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) ∧
  (9 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) < (19 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) ∧
  (19 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) ≤ 19 ∧
  parallel_generatedRoot_8.constraints_1 backend hashModel tape path i_0 w_2_0 w_11_0 w_12_0 w_14_0 w_18_0 w_20_0 w_28_0 w_9_0 w_10_0 w_13_0 w_15_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_32_0 w_33_0 w_34_0 w_36_0 w_41_0 w_42_0 outputs

def parallel_generatedRoot_8 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Int × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_2_0 : ByteArray := inputs.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.2.2.1
  let w_11_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.1
  let w_12_0 : Int := inputs.2.2.2.2.1
  let w_14_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.1
  let w_18_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.1
  let w_20_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.1
  let w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_1_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
    ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_1_0 w_3_0
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
        let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_5_0 w_6_0
        let w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_4_0 w_7_0
        let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19 := (MxxRuntime.unitRow (9) : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19)
        ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
          ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
            ∃ (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
              ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
                ∃ (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
                  ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                    ∃ (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                      ∃ (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
                        ∃ (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
                          ∃ (w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2),
                            ∃ (w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
                              ∃ (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1),
                                let w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_10_0 w_26_0
                                ∃ (w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                                  let w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_1_0 w_29_0
                                  let w_31_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_8_0 w_30_0
                                  ∃ (w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                                    let w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_31_0 w_32_0
                                    ∃ (w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                                      let w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_1_0 w_34_0
                                      ∃ (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                                        let w_37_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_5_0 w_36_0
                                        let w_38_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_35_0 w_37_0
                                        let w_39_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_5_0 w_29_0
                                        let w_40_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_38_0 w_39_0
                                        ∃ (w_41_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                                          let w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_40_0 w_41_0
                                          parallel_generatedRoot_8.constraints_0 backend hashModel tape path i_0 w_0_0 w_2_0 w_11_0 w_12_0 w_14_0 w_18_0 w_20_0 w_28_0 w_1_0 w_3_0 w_6_0 w_9_0 w_10_0 w_13_0 w_15_0 w_16_0 w_17_0 w_19_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_29_0 w_32_0 w_33_0 w_34_0 w_36_0 w_41_0 w_42_0 outputs

abbrev parallel_generatedRoot_37.constraints_0 (w_2_0 : Int) (outputs : Int) : Prop :=
  (400) ≠ 0 ∧
  outputs = w_2_0

def parallel_generatedRoot_37 (backend : MxxRuntime.BackendContext) (_ : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (_ : Unit) (outputs : Int) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Int := (Int.fdiv ((Int.ofNat i_0)) (400))
  let w_1_0 : Int := 0
  let w_2_0 : Int := (w_0_0 + w_1_0)
  parallel_generatedRoot_37.constraints_0 w_2_0 outputs

abbrev parallel_generatedRoot_43.constraints_2 (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (400))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_28_0 ∧
  (400) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (400))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_30_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [35]) ((319/100)) (20) w_35_0 ∧
  outputs = (w_27_0, (w_36_0, ()))

abbrev parallel_generatedRoot_43.constraints_1 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2) (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1) (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  (Int.fmod ((Int.ofNat i_0)) (20)) < (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ≤ 20 ∧
  MxxRuntime.sliceMatrix w_12_0 0 1 (Int.fmod ((Int.ofNat i_0)) (20)) (1 + (Int.fmod ((Int.ofNat i_0)) (20))) w_13_0 ∧
  MxxRuntime.concatColumns w_14_0 w_15_0 w_16_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (20)) ∧
  (Int.fmod ((Int.ofNat i_0)) (20)) < (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ∧
  (1 + (Int.fmod ((Int.ofNat i_0)) (20))) ≤ 20 ∧
  MxxRuntime.sliceMatrix w_16_0 0 1 (Int.fmod ((Int.ofNat i_0)) (20)) (1 + (Int.fmod ((Int.ofNat i_0)) (20))) w_17_0 ∧
  MxxRuntime.concatColumns w_13_0 w_17_0 w_18_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  0 ≤ (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)) ∧
  (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)) < (1 + (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10))) ∧
  (1 + (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10))) ≤ 2 ∧
  MxxRuntime.sliceMatrix w_18_0 0 1 (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)) (1 + (Int.fdiv ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10))) w_19_0 ∧
  MxxRuntime.gadgetDecomposeRuns backend 32768 10 w_19_0 w_20_0 ∧
  MxxRuntime.tensorRuns w_21_0 w_22_0 w_23_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [26]) ((319/100)) (20) w_26_0 ∧
  (400) ≠ 0 ∧
  (20) ≠ 0 ∧
  parallel_generatedRoot_43.constraints_2 hashModel tape path i_0 w_1_0 w_27_0 w_28_0 w_30_0 w_35_0 w_36_0 outputs

abbrev parallel_generatedRoot_43.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (i_0 : Nat) (w_1_0 : ByteArray) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2) (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1) (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  (400) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (400))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_2_0 ∧
  (400) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [99, 111, 109, 112, 97, 110, 105, 111, 110, 45, 107, 101, 121], .bytes [114, 104, 115], .u64Le (0), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.fdiv ((Int.ofNat i_0)) (400))), .bytes [99, 111, 108, 117, 109, 110], .u64Le ((Int.fmod ((Int.ofNat i_0)) (20))), .bytes [100, 105, 103, 105, 116, 45, 114, 111, 119], .u64Le ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_5_0 ∧
  0 ≤ (9) ∧
  (9) < 19 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  (10) ≠ 0 ∧
  0 ≤ (9 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) ∧
  (9 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) < (19 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) ∧
  (19 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) ≤ 19 ∧
  MxxRuntime.sliceMatrix w_8_0 0 1 (9 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) (19 + (-1 * (Int.fmod ((Int.fmod ((Int.fdiv ((Int.ofNat i_0)) (20))) (20))) (10)))) w_9_0 ∧
  MxxRuntime.concatColumns w_10_0 w_11_0 w_12_0 ∧
  0 ≤ 0 ∧
  0 < 1 ∧
  1 ≤ 1 ∧
  (20) ≠ 0 ∧
  (20) ≠ 0 ∧
  0 ≤ (Int.fmod ((Int.ofNat i_0)) (20)) ∧
  parallel_generatedRoot_43.constraints_1 backend hashModel tape path i_0 w_1_0 w_14_0 w_15_0 w_22_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_23_0 w_26_0 w_27_0 w_28_0 w_30_0 w_35_0 w_36_0 outputs

def parallel_generatedRoot_43 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × Int × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Unit) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.2.2.1
  let _ : Int := inputs.2.2.2.1
  let w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.1
  let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.1
  let w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.1
  let w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.1
  let w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.2.1
  ∃ (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
    let w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_2_0
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      let w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_4_0 w_5_0
      let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_3_0 w_6_0
      let w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19 := (MxxRuntime.unitRow (9) : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 19)
      ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
        ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
          ∃ (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
            ∃ (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
              ∃ (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
                ∃ (w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 2),
                  ∃ (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1),
                    ∃ (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 10 1),
                      let w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_9_0 w_20_0
                      ∃ (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                        let w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_23_0
                        let w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_7_0 w_24_0
                        ∃ (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                          let w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_25_0 w_26_0
                          ∃ (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                            let w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_28_0
                            ∃ (w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                              let w_31_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_4_0 w_30_0
                              let w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_29_0 w_31_0
                              let w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_4_0 w_23_0
                              let w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_32_0 w_33_0
                              ∃ (w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
                                let w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_34_0 w_35_0
                                parallel_generatedRoot_43.constraints_0 backend hashModel tape path i_0 w_1_0 w_10_0 w_11_0 w_14_0 w_15_0 w_22_0 w_2_0 w_5_0 w_8_0 w_9_0 w_12_0 w_13_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_23_0 w_26_0 w_27_0 w_28_0 w_30_0 w_35_0 w_36_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_4_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_4_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_5_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_5_2 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_5_3 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_6_0 : Fin 400 → Int
  w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_8_0 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_8_1 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_37_0 : Fin 400 → Int
  w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_43_0 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_43_1 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_1_0 : ByteArray) (w_38_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_39_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_40_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_41_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_1 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_2 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_3 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_6_0 : Fin 400 → Int) (w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_8_0 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_8_1 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_37_0 : Fin 400 → Int) (w_42_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_43_0 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_43_1 : Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (outputs : (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Unit) : Prop :=
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_3_0 ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_4 backend hashModel tape (path ++ [4, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_3_0, ())))) ((w_4_0 i), ((w_4_1 i), ()))) ∧
  (1) = 1 ∧
  (∀ i : Fin 1, parallel_generatedRoot_5 backend hashModel tape (path ++ [5, i.val]) params i (w_1_0, (w_4_0, (w_4_1, ()))) ((w_5_0 i), ((w_5_1 i), ((w_5_2 i), ((w_5_3 i), ()))))) ∧
  (400) = 400 ∧
  (∀ i : Fin 400, parallel_generatedRoot_6 backend hashModel tape (path ++ [6, i.val]) params i () (w_6_0 i)) ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_7_0 ∧
  (400) = 400 ∧
  (∀ i : Fin 400, parallel_generatedRoot_8 backend hashModel tape (path ++ [8, i.val]) params i (w_0_0, (w_1_0, (w_2_0, (w_5_0, ((w_6_0 i), (w_5_1, (w_5_2, (w_5_3, (w_7_0, ()))))))))) ((w_8_0 i), ((w_8_1 i), ()))) ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (w_1_0) w_9_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_1_0) w_11_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_14_0 ∧
  MxxRuntime.centeredRebaseRuns w_14_0 w_15_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [25]) ((319/100)) (20) w_25_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (w_1_0) w_27_0 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [48, 49, 56, 55, 99, 101, 49, 53, 57, 49, 53, 100, 54, 51, 53, 54, 56, 48, 102, 99, 52, 53, 49, 100, 54, 56, 102, 55, 55, 101, 101, 50, 99, 51, 57, 102, 50, 50, 50, 52, 55, 48, 51, 49, 52, 56, 54, 101, 54, 97, 102, 100, 53, 50, 97, 99, 53, 53, 52, 99, 102, 57, 98, 52], .bytes [105, 110, 112, 117, 116, 45, 107, 101, 121], .u64Le (0), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_1_0) w_29_0 ∧
  MxxRuntime.gaussianSampleAt tape (path ++ [35]) ((319/100)) (20) w_35_0 ∧
  (400) = 400 ∧
  (∀ i : Fin 400, parallel_generatedRoot_37 backend hashModel tape (path ++ [37, i.val]) params i () (w_37_0 i)) ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_42_0 ∧
  (400) = 400 ∧
  (∀ i : Fin 400, ∃ w_38_0_gathered : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10, ∃ w_39_0_gathered : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10, ∃ w_40_0_gathered : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10, ∃ w_41_0_gathered : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10, 0 ≤ (w_37_0 i) ∧
  (w_37_0 i) < 1 ∧
  MxxRuntime.familyGetDynamic w_38_0 (w_37_0 i) w_38_0_gathered ∧
  0 ≤ (w_37_0 i) ∧
  (w_37_0 i) < 1 ∧
  MxxRuntime.familyGetDynamic w_39_0 (w_37_0 i) w_39_0_gathered ∧
  0 ≤ (w_37_0 i) ∧
  (w_37_0 i) < 1 ∧
  MxxRuntime.familyGetDynamic w_40_0 (w_37_0 i) w_40_0_gathered ∧
  0 ≤ (w_37_0 i) ∧
  (w_37_0 i) < 1 ∧
  MxxRuntime.familyGetDynamic w_41_0 (w_37_0 i) w_41_0_gathered ∧
  parallel_generatedRoot_43 backend hashModel tape (path ++ [43, i.val]) params i (w_0_0, (w_1_0, (w_2_0, ((w_37_0 i), (w_38_0_gathered, (w_39_0_gathered, (w_40_0_gathered, (w_41_0_gathered, (w_42_0, ()))))))))) ((w_43_0 i), ((w_43_1 i), ()))) ∧
  outputs = (w_8_0, (w_8_1, (w_4_0, (w_4_1, (w_26_0, (w_36_0, (w_43_0, (w_43_1, ()))))))))

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Int × Int × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Unit) (outputs : (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Unit) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_18_0 : Int := inputs.2.2.1
  let w_20_0 : Int := inputs.2.2.2.1
  let w_38_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.1
  let w_39_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.1
  let w_40_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.1
  let w_41_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.1
  let w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (1 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
  let w_3_0 := witness.w_3_0
  let w_4_0 := witness.w_4_0
  let w_4_1 := witness.w_4_1
  let w_5_0 := witness.w_5_0
  let w_5_1 := witness.w_5_1
  let w_5_2 := witness.w_5_2
  let w_5_3 := witness.w_5_3
  let w_6_0 := witness.w_6_0
  let w_7_0 := witness.w_7_0
  let w_8_0 := witness.w_8_0
  let w_8_1 := witness.w_8_1
  let w_9_0 := witness.w_9_0
  let w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_9_0
  let w_11_0 := witness.w_11_0
  let w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_2_0 w_11_0
  let w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_10_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 := witness.w_15_0
  let w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_15_0
  let w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (MxxRuntime.matrixPolynomial [(1)] : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
  let w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.liftInteger w_18_0
  let w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.multiplyMonomial w_19_0 w_20_0
  let w_22_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMulScalarLeft w_17_0 w_21_0
  let w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarRight w_16_0 w_22_0
  let w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_13_0 w_23_0
  let w_25_0 := witness.w_25_0
  let w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_24_0 w_25_0
  let w_27_0 := witness.w_27_0
  let w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_0_0 w_27_0
  let w_29_0 := witness.w_29_0
  let w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_2_0 w_29_0
  let w_31_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_28_0 w_30_0
  let w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarLeft w_2_0 w_15_0
  let w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixMulScalarRight w_32_0 w_22_0
  let w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixSub w_31_0 w_33_0
  let w_35_0 := witness.w_35_0
  let w_36_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := MxxRuntime.matrixAdd w_34_0 w_35_0
  let w_37_0 := witness.w_37_0
  let w_42_0 := witness.w_42_0
  let w_43_0 := witness.w_43_0
  let w_43_1 := witness.w_43_1
  generatedRoot.constraints_0 backend hashModel tape path params w_0_0 w_1_0 w_38_0 w_39_0 w_40_0 w_41_0 w_2_0 w_3_0 w_4_0 w_4_1 w_5_0 w_5_1 w_5_2 w_5_3 w_6_0 w_7_0 w_8_0 w_8_1 w_9_0 w_11_0 w_14_0 w_15_0 w_25_0 w_26_0 w_27_0 w_29_0 w_35_0 w_36_0 w_37_0 w_42_0 w_43_0 w_43_1 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Int × Int × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Unit) (outputs : (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 400 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × Unit) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_encrypt
