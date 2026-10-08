import MxxIR
import MxxRuntime

set_option maxRecDepth 16384
set_option maxHeartbeats 2000000

namespace Stage_decode

structure Params where
  unit : Unit

abbrev sequential_generatedRoot_23.constraints_0 (hashModel : MxxRuntime.HashModel) (i_0 : Nat) (w_1_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_4_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_7_0 : ByteArray) (w_15_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_2_0 : Int) (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_16_0 : Int) (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) : Prop :=
  (1) ≠ 0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_1_0 w_2_0 w_3_0 ∧
  0 ≤ w_2_0 ∧
  w_2_0 < 1 ∧
  MxxRuntime.familyGetDynamic w_4_0 w_2_0 w_5_0 ∧
  MxxRuntime.concatColumns w_3_0 w_5_0 w_6_0 ∧
  (1) ≠ 0 ∧
  0 ≤ ((Int.fmod ((Int.ofNat i_0)) (1))) ∧
  ((Int.fmod ((Int.ofNat i_0)) (1))) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [54, 98, 53, 99, 56, 55, 50, 102, 98, 52, 56, 101, 102, 97, 57, 54, 54, 100, 57, 57, 57, 52, 101, 48, 55, 56, 57, 51, 53, 97, 55, 53, 57, 55, 101, 98, 55, 48, 55, 50, 55, 57, 98, 52, 51, 100, 54, 51, 99, 54, 101, 54, 51, 53, 102, 49, 54, 102, 55, 99, 100, 50, 99, 53], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.fmod ((Int.ofNat i_0)) (1))) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (w_7_0) w_8_0 ∧
  (1) ≠ 0 ∧
  0 ≤ ((Int.fmod ((Int.ofNat i_0)) (1))) ∧
  ((Int.fmod ((Int.ofNat i_0)) (1))) < 1 ∧
  MxxRuntime.hashSample (hashModel) ([114, 110, 115, 45, 98, 103, 103, 47, 100, 121, 110, 97, 109, 105, 99]) ([.bytes [54, 98, 53, 99, 56, 55, 50, 102, 98, 52, 56, 101, 102, 97, 57, 54, 54, 100, 57, 57, 57, 52, 101, 48, 55, 56, 57, 51, 53, 97, 55, 53, 57, 55, 101, 98, 55, 48, 55, 50, 55, 57, 98, 52, 51, 100, 54, 51, 99, 54, 101, 54, 51, 53, 102, 49, 54, 102, 55, 99, 100, 50, 99, 53], .bytes [104, 101, 108, 112, 101, 114, 45, 107, 101, 121], .u64Le (MxxIR.selectInt ((Int.fmod ((Int.ofNat i_0)) (1))) [(8193)]), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (w_7_0) w_9_0 ∧
  MxxRuntime.concatColumns w_8_0 w_9_0 w_10_0 ∧
  (1) ≠ 0 ∧
  0 ≤ ((Int.fdiv ((Int.ofNat i_0)) (1))) ∧
  ((Int.fdiv ((Int.ofNat i_0)) (1))) < 2 ∧
  MxxRuntime.ringAutomorphismRuns (MxxIR.selectInt ((Int.fdiv ((Int.ofNat i_0)) (1))) [(1), (8193)]) w_11_0 w_12_0 ∧
  0 ≤ w_16_0 ∧
  w_16_0 < 2 ∧
  MxxRuntime.familyGetDynamic w_15_0 w_16_0 w_17_0 ∧
  outputs = w_19_0

def sequential_generatedRoot_23 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (_ : MxxRuntime.SampleTape) (_ : List Nat) (params : Params) (i_0 : Nat) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Unit) (outputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_1_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.1
  let w_4_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.1
  let w_7_0 : ByteArray := inputs.2.2.2.1
  let w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := inputs.2.2.2.2.1
  let w_15_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 := inputs.2.2.2.2.2.1
  let w_2_0 : Int := (Int.fmod ((Int.ofNat i_0)) (1))
  ∃ (w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
    ∃ (w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
      ∃ (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
        ∃ (w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
          ∃ (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10),
            ∃ (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
              let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := MxxRuntime.matrixSub w_6_0 w_10_0
              ∃ (w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20),
                let w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := MxxRuntime.matrixAdd w_12_0 w_13_0
                let w_16_0 : Int := (Int.ofNat i_0)
                ∃ (w_17_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1),
                  let w_18_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_14_0 w_17_0
                  let w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_0_0 w_18_0
                  sequential_generatedRoot_23.constraints_0 hashModel i_0 w_1_0 w_4_0 w_7_0 w_15_0 w_2_0 w_3_0 w_5_0 w_6_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_16_0 w_17_0 w_19_0 outputs

set_option genInjectivity false in
set_option genSizeOf false in
structure generatedRoot.Witness where
  w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12
  w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10
  w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20
  w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1
  w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20
  w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20
  w_36_0 : Fin 8192 → Int

abbrev generatedRoot.constraints_0 (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (w_1_0 : ByteArray) (w_17_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_18_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_22_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) (w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_2_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_6_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12) (w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_10_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_14_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_20_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) (w_21_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_23_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_27_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_30_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20) (w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) (w_36_0 : Fin 8192 → Int) (outputs : Fin 8192 → Int) : Prop :=
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [49, 50, 102, 48, 97, 97, 48, 101, 101, 97, 100, 56, 49, 53, 55, 100, 57, 101, 98, 57, 52, 54, 98, 100, 55, 99, 54, 98, 54, 49, 54, 49, 101, 102, 100, 52, 101, 50, 56, 100, 102, 100, 101, 54, 50, 51, 49, 98, 50, 99, 53, 100, 99, 99, 50, 51, 52, 53, 53, 54, 98, 98, 99, 99], .bytes [109, 101, 115, 115, 97, 103, 101, 45, 114, 111, 119], .u64Le (1)]) (w_1_0) w_2_0 ∧
  MxxRuntime.hashSample (hashModel) ([109, 120, 120, 45, 112, 101]) ([.bytes [49, 50, 102, 48, 97, 97, 48, 101, 101, 97, 100, 56, 49, 53, 55, 100, 57, 101, 98, 57, 52, 54, 98, 100, 55, 99, 54, 98, 54, 49, 54, 49, 101, 102, 100, 52, 101, 50, 56, 100, 102, 100, 101, 54, 50, 51, 49, 98, 50, 99, 53, 100, 99, 99, 50, 51, 52, 53, 53, 54, 98, 98, 99, 99], .bytes [97, 110, 99, 104, 111, 114, 45, 115, 101, 99, 111, 110, 100, 45, 114, 111, 119]]) (w_1_0) w_6_0 ∧
  MxxRuntime.ringAutomorphismRuns (1) w_9_0 w_10_0 ∧
  MxxRuntime.ringAutomorphismRuns (8193) w_13_0 w_14_0 ∧
  MxxRuntime.gadgetMatrixRuns backend 32768 10 w_20_0 ∧
  MxxRuntime.concatColumns w_19_0 w_20_0 w_21_0 ∧
  0 ≤ (2) ∧
  MxxIR.IterRuns (fun (i : Nat) (current next : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1) => sequential_generatedRoot_23 backend hashModel tape (path ++ [23, i]) params i (current, (w_17_0, (w_18_0, (w_1_0, (w_21_0, (w_22_0, ())))))) next) (Int.toNat (2)) w_16_0 w_23_0 ∧
  MxxRuntime.concatColumns w_25_0 w_26_0 w_27_0 ∧
  MxxRuntime.concatColumns w_28_0 w_29_0 w_30_0 ∧
  (8192) = 8192 ∧
  ∀ k : Fin 8192, MxxRuntime.thresholdDecode (2) (8192) (k.val : Int) w_35_0 (w_36_0 k) ∧
  outputs = w_36_0

abbrev generatedRoot.body (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) (outputs : Fin 8192 → Int) (witness : generatedRoot.Witness) : Prop :=
  let _params := params
  let _backend := backend
  let w_0_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := inputs.1
  let w_1_0 : ByteArray := inputs.2.1
  let w_5_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := inputs.2.2.1
  let w_8_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 := inputs.2.2.2.1
  let w_12_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 := inputs.2.2.2.2.1
  let w_17_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.1
  let w_18_0 : Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.1
  let w_22_0 : Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 := inputs.2.2.2.2.2.2.2.1
  let w_25_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.2.1
  let w_26_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.2.2.1
  let w_28_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.2.2.2.1
  let w_29_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := inputs.2.2.2.2.2.2.2.2.2.2.2.1
  let w_32_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 := inputs.2.2.2.2.2.2.2.2.2.2.2.2.1
  let w_2_0 := witness.w_2_0
  let w_3_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixSub w_0_0 w_2_0
  let w_4_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
  let w_6_0 := witness.w_6_0
  let w_7_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 := MxxRuntime.matrixSub w_5_0 w_6_0
  let w_9_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_7_0 w_8_0
  let w_10_0 := witness.w_10_0
  let w_11_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_4_0 w_10_0
  let w_13_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_7_0 w_12_0
  let w_14_0 := witness.w_14_0
  let w_15_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_11_0 w_14_0
  let w_16_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := (0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1)
  let w_19_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 := (0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10)
  let w_20_0 := witness.w_20_0
  let w_21_0 := witness.w_21_0
  let w_23_0 := witness.w_23_0
  let w_24_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_15_0 w_23_0
  let w_27_0 := witness.w_27_0
  let w_30_0 := witness.w_30_0
  let w_31_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 20 := MxxRuntime.matrixSub w_27_0 w_30_0
  let w_33_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixMul w_31_0 w_32_0
  let w_34_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixAdd w_24_0 w_33_0
  let w_35_0 : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 := MxxRuntime.matrixSub w_3_0 w_34_0
  let w_36_0 := witness.w_36_0
  generatedRoot.constraints_0 backend hashModel tape path params w_1_0 w_17_0 w_18_0 w_22_0 w_25_0 w_26_0 w_28_0 w_29_0 w_2_0 w_6_0 w_9_0 w_10_0 w_13_0 w_14_0 w_16_0 w_19_0 w_20_0 w_21_0 w_23_0 w_27_0 w_30_0 w_35_0 w_36_0 outputs

def generatedRoot (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (tape : MxxRuntime.SampleTape) (path : List Nat) (params : Params) (inputs : Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 1 × ByteArray × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 12 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 12 1 × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 1 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10) × (Fin 2 → Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1) × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 1 10 × Mxx.Primitives.ExactMatrix 1425702062627056211233337151529541201382227969 8192 20 1 × Unit) (outputs : Fin 8192 → Int) : Prop :=
  ∃ witness : generatedRoot.Witness,
    generatedRoot.body backend hashModel tape path params inputs outputs witness


end Stage_decode
